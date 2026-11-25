#!/usr/bin/env python3
"""Scenario 2 original-side parity runner for BigLaw Bench (16 public tasks).

Runs a pinned CLI agent (e.g. codex@0.159.1 or claude-code) with the symmetric
prompt, CLI flags/environment, and scorer directly against the upstream repository layout:
  - 6 BLB-Core tasks (blb-core/core-samples.csv + blb-core/documents/)
  - 10 BLB-Workflows SPA tasks (blb-workflows/spa/spa-samples.csv + schema.json + documents/)

Usage:
    python3 parity_runner.py --split full --dry-run
    python3 parity_runner.py --split full --agent mock --output-dir parity_results/mock
    python3 parity_runner.py --split full --agent codex --model gpt-5-mini --runs 3
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import concurrent.futures
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "blb-core"
SPA_DIR = REPO_ROOT / "blb-workflows" / "spa"

DEFAULT_OPENAI_BASE_URL = "https://app.ppapi.ai/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://app.ppapi.ai"
DEFAULT_JUDGE_BASE_URL = "https://app.ppapi.ai/v1"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_AGENT_TIMEOUT_SEC = 1800

POS_RE = re.compile(r"\((\d+(?:\.\d+)?)\s*points?\)\s*$", re.I)
NEG_RE = re.compile(r"^\s*(?:\d+\.\s*)?-\s*(\d+(?:\.\d+)?)\s*points?\s+for\s+(?:every|each)\s+(.+?)\s*$", re.I)
_NA_STRINGS = {"", "n/a", "na", "none", "null", "not applicable", "n.a.", "-"}
_ENTITY_TYPE_RE = re.compile(
    r"^(.*?)\b(limited liability company|unlimited liability company|aktiengesellschaft|"
    r"corporation|corp\.?|llc|ulc|l\.?l\.?c\.?|l\.?p\.?|llp|partnership|company|"
    r"individual|trust)\s*$",
    re.I,
)

SPA_FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("effective_date", ("effective_date",)),
    ("purchaser", ("purchaser",)),
    ("seller", ("seller",)),
    ("company", ("company",)),
    ("price.base_purchase_price", ("price", "base_purchase_price")),
    ("price.price_per_share", ("price", "price_per_share")),
    ("price.number_of_shares", ("price", "number_of_shares")),
    ("closing_deliverables", ("closing_deliverables",)),
    ("purchase_price_adjustments", ("purchase_price_adjustments",)),
    ("capitalization.shares_outstanding", ("capitalization", "shares_outstanding")),
    ("capitalization.pro_forma_percentage", ("capitalization", "pro_forma_percentage")),
    ("transaction_agreements", ("transaction_agreements",)),
    ("indemnification.by_buyer.survival", ("indemnification", "by_buyer", "survival")),
    ("indemnification.by_buyer.liability_cap", ("indemnification", "by_buyer", "liability_cap")),
    ("indemnification.by_buyer.basket", ("indemnification", "by_buyer", "basket")),
    ("indemnification.by_seller.survival", ("indemnification", "by_seller", "survival")),
    ("indemnification.by_seller.liability_cap", ("indemnification", "by_seller", "liability_cap")),
    ("indemnification.by_seller.basket", ("indemnification", "by_seller", "basket")),
    ("conditions_to_sellers_obligations", ("conditions_to_sellers_obligations",)),
    ("conditions_to_buyers_obligations", ("conditions_to_buyers_obligations",)),
    ("governing_law", ("governing_law",)),
    ("dispute_resolution_jurisdiction", ("dispute_resolution_jurisdiction",)),
    ("termination_rights.by_buyer", ("termination_rights", "by_buyer")),
    ("termination_rights.by_seller", ("termination_rights", "by_seller")),
    ("effect_of_termination", ("effect_of_termination",)),
    ("transfer_taxes", ("transfer_taxes",)),
    ("fees_and_expenses", ("fees_and_expenses",)),
    ("notices.to_buyer", ("notices", "to_buyer")),
    ("notices.to_seller_or_company", ("notices", "to_seller_or_company")),
]

_CLAUDE_ALIAS_ENV_VARS = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)


def _parse_single_party(line: str) -> dict[str, str]:
    line = re.sub(r'\s*-\s*the\s+"[^"]+"\s*$', "", line.strip()).rstrip(";,").strip()
    m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", line)
    if m:
        name = m.group(1).strip().rstrip(",")
        inside = re.sub(r"^a\s+", "", m.group(2).strip(), flags=re.I)
        em = _ENTITY_TYPE_RE.match(inside)
        if em:
            return {
                "name": name,
                "entity_type": em.group(2).strip() or "N/A",
                "jurisdiction": em.group(1).strip().rstrip(",") or "N/A",
            }
    return {"name": line or "N/A", "entity_type": "N/A", "jurisdiction": "N/A"}


def parse_party(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if not raw or raw.upper() == "N/A":
        return {"name": "N/A", "entity_type": "N/A", "jurisdiction": "N/A"}
    lines = [
        ln.strip()
        for ln in raw.splitlines()
        if ln.strip() and not re.match(r'^"?(buyer|seller)"?:?$', ln.strip(), re.I)
    ]
    if len(lines) == 1:
        return _parse_single_party(lines[0])
    parsed = [_parse_single_party(ln) for ln in lines]
    return {
        "name": "; ".join(lines),
        "entity_type": "; ".join(dict.fromkeys(p["entity_type"] for p in parsed if p["entity_type"] != "N/A")) or "N/A",
        "jurisdiction": "; ".join(dict.fromkeys(p["jurisdiction"] for p in parsed if p["jurisdiction"] != "N/A")) or "N/A",
    }


def parse_list_field(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw or raw.upper() == "N/A":
        return []
    return [re.sub(r"^\s*[-*\u2022]\s*", "", ln.strip()) for ln in raw.splitlines() if ln.strip()]


def parse_conditions(raw: str) -> tuple[list[str], list[str]]:
    raw = raw.strip()
    if not raw or raw.upper() == "N/A":
        return [], []
    mutual, buyer, seller = [], [], []
    bucket = None
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        hdr = s.rstrip(":").strip().lower()
        if hdr == "mutual":
            bucket = mutual
            continue
        if hdr in ("buyer", "purchaser", "conditions to obligations of buyer", "conditions to buyer's obligations"):
            bucket = buyer
            continue
        if hdr in ("seller", "conditions to obligations of seller", "conditions to seller's obligations"):
            bucket = seller
            continue
        item = re.sub(r"^\s*[-*\u2022]\s*", "", s)
        (mutual if bucket is None else bucket).append(item)
    return mutual + seller, mutual + buyer


def spa_row_to_json(row: dict[str, str]) -> dict:
    to_sellers, to_buyers = parse_conditions(row["Conditions Precedent to Closing Transaction"])
    return {
        "effective_date": row["Effective Date"].strip(),
        "purchaser": parse_party(row["Purchaser (Name + entity type)"]),
        "seller": parse_party(row["Seller (name + entity)"]),
        "company": parse_party(row["Company (Name, jurisdiction and entity type)"]),
        "price": {
            "base_purchase_price": row["Base Purchase Price"].strip(),
            "price_per_share": row["Price Per Share"].strip(),
            "number_of_shares": row["Number of Shares"].strip(),
        },
        "closing_deliverables": parse_list_field(row["Closing Deliverables List"]),
        "purchase_price_adjustments": row["Post-Closing Payments Adjustments"].strip(),
        "capitalization": {
            "shares_outstanding": row["Shares Outstanding"].strip(),
            "pro_forma_percentage": row["Pro Forma Ownership Percentage"].strip(),
        },
        "transaction_agreements": parse_list_field(row["List of Other Transaction Agreements"]),
        "indemnification": {
            "by_buyer": {
                "survival": row["Indemnification by Buyer Survival Period"].strip(),
                "liability_cap": row["Indemnification by Buyer Liability Cap"].strip(),
                "basket": row["Indemnification by Buyer Basket"].strip(),
            },
            "by_seller": {
                "survival": row["Indemnification by Seller or Company Survival Period"].strip(),
                "liability_cap": row["Indemnification by Seller or Company  Liability Cap"].strip(),
                "basket": row["Indemnification by Seller or Company  Basket"].strip(),
            },
        },
        "conditions_to_sellers_obligations": to_sellers,
        "conditions_to_buyers_obligations": to_buyers,
        "governing_law": row["Governing Law"].strip(),
        "dispute_resolution_jurisdiction": row["Dispute Resolution Jurisdiction"].strip(),
        "termination_rights": {
            "by_buyer": row["Termination Rights by Buyer"].strip(),
            "by_seller": row["Termination Rights by Seller"].strip(),
        },
        "effect_of_termination": parse_list_field(row["Effect of Termination"]),
        "transfer_taxes": row["Transfer Taxes"].strip(),
        "fees_and_expenses": row["Fees and Expenses"].strip(),
        "notices": {
            "to_buyer": row["Notices to Buyer"].strip(),
            "to_seller_or_company": row["Notices to Seller/Company"].strip(),
        },
    }


def load_tasks(split: str = "full", task_ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    tasks: list[dict] = []
    if split in ("full", "core"):
        core_text = (CORE_DIR / "core-samples.csv").read_bytes().decode("utf-8-sig")
        for r in csv.DictReader(io.StringIO(core_text)):
            tid = f"blb-core-{int(r['Number']):03d}"
            docs = [d.strip() for d in r["Document(s)"].splitlines() if d.strip() and d.strip().upper() != "N/A"]
            tasks.append({
                "id": tid,
                "kind": "core",
                "prompt": r["Prompt"].strip(),
                "rubric": r["Rubric"].strip() + "\n",
                "docs": [CORE_DIR / "documents" / d for d in docs],
            })
    if split in ("full", "spa"):
        schema_path = SPA_DIR / "schema.json"
        spa_text = (SPA_DIR / "spa-samples.csv").read_bytes().decode("utf-8-sig")
        for idx, r in enumerate(csv.DictReader(io.StringIO(spa_text)), 1):
            tid = f"blb-spa-{idx:03d}"
            doc_name = r["Documents"].strip()
            tasks.append({
                "id": tid,
                "kind": "spa",
                "doc_name": doc_name,
                "docs": [SPA_DIR / "documents" / doc_name],
                "schema_path": schema_path,
                "ground_truth": spa_row_to_json(r),
            })
    if task_ids:
        wanted = {t.lower() for t in task_ids}
        tasks = [t for t in tasks if t["id"].lower() in wanted]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def build_symmetric_prompt(task: dict, workdir: Path) -> tuple[str, Path]:
    """Build the exact same instruction wording as the Harbor adapter, rooted in workdir."""
    docs_dir = workdir / "documents"
    if task["kind"] == "core":
        out_path = workdir / "answer.md"
        if task["docs"]:
            doc_block = (
                f"The attached documents are available in `{docs_dir}/` (PDF files; `pdftotext` is installed):\n\n"
                + "\n".join(f"- `{docs_dir / d.name}`" for d in task["docs"])
                + "\n\n"
            )
        else:
            doc_block = "No documents are attached; everything needed is in the request above.\n\n"
        prompt = (
            f"{task['prompt']}\n\n{doc_block}"
            f"Write your complete final response (the work product) as plain text or Markdown to `{out_path}`. "
            "Cite the specific source documents (and pages/sections where possible) that support your statements.\n"
        )
        return prompt, out_path

    out_path = workdir / "answer.json"
    schema_dest = workdir / "schema.json"
    doc_dest = docs_dir / task["doc_name"]
    prompt = (
        "Extract all 29 deal-point fields from the Share Purchase Agreement (SPA) PDF into a single JSON file "
        f"conforming to the schema at `{schema_dest}`.\n\n"
        "Available files (`pdftotext` is installed):\n"
        f"- SPA document: `{doc_dest}`\n"
        f"- Output JSON schema: `{schema_dest}`\n\n"
        "Requirements:\n"
        f"- Populate every field in `{schema_dest}`, covering all 29 deal points (effective date, purchaser, "
        "seller, company, base purchase price, price per share, number of shares, closing deliverables, "
        "purchase price adjustments, shares outstanding, pro forma ownership percentage, other transaction "
        "agreements, buyer and seller indemnification survival/liability cap/basket, conditions to seller's "
        "and buyer's closing obligations, governing law, dispute resolution jurisdiction, buyer and seller "
        "termination rights, effect of termination, transfer taxes, fees and expenses, and buyer and "
        "seller/company notices).\n"
        '- Use `"N/A"` for string fields (or `[]` for array fields) when a term is not present or not applicable in the agreement.\n'
        f"- Write valid JSON matching `{schema_dest}` to `{out_path}`.\n"
    )
    return prompt, out_path


def parse_rubric(text: str):
    """Return (positives, negatives). positives: [{id,text,points}], negatives: [{id,text,penalty}]."""
    positives, negatives = [], []
    in_neg = False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^negative points", s, re.I):
            in_neg = True
            continue
        if in_neg:
            m = NEG_RE.match(s)
            if m:
                negatives.append({"id": f"n{len(negatives)+1}", "text": m.group(2), "penalty": float(m.group(1))})
            continue
        m = POS_RE.search(s)
        if m:
            body = POS_RE.sub("", s).strip()
            body = re.sub(r"^[a-zA-Z0-9]+\.?\s+(?=[A-Z])", "", body)
            positives.append({"id": f"p{len(positives)+1}", "text": body, "points": float(m.group(1))})
    return positives, negatives


def compute_score(positives, negatives, verdict: dict) -> float:
    total = sum(p["points"] for p in positives)
    if total <= 0:
        return 0.0
    met = verdict.get("criteria", {})
    counts = verdict.get("negatives", {})
    net = sum(p["points"] for p in positives if met.get(p["id"]) is True)
    for n in negatives:
        c = counts.get(n["id"], 0)
        try:
            c = max(0.0, float(c))
        except (TypeError, ValueError):
            c = 0.0
        net -= n["penalty"] * c
    return max(0.0, min(1.0, net / total))


compute_core_score = compute_score


def build_prompt(positives, negatives, answer: str) -> str:
    crit = "\n".join(f'{p["id"]}: {p["text"]}' for p in positives)
    neg = "\n".join(f'{n["id"]}: number of {n["text"]}' for n in negatives)
    return (
        "You are a strict grader of legal work product. Grade the RESPONSE against the rubric.\n"
        "For each affirmative criterion, answer true only if the response clearly satisfies it, else false.\n"
        "For each negative item, give the integer count of occurrences in the response "
        "(hallucination = a statement not supported by the source documents or false; "
        "extraneous/irrelevant = accurate but unnecessary or irrelevant to the task).\n"
        'Return ONLY JSON: {"criteria": {"p1": true, ...}, "negatives": {"n1": 0, ...}}\n\n'
        f"AFFIRMATIVE CRITERIA:\n{crit}\n\nNEGATIVE ITEMS:\n{neg}\n\n"
        f"RESPONSE:\n<<<\n{answer}\n>>>\n"
    )


build_core_prompt = build_prompt


def call_judge(prompt: str) -> dict:
    key = os.environ.get("JUDGE_API_KEY", "")
    base = os.environ.get("JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL).rstrip("/")
    model = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    if not key:
        raise RuntimeError("JUDGE_API_KEY is not set")
    url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0))


def _get_path(obj: dict, path: tuple[str, ...]):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _is_na(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in _NA_STRINGS
    if isinstance(val, list):
        return len(val) == 0 or all(_is_na(x) for x in val)
    if isinstance(val, dict):
        return len(val) == 0 or all(_is_na(v) for v in val.values())
    return False


def _normalize_text(val: str) -> str:
    s = val.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(\d)\.00\b", r"\1", s)
    return s.strip(" .;")


def _normalize_date(val) -> str | None:
    if not isinstance(val, str):
        return None
    s = re.sub(r"\s+", " ", val.strip())
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_list(val) -> list[str]:
    if isinstance(val, str):
        raw_items = val.splitlines()
    elif isinstance(val, list):
        raw_items = [str(x) for x in val]
    else:
        return []
    out = []
    for item in raw_items:
        cleaned = re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*", "", item.strip())
        if cleaned.lower() in _NA_STRINGS:
            continue
        norm = _normalize_text(cleaned)
        if norm:
            out.append(norm)
    return sorted(set(out))


def _party_tokens(val) -> list[str]:
    if isinstance(val, dict):
        parts = []
        for k in ("name", "entity_type", "jurisdiction"):
            v = val.get(k)
            if not _is_na(v):
                parts.append(_normalize_text(str(v)))
        text = " ".join(parts)
    else:
        text = _normalize_text(str(val))
    return sorted(set(re.findall(r"[a-z0-9]+", text)))


def _fields_match_deterministic(field_id: str, expected, actual) -> bool:
    if _is_na(expected) and _is_na(actual):
        return True
    if _is_na(expected) or _is_na(actual):
        return False

    if field_id == "effective_date":
        d_exp = _normalize_date(str(expected))
        d_act = _normalize_date(str(actual))
        if d_exp and d_act:
            return d_exp == d_act

    if field_id in ("purchaser", "seller", "company"):
        if isinstance(expected, dict) and isinstance(actual, dict):
            if all(_normalize_text(str(expected.get(k, ""))) == _normalize_text(str(actual.get(k, "")))
                   for k in ("name", "entity_type", "jurisdiction")):
                return True
        return _party_tokens(expected) == _party_tokens(actual)

    if isinstance(expected, list) or isinstance(actual, list):
        l_exp = _normalize_list(expected)
        l_act = _normalize_list(actual)
        return bool(l_exp) and l_exp == l_act

    return _normalize_text(str(expected)) == _normalize_text(str(actual))


_fields_match = _fields_match_deterministic


def build_spa_prompt(items: list[tuple[str, object, object]]) -> str:
    payload = [
        {"field": fid, "ground_truth": exp, "prediction": act}
        for fid, exp, act in items
    ]
    return (
        "You are a strict grader of legal contract extraction (Share Purchase Agreements).\n"
        "For each field below, decide whether the PREDICTION is semantically equivalent to "
        "and captures the material legal/factual substance of the GROUND_TRUTH.\n"
        "Ignore minor formatting, capitalization, punctuation, date format, or phrasing differences, "
        "and treat equivalent N/A phrasings as true. Mark false if material terms, numbers, parties, "
        "lists, or legal rights/obligations are missing, contradicted, or inaccurate.\n"
        'Return ONLY JSON: {"fields": {"<field>": true, ...}}\n\n'
        f"FIELDS TO GRADE:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
    )


def score_spa(ground_truth: dict, prediction: dict, judge=None) -> tuple[float, dict[str, bool]]:
    """Score all 29 SPA deal-point fields against ground_truth."""
    if judge is None:
        judge = call_judge
    results: dict[str, bool] = {}
    pending: list[tuple[str, object, object]] = []

    for fid, path in SPA_FIELDS:
        exp = _get_path(ground_truth, path)
        act = _get_path(prediction, path)
        if _fields_match_deterministic(fid, exp, act):
            results[fid] = True
        elif _is_na(act) and not _is_na(exp):
            results[fid] = False
        else:
            pending.append((fid, exp, act))

    if pending:
        verdict = None
        last_err = None
        prompt = build_spa_prompt(pending)
        for attempt in range(6):
            try:
                verdict = judge(prompt)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"judge attempt {attempt+1} failed: {e}", file=sys.stderr)
                if attempt < 5:
                    time.sleep(min(2 * (attempt + 1), 10))
        if verdict is None:
            raise RuntimeError(f"judge unavailable: {last_err}")
        judged_fields = verdict.get("fields", {})
        for fid, _, _ in pending:
            results[fid] = judged_fields.get(fid) is True

    score = sum(1 for v in results.values() if v) / len(SPA_FIELDS)
    return score, results


def score_task(task: dict, out_path: Path, judge=call_judge) -> float:
    if not out_path.exists():
        return 0.0
    if task["kind"] == "core":
        ans = out_path.read_text(encoding="utf-8", errors="replace")
        if not ans.strip():
            return 0.0
        pos, neg = parse_rubric(task["rubric"])
        prompt = build_prompt(pos, neg, ans)
        verdict = None
        last_err = None
        for attempt in range(6):
            try:
                verdict = judge(prompt)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"judge attempt {attempt+1} failed: {e}", file=sys.stderr)
                if attempt < 5:
                    time.sleep(min(2 * (attempt + 1), 10))
        if verdict is None:
            raise RuntimeError(f"judge unavailable: {last_err}")
        return compute_score(pos, neg, verdict)

    raw = out_path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return 0.0
    try:
        pred = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return 0.0
    if not isinstance(pred, dict):
        return 0.0
    score, _ = score_spa(task["ground_truth"], pred, judge=judge)
    return score


def resolve_cli_model(model: str) -> str:
    """Strip Harbor's optional '<provider>/' prefix, matching Codex/ClaudeCode._resolved_model_name."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def resolve_openai_base_url(raw: str | None) -> str:
    base = (raw or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    return base or DEFAULT_OPENAI_BASE_URL


def resolve_anthropic_base_url(raw: str | None) -> str:
    """Mirror Harbor with_messages_client_base_url: strip trailing /v1 for Anthropic Messages clients."""
    base = (raw or DEFAULT_ANTHROPIC_BASE_URL).strip().rstrip("/")
    if not base:
        return DEFAULT_ANTHROPIC_BASE_URL
    if base.endswith("/v1"):
        base = base.removesuffix("/v1")
    return base


def build_agent_cmd(
    agent: str,
    model: str,
    prompt: str,
    workdir: Path,  # noqa: ARG001
    reasoning_effort: str | None = None,
) -> list[str]:
    """Build the exact CLI command matching Harbor's installed codex and claude-code wrappers."""
    resolved_model = resolve_cli_model(model)
    if agent == "codex":
        cmd = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model",
            resolved_model,
            "--json",
            "--enable",
            "unified_exec",
        ]
        if reasoning_effort:
            cmd.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])
        cmd.extend(["--", prompt])
        return cmd
    if agent == "claude-code":
        # Harbor pipes instruction via stdin (`printf "%s" ... | claude ... --print`)
        return [
            "claude",
            "--verbose",
            "--output-format=stream-json",
            "--permission-mode=bypassPermissions",
            "--print",
        ]
    raise ValueError(f"Unsupported agent: {agent}")


def prepare_agent_env(
    agent: str,
    model: str,
    agent_home: Path,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build isolated env + config files matching Harbor's codex.py / claude_code.py wrappers."""
    env = dict(os.environ if base_env is None else base_env)
    homebrew_bin = "/opt/homebrew/bin"
    if Path(homebrew_bin).is_dir():
        cur_path = env.get("PATH", "")
        parts = [p for p in cur_path.split(os.pathsep) if p]
        if homebrew_bin not in parts:
            env["PATH"] = os.pathsep.join([homebrew_bin, *parts]) if parts else homebrew_bin
    resolved_model = resolve_cli_model(model)

    if agent == "codex":
        codex_home = agent_home / "codex-home"
        codex_home.mkdir(parents=True, exist_ok=True)
        openai_base_url = resolve_openai_base_url(env.get("OPENAI_BASE_URL"))
        api_key = env.get("OPENAI_API_KEY") or env.get("CODEX_API_KEY") or ""
        (codex_home / "auth.json").write_text(
            json.dumps({"OPENAI_API_KEY": api_key}, indent=2) + "\n",
            encoding="utf-8",
        )
        (codex_home / "config.toml").write_text(
            f'model_provider = "openai"\nopenai_base_url = "{openai_base_url}"\n',
            encoding="utf-8",
        )
        env["CODEX_HOME"] = str(codex_home)
        env["OPENAI_BASE_URL"] = openai_base_url
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        env["CODEX_API_KEY"] = ""
        return env

    if agent == "claude-code":
        claude_config_dir = agent_home / "claude-sessions"
        for sub in ("debug", "projects/-app", "shell-snapshots", "statsig", "todos", "skills"):
            (claude_config_dir / sub).mkdir(parents=True, exist_ok=True)
        anthropic_base_url = resolve_anthropic_base_url(env.get("ANTHROPIC_BASE_URL"))
        env["CLAUDE_CONFIG_DIR"] = str(claude_config_dir)
        env["IS_SANDBOX"] = "1"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env["ANTHROPIC_MODEL"] = resolved_model
        env["ANTHROPIC_BASE_URL"] = anthropic_base_url
        for alias_var in _CLAUDE_ALIAS_ENV_VARS:
            env.setdefault(alias_var, resolved_model)
        if env.get("ANTHROPIC_AUTH_TOKEN"):
            env["ANTHROPIC_API_KEY"] = ""
            env["CLAUDE_CODE_OAUTH_TOKEN"] = ""
        elif env.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_AUTH_TOKEN"] = ""
        return env

    raise ValueError(f"Unsupported agent: {agent}")


def run_single_task(
    task: dict,
    agent: str,
    model: str,
    dry_run: bool = False,
    timeout_sec: int = DEFAULT_AGENT_TIMEOUT_SEC,
    reasoning_effort: str | None = None,
) -> dict:
    with (
        tempfile.TemporaryDirectory(prefix=f"blb_{task['id']}_") as tmp,
        tempfile.TemporaryDirectory(prefix=f"blb_home_{task['id']}_") as tmp_home,
    ):
        workdir = Path(tmp)
        agent_home = Path(tmp_home)
        docs_dir = workdir / "documents"
        docs_dir.mkdir(parents=True)
        for doc in task["docs"]:
            if not doc.is_file():
                raise FileNotFoundError(f"Missing document for {task['id']}: {doc}")
            shutil.copy2(doc, docs_dir / doc.name)
        if task["kind"] == "spa":
            shutil.copy2(task["schema_path"], workdir / "schema.json")

        prompt, out_path = build_symmetric_prompt(task, workdir)

        if dry_run:
            effective_agent = "codex" if agent == "mock" else agent
            cmd = build_agent_cmd(
                effective_agent,
                model,
                "<PROMPT>",
                workdir,
                reasoning_effort=reasoning_effort,
            )
            agent_env = prepare_agent_env(effective_agent, model, agent_home)
            if task["kind"] == "core":
                pos, neg = parse_rubric(task["rubric"])
                assert pos and len(neg) >= 2, f"Invalid rubric in {task['id']}"
            else:
                assert len(SPA_FIELDS) == 29
            base_url_key = "OPENAI_BASE_URL" if effective_agent == "codex" else "ANTHROPIC_BASE_URL"
            return {
                "task_id": task["id"],
                "kind": task["kind"],
                "docs_count": len(task["docs"]),
                "prompt_chars": len(prompt),
                "cmd_preview": cmd,
                "base_url": agent_env.get(base_url_key),
                "status": "dry_run_ok",
            }

        if agent == "mock":
            if task["kind"] == "core":
                pos, _ = parse_rubric(task["rubric"])
                out_path.write_text("# Mock Response\n\n" + "\n".join(f"- {p['text']}" for p in pos))
                score = score_task(task, out_path, judge=lambda _p: {
                    "criteria": {p["id"]: True for p in pos},
                    "negatives": {"n1": 0, "n2": 0},
                })
            else:
                out_path.write_text(json.dumps(task["ground_truth"], indent=2))
                score = score_task(task, out_path, judge=lambda _p: {"fields": {}})
            return {"task_id": task["id"], "kind": task["kind"], "score": score, "status": "mock_ok"}

        cmd = build_agent_cmd(agent, model, prompt, workdir, reasoning_effort=reasoning_effort)
        agent_env = prepare_agent_env(agent, model, agent_home)
        t0 = time.monotonic()
        timed_out = False
        returncode = -1
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                env=agent_env,
                input=(prompt if agent == "claude-code" else None),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -1
        score = score_task(task, out_path)
        elapsed_sec = round(time.monotonic() - t0, 2)
        status = "timed_out" if timed_out else ("ok" if out_path.exists() else "missing_output")
        return {
            "task_id": task["id"],
            "kind": task["kind"],
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_sec": elapsed_sec,
            "score": score,
            "status": status,
        }


def _write_summary(args, tasks: list[dict], run_means: list[float], all_runs: list[dict]) -> Path:
    summary = {
        "split": args.split,
        "agent": args.agent,
        "model": args.model,
        "resolved_model": resolve_cli_model(args.model),
        "reasoning_effort": args.reasoning_effort,
        "timeout_sec": args.timeout,
        "judge_base_url": os.environ.get("JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL).rstrip("/"),
        "judge_model": os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        "dry_run": args.dry_run,
        "task_count": len(tasks),
        "run_means": run_means,
        "runs": all_runs,
    }
    if len(run_means) >= 2:
        mean = statistics.mean(run_means)
        sem = statistics.stdev(run_means) / (len(run_means) ** 0.5)
        summary["mean"] = mean
        summary["sample_sem"] = sem
    out_file = args.output_dir / ("dry_run.json" if args.dry_run else "parity_summary.json")
    out_file.write_text(json.dumps(summary, indent=2))
    return out_file


def main() -> int:
    ap = argparse.ArgumentParser(description="BigLaw Bench Scenario 2 original-side parity runner")
    ap.add_argument("--split", choices=["full", "core", "spa"], default="full")
    ap.add_argument("--task-ids", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--agent", choices=["codex", "claude-code", "mock"], default="codex")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=DEFAULT_AGENT_TIMEOUT_SEC)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "parity_results")
    ap.add_argument("--dry-run", action="store_true", help="Validate tasks/prompts without invoking agent or judge")
    args = ap.parse_args()

    tasks = load_tasks(split=args.split, task_ids=args.task_ids, limit=args.limit)
    print(
        f"Loaded {len(tasks)} tasks (split={args.split}, agent={args.agent}, model={args.model}, "
        f"reasoning_effort={args.reasoning_effort}, concurrency={args.concurrency}, dry_run={args.dry_run})",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_means = []
    all_runs = []
    for run_idx in range(1, args.runs + 1):
        if args.concurrency > 1 and not args.dry_run and len(tasks) > 1:
            results_by_idx: dict[int, dict] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                fut_map = {
                    pool.submit(
                        run_single_task,
                        t,
                        args.agent,
                        args.model,
                        dry_run=False,
                        timeout_sec=args.timeout,
                        reasoning_effort=args.reasoning_effort,
                    ): (idx, t["id"])
                    for idx, t in enumerate(tasks)
                }
                for fut in concurrent.futures.as_completed(fut_map):
                    idx, tid = fut_map[fut]
                    res = fut.result()
                    results_by_idx[idx] = res
                    print(
                        f"  [run {run_idx}/{args.runs}] {tid}: status={res['status']} score={res['score']:.4f} elapsed={res.get('elapsed_sec')}s",
                        flush=True,
                    )
            results = [results_by_idx[i] for i in range(len(tasks))]
        else:
            results = []
            for t in tasks:
                res = run_single_task(
                    t,
                    args.agent,
                    args.model,
                    dry_run=args.dry_run,
                    timeout_sec=args.timeout,
                    reasoning_effort=args.reasoning_effort,
                )
                results.append(res)
                if not args.dry_run:
                    print(
                        f"  [run {run_idx}/{args.runs}] {t['id']}: status={res['status']} score={res['score']:.4f} elapsed={res.get('elapsed_sec')}s",
                        flush=True,
                    )
        all_runs.append({"run": run_idx, "results": results})
        if not args.dry_run:
            mean_score = sum(r["score"] for r in results) / len(results)
            run_means.append(mean_score)
            print(f"Run {run_idx}/{args.runs}: mean={mean_score:.4f} ({len(results)} tasks)", flush=True)
        else:
            print(f"Dry-run {run_idx}/{args.runs}: validated {len(results)} tasks", flush=True)
        out_file = _write_summary(args, tasks, run_means, all_runs)

    if len(run_means) >= 2:
        mean = statistics.mean(run_means)
        sem = statistics.stdev(run_means) / (len(run_means) ** 0.5)
        print(f"Aggregate: {mean:.4f} +/- {sem:.4f}", flush=True)
    print(f"Wrote {out_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
