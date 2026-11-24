#!/usr/bin/env python3
"""Scenario 2 original-side parity runner for BigLaw Bench (16 public tasks).

Runs a pinned CLI agent (e.g. codex@0.159.1 or claude-code) with the symmetric
prompt and scorer directly against the upstream repository layout:
  - 6 BLB-Core tasks (blb-core/core-samples.csv + blb-core/documents/)
  - 10 BLB-Workflows SPA tasks (blb-workflows/spa/spa-samples.csv + schema.json + documents/)

Usage:
    python3 parity_runner.py --split full --dry-run
    python3 parity_runner.py --split full --agent mock --output-dir parity_results/mock
    python3 parity_runner.py --split full --agent codex --model gpt-5-mini --runs 3
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "blb-core"
SPA_DIR = REPO_ROOT / "blb-workflows" / "spa"

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
        with open(CORE_DIR / "core-samples.csv", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
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
        with open(SPA_DIR / "spa-samples.csv", encoding="utf-8-sig") as f:
            for idx, r in enumerate(csv.DictReader(f), 1):
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
            "Cite the specific source documents (and pages/sections where possible) that support your statements."
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
            body = re.sub(r"^[a-zA-Z0-9]+\.?\s+(?=[A-Z])", "", POS_RE.sub("", s).strip())
            positives.append({"id": f"p{len(positives)+1}", "text": body, "points": float(m.group(1))})
    return positives, negatives


def compute_core_score(positives, negatives, verdict: dict) -> float:
    total = sum(p["points"] for p in positives)
    if total <= 0:
        return 0.0
    met = verdict.get("criteria", {})
    counts = verdict.get("negatives", {})
    net = sum(p["points"] for p in positives if met.get(p["id"]) is True)
    for n in negatives:
        try:
            c = max(0.0, float(counts.get(n["id"], 0)))
        except (TypeError, ValueError):
            c = 0.0
        net -= n["penalty"] * c
    return max(0.0, min(1.0, net / total))


def call_judge(prompt: str) -> dict:
    key = os.environ.get("JUDGE_API_KEY", "")
    base = os.environ.get("JUDGE_BASE_URL", "https://app.ppapi.ai/v1").rstrip("/")
    model = os.environ.get("JUDGE_MODEL", "gpt-5-mini")
    if not key:
        raise RuntimeError("JUDGE_API_KEY is not set")
    url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    return json.loads(m.group(0))


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


def _fields_match(fid: str, exp, act) -> bool:
    if _is_na(exp) and _is_na(act):
        return True
    if _is_na(exp) or _is_na(act):
        return False
    if fid == "effective_date":
        d1, d2 = _normalize_date(str(exp)), _normalize_date(str(act))
        if d1 and d2:
            return d1 == d2
    if fid in ("purchaser", "seller", "company"):
        t1 = sorted(set(re.findall(r"[a-z0-9]+", _normalize_text(json.dumps(exp) if isinstance(exp, dict) else str(exp)))))
        t2 = sorted(set(re.findall(r"[a-z0-9]+", _normalize_text(json.dumps(act) if isinstance(act, dict) else str(act)))))
        return t1 == t2
    if isinstance(exp, list) or isinstance(act, list):
        l1 = sorted({ _normalize_text(re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*", "", str(x))) for x in (exp if isinstance(exp, list) else str(exp).splitlines()) if str(x).strip().lower() not in _NA_STRINGS })
        l2 = sorted({ _normalize_text(re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*", "", str(x))) for x in (act if isinstance(act, list) else str(act).splitlines()) if str(x).strip().lower() not in _NA_STRINGS })
        return bool(l1) and l1 == l2
    return _normalize_text(str(exp)) == _normalize_text(str(act))


def score_task(task: dict, out_path: Path, judge=call_judge) -> float:
    if not out_path.exists():
        return 0.0
    if task["kind"] == "core":
        ans = out_path.read_text(encoding="utf-8", errors="replace")
        if not ans.strip():
            return 0.0
        pos, neg = parse_rubric(task["rubric"])
        crit = "\n".join(f'{p["id"]}: {p["text"]}' for p in pos)
        neg_s = "\n".join(f'{n["id"]}: number of {n["text"]}' for n in neg)
        prompt = (
            "You are a strict grader of legal work product. Grade the RESPONSE against the rubric.\n"
            'Return ONLY JSON: {"criteria": {"p1": true, ...}, "negatives": {"n1": 0, ...}}\n\n'
            f"AFFIRMATIVE CRITERIA:\n{crit}\n\nNEGATIVE ITEMS:\n{neg_s}\n\nRESPONSE:\n<<<\n{ans}\n>>>\n"
        )
        verdict = judge(prompt)
        return compute_core_score(pos, neg, verdict)

    try:
        pred = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0.0
    if not isinstance(pred, dict):
        return 0.0
    gt = task["ground_truth"]
    results = {}
    pending = []
    for fid, path in SPA_FIELDS:
        exp = gt
        for k in path:
            exp = exp.get(k) if isinstance(exp, dict) else None
        act = pred
        for k in path:
            act = act.get(k) if isinstance(act, dict) else None
        if _fields_match(fid, exp, act):
            results[fid] = True
        elif _is_na(act) and not _is_na(exp):
            results[fid] = False
        else:
            pending.append((fid, exp, act))
    if pending:
        payload = [{"field": fid, "ground_truth": exp, "prediction": act} for fid, exp, act in pending]
        prompt = (
            "You are a strict grader of legal contract extraction (Share Purchase Agreements).\n"
            'Return ONLY JSON: {"fields": {"<field>": true, ...}}\n\n'
            f"FIELDS TO GRADE:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
        )
        verdict = judge(prompt)
        jfields = verdict.get("fields", {})
        for fid, _, _ in pending:
            results[fid] = jfields.get(fid) is True
    return sum(1 for v in results.values() if v) / len(SPA_FIELDS)


def build_agent_cmd(agent: str, model: str, prompt: str, workdir: Path) -> list[str]:
    if agent == "codex":
        return [
            "codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "-C", str(workdir),
            "-m", model,
            prompt,
        ]
    if agent == "claude-code":
        return [
            "claude", "-p", prompt,
            "--dangerously-skip-permissions",
            "--model", model,
        ]
    raise ValueError(f"Unsupported agent: {agent}")


def run_single_task(task: dict, agent: str, model: str, dry_run: bool = False) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"blb_{task['id']}_") as tmp:
        workdir = Path(tmp)
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
            cmd = build_agent_cmd("codex" if agent == "mock" else agent, model, "<PROMPT>", workdir)
            if task["kind"] == "core":
                pos, neg = parse_rubric(task["rubric"])
                assert pos and len(neg) >= 2, f"Invalid rubric in {task['id']}"
            else:
                assert len(SPA_FIELDS) == 29
            return {
                "task_id": task["id"],
                "kind": task["kind"],
                "docs_count": len(task["docs"]),
                "prompt_chars": len(prompt),
                "cmd_preview": cmd,
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

        cmd = build_agent_cmd(agent, model, prompt, workdir)
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=1800)
        score = score_task(task, out_path)
        return {
            "task_id": task["id"],
            "kind": task["kind"],
            "returncode": proc.returncode,
            "score": score,
            "status": "ok" if out_path.exists() else "missing_output",
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="BigLaw Bench Scenario 2 original-side parity runner")
    ap.add_argument("--split", choices=["full", "core", "spa"], default="full")
    ap.add_argument("--task-ids", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--agent", choices=["codex", "claude-code", "mock"], default="codex")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "parity_results")
    ap.add_argument("--dry-run", action="store_true", help="Validate tasks/prompts without invoking agent or judge")
    args = ap.parse_args()

    tasks = load_tasks(split=args.split, task_ids=args.task_ids, limit=args.limit)
    print(f"Loaded {len(tasks)} tasks (split={args.split}, agent={args.agent}, model={args.model}, dry_run={args.dry_run})")

    run_means = []
    all_runs = []
    for run_idx in range(1, args.runs + 1):
        results = [run_single_task(t, args.agent, args.model, dry_run=args.dry_run) for t in tasks]
        all_runs.append({"run": run_idx, "results": results})
        if not args.dry_run:
            mean_score = sum(r["score"] for r in results) / len(results)
            run_means.append(mean_score)
            print(f"Run {run_idx}/{args.runs}: mean={mean_score:.4f} ({len(results)} tasks)")
        else:
            print(f"Dry-run {run_idx}/{args.runs}: validated {len(results)} tasks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split,
        "agent": args.agent,
        "model": args.model,
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
        print(f"Aggregate: {mean:.4f} +/- {sem:.4f}")
    out_file = args.output_dir / ("dry_run.json" if args.dry_run else "parity_summary.json")
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
