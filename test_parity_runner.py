"""Symmetry tests for BigLaw Bench original-side parity_runner.py vs Harbor adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
import sys

import pytest

FORK_ROOT = Path(__file__).resolve().parent
BIGLAW_ROOT = FORK_ROOT.parent
DATASETS_DIR = BIGLAW_ROOT / "datasets" / "biglawbench"
JUDGE_PATH = (
    BIGLAW_ROOT
    / "repo"
    / "src"
    / "biglawbench"
    / "src"
    / "biglawbench"
    / "task-template"
    / "tests"
    / "judge.py"
)

sys.path.insert(0, str(FORK_ROOT))
import parity_runner  # noqa: E402


def _load_harbor_judge():
    spec = importlib.util.spec_from_file_location("harbor_blb_judge", JUDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["harbor_blb_judge"] = mod
    spec.loader.exec_module(mod)
    return mod


harbor_judge = _load_harbor_judge()

EXPECTED_TASK_IDS = [
    "blb-core-004",
    "blb-core-005",
    "blb-core-019",
    "blb-core-049",
    "blb-core-060",
    "blb-core-096",
    "blb-spa-001",
    "blb-spa-002",
    "blb-spa-003",
    "blb-spa-004",
    "blb-spa-005",
    "blb-spa-006",
    "blb-spa-007",
    "blb-spa-008",
    "blb-spa-009",
    "blb-spa-010",
]


def test_all_16_tasks_present_and_prompt_identical(tmp_path):
    """For all 16 tasks, Harbor instruction.md and parity_runner prompt match byte-for-byte under /app mapping."""
    assert DATASETS_DIR.is_dir(), f"Missing generated dataset at {DATASETS_DIR}"
    tasks = parity_runner.load_tasks("full")
    assert [t["id"] for t in tasks] == EXPECTED_TASK_IDS

    local_workdir = tmp_path / "blb_local_workspace"
    for task in tasks:
        tid = task["id"]
        task_dir = DATASETS_DIR / tid
        assert task_dir.is_dir(), f"Missing Harbor task dir {task_dir}"
        harbor_instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")

        # 1) Direct /app root check
        app_prompt, app_out = parity_runner.build_symmetric_prompt(task, Path("/app"))
        assert app_prompt == harbor_instruction, f"Prompt drift on {tid}"
        expected_out = Path("/app") / ("answer.md" if task["kind"] == "core" else "answer.json")
        assert app_out == expected_out

        # 2) Local TemporaryDirectory workspace mapping check:
        #    replacing str(local_workdir) -> "/app" recovers harbor_instruction identically.
        local_prompt, local_out = parity_runner.build_symmetric_prompt(task, local_workdir)
        assert local_prompt.replace(str(local_workdir), "/app") == harbor_instruction
        assert local_out.name == expected_out.name

        # 3) Dockerfile build-time downloads and schema / ground_truth match Harbor task files
        assert not list((task_dir / "environment").rglob("*.pdf")), f"PDF should not be bundled in {tid}"
        dockerfile_text = (task_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")
        tokens = shlex.split(dockerfile_text.replace("\\\n", " "))
        downloaded_targets = [
            tokens[i + 1]
            for i in range(len(tokens) - 1)
            if tokens[i] == "-o"
        ]
        harbor_docs = sorted(
            t.removeprefix("/app/documents/")
            for t in downloaded_targets
            if t.startswith("/app/documents/")
        )
        runner_docs = sorted(d.name for d in task["docs"])
        assert runner_docs == harbor_docs
        for d in task["docs"]:
            assert d.is_file()

        if task["kind"] == "spa":
            assert "/app/schema.json" in downloaded_targets
            harbor_gt = json.loads((task_dir / "tests" / "ground_truth.json").read_text(encoding="utf-8"))
            assert task["ground_truth"] == harbor_gt


def test_core_rubrics_and_scorers_identical_on_mock_verdicts(tmp_path):
    """Core rubrics, judge prompts, and mock-verdict rewards match judge.py identically across all 6 tasks."""
    tasks = parity_runner.load_tasks("core")
    assert len(tasks) == 6

    for task in tasks:
        tid = task["id"]
        harbor_rubric = (DATASETS_DIR / tid / "tests" / "rubric.md").read_text(encoding="utf-8")
        assert task["rubric"] == harbor_rubric

        pos_r, neg_r = parity_runner.parse_rubric(task["rubric"])
        pos_j, neg_j = harbor_judge.parse_rubric(harbor_rubric)
        assert pos_r == pos_j
        assert neg_r == neg_j

        answer_text = "Sample work product citing source documents.\n"
        out_path = tmp_path / f"{tid}_answer.md"
        out_path.write_text(answer_text, encoding="utf-8")

        assert parity_runner.build_prompt(pos_r, neg_r, answer_text) == harbor_judge.build_prompt(
            pos_j, neg_j, answer_text
        )

        verdicts = [
            {"criteria": {p["id"]: True for p in pos_r}, "negatives": {n["id"]: 0 for n in neg_r}},
            {
                "criteria": {p["id"]: (i % 2 == 0) for i, p in enumerate(pos_r)},
                "negatives": {neg_r[0]["id"]: 1, neg_r[1]["id"]: 2},
            },
            {"criteria": {}, "negatives": {neg_r[0]["id"]: 10}},
            {"criteria": {pos_r[0]["id"]: True}, "negatives": {neg_r[0]["id"]: "bad"}},
        ]
        for verdict in verdicts:
            captured_prompts: list[str] = []

            def fake_judge(prompt: str, _v=verdict) -> dict:
                captured_prompts.append(prompt)
                return _v

            runner_score = parity_runner.score_task(task, out_path, judge=fake_judge)
            judge_score = harbor_judge.compute_score(pos_j, neg_j, verdict)
            assert runner_score == pytest.approx(judge_score, abs=1e-12)
            assert captured_prompts == [harbor_judge.build_prompt(pos_j, neg_j, answer_text)]


def test_spa_scorers_identical_on_mock_verdicts(tmp_path):
    """SPA deterministic matching, fallback payloads, and mock-verdict rewards match judge.py across all 10 tasks."""
    tasks = parity_runner.load_tasks("spa")
    assert len(tasks) == 10

    for task in tasks:
        tid = task["id"]
        gt = task["ground_truth"]
        harbor_gt = json.loads((DATASETS_DIR / tid / "tests" / "ground_truth.json").read_text(encoding="utf-8"))

        # Case 1: Exact ground truth (no judge call)
        out_path = tmp_path / f"{tid}_exact.json"
        out_path.write_text(json.dumps(gt), encoding="utf-8")

        def fail_judge(_p: str) -> dict:
            raise AssertionError("Judge should not be called on exact GT")

        r_exact = parity_runner.score_task(task, out_path, judge=fail_judge)
        j_exact, _ = harbor_judge.score_spa(harbor_gt, gt, judge=fail_judge)
        assert r_exact == 1.0
        assert r_exact == j_exact

        # Case 2: Normalized party dict / N/A / bullet list variations (deterministic, no judge)
        norm_pred = json.loads(json.dumps(gt))
        norm_pred["purchaser"] = {
            "name": gt["purchaser"]["name"],
            "entity_type": "" if gt["purchaser"]["entity_type"] == "N/A" else gt["purchaser"]["entity_type"],
            "jurisdiction": "none" if gt["purchaser"]["jurisdiction"] == "N/A" else gt["purchaser"]["jurisdiction"],
        }
        if isinstance(norm_pred["closing_deliverables"], list) and norm_pred["closing_deliverables"]:
            norm_pred["closing_deliverables"] = [f"- {x}" for x in reversed(norm_pred["closing_deliverables"])]
        out_norm = tmp_path / f"{tid}_norm.json"
        out_norm.write_text(json.dumps(norm_pred), encoding="utf-8")
        r_norm = parity_runner.score_task(task, out_norm, judge=fail_judge)
        j_norm, _ = harbor_judge.score_spa(harbor_gt, norm_pred, judge=fail_judge)
        assert r_norm == j_norm == 1.0

        # Case 3: Partial prediction exercising LLM fallback + missing field
        partial_pred = json.loads(json.dumps(gt))
        partial_pred["governing_law"] = ""
        partial_pred["transfer_taxes"] = "Split equally by buyer and seller."
        partial_pred["fees_and_expenses"] = "Buyer pays all fees."
        out_partial = tmp_path / f"{tid}_partial.json"
        out_partial.write_text(json.dumps(partial_pred), encoding="utf-8")

        mock_verdict = {"fields": {"transfer_taxes": True, "fees_and_expenses": False}}
        runner_prompts: list[str] = []
        judge_prompts: list[str] = []

        r_partial = parity_runner.score_task(
            task,
            out_partial,
            judge=lambda p: (runner_prompts.append(p), mock_verdict)[1],
        )
        j_partial, _ = harbor_judge.score_spa(
            harbor_gt,
            partial_pred,
            judge=lambda p: (judge_prompts.append(p), mock_verdict)[1],
        )
        assert r_partial == pytest.approx(j_partial, abs=1e-12)
        assert runner_prompts == judge_prompts

        # Case 4: Malformed / missing output
        missing_path = tmp_path / f"{tid}_missing.json"
        assert parity_runner.score_task(task, missing_path, judge=fail_judge) == 0.0
        bad_json = tmp_path / f"{tid}_bad.json"
        bad_json.write_text("{not json", encoding="utf-8")
        assert parity_runner.score_task(task, bad_json, judge=fail_judge) == 0.0


def test_cli_flags_and_env_symmetry(tmp_path):
    """CLI commands and env vars match Harbor's codex and claude-code wrappers and ppapi endpoints."""
    codex_cmd = parity_runner.build_agent_cmd("codex", "openai/gpt-5-mini", "hello", tmp_path)
    assert codex_cmd == [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--model",
        "gpt-5-mini",
        "--json",
        "--enable",
        "unified_exec",
        "--",
        "hello",
    ]

    claude_cmd = parity_runner.build_agent_cmd("claude-code", "anthropic/claude-sonnet-4-6", "hello", tmp_path)
    assert claude_cmd == [
        "claude",
        "--verbose",
        "--output-format=stream-json",
        "--permission-mode=bypassPermissions",
        "--print",
    ]

    codex_cmd_effort = parity_runner.build_agent_cmd(
        "codex", "openai/gpt-5-mini", "hello", tmp_path, reasoning_effort="low"
    )
    assert codex_cmd_effort == [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--model",
        "gpt-5-mini",
        "--json",
        "--enable",
        "unified_exec",
        "-c",
        "model_reasoning_effort=low",
        "--",
        "hello",
    ]

    codex_env = parity_runner.prepare_agent_env(
        "codex",
        "openai/gpt-5-mini",
        tmp_path / "cx",
        base_env={"OPENAI_API_KEY": "sk-test", "CODEX_API_KEY": "stale", "PATH": "/usr/bin"},
    )
    if Path("/opt/homebrew/bin").is_dir():
        assert "/opt/homebrew/bin" in codex_env["PATH"].split(":")
    assert codex_env["OPENAI_BASE_URL"] == "https://app.ppapi.ai/v1"
    assert codex_env["OPENAI_API_KEY"] == "sk-test"
    assert codex_env["CODEX_API_KEY"] == ""
    codex_home = Path(codex_env["CODEX_HOME"])
    assert json.loads((codex_home / "auth.json").read_text()) == {"OPENAI_API_KEY": "sk-test"}
    assert 'openai_base_url = "https://app.ppapi.ai/v1"' in (codex_home / "config.toml").read_text()

    claude_env = parity_runner.prepare_agent_env(
        "claude-code",
        "anthropic/claude-sonnet-4-6",
        tmp_path / "cc",
        base_env={"ANTHROPIC_AUTH_TOKEN": "tok-test", "ANTHROPIC_BASE_URL": "https://app.ppapi.ai/v1"},
    )
    assert claude_env["ANTHROPIC_BASE_URL"] == "https://app.ppapi.ai"
    assert claude_env["ANTHROPIC_MODEL"] == "claude-sonnet-4-6"
    assert claude_env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-sonnet-4-6"
    assert claude_env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "claude-sonnet-4-6"
    assert claude_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-sonnet-4-6"
    assert claude_env["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "claude-sonnet-4-6"
    assert claude_env["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-sonnet-4-6"
    assert claude_env["IS_SANDBOX"] == "1"
    assert claude_env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert claude_env["ANTHROPIC_AUTH_TOKEN"] == "tok-test"
    assert claude_env["ANTHROPIC_API_KEY"] == ""
    assert Path(claude_env["CLAUDE_CONFIG_DIR"]).is_dir()


def test_dry_run_and_mock_all_16_tasks():
    """Dry-run and mock runs succeed across all 16 tasks with 1.0 mock score."""
    tasks = parity_runner.load_tasks("full")
    for task in tasks:
        dry = parity_runner.run_single_task(task, "codex", "gpt-5-mini", dry_run=True)
        assert dry["status"] == "dry_run_ok"
        assert dry["base_url"] == "https://app.ppapi.ai/v1"

        mock = parity_runner.run_single_task(task, "mock", "gpt-5-mini", dry_run=False)
        assert mock["status"] == "mock_ok"
        assert mock["score"] == 1.0


def test_timeout_expired_still_scores_output_and_records_status(monkeypatch):
    """If subprocess.run times out after writing answer.json, parity_runner still scores it cleanly."""
    import subprocess

    spa_task = next(t for t in parity_runner.load_tasks("spa") if t["id"] == "blb-spa-001")

    def fake_run(cmd, cwd, **kwargs):
        (Path(cwd) / "answer.json").write_text(json.dumps(spa_task["ground_truth"]))
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr(parity_runner.subprocess, "run", fake_run)
    res = parity_runner.run_single_task(
        spa_task,
        "codex",
        "openai/gpt-5-mini",
        timeout_sec=1,
        reasoning_effort="low",
    )
    assert res["timed_out"] is True
    assert res["returncode"] == -1
    assert res["status"] == "timed_out"
    assert res["score"] == 1.0
