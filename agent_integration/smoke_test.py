"""P4 integration smoke test with traceability and a feasibility preflight.

Every run gets a ``run_id``, timestamps per stage, an environment snapshot, and
a JSON report written to ``agent_integration/smoke_report.json`` so any failure
can be located and reproduced.

Stages:
    0_preflight       check whether each stage can run on this machine
    1_imports         openhands.sdk + P4 modules import
    2_data            local knowledge base loads and looks up correctly
    3_decision        DecisionPolicy calibrated/raw/unknown routing
    4_vllm_mcp        vLLM answer + logprobs + CalibrationClient -> MCP
    5_registration    Agent + add_runtime_tools exposes both tools
    6_e2e             full Conversation loop (needs live LLM)

Usage:
    python agent_integration/smoke_test.py              # preflight + stages 1..5
    python agent_integration/smoke_test.py --stage 6_e2e
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import traceback
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))

from pydantic import SecretStr  # noqa: E402

VLLM_BASE_URL = os.getenv("M3_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL = os.getenv("M3_MODEL", "qwen2.5-7b")
AGENT_LLM_MODEL = os.getenv("M3_AGENT_LLM_MODEL", f"openai/{VLLM_MODEL}")
MCP_PYTHON = os.getenv("M3_MCP_PYTHON", sys.executable)
PAPER_REPO = os.getenv(
    "POPULAR_BUT_WRONG_REPO",
    str(PROJECT_ROOT / "reference" / "Popular-but-Wrong"),
)
DATA_DIR = str(Path(PAPER_REPO) / "data")
REPORT_PATH = HERE / "smoke_report.json"

SMOKE_PROMPT = "SMOKE-TEST: answer factual questions via calibrated_qa."

RUN_ID = str(uuid.uuid4())
STARTED_AT = datetime.now(timezone.utc).isoformat()

REPORT: dict = {
    "run_id": RUN_ID,
    "started_at": STARTED_AT,
    "platform": platform.platform(),
    "python": sys.version.split()[0],
    "environment": {
        "M3_VLLM_BASE_URL": VLLM_BASE_URL,
        "M3_MODEL": VLLM_MODEL,
        "M3_AGENT_LLM_MODEL": AGENT_LLM_MODEL,
        "M3_MCP_PYTHON": MCP_PYTHON,
        "POPULAR_BUT_WRONG_REPO": PAPER_REPO,
        "DATA_DIR": DATA_DIR,
        "PYTHONPATH": os.getenv("PYTHONPATH"),
    },
    "stages": [],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latency_ms(started: str, finished: str) -> int:
    try:
        delta = datetime.fromisoformat(finished) - datetime.fromisoformat(started)
        return int(delta.total_seconds() * 1000)
    except ValueError:
        return -1


def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False


def _mcp_server_importable() -> bool:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return _module_importable("mcp") and _module_importable("mcp_server")


def _vllm_reachable(base_url: str) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def stage0_preflight() -> str:
    checks = {
        "openhands_sdk_importable": _module_importable("openhands.sdk"),
        "data_dir": Path(DATA_DIR).is_dir(),
        "vllm_reachable": _vllm_reachable(VLLM_BASE_URL),
        "mcp_server_importable": _mcp_server_importable(),
    }
    feasibility = {
        "stage_1_imports": checks["openhands_sdk_importable"],
        "stage_2_data": checks["data_dir"],
        "stage_3_decision": checks["openhands_sdk_importable"],
        "stage_4_vllm_mcp": checks["vllm_reachable"] and checks["mcp_server_importable"],
        "stage_5_registration": checks["openhands_sdk_importable"],
        "stage_6_e2e": (
            checks["openhands_sdk_importable"]
            and checks["vllm_reachable"]
            and checks["mcp_server_importable"]
        ),
    }
    return json.dumps({"checks": checks, "feasibility": feasibility}, ensure_ascii=False)


def stage1_imports() -> str:
    from openhands.sdk import Agent, Conversation, LLM  # noqa: F401

    import calibrated_qa  # noqa: F401
    import search_tool  # noqa: F401

    return "openhands.sdk and P4 modules import OK"


def stage2_data() -> str:
    from search_tool import KnowledgeBase

    kb = KnowledgeBase(DATA_DIR)
    recs = kb.search("Who is the director of the movie The Intouchables")
    assert recs and recs[0]["references"], "Intouchables lookup returned nothing"
    assert recs[0]["references"][0] == "Olivier Nakache", recs[0]["references"]
    kb._load()
    entity_count = sum(len(v) for v in kb._by_entity.values())
    return f"KB loaded: {len(kb._by_question)} questions, {entity_count} entity entries"


def stage3_decision() -> str:
    from calibrated_qa import DecisionPolicy

    d1, s1 = DecisionPolicy.decide(
        mode="calibrated", calibrated_probability=0.9, threshold=0.34,
        raw_confidence=0.8, raw_threshold=0.5,
    )
    assert (d1, s1) == ("answer", "calibrated"), (d1, s1)
    d2, s2 = DecisionPolicy.decide(
        mode="raw", calibrated_probability=None, threshold=None,
        raw_confidence=0.8, raw_threshold=0.5,
    )
    assert (d2, s2) == ("answer", "raw"), (d2, s2)
    d3, s3 = DecisionPolicy.decide(
        mode="calibrated", calibrated_probability=None, threshold=0.34,
        raw_confidence=0.8, raw_threshold=0.5,
    )
    assert (d3, s3) == ("unknown", "calibrated"), (d3, s3)
    return "DecisionPolicy calibrated/raw/unknown routing OK"


def stage4_vllm_mcp() -> str:
    from calibrated_qa import CalibrationClient, generate_answer

    question = "Who is the director of the movie The Intouchables"
    generation = generate_answer(
        base_url=VLLM_BASE_URL, model=VLLM_MODEL, question=question
    )
    client = CalibrationClient(
        mcp_python=MCP_PYTHON,
        project_root=str(PROJECT_ROOT),
        paper_repo=PAPER_REPO,
    )
    result = client.calibrate(
        question=question,
        answer=generation.answer,
        confidence=generation.raw_confidence,
        model=VLLM_MODEL,
        domain="movies",
    )
    assert result.get("calibrated_probability") is not None, result
    return (
        f"answer={generation.answer!r} c={generation.raw_confidence:.4f} "
        f"p={result['calibrated_probability']:.4f} "
        f"decision={result.get('decision')} artifact={result.get('artifact')}"
    )


def stage5_registration() -> str:
    from openhands.sdk import Agent, Conversation, LLM
    from openhands.sdk.tool import Tool
    from openhands.sdk.tool.registry import list_registered_tools

    import calibrated_qa  # noqa: F401  (registers calibrated_qa and, transitively, search_qa)

    llm = LLM(
        usage_id="smoke",
        model=AGENT_LLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=SecretStr("local-dev"),
    )
    agent = Agent(
        llm=llm,
        tools=[
            Tool(
                name="calibrated_qa",
                params={
                    "base_url": VLLM_BASE_URL,
                    "model": VLLM_MODEL,
                    "mcp_python": MCP_PYTHON,
                    "project_root": str(PROJECT_ROOT),
                    "paper_repo": PAPER_REPO,
                    "data_dir": DATA_DIR,
                },
            ),
        ],
        system_prompt=SMOKE_PROMPT,
    )
    conversation = Conversation(agent=agent, workspace=str(PROJECT_ROOT))
    agent._initialize(conversation.state)  # force tool resolution for the check
    names = sorted(agent._tools.keys())
    assert "calibrated_qa" in names, names
    registered = list_registered_tools()
    assert "search_qa" in registered, registered  # registered but not exposed to agent
    return f"agent tools={names}; registered={sorted(registered)}"


def stage6_e2e() -> str:
    from openhands.sdk import Agent, Conversation, LLM
    from openhands.sdk.tool import Tool

    import calibrated_qa  # noqa: F401

    llm = LLM(
        usage_id="e2e",
        model=AGENT_LLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=SecretStr("local-dev"),
    )
    agent = Agent(
        llm=llm,
        tools=[
            Tool(
                name="calibrated_qa",
                params={
                    "base_url": VLLM_BASE_URL,
                    "model": VLLM_MODEL,
                    "mcp_python": MCP_PYTHON,
                    "project_root": str(PROJECT_ROOT),
                    "paper_repo": PAPER_REPO,
                    "data_dir": DATA_DIR,
                },
            ),
        ],
        system_prompt=SMOKE_PROMPT,
    )
    conversation = Conversation(agent=agent, workspace=str(PROJECT_ROOT))
    conversation.send_message("Who is the director of the movie The Intouchables?")
    conversation.run()
    return f"conversation status={conversation.state.execution_status}"


STAGES = {
    "0_preflight": stage0_preflight,
    "1_imports": stage1_imports,
    "2_data": stage2_data,
    "3_decision": stage3_decision,
    "4_vllm_mcp": stage4_vllm_mcp,
    "5_registration": stage5_registration,
    "6_e2e": stage6_e2e,
}


def run_stage(name: str, fn) -> bool:
    started = _now()
    try:
        message = fn()
        status, error = "PASS", None
    except Exception as exc:  # noqa: BLE001
        status, message = "FAIL", f"{type(exc).__name__}: {exc}"
        error = traceback.format_exc()
    finished = _now()
    entry = {
        "stage": name,
        "status": status,
        "message": message,
        "error": error,
        "started_at": started,
        "finished_at": finished,
        "duration_ms": _latency_ms(started, finished),
    }
    REPORT["stages"].append(entry)
    print(f"[{status}] {name} ({entry['duration_ms']}ms): {message}")
    if error:
        print(error)
    return status == "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), help="Run a single stage.")
    args = parser.parse_args()

    if args.stage:
        ordered = [args.stage]
    else:
        ordered = [n for n in sorted(STAGES) if n not in {"0_preflight", "6_e2e"}]
        ordered.insert(0, "0_preflight")

    failed = False
    for name in ordered:
        if not run_stage(name, STAGES[name]):
            failed = True
            break

    REPORT["finished_at"] = _now()
    REPORT["result"] = "FAILED" if failed else "PASSED"
    REPORT_PATH.write_text(json.dumps(REPORT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {REPORT_PATH}")
    print("SMOKE FAILED" if failed else "SMOKE PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
