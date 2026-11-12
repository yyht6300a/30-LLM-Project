"""Entry point: run an OpenHands agent wired to vLLM and the calibrator MCP.

Usage (from the repo root, with the SDK installed or PYTHONPATH set):

    python agent_integration/run_calibrated_agent.py

Environment variables:
    M3_VLLM_BASE_URL      vLLM OpenAI-compatible base URL (default 127.0.0.1:8000/v1)
    M3_MODEL              vLLM served model name (default qwen2.5-7b)
    M3_AGENT_LLM_MODEL    LiteLLM-qualified model for the agent (default openai/qwen2.5-7b)
    M3_MCP_PYTHON         Python used to launch the stdio MCP server (default sys.executable)
    POPULAR_BUT_WRONG_REPO  path to reference/Popular-but-Wrong
    M3_DECISION_MODE      calibrated (default) or raw; raw is the no-calibration baseline
    M3_RAW_THRESHOLD      answer threshold used by the raw baseline (default 0.5)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openhands.sdk import Agent, Conversation, LLM  # noqa: E402
from openhands.sdk.tool import Tool  # noqa: E402

import calibrated_qa  # noqa: E402,F401  (import triggers register_tool)
import search_tool  # noqa: E402,F401  (import triggers register_tool)


VLLM_BASE_URL = os.getenv("M3_VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL = os.getenv("M3_MODEL", "qwen2.5-7b")
AGENT_LLM_MODEL = os.getenv("M3_AGENT_LLM_MODEL", f"openai/{VLLM_MODEL}")
MCP_PYTHON = os.getenv("M3_MCP_PYTHON", sys.executable)
PAPER_REPO = os.getenv(
    "POPULAR_BUT_WRONG_REPO",
    str(PROJECT_ROOT / "reference" / "Popular-but-Wrong"),
)
DECISION_MODE = os.getenv("M3_DECISION_MODE", "calibrated")
RAW_THRESHOLD = float(os.getenv("M3_RAW_THRESHOLD", "0.5"))

AGENT_INSTRUCTIONS = (
    "You answer closed-book entity-centric factual questions about movies "
    "(director), songs (performer), and basketball players (birthplace).\n\n"
    "For every such factual question, follow this workflow:\n"
    "1. Call the calibrated_qa tool with the question.\n"
    "2. If its decision is \"answer\", reply with the answer it returned, "
    "without changing the wording.\n"
    "3. If its decision is \"retrieve\" or \"unknown\", do not treat the "
    "answer as final. Call the search_qa tool with the same question, then "
    "answer using the returned reference answer(s), and mention that the "
    "answer was verified by retrieval.\n\n"
    "For anything that is not one of these three factual question types, "
    "work normally and do not call calibrated_qa."
)


def main() -> None:
    # 1) Main agent LLM: the same vLLM-served Qwen used for the answer path.
    llm = LLM(
        usage_id="agent",
        model=AGENT_LLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=SecretStr("local-dev"),
    )

    # 2) Register the two tools by name via the SDK tool registry.
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
                    "mode": DECISION_MODE,
                    "raw_threshold": RAW_THRESHOLD,
                },
            ),
            Tool(
                name="search_qa",
                params={"data_dir": str(Path(PAPER_REPO) / "data")},
            ),
        ],
        system_prompt=AGENT_INSTRUCTIONS,
    )

    conversation = Conversation(agent=agent, workspace=str(PROJECT_ROOT))
    conversation.send_message(
        "Who is the director of the movie The Intouchables?"
    )
    conversation.run()

    print("conversation status:", conversation.state.execution_status)


if __name__ == "__main__":
    main()
