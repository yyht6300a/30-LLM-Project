"""OpenHands runtime tool: answer a factual question with calibrated confidence.

The executor is split into small, independently replaceable layers so the final
ablation study can toggle each one without touching the others:

    Layer 0  confidence computation  (pure function, no I/O)
    Layer 1  answer generation       (vLLM, owns token logprobs)
    Layer 2  calibration client      (stdio MCP ``calibrate`` tool)
    Layer 3  decision policy         (pure function, ablation-toggleable)

Every result carries a ``trace`` with request id, model, domain, artifact,
feature values/coverage, decision source and timing so each decision can be
located and reproduced.
"""
from __future__ import annotations

import math
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

import anyio
from mcp import Client, StdioServerParameters
from openai import OpenAI
from pydantic import Field

from openhands.sdk.llm import TextContent
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from openhands.sdk.tool.registry import register_tool

if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_REPO = PROJECT_ROOT / "reference" / "Popular-but-Wrong"

SPECIAL_TOKENS = {"<|im_end|>", "<|endoftext|>"}

_SYSTEM_PROMPT = (
    "Answer with only the requested entity name. Do not explain your answer."
)

_DESCRIPTION = (
    "Answer a factual question and return a calibrated confidence for the "
    "answer. Use this tool for entity-centric questions such as the director "
    "of a movie, the performer of a song, or the birthplace of a basketball "
    "player. It returns the answer, the raw confidence, the calibrated "
    "probability, and a decision (answer / retrieve / unknown). When the "
    "decision is retrieve or unknown, do not present the answer as final; "
    "search for confirmation instead."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Layer 0: pure confidence computation
# --------------------------------------------------------------------------
def calculate_confidence(logprob_content) -> tuple[float, list]:
    """Compute the paper's raw confidence from per-token logprobs."""
    answer_tokens = [
        item for item in logprob_content if item.token not in SPECIAL_TOKENS
    ]
    if not answer_tokens:
        raise ValueError("No answer tokens with logprobs returned")
    probabilities = [math.exp(item.logprob) for item in answer_tokens]
    confidence = sum(probabilities) / len(probabilities)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence outside [0, 1]: {confidence}")
    return confidence, answer_tokens


# --------------------------------------------------------------------------
# Layer 1: answer generation (vLLM, owns logprobs)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationResult:
    """Locatable answer-generation result, independent of calibration."""

    question: str
    answer: str
    raw_confidence: float
    model: str
    answer_token_count: int


def generate_answer(
    *, base_url: str, model: str, question: str
) -> GenerationResult:
    """Call vLLM for an answer-only generation with logprobs."""
    client = OpenAI(base_url=base_url, api_key="local-dev")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        top_p=1,
        max_tokens=32,
        logprobs=True,
    )

    choice = response.choices[0]
    answer = choice.message.content
    if not answer:
        raise ValueError("vLLM returned an empty answer")
    if choice.logprobs is None or not choice.logprobs.content:
        raise ValueError("vLLM did not return logprobs")

    confidence, answer_tokens = calculate_confidence(choice.logprobs.content)
    return GenerationResult(
        question=question,
        answer=answer,
        raw_confidence=confidence,
        model=model,
        answer_token_count=len(answer_tokens),
    )


# --------------------------------------------------------------------------
# Layer 2: calibration client (stdio MCP calibrate tool)
# --------------------------------------------------------------------------
class CalibrationClient:
    """Thin client for the stdio MCP ``calibrate`` tool."""

    def __init__(
        self,
        *,
        mcp_python: str | None = None,
        project_root: str | Path | None = None,
        paper_repo: str | Path | None = None,
    ) -> None:
        self.mcp_python = mcp_python or sys.executable
        self.project_root = str(project_root or PROJECT_ROOT)
        self.paper_repo = str(paper_repo or DEFAULT_PAPER_REPO)

    def calibrate(
        self,
        *,
        question: str,
        answer: str,
        confidence: float,
        model: str,
        domain: str | None,
    ) -> dict:
        """Invoke ``calibrate`` and return its structured content."""

        async def _run() -> dict:
            server = StdioServerParameters(
                command=self.mcp_python,
                args=["-m", "mcp_server.server"],
                env={
                    "PYTHONPATH": self.project_root,
                    "POPULAR_BUT_WRONG_REPO": self.paper_repo,
                },
            )
            async with Client(server) as client:
                result = await client.call_tool(
                    "calibrate",
                    arguments={
                        "question": question,
                        "answer": answer,
                        "confidence": confidence,
                        "model": model,
                        **({"domain": domain} if domain else {}),
                    },
                )
                if result.is_error:
                    messages = [
                        getattr(block, "text", repr(block))
                        for block in result.content
                    ]
                    raise RuntimeError(
                        "MCP calibrate returned tool error: " + " ".join(messages)
                    )
                if result.structured_content is None:
                    raise RuntimeError("MCP result has no structured content")
                return result.structured_content

        return anyio.run(_run)


# --------------------------------------------------------------------------
# Layer 3: decision policy (pure, ablation-toggleable)
# --------------------------------------------------------------------------
class DecisionPolicy:
    """Map confidence/calibration to answer/retrieve/unknown.

    ``mode="calibrated"`` uses the artifact's calibrated probability and its
    exported threshold (the full pipeline). ``mode="raw"`` uses the raw
    confidence and ``raw_threshold`` only, giving the no-calibration baseline
    for the final ablation study.
    """

    @staticmethod
    def decide(
        *,
        mode: str,
        calibrated_probability: float | None,
        threshold: float | None,
        raw_confidence: float,
        raw_threshold: float,
    ) -> tuple[str, str]:
        if mode == "raw":
            decision = "answer" if raw_confidence >= raw_threshold else "retrieve"
            return decision, "raw"

        # Default and "calibrated" modes use the calibrator output.
        if calibrated_probability is None:
            return "unknown", "calibrated"
        if threshold is None:
            return "unknown", "calibrated"
        decision = "answer" if calibrated_probability >= threshold else "retrieve"
        return decision, "calibrated"


# --------------------------------------------------------------------------
# Action / Observation
# --------------------------------------------------------------------------
class CalibratedQAAction(Action):
    """Arguments the agent supplies to the calibrated QA tool."""

    question: str = Field(description="The original factual question.")
    domain: str | None = Field(
        default=None,
        description="movies, songs or basketball; omit to let the service infer it.",
    )


class CalibratedQAObservation(Observation):
    """Structured, traceable result returned to the agent."""

    answer: str | None = None
    raw_confidence: float | None = None
    calibrated_probability: float | None = None
    threshold: float | None = None
    decision: str | None = None
    decision_source: str | None = None
    status: str | None = None
    artifact: str | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    request_id: str | None = None
    model: str | None = None
    domain: str | None = None
    feature_values: dict | None = None
    feature_coverage: dict | None = None
    latency_ms: int | None = None


# --------------------------------------------------------------------------
# Executor + Tool
# --------------------------------------------------------------------------
class CalibratedQAExecutor(
    ToolExecutor[CalibratedQAAction, CalibratedQAObservation]
):
    """Orchestrate generation, calibration and decision with tracing."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen2.5-7b",
        domain: str | None = None,
        mcp_python: str | None = None,
        project_root: str | Path | None = None,
        paper_repo: str | Path | None = None,
        mode: str = "calibrated",
        raw_threshold: float = 0.5,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.domain = domain
        self.mode = mode
        self.raw_threshold = raw_threshold
        self.calibration = CalibrationClient(
            mcp_python=mcp_python,
            project_root=project_root,
            paper_repo=paper_repo,
        )

    def __call__(
        self,
        action: CalibratedQAAction,
        conversation: "LocalConversation | None" = None,
    ) -> CalibratedQAObservation:
        request_id = str(uuid.uuid4())
        started = _now()
        domain = action.domain or self.domain

        try:
            generation = generate_answer(
                base_url=self.base_url,
                model=self.model,
                question=action.question,
            )
        except Exception as exc:  # noqa: BLE001 - readable agent error
            return CalibratedQAObservation.from_text(
                text=f"Answer generation failed: {type(exc).__name__}: {exc}",
                is_error=True,
                request_id=request_id,
                model=self.model,
                domain=domain,
            )

        try:
            calibrated = self.calibration.calibrate(
                question=action.question,
                answer=generation.answer,
                confidence=generation.raw_confidence,
                model=self.model,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001
            return CalibratedQAObservation.from_text(
                text=f"Calibration failed: {type(exc).__name__}: {exc}",
                is_error=True,
                request_id=request_id,
                model=self.model,
                domain=domain,
                answer=generation.answer,
                raw_confidence=generation.raw_confidence,
            )

        probability = calibrated.get("calibrated_probability")
        threshold = calibrated.get("threshold")
        decision, decision_source = DecisionPolicy.decide(
            mode=self.mode,
            calibrated_probability=probability,
            threshold=threshold,
            raw_confidence=generation.raw_confidence,
            raw_threshold=self.raw_threshold,
        )

        finished = _now()
        latency_ms = _latency_ms(started, finished)
        lines = [
            f"Question: {action.question}",
            f"Answer: {generation.answer}",
            f"Raw confidence: {generation.raw_confidence:.4f}",
            (
                f"Calibrated probability: {probability:.4f}"
                if probability is not None
                else "Calibrated probability: unavailable"
            ),
            f"Decision: {decision} (source={decision_source})",
        ]
        if calibrated.get("reason"):
            lines.append(f"Reason: {calibrated.get('reason')}")

        return CalibratedQAObservation(
            content=[TextContent(text="\n".join(lines))],
            answer=generation.answer,
            raw_confidence=generation.raw_confidence,
            calibrated_probability=probability,
            threshold=threshold,
            decision=decision,
            decision_source=decision_source,
            status=calibrated.get("status"),
            artifact=calibrated.get("artifact"),
            reason=calibrated.get("reason"),
            warnings=list(calibrated.get("warnings", [])),
            request_id=request_id,
            model=self.model,
            domain=calibrated.get("domain"),
            feature_values=calibrated.get("feature_values"),
            feature_coverage=calibrated.get("feature_coverage"),
            latency_ms=latency_ms,
        )


def _latency_ms(started: str, finished: str) -> int:
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished)
        return int((end - start).total_seconds() * 1000)
    except ValueError:
        return -1


class CalibratedQATool(ToolDefinition[CalibratedQAAction, CalibratedQAObservation]):
    """Tool for answering factual questions with calibrated confidence."""

    name: ClassVar[str] = "calibrated_qa"
    user_selectable: ClassVar[bool] = False

    @classmethod
    def create(
        cls,
        conv_state: "object | None" = None,  # noqa: ARG003
        **params,
    ) -> Sequence[Self]:
        return [
            cls(
                name="calibrated_qa",
                description=_DESCRIPTION,
                action_type=CalibratedQAAction,
                observation_type=CalibratedQAObservation,
                executor=CalibratedQAExecutor(**params),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(CalibratedQATool.name, CalibratedQATool)
