from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_REPO = PROJECT_ROOT / "reference" / "Popular-but-Wrong"

SPECIAL_TOKENS = {
    "<|im_end|>",
    "<|endoftext|>",
}


def calculate_confidence(logprob_content) -> tuple[float, list]:
    answer_tokens = [
        item
        for item in logprob_content
        if item.token not in SPECIAL_TOKENS
    ]

    if not answer_tokens:
        raise ValueError("No answer tokens with logprobs returned")

    probabilities = [
        math.exp(item.logprob)
        for item in answer_tokens
    ]
    confidence = sum(probabilities) / len(probabilities)

    if not math.isfinite(confidence):
        raise ValueError("Calculated confidence is not finite")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Calculated confidence is outside [0, 1]: {confidence}"
        )

    return confidence, answer_tokens


def generate_answer(
    *,
    base_url: str,
    model: str,
    question: str,
) -> tuple[str, float]:
    client = OpenAI(
        base_url=base_url,
        api_key="local-dev",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer with only the requested entity name. "
                    "Do not explain your answer."
                ),
            },
            {
                "role": "user",
                # Keep this canonical question unchanged so P1 lookup
                # can identify the dataset entity.
                "content": question,
            },
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

    confidence, answer_tokens = calculate_confidence(
        choice.logprobs.content
    )

    print(f"generated_answer={answer!r}")
    print(f"answer_token_count={len(answer_tokens)}")
    print(f"raw_confidence={confidence:.9f}")

    return answer, confidence


async def call_mcp(
    *,
    question: str,
    answer: str,
    confidence: float,
    model: str,
    domain: str,
) -> dict:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env={
            # MCP v2 gives subprocesses a minimal environment, so pass
            # project paths explicitly.
            "PYTHONPATH": str(PROJECT_ROOT),
            "POPULAR_BUT_WRONG_REPO": str(PAPER_REPO),
        },
    )

    async with Client(server) as client:
        listed = await client.list_tools()
        names = [tool.name for tool in listed.tools]

        if "calibrate" not in names:
            raise RuntimeError(
                f"calibrate tool not found; available tools: {names}"
            )

        result = await client.call_tool(
            "calibrate",
            arguments={
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "model": model,
                "domain": domain,
            },
        )

        if result.is_error:
            messages = [
                getattr(block, "text", repr(block))
                for block in result.content
            ]
            raise RuntimeError(
                "MCP calibrate returned tool error: "
                + " ".join(messages)
            )

        if result.structured_content is None:
            raise RuntimeError(
                "MCP result has no structured content"
            )

        return result.structured_content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-7b",
    )
    parser.add_argument(
        "--domain",
        default="movies",
        choices=["movies", "songs", "basketball"],
    )
    parser.add_argument(
        "--question",
        default=(
            "Who is the director of the movie The Intouchables"
        ),
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
    )
    args = parser.parse_args()

    answer, confidence = generate_answer(
        base_url=args.base_url,
        model=args.model,
        question=args.question,
    )

    async def run_mcp_call() -> dict:
    	return await call_mcp(
            question=args.question,
            answer=answer,
            confidence=confidence,
            model=args.model,
            domain=args.domain,
    	)
 
    result = anyio.run(run_mcp_call)



    print(json.dumps(result, indent=2, ensure_ascii=False))

    probability = result.get("calibrated_probability")
    decision = result.get("decision")

    if probability is None or decision == "unknown":
        if not args.allow_unknown:
            raise RuntimeError(
                "End-to-end call completed but calibration degraded "
                "to unknown. Inspect feature_coverage and answer_entity."
            )

    elif decision not in {"answer", "retrieve"}:
        raise RuntimeError(f"Unexpected decision: {decision!r}")

    print("M3 end-to-end smoke test passed")


if __name__ == "__main__":
    main()