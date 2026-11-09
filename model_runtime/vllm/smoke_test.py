from __future__ import annotations

import argparse
import math
import sys

from openai import OpenAI


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
        raise ValueError("No answer tokens with logprobs were returned")

    probabilities = [
        math.exp(item.logprob)
        for item in answer_tokens
    ]

    confidence = sum(probabilities) / len(probabilities)

    if not math.isfinite(confidence):
        raise ValueError("Confidence is not finite")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence outside [0, 1]: {confidence}")

    return confidence, answer_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-7b",
    )
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key="local-dev",
    )

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer with only the name, without explanation. "
                    "Who directed the movie Scream VI?"
                ),
            }
        ],
        temperature=0,
        top_p=1,
        max_tokens=32,
        logprobs=True,
    )

    choice = response.choices[0]
    answer = choice.message.content

    if not answer:
        raise ValueError("Model returned an empty answer")

    if choice.logprobs is None or not choice.logprobs.content:
        raise ValueError("Model did not return token logprobs")

    confidence, tokens = calculate_confidence(
        choice.logprobs.content
    )

    print(f"model={response.model}")
    print(f"answer={answer!r}")
    print(f"finish_reason={choice.finish_reason}")
    print(f"answer_token_count={len(tokens)}")
    print(f"confidence={confidence:.9f}")

    for index, item in enumerate(tokens):
        probability = math.exp(item.logprob)
        print(
            f"token[{index}]={item.token!r} "
            f"logprob={item.logprob:.9f} "
            f"probability={probability:.9f}"
        )

    if response.model != args.model:
        raise ValueError(
            f"Model mismatch: expected {args.model}, "
            f"received {response.model}"
        )

    if choice.finish_reason != "stop":
        raise ValueError(
            f"Unexpected finish reason: {choice.finish_reason}"
        )

    print("vLLM smoke test passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vLLM smoke test failed: {exc}", file=sys.stderr)
        raise