from __future__ import annotations

import anyio
import pytest

from mcp import Client

from mcp_server.server import mcp


def test_mcp_tool_contract() -> None:
    async def scenario() -> None:
        async with Client(
            mcp,
            raise_exceptions=True,
        ) as client:
            listed = await client.list_tools()

            tools = {
                tool.name: tool
                for tool in listed.tools
            }

            assert "calibrate" in tools

            tool = tools["calibrate"]
            properties = tool.input_schema["properties"]

            assert set(properties) == {
                "question",
                "answer",
                "confidence",
                "model",
                "domain",
            }
            assert set(tool.input_schema["required"]) == {
                "question",
                "answer",
                "confidence",
                "model",
            }

            assert properties["confidence"]["minimum"] == 0.0
            assert properties["confidence"]["maximum"] == 1.0
            assert tool.output_schema is not None

            result = await client.call_tool(
                "calibrate",
                arguments={
                    "question": (
                        "Who is the director of the movie "
                        "The Intouchables"
                    ),
                    "answer": "Olivier Nakache",
                    "confidence": 0.8954265,
                    "model": "qwen2.5-7b",
                    "domain": "movies",
                },
            )

            assert result.is_error is False
            assert result.structured_content is not None

            data = result.structured_content
            assert data["status"] == "ok"
            assert data["decision"] == "answer"
            assert data["artifact"] == "qwen2.5-7b__movies"
            assert data["calibrated_probability"] == pytest.approx(
                0.9668865976,
                abs=1e-6,
            )

    anyio.run(scenario)


def test_mcp_degraded_result_is_not_tool_error() -> None:
    async def scenario() -> None:
        async with Client(
            mcp,
            raise_exceptions=True,
        ) as client:
            result = await client.call_tool(
                "calibrate",
                arguments={
                    "question": (
                        "Who is the director of the movie "
                        "The Intouchables"
                    ),
                    "answer": "Olivier Nakache",
                    "confidence": 0.8954265,
                    "model": "unknown-model",
                    "domain": "movies",
                },
            )

            # Expected operational degradation is a valid tool result,
            # not an MCP protocol failure.
            assert result.is_error is False

            data = result.structured_content
            assert data["status"] == "degraded"
            assert data["reason"] == "artifact_not_found"
            assert data["decision"] == "unknown"
            assert data["fallback_score"] == pytest.approx(
                0.8954265
            )

    anyio.run(scenario)


def test_mcp_rejects_invalid_confidence() -> None:
    async def scenario() -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "calibrate",
                arguments={
                    "question": "A question",
                    "answer": "An answer",
                    "confidence": 1.5,
                    "model": "qwen2.5-7b",
                    "domain": "movies",
                },
            )

            assert result.is_error is True

    anyio.run(scenario)