"""OpenHands runtime tool: offline knowledge-base retrieval for supported QA domains.

The knowledge base is the three source QA files from the paper repository
(movies / songs / basketball jsonl). Each row maps a canonical question to its
accepted answer aliases. Retrieval is a structured lookup, not a semantic
search: exact question match first; if the question is reworded, a
domain-keyword gate selects the domain and a longest question-entity substring
match resolves the entity within that domain.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

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
DEFAULT_DATA_DIR = PROJECT_ROOT / "reference" / "Popular-but-Wrong" / "data"

DOMAINS = ("movies", "songs", "basketball")

DOMAIN_KEYWORDS = {
    "movies": ("movie", "film", "director", "directed"),
    "songs": ("song", "performer", "performed", "singer", "sang"),
    "basketball": ("basketball", "player", "birthplace", "born"),
}

_DESCRIPTION = (
    "Look up a factual question in the local knowledge base. Use this tool when "
    "the calibrated answer has decision retrieve or unknown: search for the "
    "question here and answer from the returned reference answer(s), which are "
    "treated as the retrieved evidence."
)


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


class KnowledgeBase:
    """In-memory index over the three source QA jsonl files."""

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self._by_question: dict[str, dict] = {}
        self._by_entity: dict[str, dict[str, list[dict]]] = {
            domain: {} for domain in DOMAINS
        }
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        for domain in DOMAINS:
            path = self.data_dir / f"{domain}.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                record = {
                    "question": row["question"],
                    "references": list(row.get("reference", [])),
                    "question_entity": row.get("question_entity", ""),
                    "domain": domain,
                }
                self._by_question[_normalize(row["question"])] = record
                entity = _normalize(row.get("question_entity", ""))
                if entity:
                    self._by_entity[domain].setdefault(entity, []).append(record)
        self._loaded = True

    def search(self, question: str) -> list[dict]:
        """Return matching records; empty list when nothing matches."""
        self._load()
        query = _normalize(question)

        # 1) Exact canonical-question match.
        if query in self._by_question:
            return [self._by_question[query]]

        # 2) Reworded-question fallback: infer the domain from keywords, then
        #    pick the longest question entity that appears in the query. The
        #    domain gate prevents cross-domain common-word false positives.
        domains = [
            domain
            for domain, keywords in DOMAIN_KEYWORDS.items()
            if any(keyword in query for keyword in keywords)
        ]

        best_records: list[dict] = []
        best_entity_len = -1
        for domain in domains:
            for entity, records in self._by_entity[domain].items():
                if len(entity) < 3:
                    continue
                if entity in query and len(entity) > best_entity_len:
                    best_records = records
                    best_entity_len = len(entity)

        seen: set[tuple[str, str]] = set()
        records: list[dict] = []
        for record in best_records:
            key = (record["domain"], record["question"])
            if key not in seen:
                seen.add(key)
                records.append(record)
        return records


class SearchAction(Action):
    """Arguments for the knowledge-base search."""

    question: str = Field(description="The factual question to look up.")
    domain: str | None = Field(
        default=None,
        description="Optional domain hint: movies, songs or basketball.",
    )


class SearchObservation(Observation):
    """Structured retrieval result."""

    found: bool = False
    references: list[str] = Field(default_factory=list)
    matched_question: str | None = None
    question_entity: str | None = None
    domain: str | None = None


class SearchExecutor(ToolExecutor[SearchAction, SearchObservation]):
    """Run the local knowledge-base lookup."""

    def __init__(self, *, data_dir: str | Path | None = None) -> None:
        self.kb = KnowledgeBase(data_dir or DEFAULT_DATA_DIR)

    def __call__(
        self,
        action: SearchAction,
        conversation: "LocalConversation | None" = None,
    ) -> SearchObservation:
        try:
            records = self.kb.search(action.question)
        except Exception as exc:  # noqa: BLE001
            return SearchObservation.from_text(
                text=f"Knowledge-base search failed: {type(exc).__name__}: {exc}",
                is_error=True,
            )

        if not records:
            return SearchObservation.from_text(
                text="No matching record found in the local knowledge base.",
                found=False,
            )

        references: list[str] = []
        for record in records:
            for reference in record["references"]:
                if reference not in references:
                    references.append(reference)

        first = records[0]
        lines = [
            "Retrieved knowledge-base evidence:",
            f"Matched question: {first['question']}",
            f"Answer(s): {', '.join(references)}",
        ]
        return SearchObservation(
            content=[TextContent(text="\n".join(lines))],
            found=True,
            references=references,
            matched_question=first["question"],
            question_entity=first["question_entity"],
            domain=first["domain"],
        )


class SearchTool(ToolDefinition[SearchAction, SearchObservation]):
    """Tool for offline knowledge-base retrieval."""

    name: ClassVar[str] = "search_qa"
    user_selectable: ClassVar[bool] = False

    @classmethod
    def create(
        cls,
        conv_state: "object | None" = None,  # noqa: ARG003
        **params,
    ) -> Sequence[Self]:
        return [
            cls(
                name="search_qa",
                description=_DESCRIPTION,
                action_type=SearchAction,
                observation_type=SearchObservation,
                executor=SearchExecutor(**params),
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


register_tool(SearchTool.name, SearchTool)
