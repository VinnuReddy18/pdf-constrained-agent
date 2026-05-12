```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from .indexer import PdfIndex, embed_texts
from .types import AgentResult, RetrievedChunk
from .prompt_handler import PromptHandler  # Corrected import to match project import pattern


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Final answer from context only.")
    citations: List[int] = Field(description="List of 1-based context evidence IDs used.")
    insufficient_context: bool = Field(description="True when context is not enough.")


@dataclass
class AgentConfig:
    embedding_model: str = "text-embedding-3-small"
    answering_model: str = "gpt-4.1-mini"
    top_k: int = 6
    min_similarity: float = 0.2
    prompt_strictness: int = 2  # New configuration for prompt strictness


class PdfGroundedAgent:  # Class names should remain PascalCase
    def __init__(self, client: OpenAI, index: PdfIndex, config: AgentConfig) -> None:
        self.client = client
        self.index = index
        self.config = config
        self.prompt_handler = PromptHandler(strictness_level=config.prompt_strictness)  # Initialize PromptHandler

    def _conversation_context(self, history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return ""

        recent = history[-6:]
        lines: List[str] = []
        for item in recent:
            role = item.get("role", "user")
            content = item.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _retrieval_query(self, question: str, history: Optional[List[Dict[str, str]]]) -> str:
        conversation = self._conversation_context(history)
        if not conversation:
            return question
        return f"Conversation so far:\n{conversation}\n\nCurrent question:\n{question}"

    def _retrieve(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> List[RetrievedChunk]:
        query_vec = embed_texts(
            self.client,
            [self._retrieval_query(question, history)],
            self.config.embedding_model,
        )[0]
        results = self.index.search(query_vec, top_k=self.config.top_k)
        return [
            RetrievedChunk(chunk=self.index.chunks[idx], score=score)
            for idx, score in results
            if score >= self.config.min_similarity
        ]

    def _format_context(self, hits: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        for i, hit in enumerate(hits, start=1):
            section = f" section={hit.chunk.section}" if hit.chunk.section else ""
            lines.append(
                f"[{i}] page={hit.chunk.page}{section} score={hit.score:.3f}\n"
                f"{hit.chunk.text}\n"
            )
        return "\n".join(lines)

    def _format_citations(self, hits: List[RetrievedChunk]) -> List[str]:
        citations: List[str] = []
        seen: set[tuple[int, str]] = set()
        ordered_hits = sorted(hits, key=lambda item: (item.chunk.page, item.chunk.section or ""))
        for hit in ordered_hits:
            page = hit.chunk.page
            section = hit.chunk.section or ""
            key = (page, section)
            if key in seen:
                continue
            seen.add(key)
            if section:
                citations.append(f"Page {page} - {section}")
            else:
                citations.append(f"Page {page}")
        return citations

    def _grounded_answer(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> GroundedAnswer:
        system = (
            "You are a strict PDF-grounded assistant.\n"
            "Answer only from provided context.\n"
            "If context is insufficient, set insufficient_context=true.\n"
            "Conversation history may clarify pronouns or follow-up questions, but it is not evidence.\n"
            "Never use outside knowledge or unsupported conversation details.\n"
            "Return only the 1-based evidence IDs, such as [1, 2], that directly support the answer.\n"
            "Do not return page numbers in the citations field."
        )
        conversation = self._conversation_context(history)
        history_block = f"Conversation history:\n{conversation}\n\n" if conversation else ""
        
        # Update the question prompt using the PromptHandler
        updated_question = self.prompt_handler.update_prompt(question)
        
        user = f"{history_block}Question:\n{updated_question}\n\nPDF context:\n{context}"
        completion = self.client.responses.parse(
            model=self.config.answering_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=GroundedAnswer,
        )
        return completion.output_parsed

    def ask(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> AgentResult:
        trace_lines: List[str] = ["Agent trace:", "1) RetrieveEvidence"]
        hits = self._retrieve(question, history)
        trace_lines.append(f"retrieved={len(hits)} above_threshold={self.config.min_similarity}")

        if not hits:
            trace_lines.append("2) RefusalHandler -> no evidence")
            return AgentResult(
                answer="I could not find enough evidence in the provided PDF to answer this.",
                citations=[],
                refusal=True,
                debug_trace="\n".join(trace_lines),
            )

        context = self._format_context(hits)
        trace_lines.append("2) GroundedAnswer")
        structured = self._grounded_answer(question, context, history)

        if structured.insufficient_context:
            trace_lines.append("3) RefusalHandler -> insufficient context")
            return AgentResult(
                answer="I could not find enough evidence in the provided PDF to answer this.",
                citations=[],
                refusal=True,
                debug_trace="\n".join(trace_lines),
            )

        evidence_ids = sorted(
            {
                evidence_id
                for evidence_id in structured.citations
                if isinstance(evidence_id, int) and 1 <= evidence_id <= len(hits)
            }
        )
        if not evidence_ids:
            trace_lines.append("3) Verifier -> missing supported citations")
            return AgentResult(
                answer="I could not find enough evidence in the provided PDF to answer this.",
                citations=[],
                refusal=True,
                debug_trace="\n".join(trace_lines),
            )

        cited_hits = [hits[evidence_id - 1] for evidence_id in evidence_ids]
        citations = self._format_citations(cited_hits)
        trace_lines.append("3) Verifier -> citations included")
        return AgentResult(
            answer=structured.answer.strip(),
            citations=citations,
            refusal=False,
            debug_trace="\n".join(trace_lines),
        )
```