from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str
    section: Optional[str] = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class AgentResult:
    answer: str
    citations: List[str]
    refusal: bool
    debug_trace: str
