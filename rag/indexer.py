from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI

from .types import Chunk


class PdfIndex:
    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 6) -> List[Tuple[int, float]]:
        vector = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        result: List[Tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            result.append((int(idx), float(score)))
        return result


def _chunks_to_json(chunks: List[Chunk]) -> List[dict]:
    return [
        {"chunk_id": c.chunk_id, "page": c.page, "text": c.text, "section": c.section}
        for c in chunks
    ]


def _chunks_from_json(items: List[dict]) -> List[Chunk]:
    return [
        Chunk(
            chunk_id=i["chunk_id"],
            page=int(i["page"]),
            text=i["text"],
            section=i.get("section"),
        )
        for i in items
    ]


def embed_texts(client: OpenAI, texts: List[str], embedding_model: str) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot create embeddings because the PDF produced no text chunks.")
    response = client.embeddings.create(model=embedding_model, input=texts)
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype=np.float32)


def build_or_load_index(
    client: OpenAI,
    chunks: List[Chunk],
    cache_dir: str,
    cache_key: str,
    embedding_model: str,
) -> PdfIndex:
    if not chunks:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "Try a text-based PDF instead of a scanned/image-only PDF."
        )

    target_dir = Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = target_dir / f"{cache_key}.chunks.json"
    emb_file = target_dir / f"{cache_key}.embeddings.npy"

    if chunks_file.exists() and emb_file.exists():
        stored_chunks = _chunks_from_json(json.loads(chunks_file.read_text(encoding="utf-8")))
        embeddings = np.load(emb_file)
        if (
            len(stored_chunks) == len(chunks)
            and any(c.section for c in chunks)
            and not any(c.section for c in stored_chunks)
        ):
            chunks_file.write_text(
                json.dumps(_chunks_to_json(chunks), ensure_ascii=True),
                encoding="utf-8",
            )
            return PdfIndex(chunks, embeddings)
        return PdfIndex(stored_chunks, embeddings)

    embeddings = embed_texts(client, [c.text for c in chunks], embedding_model)
    chunks_file.write_text(json.dumps(_chunks_to_json(chunks), ensure_ascii=True), encoding="utf-8")
    np.save(emb_file, embeddings)
    return PdfIndex(chunks, embeddings)
