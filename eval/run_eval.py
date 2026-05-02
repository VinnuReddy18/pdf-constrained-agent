from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.agent import AgentConfig, PdfGroundedAgent
from rag.indexer import build_or_load_index
from rag.pdf_loader import build_chunks, load_pdf_text_per_page


def build_agent(pdf_path: str) -> PdfGroundedAgent:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)
    config = AgentConfig()

    raw = Path(pdf_path).read_bytes()
    cache_key = hashlib.sha256(raw).hexdigest()[:16]
    pages = load_pdf_text_per_page(pdf_path)
    chunks = build_chunks(pages)
    index = build_or_load_index(
        client=client,
        chunks=chunks,
        cache_dir=".data/index_cache",
        cache_key=cache_key,
        embedding_model=config.embedding_model,
    )
    return PdfGroundedAgent(client=client, index=index, config=config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default="samples/ai_engineering_revision_notes.pdf",
        help="Path to PDF",
    )
    parser.add_argument("--tests", default="eval/test_cases.json", help="Path to tests json")
    args = parser.parse_args()

    agent = build_agent(args.pdf)
    tests = json.loads(Path(args.tests).read_text(encoding="utf-8"))

    passes = 0
    print(f"Running {len(tests)} tests")
    for t in tests:
        result = agent.ask(t["query"])
        is_ok = True
        if t["type"] == "invalid" and not result.refusal:
            is_ok = False
        if t["type"] == "valid" and result.refusal:
            # Valid test might still fail if PDF truly lacks info; keep this strict for evaluation.
            is_ok = False
        if not result.refusal and not result.citations:
            is_ok = False
        expected_keywords = t.get("expected_keywords", [])
        if t["type"] == "valid" and expected_keywords:
            answer_lower = result.answer.lower()
            matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            if len(matched) < max(1, len(expected_keywords) // 2):
                is_ok = False

        status = "PASS" if is_ok else "FAIL"
        print(f"[{status}] {t['name']} | refusal={result.refusal} | citations={result.citations}")
        if expected_keywords:
            print(f"       expected_keywords={expected_keywords}")
            print(f"       answer={result.answer[:300]}")
        if is_ok:
            passes += 1

    print(f"Score: {passes}/{len(tests)}")


if __name__ == "__main__":
    main()
