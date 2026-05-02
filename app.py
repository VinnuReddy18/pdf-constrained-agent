from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, OpenAI

from rag.agent import AgentConfig, PdfGroundedAgent
from rag.indexer import build_or_load_index
from rag.pdf_loader import build_chunks, load_pdf_text_per_page


load_dotenv()

st.set_page_config(page_title="PDF Grounded Agent", page_icon=":mag:", layout="wide")
st.title("PDF-Constrained Conversational Agent")
st.caption("Answers strictly from uploaded PDF with page citations and refusal handling.")

SAMPLE_PDF_PATH = Path("samples") / "ai_engineering_revision_notes.pdf"
VALID_TEST_QUERIES = [
    "What does the PDF define AI Engineering as?",
    "What are the core responsibilities of an AI Engineer listed in the document?",
    "How does the PDF contrast ML Engineering with AI Engineering?",
    "What data flow steps are shown for decoder-only models?",
    "What does the document say causal masking prevents?",
]
INVALID_TEST_QUERIES = [
    "Who won the FIFA World Cup in 2018?",
    "What is the weather in Delhi today?",
    "Write Python code to reverse a linked list.",
]
VALID_TEST_LABELS = [
    "Definition of AI Engineering",
    "AI Engineer responsibilities",
    "ML vs AI Engineering",
    "Decoder-only data flow",
    "Causal masking",
]
INVALID_TEST_LABELS = [
    "FIFA World Cup",
    "Weather today",
    "Linked list code",
]


def render_citations(citations: List[str]) -> None:
    if not citations:
        st.caption("Citations: None (refusal)")
        return

    st.caption("Citations: " + ", ".join(citations))


def render_source_checklist(pdf_name: str, source_label: str) -> None:
    st.markdown("**PDF status**")
    st.markdown(f"- Active PDF: `{pdf_name}`")
    st.markdown(f"- Source mode: `{source_label}`")
    st.markdown("- [x] PDF text indexed")
    st.markdown("- [x] Answers constrained to this PDF")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in environment.")
    st.stop()

if "use_sample_pdf" not in st.session_state:
    st.session_state.use_sample_pdf = True

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
sample_available = SAMPLE_PDF_PATH.exists()

source_options: List[str] = []
if sample_available:
    source_options.append("Sample PDF")
if uploaded is not None:
    source_options.append("Uploaded PDF")

if not source_options:
    st.warning("Sample PDF is missing. Upload a PDF to start.")
    st.stop()

if uploaded is None and sample_available:
    st.info("Sample PDF is selected by default. Upload another PDF if you want to ask questions from it.")

preferred_source = "Sample PDF" if st.session_state.use_sample_pdf else "Uploaded PDF"
if preferred_source not in source_options:
    preferred_source = source_options[0]

selected_source = st.radio(
    "Select PDF to answer from",
    source_options,
    index=source_options.index(preferred_source),
    horizontal=True,
)
st.session_state.use_sample_pdf = selected_source == "Sample PDF"
use_sample = selected_source == "Sample PDF"

data_dir = Path(".data")
pdf_dir = data_dir / "pdfs"
cache_dir = data_dir / "index_cache"
pdf_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

if use_sample:
    pdf_bytes = SAMPLE_PDF_PATH.read_bytes()
    active_pdf_label = SAMPLE_PDF_PATH.name
    active_source_label = "Bundled sample PDF"
else:
    pdf_bytes = uploaded.getvalue()
    active_pdf_label = uploaded.name
    active_source_label = "Uploaded PDF"

cache_key = hashlib.sha256(pdf_bytes).hexdigest()[:16]
pdf_path = pdf_dir / f"{cache_key}.pdf"
if not pdf_path.exists():
    pdf_path.write_bytes(pdf_bytes)


@st.cache_resource(show_spinner=False)
def get_client(key: str) -> OpenAI:
    return OpenAI(api_key=key)


@st.cache_resource(show_spinner=False)
def get_agent_for_pdf(cache_key_value: str, file_path: str, key: str) -> PdfGroundedAgent:
    client = get_client(key)
    config = AgentConfig()
    pages = load_pdf_text_per_page(file_path)
    chunks = build_chunks(pages)
    if not chunks:
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be scanned/image-only or protected from text extraction."
        )
    index = build_or_load_index(
        client=client,
        chunks=chunks,
        cache_dir=str(cache_dir),
        cache_key=cache_key_value,
        embedding_model=config.embedding_model,
    )
    return PdfGroundedAgent(client=client, index=index, config=config)


if "active_pdf_key" not in st.session_state:
    st.session_state.active_pdf_key = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.active_pdf_key != cache_key:
    st.session_state.active_pdf_key = cache_key
    st.session_state.messages = []

try:
    with st.spinner("Preparing agent and index..."):
        agent = get_agent_for_pdf(cache_key, str(pdf_path), api_key)
except ValueError as exc:
    st.error(str(exc))
    st.info("Use a text-based PDF, or run OCR on the PDF first and upload the OCR version.")
    st.stop()

left, right = st.columns([3, 1])
with right:
    st.success("Agent ready")
    render_source_checklist(active_pdf_label, active_source_label)
    show_trace = st.toggle("Show debug trace", value=False)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Test queries")
    st.caption("Valid")
    for idx, (label, query) in enumerate(zip(VALID_TEST_LABELS, VALID_TEST_QUERIES), start=1):
        if st.button(label, key=f"valid_query_{idx}", use_container_width=True):
            st.session_state.pending_prompt = query
            st.rerun()

    st.caption("Invalid / out-of-scope")
    for idx, (label, query) in enumerate(zip(INVALID_TEST_LABELS, INVALID_TEST_QUERIES), start=1):
        if st.button(label, key=f"invalid_query_{idx}", use_container_width=True):
            st.session_state.pending_prompt = query
            st.rerun()

with left:
    st.subheader("Chat")
    if not st.session_state.messages:
        st.info("Ask anything from this PDF. Press Enter to send.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            citations: List[str] = msg.get("citations", [])
            if msg["role"] == "assistant":
                render_citations(citations)
            if show_trace and msg["role"] == "assistant" and msg.get("trace"):
                st.code(msg["trace"])

    prompt: Optional[str] = st.session_state.pop("pending_prompt", None)
    typed_prompt = st.chat_input("Ask a question about the uploaded PDF...")
    if typed_prompt:
        prompt = typed_prompt

    if prompt and prompt.strip():
        clean_prompt = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": clean_prompt})
        with st.chat_message("user"):
            st.markdown(clean_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Reasoning over grounded evidence..."):
                try:
                    result = agent.ask(clean_prompt, history=st.session_state.messages[:-1])
                except APIConnectionError:
                    st.error(
                        "Could not reach the OpenAI API. Check your internet connection, VPN/proxy, "
                        "firewall, and API key environment, then try again."
                    )
                    st.stop()
                except APIError as exc:
                    st.error(f"OpenAI API error: {exc}")
                    st.stop()

            response_text = result.answer
            if result.refusal:
                response_text = (
                    result.answer
                    + "\n\nI only answer from the uploaded PDF. Try rephrasing the question."
            )
            st.markdown(response_text)
            render_citations(result.citations)
            if show_trace:
                st.code(result.debug_trace)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "citations": result.citations,
                "trace": result.debug_trace,
            }
        )
