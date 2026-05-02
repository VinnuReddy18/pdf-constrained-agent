# Technical Note - PDF-Constrained Conversational Agent

## Objective
Build a source-constrained agent that answers only from a provided PDF, includes citations, and refuses out-of-scope or unsupported questions.

## User Interface
The app is built with Streamlit and supports two source modes:

- **Sample PDF**: uses the bundled `samples/ai_engineering_revision_notes.pdf` for quick testing.
- **Uploaded PDF**: lets the user upload a different PDF and explicitly select it as the active source.

The sidebar shows the active PDF, source mode, indexing status, debug trace toggle, and predefined test-query buttons. Changing the active PDF resets the chat so answers are not mixed across documents.

## Agentic Design
The system is implemented as a single-controller agent with explicit decision steps:

1. **ParsePDF**  
   Extract text page-wise using `pypdf`. Section/topic labels are detected where possible from headings such as `Topic 1`, `Chapter 1`, or `Section 1`.
2. **RetrieveEvidence**  
   Chunk text, embed chunks, and retrieve top-k evidence snippets. For follow-up questions, recent conversation turns are included in the retrieval query only to resolve context.
3. **GroundedAnswer**  
   Generate an answer only from retrieved PDF evidence. Conversation history can clarify the user's intent, but it is not treated as evidence. The model is asked to return cited evidence IDs rather than free-form page numbers.
4. **RefusalHandler**  
   Refuse if evidence is insufficient or relevance is below threshold.
5. **Verifier**  
   Ensure non-refusal answers include citations. Evidence IDs are mapped back to page numbers and detected section/topic labels, for example `Page 2 - Topic 1 Introduction to A I Engineering`.

## Why this design
- Prioritizes correctness and reliability over broad coverage.
- Enforces strict source boundaries to reduce hallucinations.
- Provides transparent testability with explicit citations, refusal behavior, and trace output.
- Keeps conversation support separate from source evidence so follow-up questions do not introduce unsupported facts.

## Trade-offs
- Fixed chunking is simple and robust but may split context awkwardly near boundaries.
- Similarity threshold tuning affects precision/recall.
- Single-stage retrieval is faster but may miss edge cases without reranking.
- Section detection is heuristic and depends on extractable PDF text quality.
- Scanned/image-only PDFs are not OCR-processed. The app detects this case and asks the user to upload a text-based or OCR version.

## Observability
The app exposes a debug trace showing retrieval count, routing decision (answer/refusal), and final verifier step. The UI also shows the active PDF and source mode so evaluators can confirm which document is used for answering.

## Testability
- Includes a sample PDF (`samples/ai_engineering_revision_notes.pdf`).
- Includes a predefined evaluation set (`eval/test_cases.json`) with 5 valid and 3 invalid queries.
- Includes reproducible runner (`eval/run_eval.py`) for automated checks.
- Includes sidebar test buttons for the same 5 valid and 3 invalid queries.
- Includes a generated report PDF (`PDF_Constrained_Agent_Evaluation_Report.pdf`) with screenshots of valid, invalid, and multilingual behavior.

## Multilingual Support
The system can answer multilingual user queries when the retrieved PDF evidence supports the answer. The grounding rule remains unchanged: multilingual responses must still be based only on the PDF context and include citations.

## Failure Handling
- If an uploaded PDF produces no readable text chunks, the app stops with a clear message instead of sending an empty embedding request.
- If the OpenAI API cannot be reached during a query, the app shows a connection error message instead of a traceback.
- Refusal responses include no citations and explicitly state that the answer could not be found in the uploaded PDF.
