# PDF-Constrained Conversational Agent

Source-grounded conversational agent for PDFs with strict refusal behavior and page citations.

## Features
- Upload any PDF and ask questions conversationally.
- Answers only from PDF evidence.
- Refuses unsupported or out-of-scope queries.
- Provides page-level citations.
- Shows agent trace for observability.

## Project Structure
- `app.py` - Streamlit app
- `rag/` - agent and RAG modules
- `samples/ai_engineering_revision_notes.pdf` - sample PDF for testing
- `eval/test_cases.json` - predefined tests
- `eval/run_eval.py` - evaluation runner
- `TECH_NOTE.md` - architecture and trade-offs

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Set environment variable:
   - `OPENAI_API_KEY=<your_key>`

## Run App
- `streamlit run app.py`

## Run Evaluation
- `python eval/run_eval.py`
- Or evaluate another PDF with matching tests:
  - `python eval/run_eval.py --pdf path/to/your.pdf --tests path/to/tests.json`

## Notes for Assessors
- The default evaluation uses the included sample PDF with 5 valid and 3 invalid queries.
- For each valid response, confirm page citations map to the PDF source.
- For invalid questions, confirm refusal behavior.
