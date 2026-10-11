"""
Seeds a real (not fabricated) benchmark run for the faculty demo.

This runs the ACTUAL pipeline -- the same claim extraction, retrieval, and
six metrics the app uses live -- against the bundled sample_data/rag_overview.pdf
and the curated questions in sample_data/sample_questions.json. Nothing about
the scores it produces is invented: it's the equivalent of manually clicking
through the Experiment -> Verification -> Evaluation flow five times, scripted
for convenience.

What this script does NOT do: assign human_label. That's real research labor
this project's own rule (never fabricate results) doesn't let a script shortcut.
Run scripts/label_demo.py afterwards to label the claims yourself -- only then
will the Dashboard's Precision/Recall/F1 table have real numbers instead of
"Not evaluated yet".

Usage (from backend/):
    python3 -m scripts.seed_demo
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.db import database, repository
from app.services import vector_store
from app.services.benchmark_runner import run_benchmark_question
from app.services.chunker import chunk_document
from app.services.embedding_service import embed_texts
from app.services.pdf_extractor import extract_text_by_page

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed_demo")

SAMPLE_DOC_ID = "sample-rag-overview"
SAMPLE_PDF = Path(__file__).parent.parent / "sample_data" / "rag_overview.pdf"
SAMPLE_QUESTIONS = Path(__file__).parent.parent / "sample_data" / "sample_questions.json"


def _ensure_sample_doc_embedded():
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    dest_path = settings.upload_dir / f"{SAMPLE_DOC_ID}.pdf"
    dest_path.write_bytes(SAMPLE_PDF.read_bytes())

    pages = extract_text_by_page(dest_path)
    chunks = chunk_document(
        pages, doc_id=SAMPLE_DOC_ID,
        chunk_size=settings.chunk_size_words, overlap=settings.chunk_overlap_words,
    )
    (settings.processed_dir / f"{SAMPLE_DOC_ID}_chunks.json").write_text(json.dumps(chunks, indent=2))

    embeddings = embed_texts([c["text"] for c in chunks])
    vector_store.delete_doc(SAMPLE_DOC_ID)  # idempotent: safe to re-run this script
    vector_store.add_chunks(SAMPLE_DOC_ID, chunks, embeddings)
    logger.info("Processed and embedded %d chunks from %s", len(chunks), SAMPLE_PDF.name)


def main():
    database.init_db()
    _ensure_sample_doc_embedded()

    questions = json.loads(SAMPLE_QUESTIONS.read_text())
    total_claims = 0
    for item in questions:
        question_row = repository.add_benchmark_question(
            SAMPLE_DOC_ID, item["question"], item.get("notes")
        )
        result = run_benchmark_question(question_row, top_k=5)
        total_claims += result["claims_recorded"]
        logger.info("  %-70s -> %s", item["question"], result["message"])

    logger.info("")
    logger.info("Seeded %d claims across %d questions.", total_claims, len(questions))
    if settings.llm_provider == "mock":
        logger.info(
            "NOTE: LLM_PROVIDER is 'mock' -- these are placeholder answers, not real scores. "
            "Set a real provider in backend/.env and re-run this script before presenting."
        )
    logger.info("Next: run `python3 -m scripts.label_demo` to label these claims yourself, "
                "then the Dashboard's Precision/Recall/F1 table will show real numbers.")


if __name__ == "__main__":
    main()
