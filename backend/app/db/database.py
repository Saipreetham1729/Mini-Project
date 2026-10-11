"""
Lightweight SQLite persistence for the benchmark/reproducibility pipeline.

Uses the standard library sqlite3 module directly rather than an ORM -- the
schema is small and stable enough that raw SQL stays readable, and it
avoids adding a new heavy dependency this late in the project.
"""
import logging
import sqlite3

from app.config import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.processed_dir.parent / "experiments.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS benchmark_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            question TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS benchmark_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            benchmark_question_id INTEGER,
            doc_id TEXT NOT NULL,
            question TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            claim_text TEXT NOT NULL,
            answer_text TEXT NOT NULL,
            retrieved_chunk_ids TEXT,
            baseline_status TEXT,
            baseline_score REAL,
            likely_failure_type TEXT,
            rouge_l_score REAL,
            nli_score REAL,
            llm_judge_score REAL,
            llm_judge_verdict TEXT,
            ragas_score REAL,
            self_consistency_score REAL,
            human_label TEXT,
            model_name TEXT,
            latency_seconds REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (benchmark_question_id) REFERENCES benchmark_questions (id)
        );
        """
    )
    conn.commit()

    # Migration for databases created before the RAGAS/self-consistency
    # extended-scope phase: CREATE TABLE IF NOT EXISTS won't add columns to
    # a table that already exists, so add them here if they're missing,
    # rather than requiring anyone to delete their experiments.db.
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(benchmark_claims)").fetchall()}
    for column in ("ragas_score", "self_consistency_score"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE benchmark_claims ADD COLUMN {column} REAL")
            logger.info("Migrated experiments.db: added benchmark_claims.%s", column)
    conn.commit()
    conn.close()
    logger.info("Initialized experiment database at %s", DB_PATH)
