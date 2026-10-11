"""
CRUD helpers for the benchmark tables. Plain functions returning dicts, not
an ORM -- consistent with the rest of the project's "explainable over
clever" approach.
"""
import logging
from typing import List, Optional

from app.db.database import get_connection

logger = logging.getLogger(__name__)


def add_benchmark_question(doc_id: str, question: str, notes: Optional[str] = None) -> dict:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO benchmark_questions (doc_id, question, notes) VALUES (?, ?, ?)",
        (doc_id, question, notes),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM benchmark_questions WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def list_benchmark_questions() -> List[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM benchmark_questions ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_benchmark_question(question_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM benchmark_questions WHERE id = ?", (question_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def insert_benchmark_claim(record: dict) -> int:
    columns = [
        "run_id", "benchmark_question_id", "doc_id", "question", "claim_id", "claim_text",
        "answer_text", "retrieved_chunk_ids", "baseline_status", "baseline_score",
        "likely_failure_type", "rouge_l_score", "nli_score", "llm_judge_score",
        "llm_judge_verdict", "ragas_score", "self_consistency_score", "model_name", "latency_seconds",
    ]
    values = [record.get(c) for c in columns]
    placeholders = ", ".join("?" for _ in columns)

    conn = get_connection()
    cur = conn.execute(
        f"INSERT INTO benchmark_claims ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def list_benchmark_claims(labeled_only: Optional[bool] = None, run_id: Optional[str] = None) -> List[dict]:
    conn = get_connection()
    query = "SELECT * FROM benchmark_claims"
    conditions = []
    params: list = []

    if labeled_only is True:
        conditions.append("human_label IS NOT NULL")
    elif labeled_only is False:
        conditions.append("human_label IS NULL")
    if run_id:
        conditions.append("run_id = ?")
        params.append(run_id)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_human_label(claim_row_id: int, label: str) -> Optional[dict]:
    conn = get_connection()
    conn.execute("UPDATE benchmark_claims SET human_label = ? WHERE id = ?", (label, claim_row_id))
    conn.commit()
    row = conn.execute("SELECT * FROM benchmark_claims WHERE id = ?", (claim_row_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
