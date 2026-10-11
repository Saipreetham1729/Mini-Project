import pytest

from app.db import database, repository


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Points the repository at a throwaway SQLite file so tests never touch the real dev database."""
    test_db_path = tmp_path / "test_experiments.db"
    monkeypatch.setattr(database, "DB_PATH", test_db_path)
    database.init_db()
    return test_db_path


def test_add_and_list_question(isolated_db):
    q = repository.add_benchmark_question("doc1", "What is RAG?")
    assert q["doc_id"] == "doc1"
    assert q["question"] == "What is RAG?"

    questions = repository.list_benchmark_questions()
    assert len(questions) == 1


def test_insert_and_label_claim(isolated_db):
    q = repository.add_benchmark_question("doc1", "What is RAG?")
    repository.insert_benchmark_claim(
        {
            "run_id": "r1",
            "benchmark_question_id": q["id"],
            "doc_id": "doc1",
            "question": "What is RAG?",
            "claim_id": "claim_0",
            "claim_text": "RAG combines retrieval and generation.",
            "answer_text": "RAG combines retrieval and generation.",
            "retrieved_chunk_ids": "c1",
            "baseline_status": "supported",
            "baseline_score": 0.9,
            "likely_failure_type": None,
            "rouge_l_score": 0.8,
            "nli_score": 0.85,
            "llm_judge_score": 1.0,
            "llm_judge_verdict": "supported",
            "model_name": "mock",
            "latency_seconds": 0.5,
        }
    )

    claims = repository.list_benchmark_claims()
    assert len(claims) == 1
    assert claims[0]["human_label"] is None

    labeled = repository.set_human_label(claims[0]["id"], "supported")
    assert labeled["human_label"] == "supported"


def test_labeled_only_filter(isolated_db):
    q = repository.add_benchmark_question("doc1", "Q?")
    for i in range(2):
        repository.insert_benchmark_claim(
            {
                "run_id": "r1",
                "benchmark_question_id": q["id"],
                "doc_id": "doc1",
                "question": "Q?",
                "claim_id": f"claim_{i}",
                "claim_text": f"Claim {i}",
                "answer_text": "Answer",
                "retrieved_chunk_ids": "c1",
                "baseline_status": "supported",
                "baseline_score": 0.9,
                "likely_failure_type": None,
                "rouge_l_score": 0.8,
                "nli_score": 0.85,
                "llm_judge_score": 1.0,
                "llm_judge_verdict": "supported",
                "model_name": "mock",
                "latency_seconds": 0.5,
            }
        )

    all_claims = repository.list_benchmark_claims()
    repository.set_human_label(all_claims[0]["id"], "supported")

    labeled = repository.list_benchmark_claims(labeled_only=True)
    unlabeled = repository.list_benchmark_claims(labeled_only=False)
    assert len(labeled) == 1
    assert len(unlabeled) == 1


def test_ragas_and_self_consistency_scores_round_trip(isolated_db):
    q = repository.add_benchmark_question("doc1", "Q?")
    repository.insert_benchmark_claim(
        {
            "run_id": "r1",
            "benchmark_question_id": q["id"],
            "doc_id": "doc1",
            "question": "Q?",
            "claim_id": "claim_0",
            "claim_text": "Claim",
            "answer_text": "Answer",
            "retrieved_chunk_ids": "c1",
            "baseline_status": "supported",
            "baseline_score": 0.9,
            "likely_failure_type": None,
            "ragas_score": 0.75,
            "self_consistency_score": 0.6,
            "model_name": "mock",
            "latency_seconds": 0.5,
        }
    )
    claim = repository.list_benchmark_claims()[0]
    assert claim["ragas_score"] == 0.75
    assert claim["self_consistency_score"] == 0.6


def test_init_db_migrates_pre_extended_scope_schema(tmp_path, monkeypatch):
    """
    Simulates a database created before ragas_score/self_consistency_score
    existed: a bare benchmark_claims table without those two columns.
    init_db() should add them rather than requiring the file to be deleted.
    """
    import sqlite3

    old_db_path = tmp_path / "old_experiments.db"
    conn = sqlite3.connect(old_db_path)
    conn.executescript(
        """
        CREATE TABLE benchmark_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL, question TEXT NOT NULL,
            notes TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE benchmark_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, benchmark_question_id INTEGER,
            doc_id TEXT NOT NULL, question TEXT NOT NULL, claim_id TEXT NOT NULL, claim_text TEXT NOT NULL,
            answer_text TEXT NOT NULL, retrieved_chunk_ids TEXT, baseline_status TEXT, baseline_score REAL,
            likely_failure_type TEXT, rouge_l_score REAL, nli_score REAL, llm_judge_score REAL,
            llm_judge_verdict TEXT, human_label TEXT, model_name TEXT, latency_seconds REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(database, "DB_PATH", old_db_path)
    database.init_db()

    conn = sqlite3.connect(old_db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(benchmark_claims)").fetchall()}
    conn.close()
    assert "ragas_score" in columns
    assert "self_consistency_score" in columns
