"""
Aggregates data across all recorded benchmark claims into summary
statistics for the research dashboard -- counts, distributions, and the
same P/R/F1 comparison as /benchmark/metrics, gathered in one place so the
frontend dashboard needs a single request.
"""
import logging
from collections import Counter

from app.db import repository
from app.services.benchmark_metrics import compute_prf_for_all_methods

logger = logging.getLogger(__name__)


def build_dashboard_summary() -> dict:
    all_claims = repository.list_benchmark_claims()
    labeled_claims = [c for c in all_claims if c.get("human_label")]
    questions = repository.list_benchmark_questions()

    status_counts = Counter(c["baseline_status"] for c in all_claims if c.get("baseline_status"))
    failure_type_counts = Counter(
        c["likely_failure_type"] for c in all_claims if c.get("likely_failure_type")
    )
    model_counts = Counter(c["model_name"] for c in all_claims if c.get("model_name"))

    latencies = [c["latency_seconds"] for c in all_claims if c.get("latency_seconds") is not None]
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else None

    prf = compute_prf_for_all_methods(labeled_claims)

    return {
        "total_questions": len(questions),
        "total_claims": len(all_claims),
        "labeled_claims": len(labeled_claims),
        "unlabeled_claims": len(all_claims) - len(labeled_claims),
        "status_counts": dict(status_counts),
        "failure_type_counts": dict(failure_type_counts),
        "model_counts": dict(model_counts),
        "avg_latency_seconds": avg_latency,
        "prf_by_method": prf,
    }
