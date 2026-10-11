"""
Computes Precision/Recall/F1 for each automated method against human-labeled
ground truth, treating "is this claim hallucinated (i.e. NOT supported)?" as
the binary classification task.

Only claims with a human_label are used -- unlabeled claims are excluded,
never guessed. If there aren't enough labeled examples yet, or the label
distribution makes precision/recall undefined, the result reports that
plainly rather than fabricating a number.

Simplification worth stating explicitly: all six methods are binarized at
the same 0.5 score threshold here, for a uniform side-by-side comparison --
even though the baseline heuristic's own internal operating thresholds
(support_threshold=0.55, partial_support_threshold=0.35) differ slightly.
Self-consistency in particular measures something conceptually different
(agreement across independent samples, not grounding in retrieved evidence)
so its PRF numbers answer "how well does sampling agreement predict human
labels" rather than "how well does evidence-grounding predict them" --
worth keeping in mind when comparing it against the other five.
"""
import logging
from typing import List, Optional, TypedDict

logger = logging.getLogger(__name__)

_SCORE_THRESHOLD = 0.5  # score >= threshold => predicted "supported" (not hallucinated)


class PrfResult(TypedDict):
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    labeled_examples: int
    note: Optional[str]


def _prf_from_score_field(rows: List[dict], score_field: str) -> PrfResult:
    labeled = [
        r for r in rows
        if r.get("human_label") in ("supported", "unsupported") and r.get(score_field) is not None
    ]
    if not labeled:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "labeled_examples": 0,
            "note": "Not evaluated yet -- no labeled examples with a score for this metric.",
        }

    tp = fp = fn = tn = 0
    for r in labeled:
        predicted_hallucinated = r[score_field] < _SCORE_THRESHOLD
        actual_hallucinated = r["human_label"] == "unsupported"
        if predicted_hallucinated and actual_hallucinated:
            tp += 1
        elif predicted_hallucinated and not actual_hallucinated:
            fp += 1
        elif not predicted_hallucinated and actual_hallucinated:
            fn += 1
        else:
            tn += 1

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall and (precision + recall) > 0
        else None
    )

    note = None
    if precision is None or recall is None:
        note = "Precision or recall undefined for this label distribution (e.g. no positive predictions or no positive ground truth in this sample)."

    return {"precision": precision, "recall": recall, "f1": f1, "labeled_examples": len(labeled), "note": note}


def compute_prf_for_all_methods(rows: List[dict]) -> dict:
    return {
        "baseline_embedding_similarity": _prf_from_score_field(rows, "baseline_score"),
        "rouge_l": _prf_from_score_field(rows, "rouge_l_score"),
        "nli_entailment": _prf_from_score_field(rows, "nli_score"),
        "llm_judge": _prf_from_score_field(rows, "llm_judge_score"),
        "ragas_faithfulness": _prf_from_score_field(rows, "ragas_score"),
        "self_consistency": _prf_from_score_field(rows, "self_consistency_score"),
    }
