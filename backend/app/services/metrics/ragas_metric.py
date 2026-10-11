"""
RAGAS-style faithfulness metric.

RAGAS's actual faithfulness score works in two stages: decompose the answer
into atomic statements, then check each statement for whether it can be
*directly inferred* from the given context -- a strictly binary judgement
(yes/no), unlike this project's LLM-judge metric, which allows a
"partially_supported" middle verdict. The final score is
supported_statements / total_statements.

We reuse this project's existing claim extractor for the statement
decomposition step (claims already are atomic, checkable units -- the
same thing RAGAS calls "statements"), and implement the binary inference
check ourselves rather than depending on the `ragas` package directly, to
keep this pipeline's LLM-provider abstraction (mock/openai) as the single
place that talks to a model. This makes ragas_faithfulness a genuinely
distinct measurement from llm_judge for the comparison this project is
about: same underlying idea, different verdict granularity and prompt.

Per the project's "never fabricate results" rule: if the LLM's response
can't be parsed, every claim gets score=None with an explanation, not a
guessed number.
"""
import json
import logging
from typing import List, Optional, TypedDict

from app.services.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are evaluating whether statements can be directly inferred from a given context, "
    "following the RAGAS faithfulness methodology. You will be given a list of numbered "
    "statements and a block of context. For EACH statement, answer STRICTLY 'yes' or 'no': "
    "'yes' only if the statement can be directly inferred from the context alone, with no "
    "outside knowledge or assumptions. There is no partial credit -- if any part of the "
    "statement is not directly supported, answer 'no'. Respond with ONLY a JSON array, one "
    "object per statement, in this exact format: "
    '[{"claim_id": "...", "verdict": "yes"|"no", "reason": "..."}]. No text outside the JSON array.'
)


class ClaimMetricScore(TypedDict):
    claim_id: str
    score: Optional[float]
    verdict: Optional[str]
    reason: Optional[str]


def _build_user_prompt(claims: List[dict], evidence_chunks: List[dict]) -> str:
    statements_block = "\n".join(f'{c["claim_id"]}: {c["text"]}' for c in claims)
    context_block = "\n\n".join(f"[Page {c['page_number']}] {c['text']}" for c in evidence_chunks)
    return f"Context:\n{context_block}\n\nStatements:\n{statements_block}\n\nRespond with the JSON array now."


def _parse_response(raw_response: str) -> Optional[dict]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        parsed = json.loads(cleaned)
        return {item["claim_id"]: item for item in parsed}
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("Could not parse RAGAS faithfulness response as JSON: %s", exc)
        return None


def score_claims(claims: List[dict], evidence_chunks: List[dict]) -> List[ClaimMetricScore]:
    if not claims:
        return []

    if not evidence_chunks:
        return [
            {"claim_id": c["claim_id"], "score": 0.0, "verdict": "no", "reason": "No context was retrieved."}
            for c in claims
        ]

    llm = get_llm_provider()
    raw_response = llm.generate(_SYSTEM_PROMPT, _build_user_prompt(claims, evidence_chunks))
    by_id = _parse_response(raw_response)

    if by_id is None:
        return [
            {
                "claim_id": c["claim_id"],
                "score": None,
                "verdict": None,
                "reason": "Not evaluated yet -- RAGAS faithfulness response could not be parsed.",
            }
            for c in claims
        ]

    results: List[ClaimMetricScore] = []
    for claim in claims:
        item = by_id.get(claim["claim_id"])
        if item is None:
            results.append(
                {
                    "claim_id": claim["claim_id"],
                    "score": None,
                    "verdict": None,
                    "reason": "Not evaluated yet -- missing from RAGAS faithfulness response.",
                }
            )
            continue
        verdict = item.get("verdict")
        score = 1.0 if verdict == "yes" else 0.0 if verdict == "no" else None
        results.append(
            {
                "claim_id": claim["claim_id"],
                "score": score,
                "verdict": verdict,
                "reason": item.get("reason"),
            }
        )
    return results
