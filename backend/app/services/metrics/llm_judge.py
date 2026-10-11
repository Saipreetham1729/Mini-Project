"""
LLM-based verification: ask the configured LLM to judge, for every claim
at once, whether it's supported by the retrieved evidence. One of the
three core faithfulness metrics for the research comparison.

Design choice: a single LLM call judges all claims together (rather than
one call per claim) to keep API cost and latency bounded -- this pipeline
gets re-run constantly during development and evaluation.

Per the project's "never fabricate results" rule: if the LLM's response
can't be parsed as the expected JSON, every claim gets score=None with an
explanation saying so -- never a guessed number.
"""
import json
import logging
from typing import List, Optional, TypedDict

from app.services.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

_VERDICT_SCORES = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}

_SYSTEM_PROMPT = (
    "You are a strict fact-checking assistant. You will be given a list of numbered claims "
    "and a block of evidence text. For EACH claim, decide whether it is 'supported', "
    "'partially_supported', or 'unsupported' by the evidence -- based only on the evidence, "
    "not outside knowledge. Respond with ONLY a JSON array, one object per claim, in this "
    "exact format: [{\"claim_id\": \"...\", \"verdict\": \"...\", \"explanation\": \"...\"}]. "
    "No text outside the JSON array."
)


class ClaimMetricScore(TypedDict):
    claim_id: str
    score: Optional[float]
    verdict: Optional[str]
    explanation: Optional[str]


def _build_user_prompt(claims: List[dict], evidence_chunks: List[dict]) -> str:
    claims_block = "\n".join(f'{c["claim_id"]}: {c["text"]}' for c in claims)
    evidence_block = "\n\n".join(f"[Page {c['page_number']}] {c['text']}" for c in evidence_chunks)
    return f"Evidence:\n{evidence_block}\n\nClaims:\n{claims_block}\n\nRespond with the JSON array now."


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
        logger.warning("Could not parse LLM judge response as JSON: %s", exc)
        return None


def score_claims(claims: List[dict], evidence_chunks: List[dict]) -> List[ClaimMetricScore]:
    if not claims:
        return []

    if not evidence_chunks:
        return [
            {"claim_id": c["claim_id"], "score": 0.0, "verdict": "unsupported", "explanation": "No evidence was retrieved."}
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
                "explanation": "Not evaluated yet -- LLM judge response could not be parsed.",
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
                    "explanation": "Not evaluated yet -- missing from LLM judge response.",
                }
            )
            continue
        verdict = item.get("verdict")
        results.append(
            {
                "claim_id": claim["claim_id"],
                "score": _VERDICT_SCORES.get(verdict),
                "verdict": verdict,
                "explanation": item.get("explanation"),
            }
        )
    return results
