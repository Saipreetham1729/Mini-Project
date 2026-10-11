"""
Splits a generated answer into individual claims to be checked against evidence.

Baseline approach: one claim per sentence. This is intentionally simple and
explainable -- a real claim can span multiple sentences, or a single sentence
can bundle multiple claims -- but sentence-level segmentation is a reasonable,
inspectable starting point, and it's easy to see exactly why each "claim" was
extracted. Refining this (e.g. LLM-based claim decomposition) is a natural
extension if time permits later.
"""
import re
from typing import List, TypedDict


class Claim(TypedDict):
    claim_id: str
    text: str


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS = 3  # skip fragments like "Yes." or "Okay." -- not really checkable claims


def extract_claims(answer: str) -> List[Claim]:
    if not answer or not answer.strip():
        return []

    sentences = _SENTENCE_SPLIT_RE.split(answer.strip())
    claims: List[Claim] = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) < _MIN_WORDS:
            continue
        claims.append({"claim_id": f"claim_{i}", "text": sentence})
    return claims
