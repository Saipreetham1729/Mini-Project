"""
Interactive labeling for benchmark claims -- deliberately the slow, honest
step this project's "never fabricate results" rule requires. This script
only makes the labeling loop faster to run; every label is still your own
judgment, typed in by you, one claim at a time.

Deliberately blind by design: this shows you the question, the generated
answer, and the claim -- NOT the baseline status or any metric's score.
Labeling blind to the automated verdicts avoids anchoring your own judgment
on them, which is standard practice for building a credible evaluation set
(and a sentence worth keeping for your IEEE paper's methods section).

Usage (from backend/):
    python3 -m scripts.label_demo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import repository


def main():
    unlabeled = repository.list_benchmark_claims(labeled_only=False)
    unlabeled = [c for c in unlabeled if not c.get("human_label")]

    if not unlabeled:
        print("Nothing left to label -- every recorded claim already has a human_label.")
        return

    print(f"{len(unlabeled)} claim(s) to label. For each: is the claim actually supported")
    print("by the answer's context, in your own judgment? [s]upported / [u]nsupported / [k]ip / [q]uit\n")

    labeled_count = 0
    for claim in unlabeled:
        print(f"Q: {claim['question']}")
        print(f"Answer: {claim['answer_text']}")
        print(f"Claim:  {claim['claim_text']}")
        choice = input("  [s/u/k/q] > ").strip().lower()

        if choice == "q":
            break
        if choice == "k":
            print()
            continue
        if choice not in ("s", "u"):
            print("  (not recognized, skipping)\n")
            continue

        label = "supported" if choice == "s" else "unsupported"
        repository.set_human_label(claim["id"], label)
        labeled_count += 1
        print()

    print(f"Labeled {labeled_count} claim(s). Check the Dashboard page for the updated P/R/F1 table.")


if __name__ == "__main__":
    main()
