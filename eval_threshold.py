"""
Empirically tunes MAX_DISTANCE using the known-answer test set (pdfs/_answer_key.json).

For each company in the answer key, asks a question whose answer lives in
exactly one file, then checks: at various thresholds, does retrieval return
ONLY chunks from the correct file, or does noise from other companies leak in?

This turns threshold-picking from guesswork into something you can actually
measure -- the same idea applies once you're on your real 2000 PDFs: build a
small set of questions with known correct source files, then run this against
your own real vector store.

Run: python eval_threshold.py
Requires: the 20 test PDFs already indexed (POST /index or CLI).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rag_pipeline as rag

ANSWER_KEY = Path(__file__).parent / "pdfs" / "_answer_key.json"
THRESHOLDS_TO_TEST = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def main():
    facts = json.loads(ANSWER_KEY.read_text())

    print(f"Testing {len(THRESHOLDS_TO_TEST)} thresholds against {len(facts)} known questions...\n")
    print(f"{'threshold':>10} | {'correct-only':>12} | {'noise leaked':>12} | {'missed (too strict)':>20}")
    print("-" * 65)

    for threshold in THRESHOLDS_TO_TEST:
        correct_only = 0   # question retrieved ONLY chunks from the right file -- ideal
        noise_leaked = 0   # question retrieved the right file PLUS wrong ones -- what you saw
        missed = 0         # question retrieved nothing, or missed the right file entirely -- threshold too strict

        for fact in facts:
            expected_file = fact["file"]
            question = f"What was {fact['company']}'s Q3 revenue in {fact['year']}?"

            chunks = rag.retrieve(question, top_k=5, max_distance=threshold)
            sources = {Path(c["source"]).name for c in chunks}

            if not sources or expected_file not in sources:
                missed += 1
            elif sources == {expected_file}:
                correct_only += 1
            else:
                noise_leaked += 1

        print(f"{threshold:>10.2f} | {correct_only:>12} | {noise_leaked:>12} | {missed:>20}")

    print("\nPick the threshold with the highest 'correct-only' count and lowest "
          "'missed' count. If every threshold has a high 'missed' count, top_k or "
          "chunking may need attention too, not just the distance cutoff.")


if __name__ == "__main__":
    main()
