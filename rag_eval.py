#!/usr/bin/env python3
"""
RAG evaluation runner (Option 6).

Usage:
  python run_rag_eval.py                          # against http://localhost:8000
  python run_rag_eval.py --base-url http://localhost:8000 --k 5
  python run_rag_eval.py --min-score-override 0.0  # see everything the service returns before filter

Measures:
  - hit@1  : expected fact appears in rank-1 context
  - hit@3  : expected fact appears in any of top-3 contexts
  - hit@5  : expected fact appears in any of top-5 contexts
  - empty  : service returned zero contexts (all below MIN_SCORE)
  - source_match : at least one returned context is from the expected source_file
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import urllib.request
except ImportError:
    urllib = None

EVAL_PATH = Path(__file__).resolve().parent / "_answer_key.json"


def ask(base_url: str, question: str, k: int | None = None) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/ask"
    body: dict[str, Any] = {"question": question}
    if k is not None:
        body["k"] = k
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fact_found(contexts: list[dict], expected_facts: list[str]) -> tuple[bool, int | None, float | None]:
    """Return (found, best_rank, best_score) if any expected fact substring appears (case-insensitive)."""
    for ctx in contexts:
        text = (ctx.get("text") or "").lower()
        for fact in expected_facts:
            if fact.lower() in text:
                return True, ctx.get("rank"), ctx.get("score")
    return False, None, None


def source_found(contexts: list[dict], source_file: str) -> bool:
    for ctx in contexts:
        if (ctx.get("source_file") or "") == source_file:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Nexlify RAG evaluation set")
    parser.add_argument("--base-url", default="http://localhost:8000", help="RAG service base URL")
    parser.add_argument("--k", type=int, default=5, help="top-k to request (passed to /ask)")
    parser.add_argument("--eval-file", default=str(EVAL_PATH), help="Path to _answer_key.json")
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests (seconds)")
    parser.add_argument("--json-out", default="", help="Optional path to write full results JSON")
    args = parser.parse_args()

    with open(args.eval_file, encoding="utf-8") as f:
        suite = json.load(f)

    questions = suite["questions"]
    results = []
    hits_at_1 = hits_at_3 = hits_at_5 = empty_count = source_hits = 0

    print(f"RAG eval | base={args.base_url} | k={args.k} | n={len(questions)}")
    print("-" * 88)

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected = q["expected_facts"]
        source = q["source_file"]
        category = q.get("category", "")

        try:
            resp = ask(args.base_url, question, k=args.k)
            contexts = resp.get("contexts") or []
        except Exception as e:
            print(f"{qid}  ERROR  {e}")
            results.append({"id": qid, "error": str(e)})
            time.sleep(args.sleep)
            continue

        found, rank, score = fact_found(contexts, expected)
        src_ok = source_found(contexts, source)
        n_ctx = len(contexts)

        if n_ctx == 0:
            empty_count += 1
            status = "EMPTY"
        elif found and rank is not None and rank <= 1:
            hits_at_1 += 1
            hits_at_3 += 1
            hits_at_5 += 1
            status = "HIT@1"
        elif found and rank is not None and rank <= 3:
            hits_at_3 += 1
            hits_at_5 += 1
            status = "HIT@3"
        elif found and rank is not None and rank <= 5:
            hits_at_5 += 1
            status = "HIT@5"
        elif found:
            status = "HIT@>5"
        else:
            status = "MISS"

        if src_ok:
            source_hits += 1

        score_str = f"{score:.4f}" if score is not None else "-"
        rank_str = str(rank) if rank is not None else "-"
        print(
            f"{qid}  {status:<7}  rank={rank_str:<3} score={score_str:<7}  "
            f"ctx={n_ctx}  src={'Y' if src_ok else 'N'}  [{category}]  {question[:55]}"
        )

        results.append({
            "id": qid,
            "question": question,
            "status": status,
            "rank": rank,
            "score": score,
            "num_contexts": n_ctx,
            "source_match": src_ok,
            "expected_facts": expected,
            "source_file": source,
            "category": category,
            "top_contexts_preview": [
                {
                    "rank": c.get("rank"),
                    "score": c.get("score"),
                    "source_file": c.get("source_file"),
                    "text_preview": (c.get("text") or "")[:180],
                }
                for c in contexts[:3]
            ],
        })
        time.sleep(args.sleep)

    n = len(questions)
    print("-" * 88)
    print(f"SUMMARY  n={n}")
    print(f"  hit@1        {hits_at_1:3d} / {n}  ({100.0 * hits_at_1 / n:.1f}%)")
    print(f"  hit@3        {hits_at_3:3d} / {n}  ({100.0 * hits_at_3 / n:.1f}%)")
    print(f"  hit@5        {hits_at_5:3d} / {n}  ({100.0 * hits_at_5 / n:.1f}%)")
    print(f"  empty        {empty_count:3d} / {n}  ({100.0 * empty_count / n:.1f}%)  ← all chunks below MIN_SCORE")
    print(f"  source_match {source_hits:3d} / {n}  ({100.0 * source_hits / n:.1f}%)  ← correct PDF among returned contexts")
    print()
    print("Interpretation guide:")
    print("  High EMPTY%     → threshold too strict or retrieval too weak (dense-only on short facts).")
    print("  High hit@5, low hit@1 → correct chunk is retrieved but ranking is weak (reranker / hybrid help).")
    print("  Low source_match → wrong documents dominating; check chunk quality and embeddings.")

    if args.json_out:
        out = {
            "summary": {
                "n": n,
                "hit_at_1": hits_at_1,
                "hit_at_3": hits_at_3,
                "hit_at_5": hits_at_5,
                "empty": empty_count,
                "source_match": source_hits,
            },
            "results": results,
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nFull results written to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
