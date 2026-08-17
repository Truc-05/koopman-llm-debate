"""Tải MMLU (cais/mmlu, split test) qua HuggingFace datasets-server API —
không cần cài `datasets`, không cần token — rồi convert về format topics của
run_debate_groq.py: [{question, answers, a_star, subject}].

Chạy :  python experiments/prepare_mmlu.py --n 40
Ra   :  experiments/topics/mmlu_40.json
Sau đó: python experiments/run_debate_groq.py --topics experiments/topics/mmlu_40.json
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import requests

API = "https://datasets-server.huggingface.co/rows"
DATASET = "cais/mmlu"
TOTAL_TEST = 14042            # num_rows_total của config 'all', split test
PAGE = 100                    # max length mỗi request của datasets-server


def fetch_page(offset, split="test"):
    r = requests.get(API, params={"dataset": DATASET, "config": "all",
                                  "split": split, "offset": int(offset),
                                  "length": PAGE},
                     timeout=60)
    r.raise_for_status()
    return [item["row"] for item in r.json()["rows"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="số câu hỏi lấy ra")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_pages = max(2, -(-2 * args.n // PAGE))          # ~2n dòng để lọc/dedupe
    offsets = rng.choice(TOTAL_TEST - PAGE, size=n_pages, replace=False)

    pool, seen = [], set()
    for off in offsets:
        print(f"fetch offset {int(off)} ...")
        for row in fetch_page(off, split=args.split):
            key = row["question"][:100]
            if key in seen or len(row["choices"]) != 4:
                continue
            seen.add(key)
            pool.append({
                "question": row["question"],
                "answers": [str(c) for c in row["choices"]],
                "a_star": int(row["answer"]),
                "subject": row.get("subject", ""),
            })

    order = rng.permutation(len(pool))
    topics = [pool[i] for i in order[:args.n]]

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "topics",
        f"mmlu_{len(topics)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(topics, f, ensure_ascii=False, indent=1)

    print(f"\nsaved {len(topics)} topics -> {out}")
    for subject, c in Counter(t["subject"] for t in topics).most_common():
        print(f"  {c:2d}  {subject}")


if __name__ == "__main__":
    main()
