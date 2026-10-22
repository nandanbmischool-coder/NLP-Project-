"""Build data/dataset.csv from artem9k/ai-text-detection-pile.

The Pile contains a balanced human/AI source label and is streamed so the
full corpus can be prepared without loading 1.39M rows into memory.
"""

import argparse
import hashlib
import random
import re
from pathlib import Path

from datasets import load_dataset

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "dataset.csv"
REPO = "artem9k/ai-text-detection-pile"
MIN_WORDS, MAX_WORDS = 20, 500
URL = re.compile(r"https?://\S+")
rng = random.Random(42)


def clean(text) -> str:
    text = URL.sub(" ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:MAX_WORDS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=700_000)
    args = parser.parse_args()
    counts = {"human": 0, "ai": 0}
    seen: set[str] = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as output:
        output.write("text,label,group,weight\n")
        dataset = load_dataset(REPO, split="train", streaming=True)
        for row in dataset:
            source = str(row.get("source", "")).lower()
            if source not in counts or counts[source] >= args.max_per_class:
                if all(value >= args.max_per_class for value in counts.values()):
                    break
                continue
            text = clean(row.get("text", ""))
            if len(text.split()) < MIN_WORDS:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            label = 1 if source == "ai" else 0
            escaped = '"' + text.replace('"', '""') + '"'
            output.write(f"{escaped},{label},{source}-{row.get('id')},1.0\n")
            counts[source] += 1
            if sum(counts.values()) % 10_000 == 0:
                print(f"Prepared human={counts['human']:,}, ai={counts['ai']:,}", flush=True)
    print(f"Prepared human={counts['human']:,}, ai={counts['ai']:,}")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
