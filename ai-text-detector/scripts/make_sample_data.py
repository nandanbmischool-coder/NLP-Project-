import random
from pathlib import Path

import pandas as pd

random.seed(42)

human_bits = [
    "honestly i dunno, it was kinda weird",
    "we got lost twice lol",
    "my phone died halfway through",
    "can't believe it took that long",
    "yeah so that happened",
    "the coffee was awful but whatever",
    "I mean, it's fine I guess",
    "no idea why she left so early",
    "ugh, Mondays",
    "we just winged it",
]
ai_bits = [
    "In conclusion, it is important to note that this approach offers several advantages",
    "Furthermore, this comprehensive framework facilitates effective decision-making",
    "Overall, these factors collectively contribute to a more robust outcome",
    "It is essential to consider the broader implications of this development",
    "Additionally, leveraging these strategies can significantly enhance productivity",
    "Moreover, the integration of such solutions ensures sustainable growth",
]


def make(bits, n):
    return [
        ". ".join(random.sample(bits, k=random.randint(3, 5))) + "."
        for _ in range(n)
    ]


rows = [(text, 0) for text in make(human_bits, 200)] + [
    (text, 1) for text in make(ai_bits, 200)
]
random.shuffle(rows)
dataset_path = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
dataset_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows, columns=["text", "label"]).to_csv(dataset_path, index=False)
print(f"Wrote {dataset_path} with {len(rows)} rows")
