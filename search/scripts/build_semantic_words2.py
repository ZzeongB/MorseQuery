import argparse
import json
import random
import re
from pathlib import Path


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def pick_words_to_remove(interruptions, rng, count_per_type):
    selected = {}
    for delay_type in ("short", "long"):
        candidates = [
            item["target_word"]
            for item in interruptions
            if item.get("delay_type") == delay_type
        ]
        if len(candidates) < count_per_type:
            raise ValueError(
                f"Not enough {delay_type} candidates: "
                f"expected {count_per_type}, found {len(candidates)}"
            )
        selected[delay_type] = rng.sample(candidates, count_per_type)
    return selected


def filter_semantic_words(semantic_words, words_to_remove):
    remove_set = {normalize_word(word) for word in words_to_remove}
    return [
        item
        for item in semantic_words
        if normalize_word(item["word"]) not in remove_set
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic-dir",
        default="data/semantic_words",
        help="Input semantic words directory",
    )
    parser.add_argument(
        "--target-dir",
        default="data/study/target_words",
        help="Target words directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/semantic_words2",
        help="Output directory",
    )
    parser.add_argument(
        "--count-per-type",
        type=int,
        default=2,
        help="How many words to remove per delay type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    semantic_dir = Path(args.semantic_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target_path in sorted(target_dir.glob("*.json")):
        stem = target_path.stem
        semantic_path = semantic_dir / f"{stem}.json"
        if not semantic_path.exists():
            raise FileNotFoundError(f"Missing semantic words file: {semantic_path}")

        target_data = load_json(target_path)
        semantic_words = load_json(semantic_path)
        selected = pick_words_to_remove(
            target_data.get("interruptions", []),
            rng,
            args.count_per_type,
        )

        words_to_remove = selected["short"] + selected["long"]
        filtered_words = filter_semantic_words(semantic_words, words_to_remove)

        output_path = output_dir / semantic_path.name
        with output_path.open("w") as f:
            json.dump(filtered_words, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(
            f"{stem}: removed short={selected['short']} "
            f"long={selected['long']} -> {output_path}"
        )


if __name__ == "__main__":
    main()
