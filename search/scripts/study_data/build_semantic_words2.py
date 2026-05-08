import argparse
import json
import re
from pathlib import Path


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def collect_target_words(interruptions):
    seen = set()
    words = []
    for item in interruptions:
        word = item.get("target_word")
        if not isinstance(word, str):
            continue
        normalized = normalize_word(word)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        words.append(word)
    return words


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
        help="Unused legacy option kept for compatibility.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Unused legacy option kept for compatibility.",
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional list of file stems to process",
    )
    args = parser.parse_args()
    semantic_dir = Path(args.semantic_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_paths = sorted(target_dir.glob("*.json"))
    if args.stems:
        allowed_stems = set(args.stems)
        target_paths = [path for path in target_paths if path.stem in allowed_stems]

    for target_path in target_paths:
        stem = target_path.stem
        semantic_path = semantic_dir / f"{stem}.json"
        if not semantic_path.exists():
            raise FileNotFoundError(f"Missing semantic words file: {semantic_path}")

        target_data = load_json(target_path)
        semantic_words = load_json(semantic_path)
        words_to_remove = collect_target_words(target_data.get("interruptions", []))
        filtered_words = filter_semantic_words(semantic_words, words_to_remove)

        output_path = output_dir / semantic_path.name
        with output_path.open("w") as f:
            json.dump(filtered_words, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"{stem}: removed target_words={words_to_remove} -> {output_path}")


if __name__ == "__main__":
    main()
