import argparse
import json
import re
from pathlib import Path

DEFAULT_INTERRUPTION_BUFFER_SECONDS = 1.0


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


def filter_near_interruption_times(
    semantic_words,
    interruptions,
    buffer_seconds: float,
    words_to_keep: list[str] | None = None,
):
    """Filter out keywords that are within buffer_seconds of any search_start_time.

    Args:
        semantic_words: List of keyword items with "word" and "time" fields.
        interruptions: List of interruption configs with "search_start_time".
        buffer_seconds: Keywords within this many seconds of search_start_time are removed.
        words_to_keep: Optional list of words to never remove (e.g., target_words).

    Returns:
        Tuple of (filtered_words, removed_count, removed_words).
    """
    search_times = [item["search_start_time"] for item in interruptions]
    keep_set = {normalize_word(word) for word in (words_to_keep or [])}

    filtered = []
    removed = []
    for item in semantic_words:
        word_time = item.get("time", 0)
        word_normalized = normalize_word(item["word"])

        # Never remove words in keep_set (e.g., target_words)
        if word_normalized in keep_set:
            filtered.append(item)
            continue

        # Check if too close to any search_start_time
        too_close = any(
            abs(word_time - search_time) <= buffer_seconds
            for search_time in search_times
        )
        if too_close:
            removed.append(item["word"])
        else:
            filtered.append(item)

    return filtered, len(removed), removed


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
    parser.add_argument(
        "--interruption-buffer-seconds",
        type=float,
        default=DEFAULT_INTERRUPTION_BUFFER_SECONDS,
        help="Remove keywords within this many seconds of search_start_time.",
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
        interruptions = target_data.get("interruptions", [])
        semantic_words = load_json(semantic_path)
        target_words_list = collect_target_words(interruptions)

        # Step 1: Filter semantic_words by time proximity (keep target_words)
        semantic_filtered, semantic_near_count, semantic_near_words = (
            filter_near_interruption_times(
                semantic_words,
                interruptions,
                args.interruption_buffer_seconds,
                words_to_keep=target_words_list,
            )
        )

        # Write filtered semantic_words back to semantic_dir
        with semantic_path.open("w") as f:
            json.dump(semantic_filtered, f, ensure_ascii=False, indent=2)
            f.write("\n")

        # Step 2: Remove target_words from filtered semantic_words -> semantic_words2
        final_filtered = filter_semantic_words(semantic_filtered, target_words_list)

        output_path = output_dir / semantic_path.name
        with output_path.open("w") as f:
            json.dump(final_filtered, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(
            f"{stem}: "
            f"removed {semantic_near_count} near-interruption keywords {semantic_near_words}, "
            f"removed target_words={target_words_list} "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
