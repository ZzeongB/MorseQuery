import argparse
import json
import random
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def sample_positive_normal(rng: random.Random, mean: float, sigma: float) -> float:
    while True:
        value = rng.gauss(mean, sigma)
        if value > 0:
            return round(value, 2)


def pick_interruption(
    candidates,
    used_indexes,
    rng: random.Random,
    *,
    delay_type: str,
    delay_mean: float,
    delay_sigma: float,
    audio_start_time: float,
    min_search_offset_seconds: float,
    max_attempts: int = 500,
):
    for _ in range(max_attempts):
        delay_seconds = sample_positive_normal(rng, delay_mean, delay_sigma)
        min_target_time = audio_start_time + min_search_offset_seconds - delay_seconds
        eligible_indexes = [
            idx
            for idx, item in enumerate(candidates)
            if idx not in used_indexes
            and float(item["time"]) >= min_target_time
        ]
        if not eligible_indexes:
            continue

        chosen_idx = rng.choice(eligible_indexes)
        chosen = candidates[chosen_idx]
        target_word_time = round(float(chosen["time"]), 2)
        search_start_time = round(target_word_time + delay_seconds, 2)
        return chosen_idx, {
            "target_word": chosen["word"],
            "target_word_time": target_word_time,
            "delay_type": delay_type,
            "delay_seconds": delay_seconds,
            "search_start_time": search_start_time,
        }

    raise ValueError(
        f"Could not find a valid {delay_type} interruption after {max_attempts} attempts"
    )


def generate_interruptions(
    candidates,
    rng: random.Random,
    *,
    audio_start_time: float,
    target_window_seconds: float,
    min_search_offset_seconds: float,
    short_count: int,
    long_count: int,
    short_mean: float,
    long_mean: float,
    delay_sigma: float,
    allow_partial: bool,
):
    used_indexes = set()
    interruptions = []

    for delay_type, count, mean in (
        ("short", short_count, short_mean),
        ("long", long_count, long_mean),
    ):
        for _ in range(count):
            try:
                chosen_idx, interruption = pick_interruption(
                    candidates,
                    used_indexes,
                    rng,
                    delay_type=delay_type,
                    delay_mean=mean,
                    delay_sigma=delay_sigma,
                    audio_start_time=audio_start_time,
                    min_search_offset_seconds=min_search_offset_seconds,
                )
            except ValueError:
                if not allow_partial:
                    raise
                break
            used_indexes.add(chosen_idx)
            interruptions.append(interruption)

    interruptions.sort(key=lambda item: item["search_start_time"])
    for idx, interruption in enumerate(interruptions, start=1):
        interruption["id"] = idx

    search_window_end_time = max(
        (item["search_start_time"] for item in interruptions),
        default=round(audio_start_time + target_window_seconds, 2),
    )
    return interruptions, search_window_end_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keywords-dir",
        default="data/semantic_words",
        help="Input directory containing jargon JSON files",
    )
    parser.add_argument(
        "--keywords-suffix",
        default=".jargon.json",
        help="File suffix used to find jargon JSON inputs.",
    )
    parser.add_argument(
        "--audio-windows",
        default="data/study/audio_windows.json",
        help="Path to study audio window config",
    )
    parser.add_argument(
        "--output-dir",
        default="data/study/target_words",
        help="Output directory",
    )
    parser.add_argument(
        "--short-count",
        type=int,
        default=5,
        help="Number of short-delay interruptions per file",
    )
    parser.add_argument(
        "--long-count",
        type=int,
        default=5,
        help="Number of long-delay interruptions per file",
    )
    parser.add_argument(
        "--short-mean",
        type=float,
        default=15.0,
        help="Short delay Gaussian mean in seconds",
    )
    parser.add_argument(
        "--long-mean",
        type=float,
        default=50.0,
        help="Long delay Gaussian mean in seconds",
    )
    parser.add_argument(
        "--delay-sigma",
        type=float,
        default=5.0,
        help="Delay Gaussian sigma in seconds",
    )
    parser.add_argument(
        "--min-search-offset-seconds",
        type=float,
        default=120.0,
        help="Minimum gap between audio_start_time and search_start_time",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write as many valid interruptions as possible instead of failing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output",
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional file stems to process",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keywords_dir = Path(args.keywords_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_windows = load_json(Path(args.audio_windows))
    default_target_window_seconds = float(
        audio_windows.get("default_target_window_seconds", 300)
    )
    video_configs = audio_windows.get("videos", {})

    keyword_paths = sorted(keywords_dir.glob(f"*{args.keywords_suffix}"))
    if args.stems:
        allowed_stems = set(args.stems)
        keyword_paths = [
            path
            for path in keyword_paths
            if path.name[: -len(args.keywords_suffix)] in allowed_stems
        ]

    for keyword_path in keyword_paths:
        video_id = keyword_path.name[: -len(args.keywords_suffix)]
        candidates = load_json(keyword_path)
        video_config = video_configs.get(video_id, {})
        audio_start_time = float(video_config.get("audio_start_time", 0))
        target_window_seconds = float(
            video_config.get("target_window_seconds", default_target_window_seconds)
        )

        interruptions, search_window_end_time = generate_interruptions(
            candidates,
            rng,
            audio_start_time=audio_start_time,
            target_window_seconds=target_window_seconds,
            min_search_offset_seconds=args.min_search_offset_seconds,
            short_count=args.short_count,
            long_count=args.long_count,
            short_mean=args.short_mean,
            long_mean=args.long_mean,
            delay_sigma=args.delay_sigma,
            allow_partial=args.allow_partial,
        )

        output_path = output_dir / f"{video_id}.json"
        payload = {
            "video_id": video_id,
            "audio_start_time": audio_start_time,
            "target_window_seconds": target_window_seconds,
            "search_window_end_time": search_window_end_time,
            "interruptions": interruptions,
        }
        with output_path.open("w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(
            f"{video_id}: wrote {len(interruptions)} interruptions to {output_path}"
        )


if __name__ == "__main__":
    main()
