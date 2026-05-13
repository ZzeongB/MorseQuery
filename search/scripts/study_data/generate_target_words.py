import argparse
import json
import random
import math
import subprocess
import sys
from pathlib import Path

STRICT_MAX_SEARCH_OFFSET_SECONDS = 299.99
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_UNIQUENESS_SCORE = 7.0


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_uniqueness_lookup(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}

    data = load_json(path)
    if not isinstance(data, dict):
        return {}

    analysis_results = data.get("analysis_results", [])
    if not isinstance(analysis_results, list):
        return {}

    lookup: dict[str, dict[str, float]] = {}
    for item in analysis_results:
        if not isinstance(item, dict):
            continue
        srt_file = item.get("srt_file")
        word_data = item.get("word_data")
        if not isinstance(srt_file, str) or not isinstance(word_data, list):
            continue

        stem = Path(srt_file).stem
        stem_lookup: dict[str, float] = {}
        for word_item in word_data:
            if not isinstance(word_item, dict):
                continue
            word = word_item.get("word")
            uniqueness = word_item.get("uniqueness")
            if not isinstance(word, str) or not isinstance(uniqueness, (int, float)):
                continue
            stem_lookup[word.lower()] = float(uniqueness)

        lookup[stem] = stem_lookup

    return lookup


def attach_uniqueness_scores(
    interruptions: list[dict],
    *,
    video_id: str,
    uniqueness_lookup: dict[str, dict[str, float]],
) -> None:
    stem_lookup = uniqueness_lookup.get(video_id, {})
    for interruption in interruptions:
        target_word = interruption.get("target_word")
        if not isinstance(target_word, str):
            interruption["target_word_uniqueness"] = DEFAULT_UNIQUENESS_SCORE
            continue
        interruption["target_word_uniqueness"] = stem_lookup.get(
            target_word.lower(),
            DEFAULT_UNIQUENESS_SCORE,
        )


def candidate_uniqueness(item: dict) -> float:
    uniqueness = item.get("uniqueness")
    if isinstance(uniqueness, (int, float)):
        return float(uniqueness)
    target_word = item.get("word")
    if isinstance(target_word, str):
        return DEFAULT_UNIQUENESS_SCORE
    return DEFAULT_UNIQUENESS_SCORE


def filter_candidates_by_uniqueness(
    candidates: list[dict],
    *,
    min_uniqueness: float,
    max_uniqueness: float,
) -> list[dict]:
    return [
        item
        for item in candidates
        if isinstance(item, dict)
        and min_uniqueness <= candidate_uniqueness(item) <= max_uniqueness
    ]


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return DEFAULT_UNIQUENESS_SCORE, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def uniqueness_distribution_penalty(
    selected_scores: list[float],
    candidate_score: float,
    *,
    target_mean: float,
    target_sd: float,
) -> float:
    new_scores = selected_scores + [candidate_score]
    new_mean, new_sd = mean_and_sd(new_scores)
    return abs(new_mean - target_mean) + abs(new_sd - target_sd)


def sample_positive_normal(rng: random.Random, mean: float, sigma: float) -> float:
    while True:
        value = rng.gauss(mean, sigma)
        if value > 0:
            return round(value, 2)


def capped_search_window_end_time(
    interruptions, *, audio_start_time: float, target_window_seconds: float
) -> float:
    default_end_time = round(
        min(
            audio_start_time + target_window_seconds,
            audio_start_time + STRICT_MAX_SEARCH_OFFSET_SECONDS,
        ),
        2,
    )
    if not interruptions:
        return default_end_time
    return round(
        min(
            max(item["search_start_time"] for item in interruptions),
            audio_start_time + STRICT_MAX_SEARCH_OFFSET_SECONDS,
        ),
        2,
    )


def resolve_video_config(video_configs, video_id: str):
    config = video_configs.get(video_id)
    if config is not None:
        return config
    base_id = video_id.split("_", 1)[0]
    return video_configs.get(base_id, {})


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
    preferred_max_offset_seconds: float,
    hard_max_offset_seconds: float,
    max_attempts: int = 500,
):
    capped_preferred_offset = min(
        preferred_max_offset_seconds, STRICT_MAX_SEARCH_OFFSET_SECONDS
    )
    capped_hard_offset = min(hard_max_offset_seconds, STRICT_MAX_SEARCH_OFFSET_SECONDS)
    soft_limit = audio_start_time + capped_preferred_offset
    hard_limit = audio_start_time + capped_hard_offset

    for _ in range(max_attempts):
        delay_seconds = sample_positive_normal(rng, delay_mean, delay_sigma)
        min_target_time = audio_start_time + min_search_offset_seconds - delay_seconds
        eligible_indexes = [
            idx
            for idx, item in enumerate(candidates)
            if idx not in used_indexes
            and float(item["time"]) >= min_target_time
            and float(item["time"]) + delay_seconds <= hard_limit
        ]
        if not eligible_indexes:
            continue

        preferred_indexes = [
            idx
            for idx in eligible_indexes
            if float(candidates[idx]["time"]) + delay_seconds <= soft_limit
        ]
        pool = preferred_indexes or eligible_indexes
        chosen_idx = rng.choice(pool)
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

    fallback_candidates = []
    for idx, item in enumerate(candidates):
        if idx in used_indexes:
            continue
        target_word_time = round(float(item["time"]), 2)
        min_delay_seconds = round(
            max(0.01, audio_start_time + min_search_offset_seconds - target_word_time),
            2,
        )
        max_delay_seconds = round(hard_limit - target_word_time, 2)
        if min_delay_seconds > max_delay_seconds:
            continue
        delay_seconds = round(
            min(max(delay_mean, min_delay_seconds), max_delay_seconds), 2
        )
        fallback_candidates.append(
            (abs(delay_seconds - delay_mean), idx, target_word_time, delay_seconds)
        )

    if fallback_candidates:
        _distance, chosen_idx, target_word_time, delay_seconds = min(
            fallback_candidates, key=lambda item: item[0]
        )
        chosen = candidates[chosen_idx]
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


def pick_minute_slot_interruption(
    candidates,
    used_indexes,
    rng: random.Random,
    *,
    search_start_time: float,
    forced_delay_type: str,
    short_delay_mean: float,
    short_delay_tolerance: float,
    long_delay_mean: float,
    long_delay_tolerance: float,
):
    # Determine target delay range based on delay_type
    if forced_delay_type == "short":
        delay_min = short_delay_mean - short_delay_tolerance
        delay_max = short_delay_mean + short_delay_tolerance
    else:  # long
        delay_min = long_delay_mean - long_delay_tolerance
        delay_max = long_delay_mean + long_delay_tolerance

    # Target word must be in range: [search_start_time - delay_max, search_start_time - delay_min]
    target_time_min = search_start_time - delay_max
    target_time_max = search_start_time - delay_min

    scored_candidates = []
    for idx, item in enumerate(candidates):
        if idx in used_indexes:
            continue
        target_time = float(item["time"])
        if target_time < target_time_min or target_time > target_time_max:
            continue
        # Score by how close to the mean delay
        delay = search_start_time - target_time
        if forced_delay_type == "short":
            distance = abs(delay - short_delay_mean)
        else:
            distance = abs(delay - long_delay_mean)
        scored_candidates.append((distance, idx, target_time))

    if not scored_candidates:
        raise ValueError(f"Could not find a valid {forced_delay_type} interruption")

    best_distance = min(item[0] for item in scored_candidates)
    closest = [item for item in scored_candidates if item[0] == best_distance]
    _distance, chosen_idx, target_time = rng.choice(closest)
    chosen = candidates[chosen_idx]
    target_word_time = round(target_time, 2)
    fixed_search_start_time = round(search_start_time, 2)
    delay_seconds = round(fixed_search_start_time - target_word_time, 2)

    return chosen_idx, {
        "target_word": chosen["word"],
        "target_word_time": target_word_time,
        "delay_type": forced_delay_type,
        "delay_seconds": delay_seconds,
        "search_start_time": fixed_search_start_time,
    }


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
    preferred_max_offset_seconds: float,
    hard_max_offset_seconds: float,
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
                    preferred_max_offset_seconds=preferred_max_offset_seconds,
                    hard_max_offset_seconds=hard_max_offset_seconds,
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

    search_window_end_time = capped_search_window_end_time(
        interruptions,
        audio_start_time=audio_start_time,
        target_window_seconds=target_window_seconds,
    )
    return interruptions, search_window_end_time


def generate_minute_slot_interruptions(
    candidates,
    rng: random.Random,
    *,
    audio_start_time: float,
    target_window_seconds: float,
    allow_partial: bool,
    long_count: int = 3,
    short_count: int = 3,
    short_delay_mean: float = 10.0,
    short_delay_tolerance: float = 5.0,
    long_delay_mean: float = 50.0,
    long_delay_tolerance: float = 5.0,
    target_uniqueness_mean: float = DEFAULT_UNIQUENESS_SCORE,
    target_uniqueness_sd: float = 0.0,
    uniqueness_balance_weight: float = 1.0,
):
    minute_count = max(1, math.floor(target_window_seconds / 60.0))
    total_count = long_count + short_count

    if minute_count != total_count and not allow_partial:
        raise ValueError(
            f"minute_count ({minute_count}) must equal long_count + short_count ({total_count})"
        )

    # Step 1: Build candidate sets for each minute and delay type
    short_min_delay = short_delay_mean - short_delay_tolerance
    short_max_delay = short_delay_mean + short_delay_tolerance
    long_min_delay = long_delay_mean - long_delay_tolerance
    long_max_delay = long_delay_mean + long_delay_tolerance

    # For each minute, collect eligible candidates for long and short
    minute_candidates = []  # list of (minute_idx, search_start_time, long_candidates, short_candidates)
    for minute_idx in range(minute_count):
        search_start_time = audio_start_time + (minute_idx + 1) * 60.0

        long_cands = []
        short_cands = []
        for idx, item in enumerate(candidates):
            target_time = float(item["time"])
            delay = search_start_time - target_time

            # Check long eligibility
            if long_min_delay <= delay <= long_max_delay:
                distance = abs(delay - long_delay_mean)
                long_cands.append((distance, idx, target_time, delay))

            # Check short eligibility
            if short_min_delay <= delay <= short_max_delay:
                distance = abs(delay - short_delay_mean)
                short_cands.append((distance, idx, target_time, delay))

        minute_candidates.append(
            {
                "minute_idx": minute_idx,
                "search_start_time": search_start_time,
                "long": long_cands,
                "short": short_cands,
            }
        )

    # Step 2: Backtracking assignment. The previous greedy pass could miss valid
    # minute/type combinations even when a full solution exists.
    minute_order = sorted(
        minute_candidates,
        key=lambda mc: (
            int(bool(mc["long"])) + int(bool(mc["short"])),
            len(mc["long"]) + len(mc["short"]),
            mc["minute_idx"],
        ),
    )

    best_exact: tuple[float, list[dict]] | None = None
    best_partial: tuple[int, float, list[dict]] | None = None

    def search(
        pos: int,
        remaining_long: int,
        remaining_short: int,
        used_indexes: set[int],
        selected_scores: list[float],
        chosen_interruptions: list[dict],
        total_cost: float,
    ) -> None:
        nonlocal best_exact, best_partial

        if remaining_long < 0 or remaining_short < 0:
            return

        minutes_left = len(minute_order) - pos
        required_left = remaining_long + remaining_short
        if not allow_partial and required_left > minutes_left:
            return

        if pos >= len(minute_order):
            long_selected = long_count - remaining_long
            short_selected = short_count - remaining_short
            selected_count = long_selected + short_selected
            if remaining_long == 0 and remaining_short == 0:
                if best_exact is None or total_cost < best_exact[0]:
                    best_exact = (total_cost, list(chosen_interruptions))
            elif allow_partial:
                candidate = (selected_count, total_cost, list(chosen_interruptions))
                if best_partial is None or selected_count > best_partial[0] or (
                    selected_count == best_partial[0] and total_cost < best_partial[1]
                ):
                    best_partial = candidate
            return

        mc = minute_order[pos]
        branch_options: list[tuple[str | None, list[tuple[float, int, float, float]]]] = []
        if remaining_long > 0:
            branch_options.append(("long", mc["long"]))
        if remaining_short > 0:
            branch_options.append(("short", mc["short"]))
        if allow_partial or required_left < minutes_left:
            branch_options.append((None, []))

        def branch_priority(
            option: tuple[str | None, list[tuple[float, int, float, float]]]
        ) -> tuple[int, int]:
            delay_type, bucket = option
            if delay_type is None:
                return (2, 0)
            needed = remaining_long if delay_type == "long" else remaining_short
            return (0 if len(bucket) <= needed else 1, len(bucket))

        for delay_type, bucket in sorted(branch_options, key=branch_priority):
            if delay_type is None:
                search(
                    pos + 1,
                    remaining_long,
                    remaining_short,
                    used_indexes,
                    selected_scores,
                    chosen_interruptions,
                    total_cost,
                )
                continue

            available = [c for c in bucket if c[1] not in used_indexes]
            ranked = sorted(
                available,
                key=lambda c: (
                    c[0]
                    + uniqueness_balance_weight
                    * uniqueness_distribution_penalty(
                        selected_scores,
                        candidate_uniqueness(candidates[c[1]]),
                        target_mean=target_uniqueness_mean,
                        target_sd=target_uniqueness_sd,
                    ),
                    c[1],
                ),
            )
            for distance, chosen_idx, target_time, delay in ranked:
                chosen = candidates[chosen_idx]
                score = candidate_uniqueness(chosen)
                penalty = uniqueness_balance_weight * uniqueness_distribution_penalty(
                    selected_scores,
                    score,
                    target_mean=target_uniqueness_mean,
                    target_sd=target_uniqueness_sd,
                )
                next_interruptions = chosen_interruptions + [
                    {
                        "target_word": chosen["word"],
                        "target_word_time": round(target_time, 2),
                        "delay_type": delay_type,
                        "delay_seconds": round(delay, 2),
                        "search_start_time": round(mc["search_start_time"], 2),
                    }
                ]
                search(
                    pos + 1,
                    remaining_long - (1 if delay_type == "long" else 0),
                    remaining_short - (1 if delay_type == "short" else 0),
                    used_indexes | {chosen_idx},
                    selected_scores + [score],
                    next_interruptions,
                    total_cost + distance + penalty,
                )

    search(
        0,
        long_count,
        short_count,
        set(),
        [],
        [],
        0.0,
    )

    chosen_solution: list[dict] | None = None
    if best_exact is not None:
        chosen_solution = best_exact[1]
    elif allow_partial and best_partial is not None:
        chosen_solution = best_partial[2]

    interruptions = chosen_solution or []
    long_selected = sum(1 for item in interruptions if item["delay_type"] == "long")
    short_selected = sum(1 for item in interruptions if item["delay_type"] == "short")

    if not allow_partial:
        if long_selected < long_count:
            raise ValueError(
                f"Could only find {long_selected}/{long_count} long interruptions"
            )
        if short_selected < short_count:
            raise ValueError(
                f"Could only find {short_selected}/{short_count} short interruptions"
            )

    interruptions.sort(key=lambda item: item["target_word_time"])
    for idx, interruption in enumerate(interruptions, start=1):
        interruption["id"] = idx

    search_window_end_time = capped_search_window_end_time(
        interruptions,
        audio_start_time=audio_start_time,
        target_window_seconds=target_window_seconds,
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
        default=3,
        help="Number of short-delay interruptions per file",
    )
    parser.add_argument(
        "--long-count",
        type=int,
        default=3,
        help="Number of long-delay interruptions per file",
    )
    parser.add_argument(
        "--short-mean",
        type=float,
        default=10.0,
        help="Short delay mean in seconds",
    )
    parser.add_argument(
        "--long-mean",
        type=float,
        default=50.0,
        help="Long delay mean in seconds",
    )
    parser.add_argument(
        "--delay-sigma",
        type=float,
        default=5.0,
        help="Delay Gaussian sigma in seconds (for gaussian_delay mode)",
    )
    parser.add_argument(
        "--delay-tolerance",
        type=float,
        default=10.0,
        help="Delay tolerance in seconds (for minute_slots mode): delay = mean ± tolerance",
    )
    parser.add_argument(
        "--min-search-offset-seconds",
        type=float,
        default=120.0,
        help="Minimum gap between audio_start_time and search_start_time",
    )
    parser.add_argument(
        "--preferred-max-offset-seconds",
        type=float,
        default=300.0,
        help="Soft preference for latest search_start_time relative to audio_start_time.",
    )
    parser.add_argument(
        "--hard-max-offset-seconds",
        type=float,
        default=420.0,
        help="Hard cap for latest search_start_time relative to audio_start_time.",
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
    parser.add_argument(
        "--selection-mode",
        choices=("gaussian_delay", "minute_slots"),
        default="minute_slots",
        help="How to select target words from jargon candidates.",
    )
    parser.add_argument(
        "--search-window-seconds",
        type=float,
        default=60.0,
        help="Search/navigation window size exposed to the study UI.",
    )
    parser.add_argument(
        "--strip-video-suffix",
        default=None,
        help="Suffix to strip from video_id for output filename (e.g., '.zero').",
    )
    parser.add_argument(
        "--word-uniqueness-analysis",
        default="data/analysis/word_uniqueness_analysis.json",
        help="Path to word uniqueness analysis JSON used to attach target-word scores.",
    )
    parser.add_argument(
        "--uniqueness-balance-weight",
        type=float,
        default=1.0,
        help="Weight for keeping selected target-word uniqueness distributions similar.",
    )
    parser.add_argument(
        "--min-target-uniqueness",
        type=float,
        default=4.0,
        help="Minimum allowed uniqueness score for target-word candidates.",
    )
    parser.add_argument(
        "--max-target-uniqueness",
        type=float,
        default=7.0,
        help="Inclusive maximum allowed uniqueness score for target-word candidates.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    keywords_dir = Path(args.keywords_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_windows = load_json(Path(args.audio_windows))
    uniqueness_lookup = load_uniqueness_lookup(Path(args.word_uniqueness_analysis))
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

    pooled_candidate_scores: list[float] = []
    for keyword_path in keyword_paths:
        candidates = filter_candidates_by_uniqueness(
            load_json(keyword_path),
            min_uniqueness=args.min_target_uniqueness,
            max_uniqueness=args.max_target_uniqueness,
        )
        if not isinstance(candidates, list):
            continue
        pooled_candidate_scores.extend(
            candidate_uniqueness(item) for item in candidates if isinstance(item, dict)
        )
    target_uniqueness_mean, target_uniqueness_sd = mean_and_sd(pooled_candidate_scores)

    for keyword_path in keyword_paths:
        video_id = keyword_path.name[: -len(args.keywords_suffix)]
        candidates = filter_candidates_by_uniqueness(
            load_json(keyword_path),
            min_uniqueness=args.min_target_uniqueness,
            max_uniqueness=args.max_target_uniqueness,
        )
        video_config = resolve_video_config(video_configs, video_id)
        audio_start_time = float(video_config.get("audio_start_time", 0))
        target_window_seconds = float(
            video_config.get("target_window_seconds", default_target_window_seconds)
        )

        if args.selection_mode == "minute_slots":
            interruptions, search_window_end_time = generate_minute_slot_interruptions(
                candidates,
                rng,
                audio_start_time=audio_start_time,
                target_window_seconds=target_window_seconds,
                allow_partial=args.allow_partial,
                long_count=args.long_count,
                short_count=args.short_count,
                short_delay_mean=args.short_mean,
                short_delay_tolerance=args.delay_tolerance,
                long_delay_mean=args.long_mean,
                long_delay_tolerance=args.delay_tolerance,
                target_uniqueness_mean=target_uniqueness_mean,
                target_uniqueness_sd=target_uniqueness_sd,
                uniqueness_balance_weight=args.uniqueness_balance_weight,
            )
        else:
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
                preferred_max_offset_seconds=args.preferred_max_offset_seconds,
                hard_max_offset_seconds=args.hard_max_offset_seconds,
                allow_partial=args.allow_partial,
            )

        output_video_id = video_id
        if args.strip_video_suffix and video_id.endswith(args.strip_video_suffix):
            output_video_id = video_id[: -len(args.strip_video_suffix)]

        attach_uniqueness_scores(
            interruptions,
            video_id=output_video_id,
            uniqueness_lookup=uniqueness_lookup,
        )

        output_path = output_dir / f"{output_video_id}.json"
        payload = {
            "video_id": output_video_id,
            "audio_start_time": audio_start_time,
            "target_window_seconds": target_window_seconds,
            "search_window_end_time": search_window_end_time,
            "search_window_seconds": float(args.search_window_seconds),
            "selection_mode": args.selection_mode,
            "interruptions": interruptions,
        }
        with output_path.open("w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"{video_id}: wrote {len(interruptions)} interruptions to {output_path}")

    plot_script = ROOT_DIR / "scripts" / "analysis" / "plot_jargon_word_distribution.py"
    subprocess.run([sys.executable, str(plot_script)], check=True, cwd=ROOT_DIR)

    uniqueness_plot_script = (
        ROOT_DIR / "scripts" / "analysis" / "plot_target_word_uniqueness_study.py"
    )
    subprocess.run([sys.executable, str(uniqueness_plot_script)], check=True, cwd=ROOT_DIR)


if __name__ == "__main__":
    main()
