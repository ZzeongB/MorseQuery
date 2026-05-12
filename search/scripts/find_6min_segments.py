"""
Find optimal 6-minute segments for user study.

Criteria:
1. Each 1-minute window must have at least 1 target word
2. Prefer target words in 0-20s or 40-60s of each minute
3. Word uniqueness distribution should be similar across all SRT files
"""

import json
import random
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_analysis_results(json_path):
    """Load the word uniqueness analysis results."""
    with open(json_path, "r") as f:
        return json.load(f)


def timestamp_to_seconds(ts):
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    hours, minutes, seconds, millis = map(int, re.split(r"[:,]", ts))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def load_srt_entries(srt_path):
    """Load SRT cues from the raw .srt file."""
    content = srt_path.read_text(encoding="utf-8")
    pattern = (
        r"(\d+)\n"
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n"
        r"(.*?)(?=\n\n|\Z)"
    )
    entries = []
    for idx, start_ts, end_ts, text in re.findall(pattern, content, re.DOTALL):
        cleaned_text = " ".join(line.strip() for line in text.splitlines()).strip()
        if not cleaned_text:
            continue
        entries.append(
            {
                "index": int(idx),
                "text": cleaned_text,
                "start": timestamp_to_seconds(start_ts),
                "end": timestamp_to_seconds(end_ts),
            }
        )
    return entries


def get_words_in_time_range(word_data, start_time, end_time):
    """Get all words within a time range."""
    return [w for w in word_data if start_time <= w["first_time"] < end_time]


def get_candidates_in_time_range(candidates, start_time, end_time):
    """Get target word candidates within a time range."""
    return [c for c in candidates if start_time <= c["first_time"] < end_time]


def is_preferred_position(time_in_minute):
    """Check if time is in preferred position (0-20s or 40-60s of the minute)."""
    second_in_minute = time_in_minute % 60
    return (0 <= second_in_minute <= 20) or (40 <= second_in_minute <= 60)


def score_6min_segment(word_data, candidates, segment_start):
    """
    Score a 6-minute segment based on criteria.

    Returns:
        dict with score details, or None if segment doesn't meet minimum criteria
    """
    segment_end = segment_start + 360  # 6 minutes

    # Get all words and candidates in this segment
    segment_words = get_words_in_time_range(word_data, segment_start, segment_end)
    segment_candidates = get_candidates_in_time_range(
        candidates, segment_start, segment_end
    )

    if not segment_words:
        return None, "no_words_in_segment"

    # Check each 1-minute bucket
    minute_details = []
    all_minutes_have_candidate = True
    preferred_count = 0

    for minute_idx in range(6):
        minute_start = segment_start + minute_idx * 60
        minute_end = minute_start + 60

        minute_candidates = get_candidates_in_time_range(
            segment_candidates, minute_start, minute_end
        )

        if not minute_candidates:
            all_minutes_have_candidate = False
            minute_details.append(
                {
                    "minute": minute_idx,
                    "abs_minute": int(minute_start // 60),
                    "has_candidate": False,
                    "candidates": [],
                    "preferred_candidate": None,
                }
            )
        else:
            # Find best candidate (prefer 0-20s or 40-60s position)
            best_candidate = None
            for c in minute_candidates:
                if is_preferred_position(c["first_time"]):
                    if (
                        best_candidate is None
                        or c["uniqueness"] > best_candidate["uniqueness"]
                    ):
                        best_candidate = c

            # If no preferred position candidate, take highest uniqueness
            if best_candidate is None:
                best_candidate = max(minute_candidates, key=lambda x: x["uniqueness"])
            else:
                preferred_count += 1

            minute_details.append(
                {
                    "minute": minute_idx,
                    "abs_minute": int(minute_start // 60),
                    "has_candidate": True,
                    "candidates": [
                        {
                            "word": c["word"],
                            "time": c["first_time"],
                            "uniqueness": c["uniqueness"],
                            "preferred_pos": is_preferred_position(c["first_time"]),
                        }
                        for c in minute_candidates
                    ],
                    "preferred_candidate": {
                        "word": best_candidate["word"],
                        "time": best_candidate["first_time"],
                        "uniqueness": best_candidate["uniqueness"],
                        "is_preferred": is_preferred_position(
                            best_candidate["first_time"]
                        ),
                    },
                }
            )

    if not all_minutes_have_candidate:
        return None, "missing_candidate_in_minute"

    # Calculate uniqueness distribution for this segment
    segment_non_stopword = [w for w in segment_words if not w.get("is_stopword", False)]
    uniqueness_scores = [w["uniqueness"] for w in segment_non_stopword]

    if not uniqueness_scores:
        return None, "no_non_stopword_uniqueness"

    # Selected target words (one per minute)
    selected_targets = [m["preferred_candidate"] for m in minute_details]
    target_uniqueness = [t["uniqueness"] for t in selected_targets]

    return {
        "segment_start": segment_start,
        "segment_end": segment_end,
        "segment_start_min": segment_start / 60,
        "segment_end_min": segment_end / 60,
        "minute_details": minute_details,
        "selected_targets": selected_targets,
        "preferred_position_count": preferred_count,
        "distribution": {
            "mean": np.mean(uniqueness_scores),
            "std": np.std(uniqueness_scores),
            "median": np.median(uniqueness_scores),
            "n_words": len(uniqueness_scores),
        },
        "target_distribution": {
            "mean": np.mean(target_uniqueness),
            "std": np.std(target_uniqueness),
            "min": min(target_uniqueness),
            "max": max(target_uniqueness),
        },
    }, None


def sample_srt_end_checks(entries, total_duration, sample_size=10, seed=42):
    """Sample SRT cue-end segment starts that still allow a full 6-minute window."""
    max_start = total_duration - 360
    valid_entries = [entry for entry in entries if entry["end"] <= max_start]

    if len(valid_entries) <= sample_size:
        sampled = valid_entries
    else:
        rng = random.Random(seed)
        sampled = rng.sample(valid_entries, sample_size)
        sampled.sort(key=lambda entry: entry["end"])

    return sampled


def find_all_valid_segments(word_data, candidates, sampled_entries):
    """Find valid 6-minute segments from sampled SRT cue-end starts."""
    valid_segments = []
    sampled_checks = []

    for entry in sampled_entries:
        start = entry["end"]
        result, invalid_reason = score_6min_segment(word_data, candidates, start)
        sampled_checks.append(
            {
                "srt_index": entry["index"],
                "srt_text": entry["text"],
                "srt_start": entry["start"],
                "srt_end": entry["end"],
                "segment_start": start,
                "segment_end": start + 360,
                "is_valid": result is not None,
                "invalid_reason": invalid_reason,
            }
        )
        if result:
            result["anchor_srt"] = {
                "index": entry["index"],
                "text": entry["text"],
                "start": entry["start"],
                "end": entry["end"],
            }
            valid_segments.append(result)

    return valid_segments, sampled_checks


def compare_distributions(segments_by_file):
    """
    Compare segment distributions across files to find similar ones.

    Returns list of segment combinations with similarity scores.
    """
    file_names = list(segments_by_file.keys())

    if len(file_names) < 2:
        return []

    # For each file, get distribution stats of all valid segments
    results = []

    # Get all valid segments per file
    for file_name, segments in segments_by_file.items():
        if segments:
            print(f"\n{file_name}: {len(segments)} valid 6-min segments")
            for seg in segments[:5]:  # Show first 5
                print(
                    f"  {seg['segment_start_min']:.0f}-{seg['segment_end_min']:.0f}min: "
                    f"mean={seg['distribution']['mean']:.2f}, "
                    f"preferred={seg['preferred_position_count']}/6, "
                    f"targets: {[t['word'] for t in seg['selected_targets']]}"
                )

    return segments_by_file


def find_best_matching_segments(segments_by_file, n_top=5):
    """
    Find segment combinations where distributions are most similar across files.
    """
    file_names = list(segments_by_file.keys())

    # Collect all segment means
    all_means = []
    for file_name, segments in segments_by_file.items():
        for seg in segments:
            all_means.append(seg["distribution"]["mean"])

    if not all_means:
        return []

    # Target: find segments with similar mean uniqueness
    global_mean = np.mean(all_means)
    global_std = np.std(all_means)

    print(f"\nGlobal distribution: mean={global_mean:.2f}, std={global_std:.2f}")

    # Score each segment by how close it is to global mean AND preferred position count
    scored_segments = {}
    for file_name, segments in segments_by_file.items():
        scored = []
        for seg in segments:
            # Distance from global mean (lower is better)
            mean_diff = abs(seg["distribution"]["mean"] - global_mean)

            # Preferred position bonus (higher is better)
            preferred_bonus = seg["preferred_position_count"] / 6

            # Combined score (lower is better)
            score = mean_diff - preferred_bonus * 0.5

            scored.append(
                {
                    "segment": seg,
                    "score": score,
                    "mean_diff": mean_diff,
                    "preferred_ratio": preferred_bonus,
                }
            )

        # Sort by score (lower is better)
        scored.sort(key=lambda x: x["score"])
        scored_segments[file_name] = scored[:n_top]

    return scored_segments


def print_recommendations(scored_segments):
    """Print recommended segments for each file."""
    print("\n" + "=" * 80)
    print("RECOMMENDED 6-MINUTE SEGMENTS")
    print("=" * 80)

    for file_name, scored in scored_segments.items():
        print(f"\n{'='*40}")
        print(f"{file_name}")
        print(f"{'='*40}")

        for i, item in enumerate(scored):
            seg = item["segment"]
            print(
                f"\n[Rank {i+1}] {seg['segment_start_min']:.0f}m - {seg['segment_end_min']:.0f}m"
            )
            print(
                f"  Score: {item['score']:.3f} (mean_diff={item['mean_diff']:.2f}, preferred={item['preferred_ratio']*100:.0f}%)"
            )
            print(
                f"  Distribution: mean={seg['distribution']['mean']:.2f}, std={seg['distribution']['std']:.2f}"
            )
            print(
                f"  Target words ({seg['preferred_position_count']}/6 in preferred position):"
            )

            for m in seg["minute_details"]:
                t = m["preferred_candidate"]
                pos_mark = "✓" if t["is_preferred"] else "✗"
                time_in_min = t["time"] % 60
                print(
                    f"    [{m['abs_minute']}m] {t['word']}: {time_in_min:.0f}s, uniq={t['uniqueness']:.2f} {pos_mark}"
                )


def create_comparison_plot(scored_segments, output_dir):
    """Create visualization comparing recommended segments across files."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    file_names = list(scored_segments.keys())

    for idx, file_name in enumerate(file_names):
        if idx >= 5:
            break

        ax = axes[idx]
        scored = scored_segments[file_name]

        if not scored:
            ax.text(0.5, 0.5, "No valid segments", ha="center", va="center")
            ax.set_title(file_name)
            continue

        # Plot top segment's target words
        top_seg = scored[0]["segment"]
        minutes = list(range(6))
        uniqueness = [
            m["preferred_candidate"]["uniqueness"] for m in top_seg["minute_details"]
        ]
        preferred = [
            m["preferred_candidate"]["is_preferred"] for m in top_seg["minute_details"]
        ]
        words = [m["preferred_candidate"]["word"] for m in top_seg["minute_details"]]

        colors = ["green" if p else "orange" for p in preferred]
        bars = ax.bar(minutes, uniqueness, color=colors, edgecolor="black")

        # Add word labels
        for i, (bar, word) in enumerate(zip(bars, words)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                word,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

        ax.set_xlabel("Minute in segment")
        ax.set_ylabel("Uniqueness")
        ax.set_title(
            f'{file_name}\n{top_seg["segment_start_min"]:.0f}m-{top_seg["segment_end_min"]:.0f}m '
            f'(mean={top_seg["distribution"]["mean"]:.2f})'
        )
        ax.set_ylim(0, 8)
        ax.axhline(y=5.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=6.8, color="gray", linestyle="--", alpha=0.5)

    # Hide unused subplot
    axes[5].axis("off")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Preferred position (0-20s, 40-60s)"),
        Patch(facecolor="orange", label="Non-preferred position (20-40s)"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.suptitle(
        "Recommended 6-Minute Segments - Target Words per Minute",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "recommended_6min_segments.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved comparison plot: {output_path}")


def main():
    # Load data
    output_dir = Path("/Users/jeongin/morsequery/search/data/analysis")
    json_path = output_dir / "word_uniqueness_analysis.json"
    srt_dir = Path("/Users/jeongin/morsequery/search/data/mp3/srt")

    data = load_analysis_results(json_path)
    analysis_results = data["analysis_results"]
    candidates_by_file = data["candidates"]

    print("=" * 80)
    print("FINDING OPTIMAL 6-MINUTE SEGMENTS")
    print("=" * 80)

    # Find valid segments for each file
    segments_by_file = {}
    sampled_checks_by_file = {}

    for result in analysis_results:
        file_name = result["srt_file"]
        word_data = result["word_data"]
        total_duration = result["total_duration"]
        candidates = candidates_by_file.get(file_name, [])
        srt_entries = load_srt_entries(srt_dir / file_name)
        sampled_entries = sample_srt_end_checks(
            srt_entries, total_duration, sample_size=10, seed=42
        )

        print(f"\nProcessing {file_name} (duration: {total_duration/60:.1f} min)...")
        print(f"  Sampled {len(sampled_entries)} SRT cue-end anchors")

        valid_segments, sampled_checks = find_all_valid_segments(
            word_data, candidates, sampled_entries
        )
        segments_by_file[file_name] = valid_segments
        sampled_checks_by_file[file_name] = sampled_checks

        print(f"  Found {len(valid_segments)} valid 6-minute segments")

    # Compare and find best matches
    compare_distributions(segments_by_file)

    # Get recommendations
    scored_segments = find_best_matching_segments(segments_by_file, n_top=3)

    # Print recommendations
    print_recommendations(scored_segments)

    # Save results to JSON before plotting so analysis survives plot issues.
    results_json = {
        "segments_by_file": {
            file_name: [
                {
                    "segment_start": s["segment"]["segment_start"],
                    "segment_end": s["segment"]["segment_end"],
                    "segment_start_min": s["segment"]["segment_start_min"],
                    "segment_end_min": s["segment"]["segment_end_min"],
                    "anchor_srt": s["segment"]["anchor_srt"],
                    "score": s["score"],
                    "mean_diff": s["mean_diff"],
                    "preferred_ratio": s["preferred_ratio"],
                    "distribution": s["segment"]["distribution"],
                    "target_distribution": s["segment"]["target_distribution"],
                    "selected_targets": s["segment"]["selected_targets"],
                }
                for s in scored
            ]
            for file_name, scored in scored_segments.items()
        },
        "sampled_srt_checks": sampled_checks_by_file,
    }

    json_output = output_dir / "recommended_segments.json"
    with open(json_output, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved recommendations to: {json_output}")

    # Create visualization
    try:
        create_comparison_plot(scored_segments, output_dir)
    except Exception as e:
        print(f"\nSkipping plot generation: {e}")


if __name__ == "__main__":
    main()
