#!/usr/bin/env python3
"""Build a 3-speaker transcript by treating unmatched realtime utterances as speaker C."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


def load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def transcript_time(entry: dict) -> float:
    for key in ("start_time", "timestamp", "end_time"):
        value = entry.get(key)
        if value is not None:
            return float(value)
    return 0.0


def similarity_score(text_a: str, text_b: str) -> float:
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a and not norm_b:
        return 1.0

    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if norm_a and norm_b and (norm_a in norm_b or norm_b in norm_a):
        ratio = max(ratio, 0.92)

    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    overlap = 0.0
    if words_a and words_b:
        overlap = len(words_a & words_b) / max(1, min(len(words_a), len(words_b)))

    prefix_ratio = 0.0
    if norm_a and norm_b:
        prefix_ratio = (
            SequenceMatcher(
                None,
                norm_a[: min(40, len(norm_a))],
                norm_b[: min(40, len(norm_b))],
            ).ratio()
            * 0.9
        )

    return max(ratio, overlap, prefix_ratio)


@dataclass
class MatchResult:
    speaker: str
    source_id: str
    score: float
    time_diff: float
    matched_indices: tuple[int, ...]
    diarized_text: str


def find_best_match(
    realtime_entry: dict,
    diarized_by_source: dict[str, list[dict]],
    used_indices: dict[str, set[int]],
    max_window_size: int = 4,
    max_candidate_time_diff: float = 12.0,
    max_gap_within_window: float = 4.0,
) -> MatchResult | None:
    best: MatchResult | None = None
    best_rank: tuple[float, float, float] | None = None
    realtime_ts = float(realtime_entry["timestamp"])
    realtime_text = realtime_entry.get("text", "")
    realtime_norm_len = len(normalize_text(realtime_text))

    for source_id, diarized_entries in diarized_by_source.items():
        for start_idx, start_entry in enumerate(diarized_entries):
            if start_idx in used_indices[source_id]:
                continue

            start_ts = transcript_time(start_entry)
            if abs(realtime_ts - start_ts) > max_candidate_time_diff:
                continue

            combined_parts: list[str] = []
            consumed: list[int] = []
            last_ts = start_ts

            for end_idx in range(start_idx, min(start_idx + max_window_size, len(diarized_entries))):
                if end_idx in used_indices[source_id]:
                    break

                diarized_entry = diarized_entries[end_idx]
                current_ts = transcript_time(diarized_entry)
                if end_idx > start_idx and current_ts - last_ts > max_gap_within_window:
                    break

                consumed.append(end_idx)
                combined_parts.append(diarized_entry.get("text", ""))
                last_ts = transcript_time(diarized_entry)

                combined_text = " ".join(part for part in combined_parts if part).strip()
                raw_score = similarity_score(realtime_text, combined_text)
                time_diff = min(abs(realtime_ts - start_ts), abs(realtime_ts - last_ts))

                combined_norm_len = len(normalize_text(combined_text))
                length_ratio = min(realtime_norm_len + 1, combined_norm_len + 1) / max(
                    realtime_norm_len + 1,
                    combined_norm_len + 1,
                )

                rank = (
                    raw_score + 0.18 * length_ratio - 0.03 * time_diff,
                    raw_score,
                    -time_diff,
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best = MatchResult(
                        speaker="A" if source_id == "sum0" else "B",
                        source_id=source_id,
                        score=raw_score,
                        time_diff=time_diff,
                        matched_indices=tuple(consumed),
                        diarized_text=combined_text,
                    )

    return best


def should_accept_match(match: MatchResult) -> bool:
    diarized_word_count = word_count(match.diarized_text)
    return (
        match.score >= 0.83
        or (match.score >= 0.72 and match.time_diff <= 2.5)
        or (match.score >= 0.60 and diarized_word_count >= 8 and match.time_diff <= 1.5)
    )


def build_speaker_c_entries(
    realtime_entries: list[dict],
    diarized_by_source: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    used_indices = {source_id: set() for source_id in diarized_by_source}
    speaker_c_entries: list[dict] = []
    match_report: list[dict] = []

    for idx, realtime_entry in enumerate(realtime_entries):
        match = find_best_match(realtime_entry, diarized_by_source, used_indices)

        if match and should_accept_match(match):
            for matched_idx in match.matched_indices:
                used_indices[match.source_id].add(matched_idx)
            match_report.append(
                {
                    "realtime_index": idx,
                    "realtime_timestamp": realtime_entry.get("timestamp"),
                    "realtime_text": realtime_entry.get("text", ""),
                    "assigned_to": match.speaker,
                    "matched_source_id": match.source_id,
                    "matched_indices": list(match.matched_indices),
                    "score": round(match.score, 4),
                    "time_diff_sec": round(match.time_diff, 4),
                    "matched_text": match.diarized_text,
                }
            )
            continue

        speaker_c_entry = dict(realtime_entry)
        speaker_c_entry["speaker"] = "C"
        speaker_c_entry["source_id"] = "realtime_unmatched"
        speaker_c_entry["derived_from"] = "realtime"
        speaker_c_entries.append(speaker_c_entry)
        match_report.append(
            {
                "realtime_index": idx,
                "realtime_timestamp": realtime_entry.get("timestamp"),
                "realtime_text": realtime_entry.get("text", ""),
                "assigned_to": "C",
            }
        )

    return speaker_c_entries, match_report


def relabel_entries(entries: list[dict], speaker: str, source_id: str) -> list[dict]:
    relabeled = []
    for entry in entries:
        item = dict(entry)
        item["speaker"] = speaker
        item["source_id"] = source_id
        relabeled.append(item)
    return relabeled


def build_stats(entries: list[dict]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for entry in entries:
        speaker = str(entry.get("speaker", ""))
        speaker_stats = stats.setdefault(speaker, {"utterances": 0, "words": 0})
        speaker_stats["utterances"] += 1
        speaker_stats["words"] += word_count(entry.get("text", ""))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a third speaker from realtime transcript by removing high-confidence sum0/sum1 matches."
    )
    parser.add_argument("realtime", type=Path)
    parser.add_argument("sum0", type=Path)
    parser.add_argument("sum1", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix. Defaults next to realtime input.",
    )
    args = parser.parse_args()

    realtime_entries = load_json_list(args.realtime)
    sum0_entries = load_json_list(args.sum0)
    sum1_entries = load_json_list(args.sum1)

    diarized_by_source = {"sum0": sum0_entries, "sum1": sum1_entries}
    speaker_c_entries, match_report = build_speaker_c_entries(realtime_entries, diarized_by_source)

    combined_entries = (
        relabel_entries(sum0_entries, "A", "sum0")
        + relabel_entries(sum1_entries, "B", "sum1")
        + speaker_c_entries
    )
    combined_entries.sort(key=transcript_time)

    stats = build_stats(combined_entries)

    prefix = args.output_prefix
    if prefix is None:
        prefix = args.realtime.with_name(args.realtime.stem.replace("_transcript_realtime", "_transcript_3speaker"))

    transcript_output = Path(f"{prefix}.json")
    stats_output = Path(f"{prefix}_stats.json")
    report_output = Path(f"{prefix}_match_report.json")

    for path, payload in (
        (transcript_output, combined_entries),
        (stats_output, stats),
        (report_output, match_report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote transcript: {transcript_output}")
    print(f"Wrote stats: {stats_output}")
    print(f"Wrote match report: {report_output}")
    for speaker in sorted(stats):
        speaker_stats = stats[speaker]
        print(
            f"{speaker}: utterances={speaker_stats['utterances']} words={speaker_stats['words']}"
        )


if __name__ == "__main__":
    main()
