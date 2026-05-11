from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TARGET_DIR = DATA_DIR / "study" / "target_words"
SEMANTIC_DIR = DATA_DIR / "semantic_words"
SEMANTIC2_DIR = DATA_DIR / "semantic_words2"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
WORDS_DIR = DATA_DIR / "words"

SHORT_MEAN = 10.0
LONG_MEAN = 50.0
DELAY_TOLERANCE = 10.0
SEMANTIC_MIN_TIME_GAP = 2.0
SEMANTIC_MIN_WORD_GAP = 3
WORD_MIN_START_GAP = 2.0
WORD_MIN_DURATION = 2.0
MATCH_TOLERANCE = 0.05
TOKEN_RE = re.compile(r"[a-z0-9']+")
NUMERIC_STEM_RE = re.compile(r"^\d+$")


@dataclass
class ValidationIssue:
    scope: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate user-study materials under data/."
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional target_words stems to validate.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_word(word: Any) -> str:
    if not isinstance(word, str):
        return ""
    tokens = TOKEN_RE.findall(word.lower())
    return "".join(tokens)


def normalize_token(word: Any) -> str:
    if not isinstance(word, str):
        return ""
    tokens = TOKEN_RE.findall(word.lower())
    return tokens[0] if tokens else ""


def resolve_related_path(directory: Path, stem: str) -> Path | None:
    direct = directory / f"{stem}.json"
    if direct.exists():
        return direct
    base_stem = stem.split("_", 1)[0]
    fallback = directory / f"{base_stem}.json"
    if fallback.exists():
        return fallback
    return None


def load_target_paths(stems: list[str] | None) -> list[Path]:
    paths = sorted(TARGET_DIR.glob("*.json"))
    if stems is None:
        return [path for path in paths if NUMERIC_STEM_RE.fullmatch(path.stem)]
    allowed = set(stems)
    return [path for path in paths if path.stem in allowed]


def is_delay_in_range(delay_type: str, delay_seconds: float) -> bool:
    if delay_type == "short":
        return SHORT_MEAN - DELAY_TOLERANCE <= delay_seconds <= SHORT_MEAN + DELAY_TOLERANCE
    if delay_type == "long":
        return LONG_MEAN - DELAY_TOLERANCE <= delay_seconds <= LONG_MEAN + DELAY_TOLERANCE
    return False


def collect_transcript_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            token = normalize_token(word.get("word"))
            start = word.get("start")
            if not token or not isinstance(start, (int, float)):
                continue
            words.append({"token": token, "start": float(start)})
    return words


def attach_transcript_indexes(
    semantic_words: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed: list[dict[str, Any]] = []
    next_search_start = 0

    for item in sorted(semantic_words, key=lambda value: float(value.get("time", 0.0))):
        token = normalize_token(item.get("word"))
        time = item.get("time")
        transcript_index = None

        if token and isinstance(time, (int, float)):
            for idx in range(next_search_start, len(transcript_words)):
                candidate = transcript_words[idx]
                if candidate["token"] != token:
                    continue
                if abs(candidate["start"] - float(time)) > MATCH_TOLERANCE:
                    continue
                transcript_index = idx
                next_search_start = idx + 1
                break

        copied = dict(item)
        copied["_transcript_index"] = transcript_index
        indexed.append(copied)

    return indexed


def merge_words_min_duration(words: list[dict[str, Any]], min_duration: float) -> list[dict[str, Any]]:
    if not words:
        return []

    groups: list[dict[str, Any]] = []
    current_group: list[dict[str, Any]] = []

    for word in reversed(words):
        current_group.insert(0, word)
        duration = current_group[-1]["end"] - current_group[0]["start"]
        if duration >= min_duration:
            groups.insert(0, build_word_group(current_group))
            current_group = []

    if current_group:
        if groups:
            merged_words = list(current_group) + list(groups[0]["words"])
            groups[0] = build_word_group(merged_words)
        else:
            groups.insert(0, build_word_group(current_group))

    return groups


def build_word_group(words: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "words": list(words),
        "word": " ".join(item["word"] for item in words),
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "freq": min(item.get("freq", -1) for item in words),
    }


def word_group_duration(group: dict[str, Any]) -> float:
    return float(group["end"]) - float(group["start"])


def word_group_starts_with_target(group: dict[str, Any], target_words: set[str]) -> bool:
    words = group.get("words")
    if not isinstance(words, list) or not words:
        return False
    first_word = words[0]
    if not isinstance(first_word, dict):
        return False
    return bool(first_word.get("is_target"))


def collect_target_occurrences(interruptions: list[dict[str, Any]]) -> list[tuple[str, float]]:
    occurrences: list[tuple[str, float]] = []
    for item in interruptions:
        token = normalize_token(item.get("target_word"))
        time = item.get("target_word_time")
        if token and isinstance(time, (int, float)):
            occurrences.append((token, float(time)))
    return occurrences


def mark_generated_word_targets(
    words: list[dict[str, Any]], target_occurrences: list[tuple[str, float]]
) -> list[dict[str, Any]]:
    occurrences_by_token: dict[str, list[float]] = {}
    for token, time in sorted(target_occurrences, key=lambda item: (item[0], item[1])):
        occurrences_by_token.setdefault(token, []).append(time)

    marked_words: list[dict[str, Any]] = []
    for word in words:
        token = normalize_token(word.get("word"))
        start = float(word.get("start", 0.0))
        candidates = occurrences_by_token.get(token, [])
        copied = dict(word)
        if candidates and abs(candidates[0] - start) <= MATCH_TOLERANCE:
            copied["is_target"] = True
            candidates.pop(0)
        marked_words.append(copied)
    return marked_words


def build_generated_words(
    transcript: dict[str, Any], target_occurrences: list[tuple[str, float]]
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            start = word.get("start")
            end = word.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            words.append(
                {
                    "word": str(word.get("word", "")).strip(),
                    "start": float(start),
                    "end": float(end),
                    "freq": word.get("freq", -1),
                }
            )
    words.sort(key=lambda item: item["start"])
    words = mark_generated_word_targets(words, target_occurrences)
    return merge_words_min_duration(words, WORD_MIN_DURATION)


def split_generated_words_on_targets(
    groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not groups:
        return groups

    split_groups: list[dict[str, Any]] = []
    pending_words: list[dict[str, Any]] = []
    for group in groups:
        words = group.get("words")
        if not isinstance(words, list) or not words:
            split_groups.append(group)
            continue

        merged_words = pending_words + list(words)
        pending_words = []

        target_positions = [idx for idx, word in enumerate(merged_words) if idx > 0 and word.get("is_target")]

        if not target_positions:
            split_groups.append(build_word_group(merged_words))
            continue

        first_target_idx = target_positions[0]
        split_groups.append(build_word_group(merged_words[:first_target_idx]))

        for start_idx, end_idx in zip(target_positions, target_positions[1:]):
            split_groups.append(build_word_group(merged_words[start_idx:end_idx]))

        pending_words = merged_words[target_positions[-1]:]

    if pending_words:
        split_groups.append(build_word_group(pending_words))

    return split_groups


def rebalance_generated_words(
    groups: list[dict[str, Any]], target_words: set[str]
) -> list[dict[str, Any]]:
    if not groups:
        return groups

    balanced_groups: list[dict[str, Any]] = []
    pending_groups = list(groups)
    index = 0

    while index < len(pending_groups):
        group = pending_groups[index]
        duration = word_group_duration(group)
        starts_with_target = word_group_starts_with_target(group, target_words)

        if duration >= WORD_MIN_DURATION:
            balanced_groups.append(group)
            index += 1
            continue

        if starts_with_target and index + 1 < len(pending_groups):
            pending_groups[index + 1] = build_word_group(
                list(group["words"]) + list(pending_groups[index + 1]["words"])
            )
            index += 1
            continue

        if not starts_with_target and balanced_groups:
            balanced_groups[-1] = build_word_group(
                list(balanced_groups[-1]["words"]) + list(group["words"])
            )
            index += 1
            continue

        if not starts_with_target and index + 1 < len(pending_groups):
            if word_group_starts_with_target(pending_groups[index + 1], target_words):
                index += 1
                continue
            pending_groups[index + 1] = build_word_group(
                list(group["words"]) + list(pending_groups[index + 1]["words"])
            )
            index += 1
            continue

        balanced_groups.append(group)
        index += 1

    return balanced_groups


def validate_target_words(
    target_path: Path, target_data: dict[str, Any], issues: list[ValidationIssue]
) -> list[dict[str, Any]]:
    interruptions = target_data.get("interruptions")
    scope = target_path.stem

    if not isinstance(interruptions, list):
        issues.append(ValidationIssue(scope, "interruptions must be a list"))
        return []

    if len(interruptions) != 6:
        issues.append(
            ValidationIssue(scope, f"interruptions count must be 6, found {len(interruptions)}")
        )

    counts = {"short": 0, "long": 0}
    for idx, item in enumerate(interruptions, start=1):
        delay_type = item.get("delay_type")
        delay_seconds = item.get("delay_seconds")
        if delay_type in counts:
            counts[delay_type] += 1
        else:
            issues.append(ValidationIssue(scope, f"interruption #{idx} has invalid delay_type={delay_type!r}"))
            continue

        if not isinstance(delay_seconds, (int, float)):
            issues.append(ValidationIssue(scope, f"interruption #{idx} missing numeric delay_seconds"))
            continue

        if not is_delay_in_range(delay_type, float(delay_seconds)):
            issues.append(
                ValidationIssue(
                    scope,
                    f"interruption #{idx} delay_seconds={float(delay_seconds):.2f} out of range for {delay_type}",
                )
            )

    if counts["short"] != 3 or counts["long"] != 3:
        issues.append(
            ValidationIssue(scope, f"delay_type counts must be short=3,long=3, found short={counts['short']}, long={counts['long']}")
        )

    return interruptions


def validate_semantic_contains_targets(
    stem: str,
    semantic_words: list[dict[str, Any]],
    interruptions: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> set[str]:
    scope = stem
    semantic_pairs = {
        (normalize_word(item.get("word")), round(float(item.get("time")), 2))
        for item in semantic_words
        if isinstance(item.get("time"), (int, float))
    }
    target_normalized_words: set[str] = set()

    for idx, item in enumerate(interruptions, start=1):
        target_word = item.get("target_word")
        target_time = item.get("target_word_time")
        normalized = normalize_word(target_word)
        target_normalized_words.add(normalized)
        if not normalized or not isinstance(target_time, (int, float)):
            issues.append(ValidationIssue(scope, f"interruption #{idx} missing target_word/target_word_time"))
            continue
        pair = (normalized, round(float(target_time), 2))
        if pair not in semantic_pairs:
            issues.append(
                ValidationIssue(
                    scope,
                    f"semantic_words missing target_word={target_word!r} at time={float(target_time):.2f}",
                )
            )

    return target_normalized_words


def validate_semantic2_excludes_targets(
    stem: str,
    semantic_words2: list[dict[str, Any]],
    target_words: set[str],
    issues: list[ValidationIssue],
) -> None:
    scope = stem
    for item in semantic_words2:
        normalized = normalize_word(item.get("word"))
        if normalized and normalized in target_words:
            issues.append(
                ValidationIssue(
                    scope,
                    f"semantic_words2 still contains target word {item.get('word')!r} at time={float(item.get('time', 0.0)):.2f}",
                )
            )


def validate_semantic_gaps(
    stem: str,
    semantic_words: list[dict[str, Any]],
    transcript: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    scope = stem
    transcript_words = collect_transcript_words(transcript)
    indexed_words = attach_transcript_indexes(semantic_words, transcript_words)

    for idx, item in enumerate(indexed_words, start=1):
        if item.get("_transcript_index") is None:
            issues.append(
                ValidationIssue(
                    scope,
                    f"semantic_words item #{idx} could not be matched to transcript: word={item.get('word')!r}, time={float(item.get('time', 0.0)):.2f}",
                )
            )

    for prev, curr in zip(indexed_words, indexed_words[1:]):
        prev_time = prev.get("time")
        curr_time = curr.get("time")
        if isinstance(prev_time, (int, float)) and isinstance(curr_time, (int, float)):
            time_gap = float(curr_time) - float(prev_time)
            if time_gap < SEMANTIC_MIN_TIME_GAP:
                issues.append(
                    ValidationIssue(
                        scope,
                        f"semantic_words time gap {time_gap:.2f}s < {SEMANTIC_MIN_TIME_GAP:.2f}s between {prev.get('word')!r} ({float(prev_time):.2f}) and {curr.get('word')!r} ({float(curr_time):.2f})",
                    )
                )

        prev_index = prev.get("_transcript_index")
        curr_index = curr.get("_transcript_index")
        if isinstance(prev_index, int) and isinstance(curr_index, int):
            word_gap = curr_index - prev_index
            if word_gap < SEMANTIC_MIN_WORD_GAP:
                issues.append(
                    ValidationIssue(
                        scope,
                        f"semantic_words transcript gap {word_gap} < {SEMANTIC_MIN_WORD_GAP} between {prev.get('word')!r} and {curr.get('word')!r}",
                    )
                )


def validate_word_groups(
    scope: str,
    label: str,
    groups: list[dict[str, Any]],
    issues: list[ValidationIssue],
) -> None:
    for group in groups:
        start = group.get("start")
        end = group.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            issues.append(ValidationIssue(scope, f"{label} contains non-numeric start/end times"))
            return
        duration = float(end) - float(start)
        if duration < WORD_MIN_DURATION:
            issues.append(
                ValidationIssue(
                    scope,
                    f"{label} duration {duration:.2f}s < {WORD_MIN_DURATION:.2f}s for {group.get('word')!r}",
                )
            )

    for prev, curr in zip(groups, groups[1:]):
        prev_start = prev.get("start")
        curr_start = curr.get("start")
        if not isinstance(prev_start, (int, float)) or not isinstance(curr_start, (int, float)):
            issues.append(ValidationIssue(scope, f"{label} contains non-numeric start times"))
            return
        gap = float(curr_start) - float(prev_start)
        if gap < WORD_MIN_START_GAP:
            issues.append(
                ValidationIssue(
                    scope,
                    f"{label} start gap {gap:.2f}s < {WORD_MIN_START_GAP:.2f}s between {prev.get('word')!r} and {curr.get('word')!r}",
                )
            )


def validate_target_word_group_starts(
    scope: str,
    label: str,
    groups: list[dict[str, Any]],
    target_occurrences: list[tuple[str, float]],
    issues: list[ValidationIssue],
) -> None:
    if not target_occurrences:
        return

    for token, time in target_occurrences:
        matched = False
        for group in groups:
            words = group.get("words")
            if not isinstance(words, list) or not words:
                continue
            first_word = words[0]
            if not isinstance(first_word, dict):
                continue
            if normalize_token(first_word.get("word")) != token:
                continue
            start = first_word.get("start")
            if not isinstance(start, (int, float)):
                continue
            if abs(float(start) - time) <= MATCH_TOLERANCE:
                matched = True
                break
        if not matched:
            issues.append(
                ValidationIssue(
                    scope,
                    f"{label} does not start a group at target word={token!r} time={time:.2f}",
                )
            )


def validate_words_cache_and_generation(
    stem: str,
    transcript: dict[str, Any],
    target_words: set[str],
    target_occurrences: list[tuple[str, float]],
    issues: list[ValidationIssue],
) -> None:
    words_path = resolve_related_path(WORDS_DIR, stem)
    if words_path is not None:
        words_data = load_json(words_path)
        items = words_data.get("items") if isinstance(words_data, dict) else None
        if not isinstance(items, list):
            issues.append(ValidationIssue(stem, f"words cache must contain an items list: {words_path.name}"))
        else:
            validate_word_groups(stem, f"words cache ({words_path.name})", items, issues)
            validate_target_word_group_starts(
                stem, f"words cache ({words_path.name})", items, target_occurrences, issues
            )

    generated_groups = rebalance_generated_words(
        split_generated_words_on_targets(
            build_generated_words(transcript, target_occurrences)
        ),
        target_words,
    )
    validate_word_groups(stem, "generated words", generated_groups, issues)
    validate_target_word_group_starts(
        stem, "generated words", generated_groups, target_occurrences, issues
    )


def validate_target_bundle(target_path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    stem = target_path.stem
    target_data = load_json(target_path)
    if not isinstance(target_data, dict):
        return [ValidationIssue(stem, "target_words JSON must contain an object")]

    semantic_path = resolve_related_path(SEMANTIC_DIR, stem)
    semantic2_path = resolve_related_path(SEMANTIC2_DIR, stem)
    transcript_path = resolve_related_path(TRANSCRIPT_DIR, stem)

    if semantic_path is None:
        issues.append(ValidationIssue(stem, "missing semantic_words JSON"))
    if semantic2_path is None:
        issues.append(ValidationIssue(stem, "missing semantic_words2 JSON"))
    if transcript_path is None:
        issues.append(ValidationIssue(stem, "missing transcript JSON"))

    interruptions = validate_target_words(target_path, target_data, issues)

    if semantic_path is None or semantic2_path is None or transcript_path is None:
        return issues

    semantic_words = load_json(semantic_path)
    semantic_words2 = load_json(semantic2_path)
    transcript = load_json(transcript_path)

    if not isinstance(semantic_words, list):
        issues.append(ValidationIssue(stem, f"semantic_words must be a list: {semantic_path.name}"))
        return issues
    if not isinstance(semantic_words2, list):
        issues.append(ValidationIssue(stem, f"semantic_words2 must be a list: {semantic2_path.name}"))
        return issues
    if not isinstance(transcript, dict):
        issues.append(ValidationIssue(stem, f"transcript must be an object: {transcript_path.name}"))
        return issues

    target_words = validate_semantic_contains_targets(
        stem, semantic_words, interruptions, issues
    )
    target_occurrences = collect_target_occurrences(interruptions)
    validate_semantic2_excludes_targets(stem, semantic_words2, target_words, issues)
    validate_semantic_gaps(stem, semantic_words, transcript, issues)
    validate_words_cache_and_generation(
        stem, transcript, target_words, target_occurrences, issues
    )
    return issues


def main() -> int:
    args = parse_args()
    target_paths = load_target_paths(args.stems)
    if not target_paths:
        print("No target_words files found to validate.", file=sys.stderr)
        return 1

    all_issues: list[ValidationIssue] = []
    for target_path in target_paths:
        all_issues.extend(validate_target_bundle(target_path))

    if all_issues:
        for issue in all_issues:
            print(f"[FAIL] {issue.scope}: {issue.message}")
        print(f"\nValidation failed with {len(all_issues)} issue(s).")
        return 1

    print(f"Validated {len(target_paths)} target_words file(s) with no issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
