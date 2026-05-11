"""Playback Search App - three playback modes for keyword search."""

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from pydub import AudioSegment

from config import (
    INTERRUPTIONS_DIR,
    KEYWORDS2_DIR,
    KEYWORDS_DIR,
    LEXICON_PATH,
    LOGS_DIR,
    MP3_DIR,
    SENTENCES_DIR,
    STUDY_AUDIO_WINDOWS_PATH,
    STUDY_DIR,
    TRANSCRIPT_DIR,
    WORDS_DIR,
)

app = Flask(__name__)
TERM_RE = re.compile(r"[a-z0-9']+")
MERGED_WORD_MIN_DURATION_SECONDS = 2.0
MIN_SENTENCE_DURATION_SECONDS = 2.0
MERGED_WORDS_CACHE_VERSION = 9
TARGET_TIME_TOLERANCE_SECONDS = 0.05
RARE_WORD_MAX_FREQ = 4.0
MAX_WORD_GROUP_DURATION_SECONDS = 3.0


def _slugify_filename_part(value: str, default: str = "unknown") -> str:
    """Normalize a filename segment to a safe, predictable token."""
    text = str(value or "").strip()
    if not text:
        return default
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or default


def _build_study_log_basename(data: dict) -> str:
    """Build the shared basename for study JSON and JSONL logs."""
    provided = data.get("logBaseName")
    if provided:
        return _slugify_filename_part(Path(str(provided)).stem)

    participant = _slugify_filename_part(data.get("participant"))
    feature = _slugify_filename_part(data.get("feature"))
    audio_value = data.get("audio") or data.get("videoId")
    audio = _slugify_filename_part(Path(str(audio_value or "unknown")).stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{participant}_{feature}_{audio}_{timestamp}"

# Stopwords for keyword extraction
STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "couldn",
    "didn",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "ma",
    "mightn",
    "mustn",
    "needn",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
    "yeah",
    "okay",
    "ok",
    "like",
    "know",
    "think",
    "going",
    "want",
    "got",
    "get",
    "go",
    "come",
    "say",
    "said",
    "would",
    "could",
    "one",
    "two",
    "let",
    "something",
    "thing",
    "things",
    "really",
    "actually",
    "basically",
    "well",
}

# Lexicon cache
_lexicon: dict[str, float] = {}


def load_lexicon() -> dict[str, float]:
    """Load OpenLexicon.xlsx and return word->frequency dict."""
    global _lexicon
    if _lexicon:
        return _lexicon

    if not LEXICON_PATH.exists():
        return {}

    import pandas as pd

    df = pd.read_excel(LEXICON_PATH)

    for _, row in df.iterrows():
        word = str(row["ortho"]).lower()
        freq = row["English_Lexicon_Project__LgSUBTLWF"]
        if pd.notna(freq):
            _lexicon[word] = float(freq)
        else:
            _lexicon[word] = 0.0

    return _lexicon


def extract_keywords(text: str, top_k: int = 3) -> list[str]:
    """Extract rare keywords from text using OpenLexicon."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS]

    lexicon = load_lexicon()

    candidates = []
    for w in words:
        freq = lexicon.get(w)
        if freq is None:
            candidates.append((w, -1.0))  # Not in lexicon = rare
        elif freq < 3.0:
            candidates.append((w, freq))

    candidates.sort(key=lambda x: x[1])

    seen = set()
    result = []
    for w, _ in candidates:
        if w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) >= top_k:
                break

    return result


def load_json_file(path):
    with open(path) as f:
        return json.load(f)


def load_study_audio_windows() -> dict:
    if not STUDY_AUDIO_WINDOWS_PATH.exists():
        return {"default_target_window_seconds": 300, "default_max_interruptions": 10, "videos": {}}
    return load_json_file(STUDY_AUDIO_WINDOWS_PATH)


def build_sentence_units(segments: list[dict]) -> list[dict]:
    """Build sentence-wise units using period-delimited word timestamps."""
    units = []
    current_words = []
    current_start = None
    fallback_segment_start = None
    fallback_segment_end = None

    def flush_sentence():
        nonlocal current_words, current_start, fallback_segment_start, fallback_segment_end
        if not current_words:
            return

        last_word = current_words[-1]
        units.append(
            {
                "text": " ".join(word["word"] for word in current_words).strip(),
                "start": current_start if current_start is not None else fallback_segment_start or 0,
                "end": last_word.get("end", fallback_segment_end or current_start or 0),
            }
        )

        current_words = []
        current_start = None
        fallback_segment_start = None
        fallback_segment_end = None

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            flush_sentence()
            units.append(
                {
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )
            continue

        for word in words:
            if not current_words:
                current_start = word["start"]
                fallback_segment_start = seg["start"]

            fallback_segment_end = seg["end"]
            current_words.append(word)

            if "." in word["word"]:
                flush_sentence()

    flush_sentence()
    return rebalance_sentence_units(units)


def sentence_unit_duration(unit: dict) -> float:
    return float(unit["end"]) - float(unit["start"])


def build_sentence_unit(text: str, start: float, end: float) -> dict:
    return {"text": text.strip(), "start": start, "end": end}


def merge_sentence_units(left: dict, right: dict) -> dict:
    return build_sentence_unit(
        f'{left.get("text", "").strip()} {right.get("text", "").strip()}'.strip(),
        float(left["start"]),
        float(right["end"]),
    )


def rebalance_sentence_units(units: list[dict]) -> list[dict]:
    if not units:
        return []

    pending_units = [
        build_sentence_unit(str(unit.get("text", "")), float(unit["start"]), float(unit["end"]))
        for unit in units
    ]
    balanced_units: list[dict] = []
    index = 0

    while index < len(pending_units):
        unit = pending_units[index]
        if sentence_unit_duration(unit) >= MIN_SENTENCE_DURATION_SECONDS:
            balanced_units.append(unit)
            index += 1
            continue

        if index + 1 < len(pending_units):
            pending_units[index + 1] = merge_sentence_units(unit, pending_units[index + 1])
            index += 1
            continue

        if balanced_units:
            balanced_units[-1] = merge_sentence_units(balanced_units[-1], unit)
            index += 1
            continue

        balanced_units.append(unit)
        index += 1

    return balanced_units


def load_or_create_sentences(video_id: str, segments: list[dict]) -> list[dict]:
    """Load sentence cache or create it from transcript segments."""
    SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
    sentence_path = SENTENCES_DIR / f"{video_id}.json"

    if sentence_path.exists():
        with open(sentence_path) as f:
            cached = json.load(f)
        if isinstance(cached, list):
            rebalanced = rebalance_sentence_units(cached)
            if rebalanced != cached:
                with open(sentence_path, "w") as f:
                    json.dump(rebalanced, f, indent=2)
            return rebalanced

    sentences = build_sentence_units(segments)
    with open(sentence_path, "w") as f:
        json.dump(sentences, f, indent=2)

    return sentences


def merge_words_min_duration(
    words: list[dict], min_duration: float = MERGED_WORD_MIN_DURATION_SECONDS
) -> list[dict]:
    """Merge words from the end so each group has duration >= min_duration."""
    if not words:
        return []

    # Process from the end
    groups = []
    current_group = []

    for w in reversed(words):
        current_group.insert(0, w)
        group_start = current_group[0]["start"]
        group_end = current_group[-1]["end"]
        duration = group_end - group_start

        if duration >= min_duration:
            # Duration is sufficient, finalize this group
            groups.insert(0, {
                "words": list(current_group),
                "word": " ".join(x["word"] for x in current_group),
                "start": group_start,
                "end": group_end,
                "freq": min(x.get("freq", -1) for x in current_group),
            })
            current_group = []

    # Fold a short leading remainder into the first finalized group so every
    # generated group still respects the minimum duration when possible.
    if current_group:
        if groups:
            first_group = groups[0]
            merged_words = list(current_group) + list(first_group["words"])
            groups[0] = {
                "words": merged_words,
                "word": " ".join(x["word"] for x in merged_words),
                "start": merged_words[0]["start"],
                "end": merged_words[-1]["end"],
                "freq": min(x.get("freq", -1) for x in merged_words),
            }
        else:
            groups.insert(0, {
                "words": list(current_group),
                "word": " ".join(x["word"] for x in current_group),
                "start": current_group[0]["start"],
                "end": current_group[-1]["end"],
                "freq": min(x.get("freq", -1) for x in current_group),
            })

    return groups


def build_word_group(words: list[dict]) -> dict:
    return {
        "words": list(words),
        "word": " ".join(x["word"] for x in words),
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "freq": min(x.get("freq", -1) for x in words),
    }


def word_group_duration(group: dict) -> float:
    return float(group["end"]) - float(group["start"])


def word_group_starts_with_target(group: dict) -> bool:
    words = group.get("words", [])
    if not isinstance(words, list) or not words:
        return False
    return bool(words[0].get("is_target"))


def is_candidate_word(word: dict) -> bool:
    token = normalize_token(str(word.get("word", "")))
    if not token or len(token) <= 2 or token in STOPWORDS:
        return False

    freq = word.get("freq", -1)
    if not isinstance(freq, (int, float)):
        return False
    return float(freq) < 0 or float(freq) < RARE_WORD_MAX_FREQ


def load_target_word_occurrences(video_id: str) -> list[tuple[str, float]]:
    target_path = preferred_data_path(INTERRUPTIONS_DIR, video_id)
    if not target_path.exists():
        return []

    with open(target_path) as f:
        data = json.load(f)

    interruptions = data.get("interruptions") if isinstance(data, dict) else None
    if not isinstance(interruptions, list):
        return []

    occurrences = []
    for item in interruptions:
        if not isinstance(item, dict):
            continue
        token = normalize_token(str(item.get("target_word", "")))
        time = item.get("target_word_time")
        if token and isinstance(time, (int, float)):
            occurrences.append((token, float(time)))
    return occurrences


def mark_target_word_occurrences(
    words: list[dict], target_occurrences: list[tuple[str, float]]
) -> list[dict]:
    occurrences_by_token: dict[str, list[float]] = {}
    for token, time in sorted(target_occurrences, key=lambda item: (item[0], item[1])):
        occurrences_by_token.setdefault(token, []).append(time)

    marked_words = []
    for word in words:
        token = normalize_token(str(word.get("word", "")))
        start = float(word.get("start", 0.0))
        candidates = occurrences_by_token.get(token, [])
        is_target = False
        if candidates and abs(candidates[0] - start) <= TARGET_TIME_TOLERANCE_SECONDS:
            is_target = True
            candidates.pop(0)

        marked_word = dict(word)
        if is_target:
            marked_word["is_target"] = True
        marked_words.append(marked_word)

    return marked_words


def split_merged_words_on_target_words(groups: list[dict]) -> list[dict]:
    if not groups:
        return groups

    split_groups = []
    pending_words: list[dict] = []
    for group in groups:
        words = group.get("words", [])
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

        pending_words = merged_words[target_positions[-1] :]

    if pending_words:
        split_groups.append(build_word_group(pending_words))

    return split_groups


def rebalance_merged_word_groups(groups: list[dict]) -> list[dict]:
    if not groups:
        return groups

    balanced_groups: list[dict] = []
    pending_groups = list(groups)
    index = 0

    while index < len(pending_groups):
        group = pending_groups[index]
        duration = word_group_duration(group)
        starts_with_target = word_group_starts_with_target(group)

        if duration >= MERGED_WORD_MIN_DURATION_SECONDS:
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
            if word_group_starts_with_target(pending_groups[index + 1]):
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


def build_candidate_anchor_groups(words: list[dict]) -> list[dict]:
    anchor_indices = [
        idx for idx, word in enumerate(words) if word.get("is_target") or is_candidate_word(word)
    ]
    if not anchor_indices:
        return []

    atomic_groups = []
    for position, start_idx in enumerate(anchor_indices):
        next_start = anchor_indices[position + 1] if position + 1 < len(anchor_indices) else len(words)
        end_idx = max(start_idx, next_start - 1)
        atomic_groups.append(build_word_group(words[start_idx : end_idx + 1]))

    merged_groups: list[dict] = []
    index = 0
    while index < len(atomic_groups):
        current_group = atomic_groups[index]

        while (
            word_group_duration(current_group) < MERGED_WORD_MIN_DURATION_SECONDS
            and index + 1 < len(atomic_groups)
        ):
            current_group = build_word_group(
                list(current_group["words"]) + list(atomic_groups[index + 1]["words"])
            )
            index += 1

        if (
            word_group_duration(current_group) < MERGED_WORD_MIN_DURATION_SECONDS
            and merged_groups
            and not word_group_starts_with_target(current_group)
        ):
            merged_groups[-1] = build_word_group(
                list(merged_groups[-1]["words"]) + list(current_group["words"])
            )
        else:
            merged_groups.append(current_group)

        index += 1

    return merged_groups


def split_word_group_if_needed(group: dict) -> list[dict]:
    duration = word_group_duration(group)
    if duration <= MAX_WORD_GROUP_DURATION_SECONDS:
        return [group]

    words = group.get("words", [])
    if not isinstance(words, list) or len(words) < 2:
        return [group]

    candidates = []
    for split_idx in range(1, len(words)):
        left_group = build_word_group(words[:split_idx])
        right_group = build_word_group(words[split_idx:])
        left_duration = word_group_duration(left_group)
        right_duration = word_group_duration(right_group)
        if (
            left_duration < MERGED_WORD_MIN_DURATION_SECONDS
            or right_duration < MERGED_WORD_MIN_DURATION_SECONDS
        ):
            continue

        prev_word = str(words[split_idx - 1].get("word", ""))
        next_word = str(words[split_idx].get("word", ""))
        punctuation_bonus = 0
        if any(mark in prev_word for mark in [".", "?", "!"]):
            punctuation_bonus = -2
        elif any(mark in prev_word for mark in [",", ";", ":"]):
            punctuation_bonus = -1
        if next_word[:1].isupper():
            punctuation_bonus -= 0.5

        candidates.append(
            (
                max(left_duration, right_duration),
                abs(left_duration - right_duration),
                punctuation_bonus,
                split_idx,
                left_group,
                right_group,
            )
        )

    if not candidates:
        return [group]

    _, _, _, _, left_group, right_group = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return split_word_group_if_needed(left_group) + split_word_group_if_needed(right_group)


def split_long_word_groups(groups: list[dict]) -> list[dict]:
    split_groups: list[dict] = []
    for group in groups:
        split_groups.extend(split_word_group_if_needed(group))
    return split_groups


def build_merged_words(
    segments: list[dict], target_occurrences: list[tuple[str, float]] | None = None
) -> list[dict]:
    """Build list of merged words from transcript segments."""
    all_words = []
    for seg in segments:
        for w in seg.get("words", []):
            all_words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
                "freq": w.get("freq", -1),
            })

    all_words.sort(key=lambda x: x["start"])
    all_words = mark_target_word_occurrences(all_words, target_occurrences or [])
    candidate_groups = build_candidate_anchor_groups(all_words)
    split_words = split_merged_words_on_target_words(candidate_groups)
    rebalanced_words = rebalance_merged_word_groups(split_words)
    return split_long_word_groups(rebalanced_words)


def load_or_create_merged_words(video_id: str, segments: list[dict]) -> list[dict]:
    """Load merged words cache or create it from transcript segments."""
    WORDS_DIR.mkdir(parents=True, exist_ok=True)
    words_path = WORDS_DIR / f"{video_id}.json"
    target_occurrences = load_target_word_occurrences(video_id)

    if words_path.exists():
        with open(words_path) as f:
            cached = json.load(f)
        if (
            isinstance(cached, dict)
            and cached.get("version") == MERGED_WORDS_CACHE_VERSION
            and cached.get("min_duration_seconds") == MERGED_WORD_MIN_DURATION_SECONDS
            and isinstance(cached.get("items"), list)
        ):
            return cached["items"]

    merged_words = build_merged_words(segments, target_occurrences)
    with open(words_path, "w") as f:
        json.dump(
            {
                "version": MERGED_WORDS_CACHE_VERSION,
                "min_duration_seconds": MERGED_WORD_MIN_DURATION_SECONDS,
                "items": merged_words,
            },
            f,
            indent=2,
        )

    return merged_words


def parse_clip_times(filename: str) -> tuple[float, float]:
    """Parse clip start/end times from filename like 'xxx_clip_680_1010.mp3'."""
    match = re.search(r"_clip_(\d+)_(\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def normalize_token(text: str) -> str:
    tokens = TERM_RE.findall((text or "").lower())
    return tokens[0] if len(tokens) == 1 else ""


def preferred_transcript_id(requested_id: str) -> str:
    normalized = edited_transcript_id(requested_id)
    if (TRANSCRIPT_DIR / f"{normalized}.json").exists():
        return normalized
    return requested_id


def base_transcript_id(transcript_id: str) -> str:
    return edited_transcript_id(transcript_id).removesuffix("_ts_edit")


def preferred_data_path(directory: Path, requested_id: str, suffix: str = ".json") -> Path:
    normalized = edited_transcript_id(requested_id)
    preferred = directory / f"{normalized}{suffix}"
    if preferred.exists():
        return preferred
    return directory / f"{base_transcript_id(requested_id)}{suffix}"


def build_word_index(segments: list[dict], *, clip_start: float) -> list[dict]:
    items = []
    for segment_index, segment in enumerate(segments):
        for word_index, word in enumerate(segment.get("words", [])):
            items.append(
                {
                    "segment_index": segment_index,
                    "word_index": word_index,
                    "text": word.get("word", ""),
                    "start": float(word.get("start", 0.0)) - clip_start,
                    "end": float(word.get("end", 0.0)) - clip_start,
                }
            )
    return items


def remap_time_entries(path: Path, words: list[dict]) -> None:
    if not path.exists():
        return

    entries = load_json_file(path)
    if not isinstance(entries, list):
        return

    updated_entries = remap_time_entries_data(entries, words)
    path.write_text(json.dumps(updated_entries, indent=2), encoding="utf-8")


def remap_time_entries_data(entries: list[dict], words: list[dict]) -> list[dict]:
    occurrences: dict[str, list[float]] = {}
    for word in words:
        token = normalize_token(str(word.get("text", "")))
        if not token:
            continue
        occurrences.setdefault(token, []).append(float(word["start"]))

    used_indices: dict[str, set[int]] = {}
    updated_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            updated_entries.append(entry)
            continue

        token = normalize_token(str(entry.get("word", "")))
        original_time = float(entry.get("time", 0.0))
        entry_copy = dict(entry)
        candidates = occurrences.get(token, [])
        if candidates:
            used = used_indices.setdefault(token, set())
            ranked = sorted(
                range(len(candidates)),
                key=lambda idx: (abs(candidates[idx] - original_time), idx),
            )
            chosen_idx = next((idx for idx in ranked if idx not in used), ranked[0])
            used.add(chosen_idx)
            entry_copy["time"] = candidates[chosen_idx]
        updated_entries.append(entry_copy)
    return updated_entries


def remap_target_words_file(source_path: Path, destination_path: Path, jargon_words_path: Path) -> None:
    if not source_path.exists():
        return

    data = load_json_file(source_path)
    interruptions = data.get("interruptions")
    if not isinstance(interruptions, list):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return

    occurrences: dict[str, list[float]] = {}
    if jargon_words_path.exists():
        jargon_words = load_json_file(jargon_words_path)
        if isinstance(jargon_words, list):
            for word in jargon_words:
                if not isinstance(word, dict):
                    continue
                token = normalize_token(str(word.get("word", "")))
                if not token or "time" not in word:
                    continue
                occurrences.setdefault(token, []).append(float(word["time"]))

    used_indices: dict[str, set[int]] = {}
    updated_interruptions = []
    for item in interruptions:
        if not isinstance(item, dict):
            updated_interruptions.append(item)
            continue

        entry = dict(item)
        token = normalize_token(str(entry.get("target_word", "")))
        original_time = float(entry.get("target_word_time", 0.0))
        candidates = occurrences.get(token, [])
        if candidates:
            used = used_indices.setdefault(token, set())
            ranked = sorted(
                range(len(candidates)),
                key=lambda idx: (abs(candidates[idx] - original_time), idx),
            )
            chosen_idx = next((idx for idx in ranked if idx not in used), ranked[0])
            used.add(chosen_idx)
            delta = candidates[chosen_idx] - original_time
            entry["target_word_time"] = candidates[chosen_idx]
            if "search_start_time" in entry and isinstance(entry["search_start_time"], (int, float)):
                entry["search_start_time"] = float(entry["search_start_time"]) + delta
        updated_interruptions.append(entry)

    output = dict(data)
    output["video_id"] = destination_path.stem
    output["interruptions"] = updated_interruptions
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def edited_transcript_id(transcript_id: str) -> str:
    match = re.match(r"^(.*?)(?:_tsedit_\d+|_ts_edit)$", transcript_id)
    base_id = match.group(1) if match else transcript_id
    return f"{base_id}_ts_edit"


def copy_json_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def copy_preferred_json_if_exists(
    directory: Path, requested_id: str, destination: Path, suffix: str = ".json"
) -> None:
    source = preferred_data_path(directory, requested_id, suffix)
    copy_json_if_exists(source, destination)


def apply_word_timestamp_updates(transcript_id: str, updated_words: list[dict]) -> str:
    source_id = preferred_transcript_id(transcript_id)
    source_transcript_path = TRANSCRIPT_DIR / f"{source_id}.json"
    target_transcript_id = edited_transcript_id(transcript_id)
    target_transcript_path = TRANSCRIPT_DIR / f"{target_transcript_id}.json"
    transcript_path = source_transcript_path
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_id}")

    data = load_json_file(transcript_path)
    clip_start = float(data.get("start_time", 0.0))
    segments = data.get("segments", [])
    indexed_words = build_word_index(segments, clip_start=clip_start)

    if len(updated_words) != len(indexed_words):
        raise ValueError("Word count mismatch while saving timestamps")

    flat_words_for_mapping = []
    for source_word, incoming_word in zip(indexed_words, updated_words):
        segment = segments[source_word["segment_index"]]
        word = segment["words"][source_word["word_index"]]
        rel_start = max(0.0, float(incoming_word["start"]))
        rel_end = max(rel_start, float(incoming_word["end"]))
        abs_start = rel_start + clip_start
        abs_end = rel_end + clip_start
        word["start"] = abs_start
        word["end"] = abs_end
        flat_words_for_mapping.append(
            {
                "text": word.get("word", ""),
                "start": rel_start,
            }
        )

    for segment in segments:
        segment_words = segment.get("words", [])
        if segment_words:
            segment["start"] = segment_words[0]["start"]
            segment["end"] = segment_words[-1]["end"]

    target_transcript_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    sentence_path = SENTENCES_DIR / f"{target_transcript_id}.json"
    sentence_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_path.write_text(
        json.dumps(build_sentence_units(load_transcript_segments_for_api(data)), indent=2),
        encoding="utf-8",
    )

    target_keywords_path = KEYWORDS_DIR / f"{target_transcript_id}.json"
    target_jargon_path = KEYWORDS_DIR / f"{target_transcript_id}.jargon.json"
    target_keywords2_path = KEYWORDS2_DIR / f"{target_transcript_id}.json"
    copy_preferred_json_if_exists(KEYWORDS_DIR, transcript_id, target_keywords_path)
    copy_preferred_json_if_exists(
        KEYWORDS_DIR, transcript_id, target_jargon_path, ".jargon.json"
    )
    copy_preferred_json_if_exists(KEYWORDS2_DIR, transcript_id, target_keywords2_path)
    remap_time_entries(target_keywords_path, flat_words_for_mapping)
    remap_time_entries(target_jargon_path, flat_words_for_mapping)
    remap_time_entries(target_keywords2_path, flat_words_for_mapping)
    remap_target_words_file(
        INTERRUPTIONS_DIR / f"{source_id}.json",
        INTERRUPTIONS_DIR / f"{target_transcript_id}.json",
        target_jargon_path,
    )
    return target_transcript_id


def load_transcript_segments_for_api(data: dict) -> list[dict]:
    clip_start = float(data.get("start_time", 0.0))
    segments = []
    lexicon = load_lexicon()
    for seg in data.get("segments", []):
        words = []
        for w in seg.get("words", []):
            word_text = w["word"].strip().lower()
            word_clean = re.sub(r"[^a-z]", "", word_text)
            freq = lexicon.get(word_clean, -1)
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": w["start"] - clip_start,
                    "end": w["end"] - clip_start,
                    "freq": freq,
                }
            )

        segments.append(
            {
                "text": seg["text"],
                "start": seg["start"] - clip_start,
                "end": seg["end"] - clip_start,
                "keywords": extract_keywords(seg["text"]),
                "words": words,
            }
        )
    return segments


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/timestamp-check")
def timestamp_check():
    return render_template("timestamp_check.html")


@app.route("/timestamp-edit")
def timestamp_edit():
    return render_template("timestamp_edit.html")


@app.route("/debug")
def debug():
    return render_template("debug.html")


@app.route("/api/files")
def get_files():
    """Return list of available mp3 files with their video IDs."""
    files = []
    for mp3_path in MP3_DIR.glob("*.mp3"):
        video_id = mp3_path.stem.split("_clip_")[0]
        transcript_paths = sorted(
            TRANSCRIPT_DIR.glob(f"{video_id}*.json"),
            key=lambda path: (
                0 if path.stem == f"{video_id}_ts_edit" else 1,
                0 if path.stem == video_id else 1,
                path.stem,
            ),
        )
        if not transcript_paths:
            continue

        clip_start, clip_end = parse_clip_times(mp3_path.name)
        for transcript_path in transcript_paths:
            files.append(
                {
                    "filename": mp3_path.name,
                    "video_id": video_id,
                    "transcript_id": transcript_path.stem,
                    "transcript_label": transcript_path.stem,
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                }
            )
    return jsonify(files)


@app.route("/api/transcript/<video_id>")
def get_transcript(video_id: str):
    """Return transcript with keywords extracted for each segment."""
    resolved_id = preferred_transcript_id(video_id)
    transcript_path = TRANSCRIPT_DIR / f"{resolved_id}.json"
    if not transcript_path.exists():
        return jsonify({"error": "Transcript not found"}), 404

    with open(transcript_path) as f:
        data = json.load(f)

    clip_start = data.get("start_time", 0)
    segments = load_transcript_segments_for_api(data)
    remap_words = build_word_index(data.get("segments", []), clip_start=float(clip_start))

    # Load custom keywords if exists
    keywords_path = preferred_data_path(KEYWORDS_DIR, video_id)
    custom_keywords = []
    if keywords_path.exists():
        with open(keywords_path) as f:
            custom_keywords = json.load(f)
        if keywords_path.stem != resolved_id and isinstance(custom_keywords, list):
            custom_keywords = remap_time_entries_data(custom_keywords, remap_words)

    jargon_path = preferred_data_path(KEYWORDS_DIR, video_id, ".jargon.json")
    jargon_keywords = []
    if jargon_path.exists():
        with open(jargon_path) as f:
            jargon_keywords = json.load(f)
        if jargon_path.stem.removesuffix(".jargon") != resolved_id and isinstance(jargon_keywords, list):
            jargon_keywords = remap_time_entries_data(jargon_keywords, remap_words)

    # Load custom keywords2 if exists
    keywords2_path = preferred_data_path(KEYWORDS2_DIR, video_id)
    custom_keywords2 = []
    if keywords2_path.exists():
        with open(keywords2_path) as f:
            custom_keywords2 = json.load(f)
        if keywords2_path.stem != resolved_id and isinstance(custom_keywords2, list):
            custom_keywords2 = remap_time_entries_data(custom_keywords2, remap_words)

    sentences = load_or_create_sentences(resolved_id, segments)
    merged_words = load_or_create_merged_words(resolved_id, segments)

    return jsonify(
        {
            "video_id": resolved_id,
            "clip_start": clip_start,
            "segments": segments,
            "sentences": sentences,
            "merged_words": merged_words,
            "custom_keywords": custom_keywords,
            "jargon_keywords": jargon_keywords,
            "custom_keywords2": custom_keywords2,
        }
    )


@app.route("/api/transcript/<transcript_id>/timestamps", methods=["POST"])
def save_transcript_timestamps(transcript_id: str):
    data = request.get_json()
    if not data or not isinstance(data.get("words"), list):
        return jsonify({"error": "Invalid payload"}), 400

    try:
        saved_transcript_id = apply_word_timestamp_updates(transcript_id, data["words"])
    except FileNotFoundError:
        return jsonify({"error": "Transcript not found"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"status": "saved", "transcript_id": saved_transcript_id})


@app.route("/mp3/<filename>")
def serve_mp3(filename: str):
    """Serve mp3 file."""
    return send_from_directory(MP3_DIR, filename)


@app.route("/api/reverse/<filename>")
def get_reversed_audio(filename: str):
    """Return reversed audio from 0 to end_sec as base64."""
    end_sec = request.args.get("end", type=float, default=0)
    if end_sec <= 0:
        return jsonify({"error": "Invalid end time"}), 400

    mp3_path = MP3_DIR / filename
    if not mp3_path.exists():
        return jsonify({"error": "File not found"}), 404

    audio = AudioSegment.from_mp3(mp3_path)
    end_ms = int(end_sec * 1000)
    chunk = audio[:end_ms]
    reversed_chunk = chunk.reverse()

    # Export to mp3 bytes
    buffer = io.BytesIO()
    reversed_chunk.export(buffer, format="mp3")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return jsonify(
        {
            "audio": b64,
            "duration": len(reversed_chunk) / 1000,
        }
    )


@app.route("/api/speedup/<filename>")
def get_speedup_audio(filename: str):
    """Return sped-up audio from start_sec for duration_sec as base64."""
    start_sec = request.args.get("start", type=float, default=0)
    duration_sec = request.args.get("duration", type=float, default=30)  # Only process 30 sec
    rate = request.args.get("rate", type=float, default=2.0)
    preserve_pitch = request.args.get("preserve_pitch", default="true").lower() == "true"

    # Clamp rate to reasonable values
    rate = max(1.1, min(4.0, rate))

    mp3_path = MP3_DIR / filename
    if not mp3_path.exists():
        return jsonify({"error": "File not found"}), 404

    audio = AudioSegment.from_mp3(mp3_path)
    start_ms = int(start_sec * 1000)
    end_ms = int((start_sec + duration_sec) * 1000)
    end_ms = min(end_ms, len(audio))  # Don't exceed audio length
    chunk = audio[start_ms:end_ms]

    # Track if this is the last segment
    is_last = end_ms >= len(audio)

    if preserve_pitch:
        # Use ffmpeg's atempo filter to preserve pitch
        # atempo only supports 0.5 to 2.0, so we chain multiple filters for higher rates
        buffer_in = io.BytesIO()
        chunk.export(buffer_in, format="mp3")
        buffer_in.seek(0)

        # Build atempo filter chain (each atempo is limited to 0.5-2.0)
        atempo_filters = []
        remaining_rate = rate
        while remaining_rate > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_rate /= 2.0
        atempo_filters.append(f"atempo={remaining_rate}")
        filter_str = ",".join(atempo_filters)

        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_in:
            tmp_in.write(buffer_in.read())
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path.replace(".mp3", "_out.mp3")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_in_path,
                    "-filter:a", filter_str,
                    "-vn", tmp_out_path
                ],
                capture_output=True,
                check=True
            )

            with open(tmp_out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            # Calculate duration
            speedup_chunk = AudioSegment.from_mp3(tmp_out_path)
            duration = len(speedup_chunk) / 1000
        finally:
            import os
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
    else:
        # Simple speedup by altering frame_rate (changes pitch)
        speedup_chunk = chunk._spawn(
            chunk.raw_data,
            overrides={"frame_rate": int(chunk.frame_rate * rate)}
        ).set_frame_rate(chunk.frame_rate)

        # Export to mp3 bytes
        buffer = io.BytesIO()
        speedup_chunk.export(buffer, format="mp3")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        duration = len(speedup_chunk) / 1000

    # original_duration = how much original audio time this covers
    original_duration = (end_ms - start_ms) / 1000

    return jsonify(
        {
            "audio": b64,
            "duration": duration,
            "rate": rate,
            "original_duration": original_duration,
            "is_last": is_last,
        }
    )


# ============== Study API Endpoints ==============


@app.route("/api/study/config")
def get_study_config():
    """Return study configuration."""
    config_path = STUDY_DIR / "study_config.json"
    if not config_path.exists():
        return jsonify({"error": "Study config not found"}), 404

    with open(config_path) as f:
        config = json.load(f)
    return jsonify(config)


@app.route("/api/study/interruptions/<video_id>")
def get_study_interruptions(video_id: str):
    """Return interruption config for a specific video."""
    resolved_id = preferred_transcript_id(video_id)
    interruptions_path = INTERRUPTIONS_DIR / f"{resolved_id}.json"
    if not interruptions_path.exists():
        interruptions_path = INTERRUPTIONS_DIR / f"{video_id}.json"
    if not interruptions_path.exists():
        return jsonify({"error": f"Interruptions for {video_id} not found"}), 404

    data = load_json_file(interruptions_path)
    interruptions = data.get("interruptions", [])
    if not interruptions:
        return jsonify({"error": f"No interruptions configured for {video_id}"}), 400

    study_windows = load_study_audio_windows()
    video_config = study_windows.get("videos", {}).get(video_id, {})
    audio_start_time = float(data.get("audio_start_time", video_config.get("audio_start_time", 0)))
    target_window_seconds = float(
        data.get(
            "target_window_seconds",
            video_config.get(
                "target_window_seconds",
                study_windows.get("default_target_window_seconds", 300),
            ),
        )
    )
    search_window_end_time = float(
        data.get("search_window_end_time", audio_start_time + target_window_seconds)
    )
    playback_end_time = max(
        float(item.get("target_word_time", 0)) + float(item.get("delay_seconds", 0))
        for item in interruptions
    )

    data["audio_start_time"] = audio_start_time
    data["target_window_seconds"] = target_window_seconds
    data["search_window_end_time"] = search_window_end_time
    data["playback_end_time"] = playback_end_time
    return jsonify(data)


@app.route("/api/study/log", methods=["POST"])
def log_study_event():
    """Log a single study event to JSONL file."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Create logs directory if not exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{_build_study_log_basename(data)}.jsonl"

    # Append event to log file
    with open(log_path, "a") as f:
        event = {"ts": datetime.now().isoformat(), **data}
        f.write(json.dumps(event) + "\n")

    return jsonify({"status": "logged", "file": log_path.name})


@app.route("/api/study/session", methods=["POST"])
def save_study_session():
    """Save complete study session data."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Create logs directory if not exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Save session summary
    session_path = LOGS_DIR / f"{_build_study_log_basename(data)}.json"
    with open(session_path, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved", "file": session_path.name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
