"""Select jargon terms from semantic words using an LLM plus local transcript context."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SYSTEM_PROMPT = """You classify which candidate words are jargon.

Jargon means domain-specific or subject-matter vocabulary that helps anchor
understanding of a particular topic, field, or discipline.

Prefer (be inclusive):
- technical or specialized terminology
- discipline-specific concepts
- words whose meaning depends on subject-matter knowledge
- terms that are useful as subject-matter anchors
- domain-relevant proper nouns (people, laws, cases, organizations)
- words that would benefit from a search to understand context

Avoid only:
- very common everyday words (e.g., "good", "thing", "people")
- generic time/place words (e.g., "today", "here")

Examples of jargon (include these kinds of words):
- homeostasis
- multicellular
- federal
- congressional
- jurisdiction
- discrimination
- conservative
- legislature

When in doubt, INCLUDE the word.

Return only valid JSON in this format:
{"jargon_ids": [1, 4, 9]}"""

TERM_RE = re.compile(r"[a-z0-9']+")
IRREGULAR_TOKEN_MAP = {
    "nuclei": "nucleus",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract jargon terms from semantic words using transcript context."
    )
    parser.add_argument(
        "--semantic-json",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words" / "4.json",
        help="Semantic words JSON with word/time pairs.",
    )
    parser.add_argument(
        "--transcript-json",
        type=Path,
        default=ROOT_DIR / "data" / "transcripts" / "4.json",
        help="Transcript JSON with segments and word timestamps.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words" / "4.jargon.json",
        help="Output path for jargon-only semantic words.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used for jargon extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Number of candidate words to classify per request.",
    )
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=12.0,
        help="Transcript window size around each word timestamp.",
    )
    return parser


def load_environment() -> None:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR.parent / ".env")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tokens(text: str) -> list[str]:
    return TERM_RE.findall(text.lower())


def canonicalize_token(token: str) -> str:
    token = IRREGULAR_TOKEN_MAP.get(token, token)
    if token.endswith("'s") and len(token) > 2:
        token = token[:-2]
    elif token.endswith("s'") and len(token) > 2:
        token = token[:-1]

    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("sses") or token.endswith("ss"):
        return token
    if token.endswith(("us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def normalize_canonical_tokens(text: str) -> list[str]:
    return [canonicalize_token(token) for token in normalize_tokens(text)]


def transcript_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    segments = transcript.get("segments", [])
    return [segment for segment in segments if isinstance(segment, dict)]


def transcript_occurrences(transcript: dict[str, Any]) -> dict[str, list[float]]:
    occurrences: dict[str, list[float]] = {}
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            raw_word = str(word.get("word", ""))
            start = word.get("start")
            if start is None:
                continue
            tokens = normalize_tokens(raw_word)
            if len(tokens) != 1:
                continue
            token = canonicalize_token(tokens[0])
            occurrences.setdefault(token, []).append(float(start))
    return occurrences


def segment_bounds(segments: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    starts: list[float] = []
    ends: list[float] = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        starts.append(start)
        ends.append(end)
    return starts, ends


def build_context_text(
    target_time: float,
    *,
    segments: list[dict[str, Any]],
    segment_starts: list[float],
    context_seconds: float,
) -> str:
    window_start = target_time - context_seconds / 2
    window_end = target_time + context_seconds / 2
    start_idx = max(bisect_left(segment_starts, window_start) - 1, 0)
    end_idx = bisect_right(segment_starts, window_end)

    snippets: list[str] = []
    for segment in segments[start_idx:end_idx]:
        segment_start = float(segment.get("start", 0.0))
        segment_end = float(segment.get("end", segment_start))
        if segment_end < window_start or segment_start > window_end:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            snippets.append(text)
    return " ".join(snippets)


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def count_occurrences_in_window(
    times: list[float],
    center: float,
    radius_seconds: float,
) -> int:
    start = center - radius_seconds
    end = center + radius_seconds
    left = bisect_left(times, start)
    right = bisect_right(times, end)

    count = right - left
    exact_left = bisect_left(times, center)
    exact_right = bisect_right(times, center)
    if exact_right > exact_left:
        count -= 1
    return max(count, 0)


def augment_duplicate_counts(
    words: list[dict[str, Any]],
    occurrences: dict[str, list[float]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for item in words:
        word = str(item.get("word", ""))
        time = item.get("time")
        normalized = [canonicalize_token(token) for token in normalize_tokens(word)]
        entry = dict(item)
        if time is None or len(normalized) != 1:
            entry["duplicate_count_1m"] = None
            entry["duplicate_count_2m"] = None
            entry["duplicate_count_3m"] = None
            augmented.append(entry)
            continue

        times = occurrences.get(normalized[0], [])
        center = float(time)
        entry["duplicate_count_1m"] = count_occurrences_in_window(times, center, 30.0)
        entry["duplicate_count_2m"] = count_occurrences_in_window(times, center, 60.0)
        entry["duplicate_count_3m"] = count_occurrences_in_window(times, center, 90.0)
        augmented.append(entry)
    return augmented


def dedupe_jargon_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(words, key=lambda item: float(item.get("time", 0.0))):
        key = " ".join(normalize_canonical_tokens(str(item.get("word", ""))))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def select_jargon_ids(
    client: OpenAI,
    *,
    model: str,
    batch: list[dict[str, Any]],
) -> set[int]:
    word_list = ", ".join(item["word"] for item in batch)
    lines = []
    for item in batch:
        lines.append(f'id={item["id"]} word="{item["word"]}"')
    prompt = (
        "Candidate words:\n"
        f"{word_list}\n\n"
        "Be inclusive - include words that are relevant to the subject matter.\n"
        "Multi-word technical terms are allowed.\n"
        "Domain-specific vocabulary at any level (basic to advanced) should be included.\n"
        "Proper nouns tied to the topic should be included.\n"
        "When in doubt, include the word.\n\n"
        + "\n".join(lines)
    )
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        text={"format": {"type": "json_object"}},
    )
    data = json.loads(response.output_text.strip())
    jargon_ids = data.get("jargon_ids", [])
    if not isinstance(jargon_ids, list):
        raise ValueError(f"Unexpected model output: {response.output_text}")
    return {int(item) for item in jargon_ids if isinstance(item, int)}


def extract_jargon_words(
    semantic_words: list[dict[str, Any]],
    transcript: dict[str, Any],
    *,
    model: str,
    batch_size: int,
    context_seconds: float,
) -> list[dict[str, Any]]:
    segments = transcript_segments(transcript)
    segment_starts, _segment_ends = segment_bounds(segments)
    occurrences = transcript_occurrences(transcript)

    candidates: list[dict[str, Any]] = []
    for idx, item in enumerate(semantic_words):
        word = str(item.get("word", "")).strip()
        time = item.get("time")
        if not word or time is None:
            continue
        candidates.append(
            {
                "id": idx,
                "word": word,
                "time": float(time),
                "context": build_context_text(
                    float(time),
                    segments=segments,
                    segment_starts=segment_starts,
                    context_seconds=context_seconds,
                ),
            }
        )

    client = OpenAI()
    selected_ids: set[int] = set()
    for batch in iter_batches(candidates, batch_size):
        if not batch:
            continue
        selected_ids.update(select_jargon_ids(client, model=model, batch=batch))

    output = [item for idx, item in enumerate(semantic_words) if idx in selected_ids]
    output = dedupe_jargon_words(output)
    output = augment_duplicate_counts(output, occurrences)
    output.sort(key=lambda item: float(item.get("time", 0.0)))
    return output


def main() -> None:
    args = build_parser().parse_args()
    load_environment()

    if not args.semantic_json.exists():
        raise SystemExit(f"Semantic JSON not found: {args.semantic_json}")
    if not args.transcript_json.exists():
        raise SystemExit(f"Transcript JSON not found: {args.transcript_json}")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Load it in the environment or ../.env."
        )

    semantic_words = read_json(args.semantic_json)
    if not isinstance(semantic_words, list):
        raise SystemExit("Semantic JSON must contain a list of word/time entries.")

    transcript = read_json(args.transcript_json)
    if not isinstance(transcript, dict):
        raise SystemExit("Transcript JSON must contain an object.")

    jargon_words = extract_jargon_words(
        semantic_words,
        transcript,
        model=args.model,
        batch_size=args.batch_size,
        context_seconds=args.context_seconds,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(jargon_words, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote jargon words: {args.output_json}")
    print(f"Selected {len(jargon_words)} jargon terms")


if __name__ == "__main__":
    main()
