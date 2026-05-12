"""Extract semantic search anchors from a transcript JSON and save start times."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SYSTEM_PROMPT = """Your job is to help users navigate and search within an audio recording.

You will be given a transcript of a speech.

Your task is to identify words that can serve as effective anchors for locating specific moments in the audio.

Selection criteria:
- Distinct and specific (not generic words)
- Likely to be remembered or searched by users

Avoid:
- Very common or generic words

Rules:
- Return single-word terms only.
- Do not return multi-word phrases.
- If a concept appears as a multi-word phrase, split it into its meaningful component words and return those as separate single-word terms instead.
- Each returned term must be a single token that could match a single transcript word timestamp.

Return only valid JSON in the format:
{"terms": ["term1", "term2"]}"""

TERM_RE = re.compile(r"[a-z0-9']+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract semantic terms plus start times from a transcript JSON."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=ROOT_DIR / "data" / "transcripts" / "4.json",
        help="Transcript JSON with segments and word timestamps.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words" / "4.json",
        help="Output path for extracted semantic terms.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used for semantic term extraction.",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=50,
        help="Maximum number of semantic terms to request from the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of segment ids to include in each model request.",
    )
    return parser


def load_environment() -> None:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR.parent / ".env")


def normalize_tokens(text: str) -> list[str]:
    return TERM_RE.findall(text.lower())


def read_transcript(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_semantic_word_gap_distribution(output_json: Path) -> None:
    stem = output_json.stem
    output_dir = ROOT_DIR / "result" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_script = (
        ROOT_DIR / "scripts" / "analysis" / "plot_semantic_word_time_gap_distribution.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--semantic-dir",
            str(output_json.parent),
            "--ids",
            stem,
            "--output",
            str(output_dir / f"{stem}_semantic_words_gap_distribution.png"),
            "--large-gap-output",
            str(output_dir / f"{stem}_semantic_words_large_gaps.txt"),
        ],
        check=True,
        cwd=ROOT_DIR,
    )


def transcript_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            raw_word = word.get("word", "")
            tokens = normalize_tokens(raw_word)
            if not tokens:
                continue
            words.append(
                {
                    "raw": raw_word,
                    "token": tokens[0],
                    "start": word.get("start"),
                }
            )
    return words


def iter_segment_batches(
    transcript: dict[str, Any],
    *,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    segments = transcript.get("segments", [])
    return [
        segments[idx : idx + batch_size] for idx in range(0, len(segments), batch_size)
    ]


def find_term_starts(term: str, words: list[dict[str, Any]]) -> list[float]:
    term_tokens = normalize_tokens(term)
    if not term_tokens:
        return []

    starts: list[float] = []
    limit = len(words) - len(term_tokens) + 1
    for idx in range(max(limit, 0)):
        candidate = [words[idx + offset]["token"] for offset in range(len(term_tokens))]
        if candidate == term_tokens:
            starts.append(float(words[idx]["start"]))

    return starts


def dedupe_word_time_entries(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for entry in entries:
        key = (entry.get("word"), entry.get("time"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def extract_candidate_terms(
    client: OpenAI,
    *,
    model: str,
    transcript_text: str,
    max_terms: int,
    batch_label: str,
) -> list[str]:
    prompt = (
        f"Transcript segment ids: {batch_label}\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Return at most {max_terms} terms."
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

    raw_text = response.output_text.strip()
    print(f"Model response for batch {batch_label}:\n{raw_text}\n")
    data = json.loads(raw_text)
    if isinstance(data, dict) and isinstance(data.get("terms"), list):
        return [item for item in data["terms"] if isinstance(item, str)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, str)]
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return [item for item in value if isinstance(item, str)]
    raise ValueError(f"Unexpected model output: {raw_text}")


def extract_semantic_words(
    transcript: dict[str, Any],
    *,
    model: str,
    max_terms: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    client = OpenAI()
    all_words = transcript_words(transcript)
    results: list[dict[str, Any]] = []
    seen_occurrences: set[tuple[str, float]] = set()
    for batch in iter_segment_batches(transcript, batch_size=batch_size):
        if not batch:
            continue

        batch_text = " ".join(
            str(segment.get("text", "")).strip() for segment in batch
        ).strip()
        if not batch_text:
            continue

        start_id = batch[0].get("id", 0)
        end_id = batch[-1].get("id", start_id)
        candidate_terms = extract_candidate_terms(
            client,
            model=model,
            transcript_text=batch_text,
            max_terms=max_terms,
            batch_label=f"{start_id}~{end_id}",
        )

        for term in candidate_terms:
            term_tokens = normalize_tokens(term)
            if not term_tokens:
                continue

            start_times = find_term_starts(term, all_words)
            if not start_times:
                continue

            for start_time in start_times:
                occurrence_key = (" ".join(term_tokens), start_time)
                if occurrence_key in seen_occurrences:
                    continue
                seen_occurrences.add(occurrence_key)
                results.append(
                    {
                        "word": term,
                        "time": start_time,
                    }
                )

    results.sort(key=lambda item: item["time"])
    return dedupe_word_time_entries(results)


def main() -> None:
    args = build_parser().parse_args()
    load_environment()

    if not args.input_json.exists():
        raise SystemExit(f"Transcript JSON not found: {args.input_json}")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Load it in the environment or ../.env."
        )

    transcript = read_transcript(args.input_json)
    semantic_words = extract_semantic_words(
        transcript,
        model=args.model,
        max_terms=args.max_terms,
        batch_size=args.batch_size,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(semantic_words, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_semantic_word_gap_distribution(args.output_json)
    print(f"Wrote semantic words: {args.output_json}")
    print(f"Extracted {len(semantic_words)} semantic terms")


if __name__ == "__main__":
    main()
