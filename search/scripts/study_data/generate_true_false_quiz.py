"""Generate true/false quiz questions from transcript JSON files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PROMPT_PATH = ROOT_DIR / "scripts" / "prompts" / "true_false_quiz_prompt.txt"
DEFAULT_INPUT_DIR = ROOT_DIR / "data" / "transcripts"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "quiz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 5 true/false quiz questions from transcript JSON files."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Single transcript JSON path. If omitted, uses --input-dir and ids.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing transcript JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where quiz JSON files will be written.",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        default=["1", "2", "3", "4", "5"],
        help="Transcript ids to process when --input-json is not provided.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model used for quiz generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def load_environment() -> None:
    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ROOT_DIR.parent / ".env")


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def transcript_text(payload: dict[str, Any]) -> str:
    text = str(payload.get("text", "")).strip()
    if text:
        return text
    parts = []
    for segment in payload.get("segments", []):
        segment_text = str(segment.get("text", "")).strip()
        if segment_text:
            parts.append(segment_text)
    return " ".join(parts).strip()


def resolve_output_id(input_path: Path, payload: dict[str, Any]) -> str:
    video_id = payload.get("video_id")
    if isinstance(video_id, str) and video_id.strip():
        return video_id.strip()
    return input_path.stem.split("_", 1)[0]


def validate_questions(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError("Model output must be a JSON object with a questions list.")

    questions = data["questions"]
    if len(questions) != 5:
        raise ValueError(f"Expected 5 questions, got {len(questions)}.")

    normalized: list[dict[str, Any]] = []
    false_count = 0
    seen_statements: set[str] = set()
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each question must be a JSON object.")
        statement = str(item.get("statement", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        answer = item.get("answer")
        if not statement:
            raise ValueError(f"Question {index} is missing a statement.")
        if not explanation:
            raise ValueError(f"Question {index} is missing an explanation.")
        if not isinstance(answer, bool):
            raise ValueError(f"Question {index} answer must be boolean.")
        if statement.lower() in seen_statements:
            raise ValueError(f"Duplicate statement detected: {statement}")
        seen_statements.add(statement.lower())
        if answer is False:
            false_count += 1
        normalized.append(
            {
                "id": index,
                "type": "true_false",
                "statement": statement,
                "answer": answer,
                "explanation": explanation,
            }
        )

    if false_count < 2:
        raise ValueError("Expected at least 2 false questions.")
    return normalized


def generate_questions(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    transcript_id: str,
    transcript: str,
) -> list[dict[str, Any]]:
    prompt = (
        f"Transcript ID: {transcript_id}\n"
        "Create the quiz from this transcript.\n\n"
        f"Transcript:\n{transcript}"
    )
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        text={"format": {"type": "json_object"}},
    )
    payload = json.loads(response.output_text.strip())
    return validate_questions(payload)


def build_output_payload(
    *,
    quiz_id: str,
    source_path: Path,
    source_payload: dict[str, Any],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": quiz_id,
        "video_id": source_payload.get("video_id", quiz_id),
        "audio_start_time": source_payload.get("audio_start_time"),
        "audio_end_time": source_payload.get("audio_end_time"),
        "target_window_seconds": source_payload.get("target_window_seconds"),
        "source_transcript_file": source_path.name,
        "question_count": len(questions),
        "questions": questions,
    }


def process_file(
    client: OpenAI,
    *,
    input_path: Path,
    output_dir: Path,
    model: str,
    system_prompt: str,
    overwrite: bool,
) -> Path:
    payload = read_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Transcript must be a JSON object: {input_path}")

    transcript = transcript_text(payload)
    if not transcript:
        raise ValueError(f"Transcript text is empty: {input_path}")

    quiz_id = resolve_output_id(input_path, payload)
    output_path = output_dir / f"{quiz_id}.json"
    if output_path.exists() and not overwrite:
        print(f"Skipping existing quiz: {output_path}")
        return output_path

    questions = generate_questions(
        client,
        model=model,
        system_prompt=system_prompt,
        transcript_id=quiz_id,
        transcript=transcript,
    )
    output_payload = build_output_payload(
        quiz_id=quiz_id,
        source_path=input_path,
        source_payload=payload,
        questions=questions,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote quiz: {output_path}")
    return output_path


def iter_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input_json:
        return [args.input_json]
    paths: list[Path] = []
    for item in args.ids:
        matches = sorted(args.input_dir.glob(f"{item}_*.json"))
        if matches:
            paths.append(matches[0])
            continue

        exact = args.input_dir / f"{item}.json"
        if exact.exists():
            paths.append(exact)
            continue

        paths.append(exact)
    return paths


def main() -> None:
    args = build_parser().parse_args()
    load_environment()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment or .env file.")
    if not PROMPT_PATH.exists():
        raise SystemExit(f"Prompt file not found: {PROMPT_PATH}")

    input_paths = iter_input_paths(args)
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Transcript JSON not found: {missing_text}")

    client = OpenAI()
    system_prompt = load_prompt()
    for input_path in input_paths:
        process_file(
            client,
            input_path=input_path,
            output_dir=args.output_dir,
            model=args.model,
            system_prompt=system_prompt,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
