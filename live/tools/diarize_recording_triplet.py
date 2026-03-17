#!/usr/bin/env python3
"""Transcribe mic2 and assign speakers using mic0/mic1 transcripts plus RMS."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import whisper


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(text_a: str, text_b: str) -> float:
    norm_a = normalize_text(text_a)
    norm_b = normalize_text(text_b)
    if not norm_a and not norm_b:
        return 1.0
    if not norm_a or not norm_b:
        return 0.0

    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    words_a = set(norm_a.split())
    words_b = set(norm_b.split())
    overlap = 0.0
    if words_a and words_b:
        overlap = len(words_a & words_b) / max(1, min(len(words_a), len(words_b)))
    if norm_a in norm_b or norm_b in norm_a:
        ratio = max(ratio, 0.92)
    return max(ratio, overlap)


def transcribe_audio(model: whisper.Whisper, audio_path: Path, language: str | None) -> dict:
    return model.transcribe(
        str(audio_path),
        word_timestamps=True,
        fp16=False,
        verbose=False,
        language=language,
    )


def rms_windows(audio_path: Path, window_sec: float) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError(f"{audio_path} must be mono")
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"{audio_path} must be 16-bit PCM")
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    window_size = max(1, int(sample_rate * window_sec))
    if len(samples) == 0:
        return np.zeros(1, dtype=np.float32)

    padded_len = int(math.ceil(len(samples) / window_size) * window_size)
    if padded_len != len(samples):
        samples = np.pad(samples, (0, padded_len - len(samples)))
    windows = samples.reshape(-1, window_size)
    return np.sqrt(np.mean(np.square(windows), axis=1))


def rms_stats(rms_values: np.ndarray, start: float, end: float, window_sec: float) -> dict[str, float]:
    start_idx = max(0, int(start / window_sec))
    end_idx = max(start_idx + 1, int(math.ceil(end / window_sec)))
    chunk = rms_values[start_idx:end_idx]
    if len(chunk) == 0:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": float(np.mean(chunk)), "max": float(np.max(chunk))}


def flatten_words(transcript: dict) -> list[dict]:
    words: list[dict] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            token = (word.get("word") or "").strip()
            start = word.get("start")
            end = word.get("end")
            if not token or start is None or end is None:
                continue
            words.append(
                {
                    "text": token,
                    "start": float(start),
                    "end": float(end),
                }
            )
    return words


def build_utterances(words: list[dict], max_gap: float = 0.8, max_duration: float = 6.0) -> list[dict]:
    utterances: list[dict] = []
    current: list[dict] = []

    for word in words:
        if not current:
            current = [word]
            continue

        gap = word["start"] - current[-1]["end"]
        duration = word["end"] - current[0]["start"]
        if gap > max_gap or duration > max_duration:
            utterances.append(words_to_utterance(current))
            current = [word]
            continue
        current.append(word)

    if current:
        utterances.append(words_to_utterance(current))
    return utterances


def words_to_utterance(words: list[dict]) -> dict:
    text = " ".join(word["text"] for word in words).strip()
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": text,
    }


@dataclass
class MatchCandidate:
    score: float
    time_diff: float
    text: str


def best_text_match(target: dict, candidates: list[dict], max_time_diff: float = 4.0) -> MatchCandidate:
    best = MatchCandidate(score=0.0, time_diff=999.0, text="")
    target_mid = (target["start"] + target["end"]) / 2
    for candidate in candidates:
        candidate_mid = (candidate["start"] + candidate["end"]) / 2
        time_diff = abs(target_mid - candidate_mid)
        if time_diff > max_time_diff:
            continue
        score = text_similarity(target["text"], candidate["text"])
        if score > best.score or (score == best.score and time_diff < best.time_diff):
            best = MatchCandidate(score=score, time_diff=time_diff, text=candidate["text"])
    return best


def assign_speaker(
    utterance: dict,
    match0: MatchCandidate,
    match1: MatchCandidate,
    rms0: dict[str, float],
    rms1: dict[str, float],
) -> tuple[int, dict]:
    mean0 = rms0["mean"]
    mean1 = rms1["mean"]
    max_rms = max(mean0, mean1, 1.0)
    abs_floor = 80.0
    rms0_norm = mean0 / max_rms
    rms1_norm = mean1 / max_rms
    ratio01 = mean0 / max(mean1, 1.0)
    ratio10 = mean1 / max(mean0, 1.0)

    score0 = match0.score + (0.18 if ratio01 >= 1.35 and mean0 >= abs_floor else 0.0)
    score1 = match1.score + (0.18 if ratio10 >= 1.35 and mean1 >= abs_floor else 0.0)

    if match0.score >= 0.85 and match1.score >= 0.85 and abs(match0.score - match1.score) <= 0.06:
        if ratio01 >= 1.8 and mean0 >= abs_floor:
            speaker = 1
        elif ratio10 >= 1.8 and mean1 >= abs_floor:
            speaker = 2
        else:
            speaker = 3
    elif score0 >= 0.78 and score0 >= score1 + 0.08:
        speaker = 1
    elif score1 >= 0.78 and score1 >= score0 + 0.08:
        speaker = 2
    elif ratio01 >= 3.0 and mean0 >= abs_floor and (match0.score >= 0.35 or match1.score <= 0.55):
        speaker = 1
    elif ratio10 >= 3.0 and mean1 >= abs_floor and (match1.score >= 0.35 or match0.score <= 0.55):
        speaker = 2
    elif ratio01 >= 1.8 and mean0 >= abs_floor and match0.score >= match1.score + 0.15:
        speaker = 1
    elif ratio10 >= 1.8 and mean1 >= abs_floor and match1.score >= match0.score + 0.15:
        speaker = 2
    else:
        speaker = 3

    meta = {
        "match0_score": round(match0.score, 4),
        "match1_score": round(match1.score, 4),
        "match0_text": match0.text,
        "match1_text": match1.text,
        "rms0_mean": round(mean0, 2),
        "rms1_mean": round(mean1, 2),
        "rms0_norm": round(rms0_norm, 4),
        "rms1_norm": round(rms1_norm, 4),
        "rms_ratio_0_over_1": round(ratio01, 4),
        "rms_ratio_1_over_0": round(ratio10, 4),
    }
    return speaker, meta


def merge_adjacent(entries: list[dict], max_gap: float = 0.6) -> list[dict]:
    if not entries:
        return []
    merged = [dict(entries[0])]
    for entry in entries[1:]:
        prev = merged[-1]
        if entry["speaker"] == prev["speaker"] and entry["start"] - prev["end"] <= max_gap:
            prev["end"] = entry["end"]
            prev["text"] = f'{prev["text"]} {entry["text"]}'.strip()
            prev.setdefault("source_segments", []).append(
                {"start": entry["start"], "end": entry["end"], "text": entry["text"]}
            )
            continue
        merged.append(dict(entry))
    return merged


def speaker_stats(entries: list[dict]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for entry in entries:
        key = str(entry["speaker"])
        bucket = stats.setdefault(key, {"utterances": 0, "words": 0})
        bucket["utterances"] += 1
        bucket["words"] += len(normalize_text(entry["text"]).split())
    return stats


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_text(path: Path, entries: list[dict]) -> None:
    lines = []
    for entry in entries:
        lines.append(
            f'[{entry["start"]:8.2f} - {entry["end"]:8.2f}] speaker{entry["speaker"]}: {entry["text"]}'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--model", default="base")
    parser.add_argument("--language", default=None)
    parser.add_argument("--window-sec", type=float, default=0.1)
    args = parser.parse_args()

    recording_dir = args.recording_dir.resolve()
    output_dir = recording_dir / "analysis"
    mic_paths = {name: recording_dir / f"{name}.wav" for name in ("mic0", "mic1", "mic2")}
    for path in mic_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    transcripts: dict[str, dict] = {}
    utterances: dict[str, list[dict]] = {}
    for mic_name, mic_path in mic_paths.items():
        transcript_path = output_dir / f"{mic_name}_transcript.json"
        utterance_path = output_dir / f"{mic_name}_utterances.json"
        if transcript_path.exists() and utterance_path.exists():
            print(f"Reusing cached transcript for {mic_name}")
            transcripts[mic_name] = load_json(transcript_path)
            utterances[mic_name] = load_json(utterance_path)
            continue

        print(f"Transcribing {mic_name}: {mic_path.name}")
        transcript = transcribe_audio(model, mic_path, args.language)
        transcripts[mic_name] = transcript
        words = flatten_words(transcript)
        utterances[mic_name] = build_utterances(words)
        write_json(transcript_path, transcript)
        write_json(utterance_path, utterances[mic_name])

    print("Computing RMS windows")
    rms0 = rms_windows(mic_paths["mic0"], args.window_sec)
    rms1 = rms_windows(mic_paths["mic1"], args.window_sec)

    diarized_entries: list[dict] = []
    for utterance in utterances["mic2"]:
        match0 = best_text_match(utterance, utterances["mic0"])
        match1 = best_text_match(utterance, utterances["mic1"])
        rms0_stats = rms_stats(rms0, utterance["start"], utterance["end"], args.window_sec)
        rms1_stats = rms_stats(rms1, utterance["start"], utterance["end"], args.window_sec)
        speaker, meta = assign_speaker(utterance, match0, match1, rms0_stats, rms1_stats)
        diarized_entries.append(
            {
                "speaker": speaker,
                "start": utterance["start"],
                "end": utterance["end"],
                "text": utterance["text"],
                "diagnostics": meta,
            }
        )

    merged_entries = merge_adjacent(diarized_entries)
    stats = speaker_stats(merged_entries)

    write_json(output_dir / "mic2_diarized_segments.json", diarized_entries)
    write_json(output_dir / "mic2_diarized_merged.json", merged_entries)
    write_json(output_dir / "mic2_diarized_stats.json", stats)
    write_text(output_dir / "mic2_diarized_merged.txt", merged_entries)

    print(f"Wrote outputs to {output_dir}")
    for speaker in sorted(stats):
        speaker_stats_entry = stats[speaker]
        print(
            f"speaker{speaker}: utterances={speaker_stats_entry['utterances']} "
            f"words={speaker_stats_entry['words']}"
        )


if __name__ == "__main__":
    main()
