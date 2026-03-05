#!/usr/bin/env python3
"""Context Similarity Explorer.

MP3 파일을 transcribe하고 문장별 context similarity를 계산합니다.
다양한 window 크기와 threshold를 실험해볼 수 있습니다.

Usage:
    python explore_context.py audio.mp3
    python explore_context.py audio.mp3 --prev-window 3 --curr-window 2
    python explore_context.py audio.mp3 --low-threshold 0.5 --high-threshold 0.8
    python explore_context.py audio.mp3 --mode rare-words --top-k 3
"""

import argparse
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

# OpenLexicon 경로 (프로젝트 루트 기준)
LEXICON_PATH = Path(__file__).parent.parent / "data" / "lexicon" / "OpenLexicon.xlsx"

EMBEDDING_MODEL = "text-embedding-3-small"

# 영어 불용어 (자주 쓰이는 일반적인 단어들)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "yeah", "okay", "ok", "like", "know", "think", "going", "want", "got",
    "get", "go", "come", "say", "said", "would", "could", "one", "two",
}


# =============================================================================
# OpenLexicon 로딩
# =============================================================================

_lexicon_dict: dict[str, float] = {}


def load_lexicon() -> dict[str, float]:
    """OpenLexicon.xlsx 로드하고 단어->빈도 딕셔너리 반환."""
    global _lexicon_dict
    if _lexicon_dict:
        return _lexicon_dict

    if not LEXICON_PATH.exists():
        print(f"⚠️  OpenLexicon not found: {LEXICON_PATH}")
        return {}

    print(f"📚 Loading OpenLexicon from {LEXICON_PATH}...")
    df = pd.read_excel(LEXICON_PATH)

    for _, row in df.iterrows():
        word = str(row["ortho"]).lower()
        freq = row["English_Lexicon_Project__LgSUBTLWF"]
        if pd.notna(freq):
            _lexicon_dict[word] = float(freq)
        else:
            _lexicon_dict[word] = 0.0  # NaN은 0으로 (희귀)

    print(f"✅ Loaded {len(_lexicon_dict)} words from OpenLexicon\n")
    return _lexicon_dict


def get_lexicon_frequency(word: str) -> float | None:
    """단어의 OpenLexicon 빈도 반환. 없으면 None."""
    lexicon = load_lexicon()
    return lexicon.get(word.lower())


# =============================================================================
# TF-IDF 계산
# =============================================================================


def compute_tfidf(segments: list[dict]) -> dict[str, dict[int, float]]:
    """각 세그먼트의 각 단어에 대한 TF-IDF 계산.

    Returns:
        {word: {segment_idx: tfidf_score}}
    """
    print("📊 Computing TF-IDF scores...")

    # 각 세그먼트를 문서로 취급
    docs = [tokenize(seg["text"]) for seg in segments]
    n_docs = len(docs)

    # Document frequency (DF): 단어가 등장한 문서 수
    df_counter = Counter()
    for doc in docs:
        unique_words = set(doc)
        df_counter.update(unique_words)

    # TF-IDF 계산
    tfidf: dict[str, dict[int, float]] = {}

    for doc_idx, doc in enumerate(docs):
        # Term frequency (TF) in this document
        tf_counter = Counter(doc)
        doc_len = len(doc) if doc else 1

        for word, count in tf_counter.items():
            tf = count / doc_len  # Normalized TF
            df = df_counter[word]
            idf = math.log(n_docs / df) if df > 0 else 0

            if word not in tfidf:
                tfidf[word] = {}
            tfidf[word][doc_idx] = tf * idf

    print(f"✅ TF-IDF computed for {len(tfidf)} unique words\n")
    return tfidf


def extract_rare_words_tfidf(
    text: str,
    segment_idx: int,
    tfidf: dict[str, dict[int, float]],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """TF-IDF 기반으로 rare words 추출.

    Args:
        text: 입력 문장
        segment_idx: 세그먼트 인덱스
        tfidf: TF-IDF 스코어 딕셔너리
        top_k: 반환할 최대 단어 수

    Returns:
        (word, tfidf_score) 리스트 (높은 순)
    """
    words = tokenize(text)

    candidates = []
    for w in words:
        if w in STOPWORDS:
            continue
        score = tfidf.get(w, {}).get(segment_idx, 0)
        if score > 0:
            candidates.append((w, score))

    # TF-IDF 높은 순 정렬
    candidates.sort(key=lambda x: -x[1])

    # 중복 제거하면서 top_k개 선택
    seen = set()
    result = []
    for w, score in candidates:
        if w not in seen:
            seen.add(w)
            result.append((w, score))
            if len(result) >= top_k:
                break

    return result


def extract_rare_words_lexicon(
    text: str,
    lexicon_threshold: float = 3.0,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """OpenLexicon 기반으로 rare words 추출.

    Args:
        text: 입력 문장
        lexicon_threshold: 이 빈도 이하인 단어만 rare로 간주
        top_k: 반환할 최대 단어 수

    Returns:
        (word, lexicon_freq) 리스트 (낮은 순)
    """
    words = tokenize(text)
    lexicon = load_lexicon()

    candidates = []
    for w in words:
        if w in STOPWORDS:
            continue
        freq = lexicon.get(w)
        if freq is None:
            # lexicon에 없으면 가장 희귀 (freq = -1로 표시)
            candidates.append((w, -1.0))
        elif freq <= lexicon_threshold:
            candidates.append((w, freq))

    # 빈도 낮은 순 정렬 (-1이 가장 앞)
    candidates.sort(key=lambda x: x[1])

    # 중복 제거하면서 top_k개 선택
    seen = set()
    result = []
    for w, freq in candidates:
        if w not in seen:
            seen.add(w)
            result.append((w, freq))
            if len(result) >= top_k:
                break

    return result


def transcribe_mp3(filepath: str, client: OpenAI) -> list[dict]:
    """MP3 파일을 Whisper로 transcribe하고 문장 리스트 반환.

    Returns:
        list of {"text": str, "start": float, "end": float}
    """
    print(f"\n📂 Loading: {filepath}")

    # Whisper API는 25MB 제한이 있으므로 필요시 분할
    audio = AudioSegment.from_file(filepath)
    duration_sec = len(audio) / 1000
    print(f"⏱️  Duration: {duration_sec:.1f}s")

    # 임시 파일로 변환 (Whisper는 mp3, wav 등 지원)
    temp_path = "/tmp/explore_context_temp.mp3"
    audio.export(temp_path, format="mp3")

    print("🎙️  Transcribing with Whisper...")

    with open(temp_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    # 세그먼트 추출
    segments = []
    for seg in response.segments:
        segments.append(
            {
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
            }
        )

    print(f"✅ Got {len(segments)} segments\n")

    # 임시 파일 삭제
    os.remove(temp_path)

    return segments


def get_embeddings(texts: list[str], client: OpenAI) -> list[np.ndarray]:
    """텍스트 리스트의 embeddings를 일괄 계산."""
    print(f"🧮 Computing embeddings for {len(texts)} texts...")

    # 빈 텍스트 처리
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not non_empty:
        return [np.zeros(1536) for _ in texts]

    indices, valid_texts = zip(*non_empty)

    response = client.embeddings.create(
        input=list(valid_texts),
        model=EMBEDDING_MODEL,
    )

    # 결과 매핑
    embeddings = [np.zeros(1536) for _ in texts]
    for idx, item in zip(indices, response.data):
        embeddings[idx] = np.array(item.embedding)

    print("✅ Embeddings computed\n")
    return embeddings


# =============================================================================
# Rare Words 방식
# =============================================================================


def tokenize(text: str) -> list[str]:
    """텍스트를 단어로 분리 (소문자, 알파벳만)."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) > 2]  # 2글자 이하 제외


def build_word_frequency(segments: list[dict]) -> Counter:
    """전체 corpus에서 단어 빈도 계산."""
    counter = Counter()
    for seg in segments:
        words = tokenize(seg["text"])
        counter.update(words)
    return counter


def extract_rare_words(
    text: str,
    word_freq: Counter,
    top_k: int = 3,
    freq_threshold: int = 2,
) -> list[str]:
    """문장에서 rare words 추출.

    Args:
        text: 입력 문장
        word_freq: 전체 corpus의 단어 빈도
        top_k: 반환할 최대 단어 수
        freq_threshold: 이 빈도 이하인 단어만 rare로 간주

    Returns:
        rare words 리스트 (빈도 낮은 순)
    """
    words = tokenize(text)

    # 불용어 제거 + 빈도 기준 필터링
    candidates = []
    for w in words:
        if w in STOPWORDS:
            continue
        freq = word_freq.get(w, 0)
        if freq <= freq_threshold:
            candidates.append((w, freq))

    # 빈도 낮은 순 정렬
    candidates.sort(key=lambda x: x[1])

    # 중복 제거하면서 top_k개 선택
    seen = set()
    result = []
    for w, _ in candidates:
        if w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) >= top_k:
                break

    return result


def get_rare_words_per_segment(
    segments: list[dict],
    word_freq: Counter,
    top_k: int = 3,
    freq_threshold: int = 2,
) -> list[list[str]]:
    """각 세그먼트에서 rare words 추출."""
    print(f"🔍 Extracting rare words (top_k={top_k}, freq_threshold={freq_threshold})...")

    rare_words_list = []
    for seg in segments:
        rare = extract_rare_words(seg["text"], word_freq, top_k, freq_threshold)
        rare_words_list.append(rare)

    # 통계 출력
    total_rare = sum(len(rw) for rw in rare_words_list)
    non_empty = sum(1 for rw in rare_words_list if rw)
    print(f"✅ Found {total_rare} rare words across {non_empty}/{len(segments)} segments\n")

    return rare_words_list


def get_rare_word_embeddings(
    rare_words_list: list[list[str]],
    client: OpenAI,
) -> tuple[list[list[np.ndarray]], dict[str, np.ndarray]]:
    """Rare words의 embeddings 계산.

    Returns:
        (세그먼트별 embedding 리스트, 단어->embedding 캐시)
    """
    # 모든 unique 단어 수집
    all_words = set()
    for words in rare_words_list:
        all_words.update(words)

    if not all_words:
        return [[]] * len(rare_words_list), {}

    print(f"🧮 Computing embeddings for {len(all_words)} unique rare words...")

    word_list = list(all_words)
    response = client.embeddings.create(
        input=word_list,
        model=EMBEDDING_MODEL,
    )

    # 캐시 생성
    word_to_emb = {}
    for word, item in zip(word_list, response.data):
        word_to_emb[word] = np.array(item.embedding)

    # 세그먼트별 embedding 리스트 생성
    embeddings_per_seg = []
    for words in rare_words_list:
        embs = [word_to_emb[w] for w in words if w in word_to_emb]
        embeddings_per_seg.append(embs)

    print("✅ Rare word embeddings computed\n")
    return embeddings_per_seg, word_to_emb


def calculate_rare_word_similarities(
    rare_embeddings: list[list[np.ndarray]],
    prev_window: int,
    curr_window: int,
) -> list[float]:
    """Rare words 기반 similarity 계산.

    각 window의 모든 rare word embeddings를 aggregate해서 비교.
    """
    similarities = []
    min_required = prev_window + curr_window

    for i in range(len(rare_embeddings)):
        if i + 1 < min_required:
            similarities.append(None)
            continue

        # 이전 window의 모든 rare words
        prev_start = i + 1 - prev_window - curr_window
        prev_end = i + 1 - curr_window
        prev_embs = []
        for j in range(prev_start, prev_end):
            prev_embs.extend(rare_embeddings[j])

        # 현재 window의 모든 rare words
        curr_embs = []
        for j in range(i + 1 - curr_window, i + 1):
            curr_embs.extend(rare_embeddings[j])

        # 둘 다 비어있으면 비교 불가
        if not prev_embs or not curr_embs:
            similarities.append(None)
            continue

        prev_agg = aggregate_embeddings(prev_embs)
        curr_agg = aggregate_embeddings(curr_embs)
        sim = cosine_similarity(prev_agg, curr_agg)
        similarities.append(sim)

    return similarities


def print_rare_words_results(
    segments: list[dict],
    rare_words_list: list[list[str]],
    similarities: list[float],
    low_threshold: float,
    high_threshold: float,
    prev_window: int,
    curr_window: int,
):
    """Rare words 방식 결과 출력."""
    print("=" * 90)
    print(f"📊 Rare Words Results (prev_window={prev_window}, curr_window={curr_window})")
    print(f"   Thresholds: low={low_threshold}, high={high_threshold}")
    print("=" * 90)
    print()

    context_changes = 0
    high_sims = 0
    skipped = 0

    for seg, rare_words, sim in zip(segments, rare_words_list, similarities):
        time_str = f"[{seg['start']:6.1f}s]"
        rare_str = ", ".join(rare_words) if rare_words else "(no rare words)"
        rare_str = rare_str[:30] + "..." if len(rare_str) > 30 else rare_str

        if sim is None:
            status = "⏳"
            sim_str = "  -   "
            skipped += 1
        elif sim < low_threshold:
            status = "🔴"
            sim_str = f"{sim:.3f}"
            context_changes += 1
        elif sim > high_threshold:
            status = "🟢"
            sim_str = f"{sim:.3f}"
            high_sims += 1
        else:
            status = "🟡"
            sim_str = f"{sim:.3f}"

        text = seg["text"][:40] + ("..." if len(seg["text"]) > 40 else "")
        print(f"{status} {time_str} sim={sim_str} | [{rare_str:30}] {text}")

    print()
    print("=" * 90)
    print("📈 Summary:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Context changes (sim < {low_threshold}): {context_changes}")
    print(f"   High similarity (sim > {high_threshold}): {high_sims}")
    print(f"   Skipped (not enough data): {skipped}")
    print("=" * 90)


def print_rare_words_results_with_scores(
    segments: list[dict],
    rare_words_with_scores: list[list[tuple[str, float]]],
    similarities: list[float],
    low_threshold: float,
    high_threshold: float,
    prev_window: int,
    curr_window: int,
    method: str,
):
    """Rare words 방식 결과 출력 (점수 포함)."""
    score_label = {"corpus": "freq", "tfidf": "tfidf", "lexicon": "lex"}[method]

    print("=" * 100)
    print(f"📊 Rare Words Results [{method.upper()}] (prev_window={prev_window}, curr_window={curr_window})")
    print(f"   Thresholds: low={low_threshold}, high={high_threshold}")
    print("=" * 100)
    print()

    context_changes = 0
    high_sims = 0
    skipped = 0

    for seg, rare_with_scores, sim in zip(segments, rare_words_with_scores, similarities):
        time_str = f"[{seg['start']:6.1f}s]"

        # 단어와 점수 함께 표시
        if rare_with_scores:
            if method == "lexicon":
                # lexicon: -1은 "N/A"로 표시
                parts = []
                for w, s in rare_with_scores:
                    if s < 0:
                        parts.append(f"{w}(N/A)")
                    else:
                        parts.append(f"{w}({s:.1f})")
                rare_str = ", ".join(parts)
            elif method == "tfidf":
                rare_str = ", ".join(f"{w}({s:.2f})" for w, s in rare_with_scores)
            else:  # corpus
                rare_str = ", ".join(f"{w}({int(s)})" for w, s in rare_with_scores)
        else:
            rare_str = "(no rare words)"

        rare_str = rare_str[:40] + "..." if len(rare_str) > 40 else rare_str

        if sim is None:
            status = "⏳"
            sim_str = "  -   "
            skipped += 1
        elif sim < low_threshold:
            status = "🔴"
            sim_str = f"{sim:.3f}"
            context_changes += 1
        elif sim > high_threshold:
            status = "🟢"
            sim_str = f"{sim:.3f}"
            high_sims += 1
        else:
            status = "🟡"
            sim_str = f"{sim:.3f}"

        text = seg["text"][:35] + ("..." if len(seg["text"]) > 35 else "")
        print(f"{status} {time_str} sim={sim_str} | [{rare_str:40}] {text}")

    print()
    print("=" * 100)
    print("📈 Summary:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Context changes (sim < {low_threshold}): {context_changes}")
    print(f"   High similarity (sim > {high_threshold}): {high_sims}")
    print(f"   Skipped (not enough data): {skipped}")
    print("=" * 100)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 cosine similarity 계산."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """여러 embedding을 평균으로 aggregate."""
    if not embeddings:
        return np.zeros(1536)
    return np.mean(embeddings, axis=0)


def calculate_similarities(
    embeddings: list[np.ndarray],
    prev_window: int,
    curr_window: int,
) -> list[float]:
    """각 위치에서의 similarity 계산.

    prev_window개 이전 문장 vs curr_window개 현재 문장 비교.
    """
    similarities = []
    min_required = prev_window + curr_window

    for i in range(len(embeddings)):
        if i + 1 < min_required:
            # 비교할 충분한 문장이 없음
            similarities.append(None)
        else:
            # 이전 window: [i+1 - prev_window - curr_window : i+1 - curr_window]
            # 현재 window: [i+1 - curr_window : i+1]
            prev_start = i + 1 - prev_window - curr_window
            prev_end = i + 1 - curr_window
            curr_start = i + 1 - curr_window
            curr_end = i + 1

            prev_emb = aggregate_embeddings(embeddings[prev_start:prev_end])
            curr_emb = aggregate_embeddings(embeddings[curr_start:curr_end])

            sim = cosine_similarity(prev_emb, curr_emb)
            similarities.append(sim)

    return similarities


def print_results(
    segments: list[dict],
    similarities: list[float],
    low_threshold: float,
    high_threshold: float,
    prev_window: int,
    curr_window: int,
):
    """결과 출력."""
    print("=" * 80)
    print(f"📊 Results (prev_window={prev_window}, curr_window={curr_window})")
    print(f"   Thresholds: low={low_threshold}, high={high_threshold}")
    print("=" * 80)
    print()

    context_changes = 0
    high_sims = 0

    for i, (seg, sim) in enumerate(zip(segments, similarities)):
        time_str = f"[{seg['start']:6.1f}s]"
        text = seg["text"][:60] + ("..." if len(seg["text"]) > 60 else "")

        if sim is None:
            status = "⏳"  # 아직 비교 불가
            sim_str = "  -   "
        elif sim < low_threshold:
            status = "🔴"  # context change
            sim_str = f"{sim:.3f}"
            context_changes += 1
        elif sim > high_threshold:
            status = "🟢"  # high similarity
            sim_str = f"{sim:.3f}"
            high_sims += 1
        else:
            status = "🟡"  # 중간
            sim_str = f"{sim:.3f}"

        print(f"{status} {time_str} sim={sim_str} | {text}")

    print()
    print("=" * 80)
    print("📈 Summary:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Context changes (sim < {low_threshold}): {context_changes}")
    print(f"   High similarity (sim > {high_threshold}): {high_sims}")
    print(
        f"   Middle range: {len(segments) - context_changes - high_sims - sum(1 for s in similarities if s is None)}"
    )
    print("=" * 80)


def run_experiment(
    segments: list[dict],
    embeddings: list[np.ndarray],
    prev_window: int,
    curr_window: int,
    low_threshold: float,
    high_threshold: float,
):
    """단일 실험 실행."""
    similarities = calculate_similarities(embeddings, prev_window, curr_window)
    print_results(
        segments, similarities, low_threshold, high_threshold, prev_window, curr_window
    )
    return similarities


def run_grid_search(
    segments: list[dict],
    embeddings: list[np.ndarray],
    prev_windows: list[int],
    curr_windows: list[int],
):
    """여러 window 설정으로 grid search 실행."""
    print("\n" + "=" * 80)
    print("🔍 Grid Search Results")
    print("=" * 80)
    print()

    results = []

    for pw in prev_windows:
        for cw in curr_windows:
            sims = calculate_similarities(embeddings, pw, cw)
            valid_sims = [s for s in sims if s is not None]

            if valid_sims:
                mean_sim = np.mean(valid_sims)
                std_sim = np.std(valid_sims)
                min_sim = np.min(valid_sims)
                max_sim = np.max(valid_sims)

                results.append(
                    {
                        "prev": pw,
                        "curr": cw,
                        "mean": mean_sim,
                        "std": std_sim,
                        "min": min_sim,
                        "max": max_sim,
                        "n": len(valid_sims),
                    }
                )

                print(
                    f"prev={pw}, curr={cw}: mean={mean_sim:.3f}, std={std_sim:.3f}, range=[{min_sim:.3f}, {max_sim:.3f}], n={len(valid_sims)}"
                )

    print()
    return results


def export_csv(
    filepath: str,
    segments: list[dict],
    embeddings: list[np.ndarray],
    prev_window: int,
    curr_window: int,
):
    """결과를 CSV로 내보내기."""
    similarities = calculate_similarities(embeddings, prev_window, curr_window)

    output_path = Path(filepath).stem + f"_pw{prev_window}_cw{curr_window}.csv"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("index,start,end,similarity,text\n")
        for i, (seg, sim) in enumerate(zip(segments, similarities)):
            sim_str = f"{sim:.4f}" if sim is not None else ""
            # CSV escape
            text = seg["text"].replace('"', '""')
            f.write(f'{i},{seg["start"]:.2f},{seg["end"]:.2f},{sim_str},"{text}"\n')

    print(f"📁 Exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MP3 파일의 context similarity 탐색",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 문장 전체 embedding 비교 (기본)
  python explore_context.py lecture.mp3
  python explore_context.py lecture.mp3 --prev-window 3 --curr-window 2

  # Rare words - corpus 빈도 기반 (기본)
  python explore_context.py lecture.mp3 --mode rare-words
  python explore_context.py lecture.mp3 --mode rare-words --top-k 5 --freq-threshold 3

  # Rare words - TF-IDF 기반
  python explore_context.py lecture.mp3 --mode rare-words --rare-method tfidf

  # Rare words - OpenLexicon 기반
  python explore_context.py lecture.mp3 --mode rare-words --rare-method lexicon --lexicon-threshold 2.5

  # 두 방식 비교
  python explore_context.py lecture.mp3 --mode compare

  # Grid search / CSV export
  python explore_context.py lecture.mp3 --grid-search
  python explore_context.py lecture.mp3 --export-csv
        """,
    )

    parser.add_argument("mp3_file", help="분석할 MP3 파일 경로")
    parser.add_argument(
        "--mode",
        choices=["sentence", "rare-words", "compare"],
        default="sentence",
        help="비교 방식: sentence(전체), rare-words(희귀단어), compare(둘 다)",
    )
    parser.add_argument(
        "--rare-method",
        choices=["corpus", "tfidf", "lexicon"],
        default="corpus",
        help="rare words 추출 방법: corpus(문서내빈도), tfidf, lexicon(OpenLexicon)",
    )
    parser.add_argument(
        "--prev-window", type=int, default=5, help="이전 문장 window 크기 (default: 5)"
    )
    parser.add_argument(
        "--curr-window", type=int, default=3, help="현재 문장 window 크기 (default: 3)"
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.3,
        help="Context change threshold (default: 0.3)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.5,
        help="High similarity threshold (default: 0.5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="각 문장에서 추출할 rare words 개수 (default: 3)",
    )
    parser.add_argument(
        "--freq-threshold",
        type=int,
        default=2,
        help="[corpus] 이 빈도 이하인 단어만 rare로 간주 (default: 2)",
    )
    parser.add_argument(
        "--lexicon-threshold",
        type=float,
        default=3.0,
        help="[lexicon] OpenLexicon 빈도 threshold (default: 3.0, 낮을수록 희귀)",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="여러 window 설정으로 grid search 실행",
    )
    parser.add_argument(
        "--export-csv", action="store_true", help="결과를 CSV로 내보내기"
    )

    args = parser.parse_args()

    # 파일 존재 확인
    if not os.path.exists(args.mp3_file):
        print(f"❌ File not found: {args.mp3_file}")
        sys.exit(1)

    # OpenAI 클라이언트 초기화
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # 1. Transcribe
    segments = transcribe_mp3(args.mp3_file, client)

    if not segments:
        print("❌ No segments found")
        sys.exit(1)

    # 2. 모드에 따라 처리
    if args.mode == "sentence" or args.mode == "compare":
        # 문장 전체 embedding
        texts = [seg["text"] for seg in segments]
        embeddings = get_embeddings(texts, client)

        if args.grid_search:
            prev_windows = [2, 3, 4, 5, 7, 10]
            curr_windows = [1, 2, 3, 4, 5]
            run_grid_search(segments, embeddings, prev_windows, curr_windows)
        else:
            print("\n" + "🔤 SENTENCE MODE " + "=" * 60)
            run_experiment(
                segments,
                embeddings,
                args.prev_window,
                args.curr_window,
                args.low_threshold,
                args.high_threshold,
            )

        if args.export_csv:
            export_csv(
                args.mp3_file, segments, embeddings, args.prev_window, args.curr_window
            )

    if args.mode == "rare-words" or args.mode == "compare":
        # Rare words 방식
        method_names = {"corpus": "CORPUS FREQ", "tfidf": "TF-IDF", "lexicon": "OPENLEXICON"}
        method_name = method_names[args.rare_method]
        print("\n" + f"🔑 RARE WORDS MODE ({method_name}) " + "=" * 50)

        # 방법에 따라 rare words 추출
        rare_words_list: list[list[str]] = []
        extra_info: list[list[tuple[str, float]]] = []  # (word, score) for display

        if args.rare_method == "corpus":
            # Corpus 빈도 기반
            word_freq = build_word_frequency(segments)
            print(f"📊 Corpus stats: {len(word_freq)} unique words, {sum(word_freq.values())} total\n")

            for seg in segments:
                rare = extract_rare_words(seg["text"], word_freq, args.top_k, args.freq_threshold)
                rare_words_list.append(rare)
                extra_info.append([(w, word_freq.get(w, 0)) for w in rare])

            print(f"🔍 Extracting rare words (top_k={args.top_k}, freq_threshold={args.freq_threshold})...")

        elif args.rare_method == "tfidf":
            # TF-IDF 기반
            tfidf = compute_tfidf(segments)

            for i, seg in enumerate(segments):
                rare_with_scores = extract_rare_words_tfidf(seg["text"], i, tfidf, args.top_k)
                rare_words_list.append([w for w, _ in rare_with_scores])
                extra_info.append(rare_with_scores)

            print(f"🔍 Extracting rare words by TF-IDF (top_k={args.top_k})...")

        elif args.rare_method == "lexicon":
            # OpenLexicon 기반
            load_lexicon()  # 미리 로드

            for seg in segments:
                rare_with_scores = extract_rare_words_lexicon(
                    seg["text"], args.lexicon_threshold, args.top_k
                )
                rare_words_list.append([w for w, _ in rare_with_scores])
                extra_info.append(rare_with_scores)

            print(f"🔍 Extracting rare words by OpenLexicon (top_k={args.top_k}, threshold={args.lexicon_threshold})...")

        # 통계 출력
        total_rare = sum(len(rw) for rw in rare_words_list)
        non_empty = sum(1 for rw in rare_words_list if rw)
        print(f"✅ Found {total_rare} rare words across {non_empty}/{len(segments)} segments\n")

        # Rare words embedding 계산
        rare_embeddings, _ = get_rare_word_embeddings(rare_words_list, client)

        # Similarity 계산
        similarities = calculate_rare_word_similarities(
            rare_embeddings, args.prev_window, args.curr_window
        )

        # 결과 출력 (with scores)
        print_rare_words_results_with_scores(
            segments,
            extra_info,
            similarities,
            args.low_threshold,
            args.high_threshold,
            args.prev_window,
            args.curr_window,
            args.rare_method,
        )

        # 방법별 추가 정보 출력
        if args.rare_method == "corpus":
            word_freq = build_word_frequency(segments)
            print("\n📋 Word frequency (top 20 non-stopwords):")
            content_words = [(w, c) for w, c in word_freq.most_common(100) if w not in STOPWORDS][:20]
            for word, count in content_words:
                print(f"   {word}: {count}")

            print("\n📋 Rarest words (bottom 20):")
            rare_sorted = sorted(word_freq.items(), key=lambda x: x[1])
            rare_content = [(w, c) for w, c in rare_sorted if w not in STOPWORDS][:20]
            for word, count in rare_content:
                print(f"   {word}: {count}")

        elif args.rare_method == "lexicon":
            lexicon = load_lexicon()
            # 이 transcript에서 lexicon 매칭 통계
            all_words = set()
            for seg in segments:
                all_words.update(tokenize(seg["text"]))
            matched = sum(1 for w in all_words if w in lexicon and w not in STOPWORDS)
            print(f"\n📋 Lexicon coverage: {matched}/{len(all_words - STOPWORDS)} content words found in OpenLexicon")


if __name__ == "__main__":
    main()
