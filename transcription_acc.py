import re
from pathlib import Path

from jiwer import cer, mer, process_words, wer, wil

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}$")
ID_RE = re.compile(r"^(?P<id>.+?)_clip_.*\.srt$")


def extract_text_from_srt(srt_path: Path) -> str:
    lines = []
    for line in srt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.isdigit():
            continue
        if TIMESTAMP_RE.match(line):
            continue
        lines.append(line)
    return " ".join(lines)


def normalize_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics_and_alignment(reference_text: str, hypothesis_text: str):
    ref = normalize_for_wer(reference_text)
    hyp = normalize_for_wer(hypothesis_text)

    out = process_words(ref, hyp)

    metrics = {
        "WER": wer(ref, hyp),
        "MER": mer(ref, hyp),
        "WIL": wil(ref, hyp),
        "CER": cer(ref, hyp),
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "ref_words": out.hits + out.substitutions + out.deletions,
        "hyp_words": out.hits + out.substitutions + out.insertions,
    }
    return metrics, out


def _get_words_from_chunk(chunk):
    """
    jiwer 버전마다 AlignmentChunk의 필드명이 달라서,
    존재하는 필드명을 순서대로 탐색해서 가져온다.
    """
    # 후보 필드명들 (버전별로 다름)
    ref_keys = ["ref_words", "reference_words", "ref", "reference", "truth_words"]
    hyp_keys = ["hyp_words", "hypothesis_words", "hyp", "hypothesis", "pred_words"]

    ref_val = None
    hyp_val = None

    for k in ref_keys:
        if hasattr(chunk, k):
            ref_val = getattr(chunk, k)
            break

    for k in hyp_keys:
        if hasattr(chunk, k):
            hyp_val = getattr(chunk, k)
            break

    # 값이 문자열일 수도/리스트일 수도 있어서 통일
    def to_str(v):
        if v is None:
            return "∅"
        if isinstance(v, str):
            return v if v.strip() else "∅"
        if isinstance(v, (list, tuple)):
            return " ".join(map(str, v)) if len(v) else "∅"
        return str(v)

    return to_str(ref_val), to_str(hyp_val)


def _get_token_lists_from_out(out):
    """
    jiwer 버전별로 토큰이 담긴 필드명이 달라서 후보를 순서대로 탐색.
    최종적으로 (ref_tokens, hyp_tokens) 를 반환. (둘 다 list[str])
    """
    ref_candidates = ["reference", "references", "truth", "truths", "ground_truth"]
    hyp_candidates = ["hypothesis", "hypotheses", "prediction", "predictions"]

    ref_obj = None
    hyp_obj = None

    for k in ref_candidates:
        if hasattr(out, k):
            ref_obj = getattr(out, k)
            break
    for k in hyp_candidates:
        if hasattr(out, k):
            hyp_obj = getattr(out, k)
            break

    def to_tokens(obj):
        # 케이스들:
        # - list[str]
        # - list[list[str]]  (pair가 1개면 [0] 사용)
        # - str
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj.split()
        if isinstance(obj, list):
            if len(obj) == 0:
                return []
            if isinstance(obj[0], list):
                return obj[0]
            if isinstance(obj[0], str):
                return obj
        return None

    ref_tokens = to_tokens(ref_obj)
    hyp_tokens = to_tokens(hyp_obj)

    # 마지막 안전장치: out.alignments[0]가 인덱스 기반이라면,
    # ref/hyp 토큰이 out에 안 담겨 있을 수도 있음 → 이 경우는 디버그 필요
    if ref_tokens is None or hyp_tokens is None:
        raise AttributeError(
            "Could not find token lists in jiwer output. "
            "Please print dir(out) to see available fields."
        )

    return ref_tokens, hyp_tokens


def _get_span_indices(chunk):
    """
    chunk에서 ref/hyp 구간 인덱스를 꺼낸다. (버전별 필드명 대응)
    반환: (r0, r1, h0, h1)  (end는 exclusive)
    """
    # 자주 쓰이는 후보들
    r0_keys = ["ref_start_idx", "ref_start", "reference_start_idx", "reference_start"]
    r1_keys = ["ref_end_idx", "ref_end", "reference_end_idx", "reference_end"]
    h0_keys = ["hyp_start_idx", "hyp_start", "hypothesis_start_idx", "hypothesis_start"]
    h1_keys = ["hyp_end_idx", "hyp_end", "hypothesis_end_idx", "hypothesis_end"]

    def pick(keys):
        for k in keys:
            if hasattr(chunk, k):
                return getattr(chunk, k)
        return None

    r0 = pick(r0_keys)
    r1 = pick(r1_keys)
    h0 = pick(h0_keys)
    h1 = pick(h1_keys)

    # 일부 버전은 start/end 대신 index range가 다른 이름일 수 있음
    if None in (r0, r1, h0, h1):
        # 디버그 힌트용: chunk가 가진 속성 중 숫자같은 후보를 보고 싶을 때
        raise AttributeError(
            f"Could not find span indices on AlignmentChunk. "
            f"Chunk attrs: {[a for a in dir(chunk) if not a.startswith('_')]}"
        )

    return int(r0), int(r1), int(h0), int(h1)


def print_alignment(out, max_edits=None):
    ref_tokens, hyp_tokens = _get_token_lists_from_out(out)

    shown = 0
    for chunk in out.alignments[0]:
        op = getattr(chunk, "type", None)
        if op == "equal":
            continue

        r0, r1, h0, h1 = _get_span_indices(chunk)
        ref_span = " ".join(ref_tokens[r0:r1]) if r1 > r0 else "∅"
        hyp_span = " ".join(hyp_tokens[h0:h1]) if h1 > h0 else "∅"

        print(f"[{(op or 'UNKNOWN').upper()}]")
        print("  REF:", ref_span)
        print("  HYP:", hyp_span)
        print()

        shown += 1
        if max_edits is not None and shown >= max_edits:
            print(f"... truncated after {max_edits} edit chunks")
            break


def collect_ids_from_srt_dir(srt_dir: Path):
    id_to_srt = {}
    for srt_path in sorted(srt_dir.glob("*.srt")):
        m = ID_RE.match(srt_path.name)
        if not m:
            continue
        example_id = m.group("id")
        # 여러 clip이 있으면 첫 번째 사용 (원하면 여기서 전략 바꿀 수 있음)
        id_to_srt.setdefault(example_id, srt_path)
    return id_to_srt


def main(max_alignment_edits=None):
    srt_dir = Path("srt")
    ref_dir = Path("transcription")

    id_to_srt = collect_ids_from_srt_dir(srt_dir)

    if not id_to_srt:
        raise RuntimeError("No valid SRT files found in srt/")

    for example_id, srt_path in id_to_srt.items():
        ref_path = ref_dir / f"{example_id}.txt"
        if not ref_path.exists():
            print(f"⚠️  Missing reference for {example_id}, skipping")
            continue

        print("=" * 80)
        print(f"ID: {example_id}")
        print("SRT:", srt_path.name)
        print("REF:", ref_path.name)
        print()

        ref_text = extract_text_from_srt(srt_path)
        hyp_text = ref_path.read_text(encoding="utf-8", errors="ignore")

        metrics, out = compute_metrics_and_alignment(ref_text, hyp_text)

        print("=== METRICS ===")
        for k in ["WER", "MER", "WIL", "CER"]:
            print(f"{k}: {metrics[k]:.4f}")

        print("\n=== COUNTS ===")
        print(
            f"ref={metrics['ref_words']} | hyp={metrics['hyp_words']} | "
            f"hits={metrics['hits']} | "
            f"sub={metrics['substitutions']} | "
            f"del={metrics['deletions']} | "
            f"ins={metrics['insertions']}"
        )

        print("\n=== ALIGNMENT (NON-EQUAL) ===")
        print_alignment(out, max_edits=max_alignment_edits)

    print("\n✅ Done.")


if __name__ == "__main__":
    # 필요하면 숫자 넣어서 alignment 출력 제한
    # 예: python wer_compare_all.py 100
    import sys

    max_edits = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_alignment_edits=max_edits)
