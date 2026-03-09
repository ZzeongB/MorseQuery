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
