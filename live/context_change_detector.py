"""
Context Change Detector
- Uses text-embedding-3-small to compute sentence embeddings
- Compares aggregated previous N sentences vs current M sentences
- Visualizes the similarity distribution
- Shows sentence pairs below threshold (context changes)
"""

import os
import sys
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Window settings
PREV_WINDOW = 5  # Number of previous sentences to aggregate
CURR_WINDOW = 2  # Number of current sentences to aggregate


def load_sentences(file_path: str) -> list[str]:
    """Load sentences from a text file, split by sentence-ending punctuation."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', content)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    return sentences


def get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embeddings for a list of texts using OpenAI API."""
    print(f"Getting embeddings for {len(texts)} text chunks...")

    response = client.embeddings.create(
        input=texts,
        model=model
    )

    embeddings = np.array([item.embedding for item in response.data])
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_windows(sentences: list[str], prev_window: int, curr_window: int) -> list[dict]:
    """Create window pairs for comparison."""
    windows = []

    for i in range(prev_window, len(sentences) - curr_window + 1):
        prev_start = i - prev_window
        prev_end = i
        curr_start = i
        curr_end = i + curr_window

        prev_text = " ".join(sentences[prev_start:prev_end])
        curr_text = " ".join(sentences[curr_start:curr_end])

        windows.append({
            'index': i,
            'prev_range': (prev_start, prev_end),
            'curr_range': (curr_start, curr_end),
            'prev_text': prev_text,
            'curr_text': curr_text,
            'prev_sentences': sentences[prev_start:prev_end],
            'curr_sentences': sentences[curr_start:curr_end]
        })

    return windows


def calculate_window_similarities(windows: list[dict]) -> tuple[list[float], np.ndarray]:
    """Calculate similarities between window pairs."""
    # Collect all texts for batch embedding
    all_texts = []
    for w in windows:
        all_texts.append(w['prev_text'])
        all_texts.append(w['curr_text'])

    # Get embeddings in batch
    embeddings = get_embeddings(all_texts)

    # Calculate similarities
    similarities = []
    for i in range(len(windows)):
        prev_emb = embeddings[i * 2]
        curr_emb = embeddings[i * 2 + 1]
        sim = cosine_similarity(prev_emb, curr_emb)
        similarities.append(sim)

    return similarities, embeddings


def visualize_similarities(similarities: list[float], low_threshold: float, high_threshold: float, output_path: str = None):
    """Visualize the distribution of cosine similarities with both thresholds."""
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of similarities
    ax1 = axes[0]
    ax1.hist(similarities, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=low_threshold, color='r', linestyle='--', linewidth=2, label=f'Low: {low_threshold:.3f}')
    ax1.axvline(x=high_threshold, color='g', linestyle='--', linewidth=2, label=f'High: {high_threshold:.3f}')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Window Similarities\n(prev {PREV_WINDOW} sentences vs curr {CURR_WINDOW} sentences)')
    ax1.legend()

    # Plot 2: Line plot of similarities over position
    ax2 = axes[1]
    indices = range(len(similarities))
    ax2.plot(indices, similarities, marker='o', markersize=4, linewidth=1, color='steelblue')
    ax2.axhline(y=low_threshold, color='r', linestyle='--', linewidth=2, label=f'Low: {low_threshold:.3f}')
    ax2.axhline(y=high_threshold, color='g', linestyle='--', linewidth=2, label=f'High: {high_threshold:.3f}')

    # Highlight points below low threshold (context change)
    below_low = [(i, s) for i, s in enumerate(similarities) if s < low_threshold]
    if below_low:
        below_indices, below_sims = zip(*below_low)
        ax2.scatter(below_indices, below_sims, color='red', s=80, zorder=5, label='Context Change')

    # Highlight points above high threshold (very similar)
    above_high = [(i, s) for i, s in enumerate(similarities) if s > high_threshold]
    if above_high:
        above_indices, above_sims = zip(*above_high)
        ax2.scatter(above_indices, above_sims, color='green', s=80, zorder=5, label='High Similarity')

    ax2.set_xlabel('Window Position (sentence index)')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Similarity Over Transcript Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()


def find_context_changes(windows: list[dict], similarities: list[float], low_threshold: float, high_threshold: float) -> tuple[list[dict], list[dict]]:
    """Find windows where similarity is below low_threshold or above high_threshold."""
    low_changes = []
    high_changes = []
    for i, (w, sim) in enumerate(zip(windows, similarities)):
        item = {
            'position': i,
            'sentence_index': w['index'],
            'similarity': sim,
            'prev_sentences': w['prev_sentences'],
            'curr_sentences': w['curr_sentences']
        }
        if sim < low_threshold:
            low_changes.append(item)
        if sim > high_threshold:
            high_changes.append(item)
    return low_changes, high_changes


def calculate_thresholds(similarities: list[float], low_percentile: int = 15, high_percentile: int = 85) -> tuple[float, float]:
    """Calculate low and high thresholds using percentiles."""
    low = np.percentile(similarities, low_percentile)
    high = np.percentile(similarities, high_percentile)
    return low, high


def calculate_threshold(similarities: list[float], method: str = 'percentile', percentile: int = 15) -> float:
    """Calculate an appropriate threshold using different methods."""
    if method == 'percentile':
        return np.percentile(similarities, percentile)
    elif method == 'std':
        return np.mean(similarities) - 1.5 * np.std(similarities)
    elif method == 'median':
        return np.median(similarities) - 0.05
    else:
        raise ValueError(f"Unknown method: {method}")


def main(file_path: str, low_threshold: float = None, high_threshold: float = None, low_percentile: int = 15, high_percentile: int = 85):
    """Main function to detect context changes in a transcript."""

    # Load sentences
    print(f"Loading sentences from: {file_path}")
    sentences = load_sentences(file_path)
    print(f"Loaded {len(sentences)} sentences")

    if len(sentences) < PREV_WINDOW + CURR_WINDOW:
        print(f"Error: Need at least {PREV_WINDOW + CURR_WINDOW} sentences.")
        return

    # Print first few sentences for verification
    print("\nFirst 5 sentences:")
    for i, s in enumerate(sentences[:5]):
        print(f"  [{i}] {s[:100]}{'...' if len(s) > 100 else ''}")

    # Create windows
    print(f"\nCreating windows: prev {PREV_WINDOW} sentences vs curr {CURR_WINDOW} sentences")
    windows = create_windows(sentences, PREV_WINDOW, CURR_WINDOW)
    print(f"Created {len(windows)} comparison windows")

    # Calculate similarities
    similarities, _ = calculate_window_similarities(windows)

    # Statistics
    print("\nSimilarity Statistics:")
    print(f"  Mean:   {np.mean(similarities):.4f}")
    print(f"  Std:    {np.std(similarities):.4f}")
    print(f"  Min:    {np.min(similarities):.4f}")
    print(f"  Max:    {np.max(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")

    # Calculate thresholds if not provided
    if low_threshold is None or high_threshold is None:
        low_threshold, high_threshold = calculate_thresholds(similarities, low_percentile, high_percentile)
        print(f"\nCalculated thresholds:")
        print(f"  Low  ({low_percentile}th percentile): {low_threshold:.4f}")
        print(f"  High ({high_percentile}th percentile): {high_threshold:.4f}")
    else:
        print(f"\nUsing provided thresholds: low={low_threshold:.4f}, high={high_threshold:.4f}")

    # Find context changes
    low_changes, high_changes = find_context_changes(windows, similarities, low_threshold, high_threshold)

    # Print LOW similarity pairs (context changes)
    print(f"\n{'='*80}")
    print(f"CONTEXT CHANGES: {len(low_changes)} pairs with similarity < {low_threshold:.4f}")
    print(f"{'='*80}")

    for i, change in enumerate(low_changes):
        print(f"\n[Low {i+1}] Position: {change['position']}, Sentence #{change['sentence_index']}, Similarity: {change['similarity']:.4f}")
        print(f"  BEFORE ({PREV_WINDOW} sentences):")
        for j, s in enumerate(change['prev_sentences']):
            print(f"    {j+1}. {s[:80]}{'...' if len(s) > 80 else ''}")
        print(f"  AFTER ({CURR_WINDOW} sentences):")
        for j, s in enumerate(change['curr_sentences']):
            print(f"    {j+1}. {s[:80]}{'...' if len(s) > 80 else ''}")

    # Print HIGH similarity pairs (very similar content)
    print(f"\n{'='*80}")
    print(f"HIGH SIMILARITY: {len(high_changes)} pairs with similarity > {high_threshold:.4f}")
    print(f"{'='*80}")

    for i, change in enumerate(high_changes):
        print(f"\n[High {i+1}] Position: {change['position']}, Sentence #{change['sentence_index']}, Similarity: {change['similarity']:.4f}")
        print(f"  BEFORE ({PREV_WINDOW} sentences):")
        for j, s in enumerate(change['prev_sentences']):
            print(f"    {j+1}. {s[:80]}{'...' if len(s) > 80 else ''}")
        print(f"  AFTER ({CURR_WINDOW} sentences):")
        for j, s in enumerate(change['curr_sentences']):
            print(f"    {j+1}. {s[:80]}{'...' if len(s) > 80 else ''}")

    # Visualize
    output_dir = os.path.dirname(file_path) or '.'
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_similarity_plot.png")
    visualize_similarities(similarities, low_threshold, high_threshold, output_path)

    return {
        'sentences': sentences,
        'windows': windows,
        'similarities': similarities,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'low_changes': low_changes,
        'high_changes': high_changes
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python context_change_detector.py <file_path> [low_threshold] [high_threshold]")
        print("Example: python context_change_detector.py ../srt/openai_transcription/bAkuNXtgrLA.txt 0.6 0.85")
        sys.exit(1)

    file_path = sys.argv[1]
    low_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None
    high_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else None

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    main(file_path, low_threshold=low_threshold, high_threshold=high_threshold)
