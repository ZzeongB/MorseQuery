"""
Plot uniqueness distributions for recommended 6-minute segments.

1. Target word uniqueness distribution
2. Overall audio uniqueness distribution
3. Relative uniqueness (target vs overall)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt


def load_data(analysis_path, recommendations_path):
    """Load analysis results and recommendations."""
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
    with open(recommendations_path, 'r') as f:
        recommendations = json.load(f)
    return analysis, recommendations


def get_segment_words(word_data, start_time, end_time):
    """Get all words in a time segment."""
    return [w for w in word_data if start_time <= w['first_time'] < end_time]


def main():
    output_dir = Path('/Users/jeongin/morsequery/search/data/analysis')

    # Load data
    analysis, recommendations = load_data(
        output_dir / 'word_uniqueness_analysis.json',
        output_dir / 'recommended_segments.json'
    )

    # Build word_data lookup
    word_data_by_file = {r['srt_file']: r['word_data'] for r in analysis['analysis_results']}

    # Get top recommended segment for each file
    segments = {}
    for file_name, scored_list in recommendations['segments_by_file'].items():
        if scored_list:
            top = scored_list[0]
            segments[file_name] = {
                'start': top['segment_start'],
                'end': top['segment_end'],
                'start_min': top['segment_start_min'],
                'end_min': top['segment_end_min'],
                'targets': top['selected_targets'],
                'distribution': top['distribution'],
                'target_distribution': top['target_distribution']
            }

    file_names = list(segments.keys())
    n_files = len(file_names)

    # =========================================================================
    # Plot 1: Target Word Uniqueness Distribution
    # =========================================================================
    fig1, axes1 = plt.subplots(1, n_files, figsize=(4*n_files, 5), sharey=True)
    if n_files == 1:
        axes1 = [axes1]

    target_data = {}
    for idx, file_name in enumerate(file_names):
        ax = axes1[idx]
        seg = segments[file_name]

        # Get target uniqueness values
        target_uniq = [t['uniqueness'] for t in seg['targets']]
        target_data[file_name] = target_uniq

        # Histogram
        ax.hist(target_uniq, bins=8, range=(4, 7), color='salmon', edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(target_uniq), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(target_uniq):.2f}')
        ax.set_xlabel('Uniqueness Score')
        ax.set_title(f'{file_name}\n{seg["start_min"]:.0f}m-{seg["end_min"]:.0f}m')
        ax.legend(fontsize=8)
        ax.set_xlim(3.5, 7.5)

    axes1[0].set_ylabel('Count')
    fig1.suptitle('1. Target Word Uniqueness Distribution (6 words per segment)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig1.savefig(output_dir / 'dist_1_target_uniqueness.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: dist_1_target_uniqueness.png")

    # =========================================================================
    # Plot 2: Overall Audio Uniqueness Distribution
    # =========================================================================
    fig2, axes2 = plt.subplots(1, n_files, figsize=(4*n_files, 5), sharey=True)
    if n_files == 1:
        axes2 = [axes2]

    overall_data = {}
    for idx, file_name in enumerate(file_names):
        ax = axes2[idx]
        seg = segments[file_name]
        word_data = word_data_by_file[file_name]

        # Get all words in segment (non-stopword)
        segment_words = get_segment_words(word_data, seg['start'], seg['end'])
        segment_words = [w for w in segment_words if not w.get('is_stopword', False)]
        overall_uniq = [w['uniqueness'] for w in segment_words]
        overall_data[file_name] = overall_uniq

        # Histogram
        ax.hist(overall_uniq, bins=20, range=(0, 8), color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(overall_uniq), color='darkblue', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(overall_uniq):.2f}')
        ax.axvline(4.0, color='green', linestyle=':', alpha=0.5)
        ax.axvline(6.5, color='green', linestyle=':', alpha=0.5)
        ax.set_xlabel('Uniqueness Score')
        ax.set_title(f'{file_name}\n{seg["start_min"]:.0f}m-{seg["end_min"]:.0f}m (n={len(overall_uniq)})')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 8)

    axes2[0].set_ylabel('Count')
    fig2.suptitle('2. Overall Audio Uniqueness Distribution (all words in 6-min segment)',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(output_dir / 'dist_2_overall_uniqueness.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: dist_2_overall_uniqueness.png")

    # =========================================================================
    # Plot 3: Relative Uniqueness (Target vs Overall) - Combined comparison
    # =========================================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
    axes3 = axes3.flatten()

    for idx, file_name in enumerate(file_names):
        ax = axes3[idx]
        seg = segments[file_name]

        target_uniq = target_data[file_name]
        overall_uniq = overall_data[file_name]

        # Overlapping histograms
        ax.hist(overall_uniq, bins=20, range=(0, 8), color='steelblue',
                edgecolor='black', alpha=0.5, label=f'All words (n={len(overall_uniq)})', density=True)
        ax.hist(target_uniq, bins=8, range=(4, 7), color='salmon',
                edgecolor='black', alpha=0.7, label=f'Target words (n={len(target_uniq)})', density=True)

        # Mean lines
        ax.axvline(np.mean(overall_uniq), color='darkblue', linestyle='--', linewidth=2)
        ax.axvline(np.mean(target_uniq), color='red', linestyle='--', linewidth=2)

        # Target range
        ax.axvspan(4.0, 6.5, alpha=0.1, color='green')

        ax.set_xlabel('Uniqueness Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{file_name} ({seg["start_min"]:.0f}m-{seg["end_min"]:.0f}m)\n'
                    f'Overall μ={np.mean(overall_uniq):.2f}, Target μ={np.mean(target_uniq):.2f}')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 8)

    # Hide unused subplot
    axes3[5].axis('off')

    fig3.suptitle('3. Relative Uniqueness: Target Words vs Overall Audio',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig3.savefig(output_dir / 'dist_3_relative_uniqueness.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: dist_3_relative_uniqueness.png")

    # =========================================================================
    # Plot 4: Cross-file Comparison (Box plots)
    # =========================================================================
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))

    # 4a: Target word uniqueness across files
    ax4a = axes4[0]
    target_box_data = [target_data[f] for f in file_names]
    bp1 = ax4a.boxplot(target_box_data, tick_labels=[f.replace('.srt', '') for f in file_names],
                       patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('salmon')
    ax4a.set_ylabel('Uniqueness Score')
    ax4a.set_xlabel('SRT File')
    ax4a.set_title('Target Word Uniqueness\n(6 words per segment)')
    ax4a.set_ylim(3.5, 7.5)

    # Calculate stats
    target_means = [np.mean(target_data[f]) for f in file_names]
    target_stds = [np.std(target_data[f]) for f in file_names]

    # ANOVA test
    f_stat, p_val = stats.f_oneway(*target_box_data)
    ax4a.text(0.02, 0.98, f'ANOVA p={p_val:.4f}', transform=ax4a.transAxes,
              fontsize=9, verticalalignment='top')

    # 4b: Overall uniqueness across files
    ax4b = axes4[1]
    overall_box_data = [overall_data[f] for f in file_names]
    bp2 = ax4b.boxplot(overall_box_data, tick_labels=[f.replace('.srt', '') for f in file_names],
                       patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('steelblue')
    ax4b.set_ylabel('Uniqueness Score')
    ax4b.set_xlabel('SRT File')
    ax4b.set_title('Overall Audio Uniqueness\n(all words in 6-min segment)')
    ax4b.set_ylim(0, 8)

    f_stat2, p_val2 = stats.f_oneway(*overall_box_data)
    ax4b.text(0.02, 0.98, f'ANOVA p={p_val2:.4f}', transform=ax4b.transAxes,
              fontsize=9, verticalalignment='top')

    # 4c: Relative position (Z-score of target mean vs overall)
    ax4c = axes4[2]
    z_scores = []
    for f in file_names:
        overall_mean = np.mean(overall_data[f])
        overall_std = np.std(overall_data[f])
        target_mean = np.mean(target_data[f])
        z = (target_mean - overall_mean) / overall_std
        z_scores.append(z)

    bars = ax4c.bar([f.replace('.srt', '') for f in file_names], z_scores,
                    color='purple', edgecolor='black', alpha=0.7)
    ax4c.axhline(np.mean(z_scores), color='red', linestyle='--',
                 label=f'Mean Z={np.mean(z_scores):.2f}')
    ax4c.set_ylabel('Z-score')
    ax4c.set_xlabel('SRT File')
    ax4c.set_title('Relative Uniqueness\n(Target Z-score vs Overall)')
    ax4c.legend()

    # CV calculation
    z_cv = np.std(z_scores) / np.mean(z_scores) * 100
    ax4c.text(0.02, 0.98, f'CV={z_cv:.1f}%', transform=ax4c.transAxes,
              fontsize=9, verticalalignment='top')

    fig4.suptitle('Cross-File Comparison of Recommended 6-Minute Segments',
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig4.savefig(output_dir / 'dist_4_cross_file_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: dist_4_cross_file_comparison.png")

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print("\n" + "="*80)
    print("DISTRIBUTION STATISTICS SUMMARY")
    print("="*80)

    print("\n1. Target Word Uniqueness:")
    print("-" * 60)
    for f in file_names:
        data = target_data[f]
        print(f"  {f}: mean={np.mean(data):.2f}, std={np.std(data):.2f}, "
              f"range=[{min(data):.2f}, {max(data):.2f}]")
    print(f"  ANOVA p-value: {p_val:.4f} ({'PASS' if p_val > 0.05 else 'FAIL'})")

    print("\n2. Overall Audio Uniqueness:")
    print("-" * 60)
    for f in file_names:
        data = overall_data[f]
        print(f"  {f}: mean={np.mean(data):.2f}, std={np.std(data):.2f}, n={len(data)}")
    print(f"  ANOVA p-value: {p_val2:.4f} ({'PASS' if p_val2 > 0.05 else 'FAIL'})")

    print("\n3. Relative Uniqueness (Z-scores):")
    print("-" * 60)
    for f, z in zip(file_names, z_scores):
        print(f"  {f}: Z={z:.2f}")
    print(f"  Mean Z-score: {np.mean(z_scores):.2f}")
    print(f"  CV: {z_cv:.1f}% ({'PASS' if z_cv < 20 else 'FAIL'})")


if __name__ == '__main__':
    main()
