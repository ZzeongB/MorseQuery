"""
Statistical control check for user study task materials.

Validates three conditions:
1. Target_words uniqueness should be consistent across all SRT files
2. Overall word uniqueness in 6-min segments should be consistent across files
3. Relative uniqueness (target words vs segment average) should be consistent

Uses ANOVA, Kruskal-Wallis, and Levene's tests for statistical validation.
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path


def load_analysis_results(json_path):
    """Load the word uniqueness analysis results."""
    with open(json_path, 'r') as f:
        return json.load(f)


def check_condition_1(candidates, alpha=0.05):
    """
    Condition 1: Target word uniqueness should be consistent across files.

    Tests:
    - Levene's test for equality of variances
    - ANOVA or Kruskal-Wallis for equality of means
    """
    print("\n" + "="*80)
    print("CONDITION 1: Target Words Uniqueness Consistency")
    print("="*80)

    # Collect uniqueness scores per file
    groups = {}
    for file_name, cands in candidates.items():
        uniqueness_scores = [c['uniqueness'] for c in cands]
        groups[file_name] = uniqueness_scores
        print(f"\n{file_name}:")
        print(f"  N candidates: {len(uniqueness_scores)}")
        print(f"  Mean: {np.mean(uniqueness_scores):.3f}")
        print(f"  Std: {np.std(uniqueness_scores):.3f}")
        print(f"  Range: [{min(uniqueness_scores):.3f}, {max(uniqueness_scores):.3f}]")

    # Prepare data for tests
    group_list = list(groups.values())

    # Levene's test for equality of variances
    stat_levene, p_levene = stats.levene(*group_list)
    print(f"\nLevene's Test (variance equality):")
    print(f"  Statistic: {stat_levene:.4f}, p-value: {p_levene:.4f}")
    print(f"  Result: {'PASS - Variances are equal' if p_levene > alpha else 'FAIL - Variances differ significantly'}")

    # ANOVA for means
    stat_anova, p_anova = stats.f_oneway(*group_list)
    print(f"\nOne-way ANOVA (mean equality):")
    print(f"  F-statistic: {stat_anova:.4f}, p-value: {p_anova:.4f}")
    print(f"  Result: {'PASS - Means are equal' if p_anova > alpha else 'FAIL - Means differ significantly'}")

    # Kruskal-Wallis (non-parametric alternative)
    stat_kw, p_kw = stats.kruskal(*group_list)
    print(f"\nKruskal-Wallis Test (non-parametric):")
    print(f"  H-statistic: {stat_kw:.4f}, p-value: {p_kw:.4f}")
    print(f"  Result: {'PASS - Distributions are equal' if p_kw > alpha else 'FAIL - Distributions differ'}")

    return {
        'levene': {'stat': stat_levene, 'p': p_levene, 'pass': p_levene > alpha},
        'anova': {'stat': stat_anova, 'p': p_anova, 'pass': p_anova > alpha},
        'kruskal': {'stat': stat_kw, 'p': p_kw, 'pass': p_kw > alpha},
        'groups': {k: {'mean': np.mean(v), 'std': np.std(v), 'n': len(v)} for k, v in groups.items()}
    }


def check_condition_2(analysis_results, alpha=0.05):
    """
    Condition 2: Overall 6-min segment word uniqueness should be consistent.

    Tests:
    - Compare distributions of all non-stopword uniqueness scores across files
    """
    print("\n" + "="*80)
    print("CONDITION 2: Overall Segment Word Uniqueness Consistency")
    print("="*80)

    groups = {}
    for result in analysis_results:
        file_name = result['srt_file']
        non_stopword = [w for w in result['word_data'] if not w['is_stopword']]
        uniqueness_scores = [w['uniqueness'] for w in non_stopword]
        groups[file_name] = uniqueness_scores

        print(f"\n{file_name}:")
        print(f"  N unique words: {len(uniqueness_scores)}")
        print(f"  Mean uniqueness: {np.mean(uniqueness_scores):.3f}")
        print(f"  Std: {np.std(uniqueness_scores):.3f}")
        print(f"  Median: {np.median(uniqueness_scores):.3f}")

    group_list = list(groups.values())

    # Levene's test
    stat_levene, p_levene = stats.levene(*group_list)
    print(f"\nLevene's Test (variance equality):")
    print(f"  Statistic: {stat_levene:.4f}, p-value: {p_levene:.4f}")
    print(f"  Result: {'PASS' if p_levene > alpha else 'FAIL'}")

    # ANOVA
    stat_anova, p_anova = stats.f_oneway(*group_list)
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic: {stat_anova:.4f}, p-value: {p_anova:.4f}")
    print(f"  Result: {'PASS' if p_anova > alpha else 'FAIL'}")

    # Kruskal-Wallis
    stat_kw, p_kw = stats.kruskal(*group_list)
    print(f"\nKruskal-Wallis Test:")
    print(f"  H-statistic: {stat_kw:.4f}, p-value: {p_kw:.4f}")
    print(f"  Result: {'PASS' if p_kw > alpha else 'FAIL'}")

    return {
        'levene': {'stat': stat_levene, 'p': p_levene, 'pass': p_levene > alpha},
        'anova': {'stat': stat_anova, 'p': p_anova, 'pass': p_anova > alpha},
        'kruskal': {'stat': stat_kw, 'p': p_kw, 'pass': p_kw > alpha},
        'groups': {k: {'mean': np.mean(v), 'std': np.std(v), 'n': len(v)} for k, v in groups.items()}
    }


def check_condition_3(analysis_results, candidates, alpha=0.05):
    """
    Condition 3: Relative uniqueness (target words compared to segment average).

    For each file, calculate:
    - Mean uniqueness of target word candidates
    - Mean uniqueness of all non-stopwords in segment
    - Ratio or difference between them

    Test if this relative measure is consistent across files.
    """
    print("\n" + "="*80)
    print("CONDITION 3: Relative Uniqueness Consistency")
    print("="*80)

    relative_scores = {}

    for result in analysis_results:
        file_name = result['srt_file']
        non_stopword = [w for w in result['word_data'] if not w['is_stopword']]
        segment_mean = np.mean([w['uniqueness'] for w in non_stopword])
        segment_std = np.std([w['uniqueness'] for w in non_stopword])

        cands = candidates.get(file_name, [])
        if not cands:
            continue

        target_mean = np.mean([c['uniqueness'] for c in cands])

        # Z-score: how many SDs above segment mean are target words?
        z_score = (target_mean - segment_mean) / segment_std

        # Simple ratio
        ratio = target_mean / segment_mean

        # Difference
        diff = target_mean - segment_mean

        relative_scores[file_name] = {
            'segment_mean': segment_mean,
            'segment_std': segment_std,
            'target_mean': target_mean,
            'z_score': z_score,
            'ratio': ratio,
            'diff': diff
        }

        print(f"\n{file_name}:")
        print(f"  Segment mean uniqueness: {segment_mean:.3f} (std: {segment_std:.3f})")
        print(f"  Target words mean: {target_mean:.3f}")
        print(f"  Z-score (target vs segment): {z_score:.3f}")
        print(f"  Ratio (target/segment): {ratio:.3f}")
        print(f"  Difference: {diff:.3f}")

    # Check consistency of relative measures
    z_scores = [v['z_score'] for v in relative_scores.values()]
    ratios = [v['ratio'] for v in relative_scores.values()]
    diffs = [v['diff'] for v in relative_scores.values()]

    print("\n--- Consistency of Relative Measures ---")
    print(f"\nZ-scores across files:")
    print(f"  Mean: {np.mean(z_scores):.3f}, Std: {np.std(z_scores):.3f}")
    print(f"  Range: [{min(z_scores):.3f}, {max(z_scores):.3f}]")
    print(f"  CV (coefficient of variation): {np.std(z_scores)/np.mean(z_scores)*100:.1f}%")

    print(f"\nRatios across files:")
    print(f"  Mean: {np.mean(ratios):.3f}, Std: {np.std(ratios):.3f}")
    print(f"  Range: [{min(ratios):.3f}, {max(ratios):.3f}]")
    print(f"  CV: {np.std(ratios)/np.mean(ratios)*100:.1f}%")

    print(f"\nDifferences across files:")
    print(f"  Mean: {np.mean(diffs):.3f}, Std: {np.std(diffs):.3f}")
    print(f"  Range: [{min(diffs):.3f}, {max(diffs):.3f}]")
    print(f"  CV: {np.std(diffs)/np.mean(diffs)*100:.1f}%")

    # Consistency threshold (CV < 20% is typically acceptable)
    cv_threshold = 20
    z_cv = np.std(z_scores)/np.mean(z_scores)*100
    ratio_cv = np.std(ratios)/np.mean(ratios)*100
    diff_cv = np.std(diffs)/np.mean(diffs)*100

    print(f"\n--- Consistency Verdict (CV < {cv_threshold}% threshold) ---")
    print(f"  Z-score CV: {z_cv:.1f}% - {'PASS' if z_cv < cv_threshold else 'FAIL'}")
    print(f"  Ratio CV: {ratio_cv:.1f}% - {'PASS' if ratio_cv < cv_threshold else 'FAIL'}")
    print(f"  Diff CV: {diff_cv:.1f}% - {'PASS' if diff_cv < cv_threshold else 'FAIL'}")

    return {
        'relative_scores': relative_scores,
        'summary': {
            'z_score': {'mean': np.mean(z_scores), 'std': np.std(z_scores), 'cv': z_cv},
            'ratio': {'mean': np.mean(ratios), 'std': np.std(ratios), 'cv': ratio_cv},
            'diff': {'mean': np.mean(diffs), 'std': np.std(diffs), 'cv': diff_cv}
        }
    }


def create_summary_plots(analysis_results, candidates, cond1, cond2, cond3, output_dir):
    """Create summary visualization of control conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Target word uniqueness distribution per file
    ax1 = axes[0, 0]
    file_names = list(candidates.keys())
    data = [np.array([c['uniqueness'] for c in candidates[f]]) for f in file_names]
    bp1 = ax1.boxplot(data, labels=[f.replace('.srt', '') for f in file_names], patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('salmon')
    ax1.set_ylabel('Uniqueness Score')
    ax1.set_xlabel('SRT File')
    ax1.set_title('Condition 1: Target Word Uniqueness\n'
                  f'ANOVA p={cond1["anova"]["p"]:.4f} ({"PASS" if cond1["anova"]["pass"] else "FAIL"})')
    ax1.axhline(y=4.0, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=6.5, color='green', linestyle='--', alpha=0.5)

    # Plot 2: Segment word uniqueness distribution per file
    ax2 = axes[0, 1]
    segment_data = []
    for result in analysis_results:
        non_stopword = [w for w in result['word_data'] if not w['is_stopword']]
        segment_data.append([w['uniqueness'] for w in non_stopword])
    bp2 = ax2.boxplot(segment_data, labels=[r['srt_file'].replace('.srt', '') for r in analysis_results],
                      patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('steelblue')
    ax2.set_ylabel('Uniqueness Score')
    ax2.set_xlabel('SRT File')
    ax2.set_title('Condition 2: Segment Word Uniqueness\n'
                  f'ANOVA p={cond2["anova"]["p"]:.4f} ({"PASS" if cond2["anova"]["pass"] else "FAIL"})')

    # Plot 3: Relative uniqueness (target vs segment)
    ax3 = axes[1, 0]
    rel_scores = cond3['relative_scores']
    files = list(rel_scores.keys())
    x = np.arange(len(files))
    width = 0.35

    segment_means = [rel_scores[f]['segment_mean'] for f in files]
    target_means = [rel_scores[f]['target_mean'] for f in files]

    ax3.bar(x - width/2, segment_means, width, label='Segment Mean', color='steelblue')
    ax3.bar(x + width/2, target_means, width, label='Target Words Mean', color='salmon')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f.replace('.srt', '') for f in files])
    ax3.set_ylabel('Uniqueness Score')
    ax3.set_xlabel('SRT File')
    ax3.legend()
    ax3.set_title('Condition 3: Target vs Segment Uniqueness')

    # Plot 4: Z-scores and ratios
    ax4 = axes[1, 1]
    z_scores = [rel_scores[f]['z_score'] for f in files]
    ax4.bar(x, z_scores, color='purple', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f.replace('.srt', '') for f in files])
    ax4.set_ylabel('Z-score (Target vs Segment)')
    ax4.set_xlabel('SRT File')
    z_cv = cond3['summary']['z_score']['cv']
    ax4.set_title(f'Relative Position: Z-scores\nCV={z_cv:.1f}% ({"PASS" if z_cv < 20 else "FAIL"})')
    ax4.axhline(y=np.mean(z_scores), color='red', linestyle='--', label=f'Mean={np.mean(z_scores):.2f}')
    ax4.legend()

    plt.tight_layout()
    output_path = output_dir / 'control_conditions_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary plot: {output_path}")


def main():
    # Load data
    output_dir = Path('/Users/jeongin/morsequery/search/data/analysis')
    json_path = output_dir / 'word_uniqueness_analysis.json'

    data = load_analysis_results(json_path)
    analysis_results = data['analysis_results']
    candidates = data['candidates']

    print("\n" + "#"*80)
    print("# STATISTICAL CONTROL CHECK FOR USER STUDY TASK MATERIALS")
    print("#"*80)

    # Run condition checks
    cond1 = check_condition_1(candidates)
    cond2 = check_condition_2(analysis_results)
    cond3 = check_condition_3(analysis_results, candidates)

    # Overall verdict
    print("\n" + "="*80)
    print("OVERALL VERDICT")
    print("="*80)

    c1_pass = cond1['anova']['pass'] and cond1['levene']['pass']
    c2_pass = cond2['anova']['pass'] or cond2['kruskal']['pass']  # Allow non-parametric
    c3_pass = cond3['summary']['z_score']['cv'] < 20

    print(f"\nCondition 1 (Target word uniqueness consistency): {'PASS' if c1_pass else 'FAIL'}")
    print(f"Condition 2 (Segment uniqueness consistency): {'PASS' if c2_pass else 'FAIL'}")
    print(f"Condition 3 (Relative uniqueness consistency): {'PASS' if c3_pass else 'FAIL'}")

    all_pass = c1_pass and c2_pass and c3_pass
    print(f"\n{'='*40}")
    print(f"FINAL VERDICT: {'ALL CONDITIONS MET - Materials are controlled' if all_pass else 'CONDITIONS NOT MET - Review needed'}")
    print(f"{'='*40}")

    # Create summary plots
    create_summary_plots(analysis_results, candidates, cond1, cond2, cond3, output_dir)

    # Save detailed stats to JSON
    stats_output = {
        'condition_1': {
            'description': 'Target word uniqueness consistency across files',
            'levene_p': float(cond1['levene']['p']),
            'anova_p': float(cond1['anova']['p']),
            'kruskal_p': float(cond1['kruskal']['p']),
            'pass': bool(c1_pass)
        },
        'condition_2': {
            'description': 'Segment uniqueness consistency across files',
            'levene_p': float(cond2['levene']['p']),
            'anova_p': float(cond2['anova']['p']),
            'kruskal_p': float(cond2['kruskal']['p']),
            'pass': bool(c2_pass)
        },
        'condition_3': {
            'description': 'Relative uniqueness (target vs segment) consistency',
            'z_score_cv': float(cond3['summary']['z_score']['cv']),
            'ratio_cv': float(cond3['summary']['ratio']['cv']),
            'diff_cv': float(cond3['summary']['diff']['cv']),
            'pass': bool(c3_pass)
        },
        'overall_pass': bool(all_pass)
    }

    stats_path = output_dir / 'control_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nSaved statistics to: {stats_path}")


if __name__ == '__main__':
    main()
