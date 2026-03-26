#!/usr/bin/env python3
"""
Quiz Session Analyzer
- Analyzes quiz_sessions/ data
- Compares Airpods vs No_Airpods conditions
- Statistical tests for significance
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from scipy import stats
import pandas as pd
import numpy as np


def load_session_logs(base_dir="quiz_sessions"):
    """Load all session_log.json files from quiz_sessions directory."""
    sessions = {}
    base_path = Path(base_dir)

    for session_dir in base_path.iterdir():
        if session_dir.is_dir():
            log_file = session_dir / "session_log.json"
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions[session_dir.name] = data
    return sessions


def extract_quiz_answers(session_data):
    """Extract quiz_question_answered events from session data."""
    answers = []
    for event in session_data.get("events", []):
        if event.get("event_type") == "quiz_question_answered":
            answers.append({
                "timestamp": event.get("timestamp"),
                "quiz_set": event.get("quiz_set"),
                "round_number": event.get("round_number"),
                "question_id": event.get("question_id"),
                "question_position": event.get("question_position"),
                "selected_index": event.get("selected_index"),
                "correct_index": event.get("correct_index"),
                "is_correct": event.get("is_correct"),
            })
    return answers


def analyze_by_round(answers):
    """Analyze answers grouped by round."""
    rounds = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})

    for ans in answers:
        r = ans["round_number"]
        rounds[r]["total"] += 1
        if ans["is_correct"]:
            rounds[r]["correct"] += 1
        else:
            rounds[r]["wrong"] += 1

    return dict(rounds)


def print_session_summary(session_name, answers):
    """Print summary for a single session."""
    print(f"\n{'='*60}")
    print(f"Session: {session_name}")
    print(f"{'='*60}")

    if not answers:
        print("No quiz answers found.")
        return

    quiz_set = answers[0].get("quiz_set", "Unknown")
    print(f"Quiz Set: {quiz_set}")

    rounds = analyze_by_round(answers)

    print(f"\n{'Round':<8} {'Total':<8} {'Correct':<10} {'Wrong':<8} {'Accuracy':<10}")
    print("-" * 50)

    total_all = 0
    correct_all = 0

    for r in sorted(rounds.keys()):
        data = rounds[r]
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"{r:<8} {data['total']:<8} {data['correct']:<10} {data['wrong']:<8} {acc:.1f}%")
        total_all += data["total"]
        correct_all += data["correct"]

    print("-" * 50)
    overall_acc = correct_all / total_all * 100 if total_all > 0 else 0
    print(f"{'Total':<8} {total_all:<8} {correct_all:<10} {total_all - correct_all:<8} {overall_acc:.1f}%")

    return {
        "session": session_name,
        "quiz_set": quiz_set,
        "total": total_all,
        "correct": correct_all,
        "wrong": total_all - correct_all,
        "accuracy": overall_acc,
        "rounds": rounds,
    }


def compare_conditions(summaries):
    """Compare Airpods vs No_Airpods conditions with statistical tests."""
    airpods = None
    no_airpods = None

    for s in summaries:
        if "Airpods" in s["session"] and "No" not in s["session"]:
            airpods = s
        elif "No_Airpods" in s["session"] or "No_airpods" in s["session"]:
            no_airpods = s

    if not airpods or not no_airpods:
        print("\nCannot compare: need both Airpods and No_Airpods sessions.")
        return

    print(f"\n{'='*60}")
    print("CONDITION COMPARISON")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Condition':<15} {'Total':<8} {'Correct':<10} {'Wrong':<8} {'Accuracy':<10}")
    print("-" * 55)
    print(f"{'Airpods':<15} {airpods['total']:<8} {airpods['correct']:<10} {airpods['wrong']:<8} {airpods['accuracy']:.1f}%")
    print(f"{'No Airpods':<15} {no_airpods['total']:<8} {no_airpods['correct']:<10} {no_airpods['wrong']:<8} {no_airpods['accuracy']:.1f}%")

    diff = airpods['accuracy'] - no_airpods['accuracy']
    print(f"\nAccuracy Difference: {diff:+.1f}%p (Airpods - No Airpods)")

    # Statistical tests
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")

    # Contingency table for Chi-square / Fisher's exact test
    #                 Correct   Wrong
    # Airpods           a         b
    # No Airpods        c         d

    contingency = [
        [airpods['correct'], airpods['wrong']],
        [no_airpods['correct'], no_airpods['wrong']]
    ]

    print("\nContingency Table:")
    print(f"{'':15} {'Correct':<10} {'Wrong':<10}")
    print(f"{'Airpods':<15} {contingency[0][0]:<10} {contingency[0][1]:<10}")
    print(f"{'No Airpods':<15} {contingency[1][0]:<10} {contingency[1][1]:<10}")

    # Fisher's exact test (better for small samples)
    odds_ratio, fisher_p = stats.fisher_exact(contingency)
    print(f"\nFisher's Exact Test:")
    print(f"  Odds Ratio: {odds_ratio:.3f}")
    print(f"  p-value: {fisher_p:.4f}")

    # Chi-square test
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square Test:")
    print(f"  Chi² statistic: {chi2:.3f}")
    print(f"  p-value: {chi_p:.4f}")
    print(f"  Degrees of freedom: {dof}")

    # Interpretation
    alpha = 0.05
    print(f"\n{'='*60}")
    print("INTERPRETATION (α = 0.05)")
    print(f"{'='*60}")

    if fisher_p < alpha:
        print(f"✓ Fisher's exact test: SIGNIFICANT (p = {fisher_p:.4f} < {alpha})")
        print("  → There IS a statistically significant difference between conditions.")
    else:
        print(f"✗ Fisher's exact test: NOT significant (p = {fisher_p:.4f} ≥ {alpha})")
        print("  → No statistically significant difference between conditions.")

    # Effect size (Phi coefficient for 2x2 table)
    n = airpods['total'] + no_airpods['total']
    phi = np.sqrt(chi2 / n) if n > 0 else 0
    print(f"\nEffect Size (Phi coefficient): {phi:.3f}")
    if phi < 0.1:
        print("  → Negligible effect")
    elif phi < 0.3:
        print("  → Small effect")
    elif phi < 0.5:
        print("  → Medium effect")
    else:
        print("  → Large effect")

    # Round-by-round comparison
    print(f"\n{'='*60}")
    print("ROUND-BY-ROUND COMPARISON")
    print(f"{'='*60}")

    all_rounds = set(airpods['rounds'].keys()) | set(no_airpods['rounds'].keys())

    print(f"\n{'Round':<8} {'Airpods':<20} {'No Airpods':<20} {'Diff':<10}")
    print("-" * 60)

    for r in sorted(all_rounds):
        air_data = airpods['rounds'].get(r, {"correct": 0, "total": 0})
        no_data = no_airpods['rounds'].get(r, {"correct": 0, "total": 0})

        air_acc = air_data["correct"] / air_data["total"] * 100 if air_data["total"] > 0 else 0
        no_acc = no_data["correct"] / no_data["total"] * 100 if no_data["total"] > 0 else 0

        air_str = f"{air_data['correct']}/{air_data['total']} ({air_acc:.1f}%)"
        no_str = f"{no_data['correct']}/{no_data['total']} ({no_acc:.1f}%)"
        diff_str = f"{air_acc - no_acc:+.1f}%p"

        print(f"{r:<8} {air_str:<20} {no_str:<20} {diff_str:<10}")


def export_to_csv(sessions, output_file="quiz_analysis.csv"):
    """Export all answer data to CSV for further analysis."""
    rows = []

    for session_name, session_data in sessions.items():
        condition = "Airpods" if "Airpods" in session_name and "No" not in session_name else "No_Airpods"
        answers = extract_quiz_answers(session_data)

        for ans in answers:
            rows.append({
                "session": session_name,
                "condition": condition,
                "timestamp": ans["timestamp"],
                "quiz_set": ans["quiz_set"],
                "round": ans["round_number"],
                "question_id": ans["question_id"],
                "question_position": ans["question_position"],
                "selected_index": ans["selected_index"],
                "correct_index": ans["correct_index"],
                "is_correct": ans["is_correct"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nData exported to: {output_file}")
    return df


def main():
    print("="*60)
    print("QUIZ SESSION ANALYSIS")
    print("="*60)

    # Load all sessions
    sessions = load_session_logs()
    print(f"\nFound {len(sessions)} session(s): {list(sessions.keys())}")

    # Analyze each session
    summaries = []
    for session_name, session_data in sessions.items():
        answers = extract_quiz_answers(session_data)
        summary = print_session_summary(session_name, answers)
        if summary:
            summaries.append(summary)

    # Compare conditions
    if len(summaries) >= 2:
        compare_conditions(summaries)

    # Export to CSV
    df = export_to_csv(sessions)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
