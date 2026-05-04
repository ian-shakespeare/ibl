"""
experiment.py — Run IB1, IB2, IB3 on Voting and Waveform datasets.

Replicates Table 6 from Aha, Kibler & Albert (1991):
  - 50 independent trials
  - Fresh random train/test split each trial (paper-specified sizes)
  - Reports: mean accuracy ± standard error, mean storage %

Usage:
    python experiment.py [--data-dir DATA_DIR] [--trials N] [--seed SEED]

Output is written to results/results.txt and printed to stdout.
"""

from src.data_loader import load_voting, load_waveform
from src.ib3 import IB3
from src.ib2 import IB2
from src.ib1 import IB1
import argparse
import math
import os
import sys
import time

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std_err(values):
    """Standard error of the mean."""
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(loader_fn, data_dir, seed):
    """
    Run one trial: load data with given seed, train & evaluate IB1/IB2/IB3.

    Returns
    -------
    dict with keys: ib1_acc, ib2_acc, ib3_acc,
                    ib1_storage, ib2_storage, ib3_storage
    """
    X_train, y_train, X_test, y_test, is_numeric = loader_fn(
        data_dir=data_dir, seed=seed
    )
    n_train = len(X_train)

    results = {}
    for name, Cls in [('ib1', IB1), ('ib2', IB2), ('ib3', IB3)]:
        model = Cls(is_numeric)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[f'{name}_acc'] = accuracy(y_test, preds) * 100.0
        results[f'{name}_storage'] = model.storage_fraction(n_train) * 100.0

    return results


# ---------------------------------------------------------------------------
# Multi-trial runner
# ---------------------------------------------------------------------------

def run_dataset(name, loader_fn, data_dir, n_trials, base_seed):
    """
    Run n_trials independent trials for one dataset and aggregate results.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}  ({n_trials} trials)")
    print(f"{'='*60}")

    keys = ['ib1_acc', 'ib2_acc', 'ib3_acc',
            'ib1_storage', 'ib2_storage', 'ib3_storage']
    accum = {k: [] for k in keys}

    for t in range(n_trials):
        seed = base_seed + t
        trial_results = run_trial(loader_fn, data_dir, seed)
        for k in keys:
            accum[k].append(trial_results[k])

        # Progress indicator
        if (t + 1) % 10 == 0 or t == 0:
            print(f"  ... trial {t+1}/{n_trials}")

    # Aggregate
    summary = {k: (mean(accum[k]), std_err(accum[k])) for k in keys}
    return summary


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_table(voting_summary, waveform_summary):
    """Format results as a table matching Table 6 in the paper."""
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("Table 6 Replication — Aha, Kibler & Albert (1991)")
    lines.append("Percent accuracy ± standard error  |  Percent storage")
    lines.append("-" * 72)
    lines.append(f"{'Dataset':<14} {'IB1 Acc':>12} {'IB1 Stor':>9} "
                 f"{'IB2 Acc':>12} {'IB2 Stor':>9} "
                 f"{'IB3 Acc':>12} {'IB3 Stor':>9}")
    lines.append("-" * 72)

    for ds_name, summary in [("Voting", voting_summary),
                             ("Waveform", waveform_summary)]:
        def fmt_acc(key):
            m, se = summary[key]
            return f"{m:5.1f} ±{se:4.1f}"

        def fmt_stor(key):
            m, _ = summary[key]
            return f"{m:5.1f}%"

        lines.append(
            f"{ds_name:<14} "
            f"{fmt_acc('ib1_acc'):>12} {fmt_stor('ib1_storage'):>9} "
            f"{fmt_acc('ib2_acc'):>12} {fmt_stor('ib2_storage'):>9} "
            f"{fmt_acc('ib3_acc'):>12} {fmt_stor('ib3_storage'):>9}"
        )

    lines.append("=" * 72)
    lines.append("")

    # Also print paper's original values for comparison
    lines.append("Paper's original Table 6 values (for reference):")
    lines.append("-" * 72)
    lines.append(f"{'Dataset':<14} {'IB1 Acc':>12} {'IB1 Stor':>9} "
                 f"{'IB2 Acc':>12} {'IB2 Stor':>9} "
                 f"{'IB3 Acc':>12} {'IB3 Stor':>9}")
    lines.append("-" * 72)
    paper = {
        "Voting":   ("91.8 ±0.4", "100.0%", "90.9 ±0.5", " 11.1%", "91.6 ±0.5", "  7.4%"),
        "Waveform": ("75.2 ±0.3", "100.0%", "69.6 ±0.4", " 32.5%", "73.8 ±0.4", " 14.6%"),
    }
    for ds_name, vals in paper.items():
        lines.append(
            f"{ds_name:<14} "
            f"{vals[0]:>12} {vals[1]:>9} "
            f"{vals[2]:>12} {vals[3]:>9} "
            f"{vals[4]:>12} {vals[5]:>9}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis notes
# ---------------------------------------------------------------------------

def analysis_notes(voting_summary, waveform_summary):
    """
    Auto-generate a brief analysis comparing IB1/IB2/IB3 outcomes,
    matching the three claims from the paper we set out to test.
    """
    lines = ["\nAnalysis", "--------"]

    for ds_name, summary in [("Voting", voting_summary),
                             ("Waveform", waveform_summary)]:
        ib1_acc = summary['ib1_acc'][0]
        ib2_acc = summary['ib2_acc'][0]
        ib3_acc = summary['ib3_acc'][0]
        ib1_stor = summary['ib1_storage'][0]
        ib2_stor = summary['ib2_storage'][0]
        ib3_stor = summary['ib3_storage'][0]

        acc_drop_ib2 = ib1_acc - ib2_acc
        stor_reduction_ib2 = ib1_stor - ib2_stor
        stor_reduction_ib3 = ib1_stor - ib3_stor

        lines.append(f"\n[{ds_name}]")
        lines.append(
            f"  Claim (a) IB2 reduces storage at small accuracy cost: "
            f"storage reduced by {stor_reduction_ib2:.1f}pp, "
            f"accuracy drop {acc_drop_ib2:.1f}pp — "
            + ("SUPPORTED" if stor_reduction_ib2 > 50 and acc_drop_ib2 < 5
               else "MIXED/NOT SUPPORTED")
        )
        lines.append(
            f"  Claim (b) IB3 handles noise better than IB2: "
            f"IB3 acc={ib3_acc:.1f}% vs IB2 acc={ib2_acc:.1f}% — "
            + ("SUPPORTED" if ib3_acc > ib2_acc else "NOT SUPPORTED")
        )
        lines.append(
            f"  Claim (c) IB3 still reduces storage vs IB1: "
            f"storage reduced by {stor_reduction_ib3:.1f}pp — "
            + ("SUPPORTED" if stor_reduction_ib3 > 50 else "MIXED/NOT SUPPORTED")
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run IB1/IB2/IB3 experiments on Voting and Waveform datasets."
    )
    parser.add_argument('--data-dir', default='data',
                        help='Directory containing voting.csv and waveform.csv')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of independent trials (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    start_time = time.time()

    print(f"Running {args.trials} trials per dataset...")
    print(f"Data directory: {args.data_dir}")

    voting_summary = run_dataset(
        "Voting", load_voting, args.data_dir, args.trials, args.seed
    )
    waveform_summary = run_dataset(
        "Waveform", load_waveform, args.data_dir, args.trials, args.seed
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    table = format_table(voting_summary, waveform_summary)
    notes = analysis_notes(voting_summary, waveform_summary)
    output = table + "\n" + notes + "\n"

    print(output)

    out_path = os.path.join('results', 'results.txt')
    with open(out_path, 'w') as f:
        f.write(output)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
