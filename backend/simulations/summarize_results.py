import argparse
import csv
import math
import statistics
from collections import defaultdict

from backend.simulations.metrics import METRICS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize benchmark CSV into mean/std tables."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input CSV from run_benchmark.py",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="experiments/results/benchmark_summary.csv",
        help="Output summary CSV path.",
    )
    return parser.parse_args()


def load_rows(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        agent = row.get("agent") or "ppo"
        key = (row["scenario"], row["reward_mode"], agent)
        grouped[key].append(row)

    summary_rows = []
    for (scenario, reward_mode, agent), items in grouped.items():
        out = {
            "scenario": scenario,
            "reward_mode": reward_mode,
            "agent": agent,
            "n": len(items),
        }
        for metric in METRICS:
            values = [to_float(i[metric]) for i in items]
            values = [v for v in values if not math.isnan(v)]
            mean = statistics.mean(values) if values else math.nan
            std = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        summary_rows.append(out)

    return summary_rows


def write_summary(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = load_rows(args.in_path)
    summary = summarize(rows)
    write_summary(args.out_path, summary)
    print(f"Wrote summary to {args.out_path}")


if __name__ == "__main__":
    main()
