import argparse
import csv
import math
from collections import defaultdict


DEFAULT_METRICS = [
    "total_reward",
    "sharpe",
    "max_drawdown",
    "turnover",
    "executed_trade_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate markdown tables from summary CSV."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input summary CSV from summarize_results.py",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output markdown path.",
    )
    parser.add_argument(
        "--reward-mode",
        dest="reward_mode",
        default="",
        help="Filter to a single reward mode (raw or risk_adjusted).",
    )
    parser.add_argument(
        "--metrics",
        dest="metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to include.",
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


def format_value(mean, std):
    if math.isnan(mean):
        return "NA"
    if math.isnan(std):
        std = 0.0
    return f"{mean:.3f} ± {std:.3f}"


def build_table(rows, metrics):
    scenarios = sorted({r["scenario"] for r in rows})
    agents = sorted({r["agent"] for r in rows})
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["scenario"], r["agent"])] = r

    headers = ["scenario", "agent"] + metrics
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for scenario in scenarios:
        for agent in agents:
            row = grouped.get((scenario, agent))
            if not row:
                continue
            cells = [scenario, agent]
            for metric in metrics:
                mean = to_float(row.get(f"{metric}_mean"))
                std = to_float(row.get(f"{metric}_std"))
                cells.append(format_value(mean, std))
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = load_rows(args.in_path)

    reward_mode_filter = args.reward_mode.strip().lower().replace("-", "_")
    if reward_mode_filter:
        rows = [
            r
            for r in rows
            if r["reward_mode"].strip().lower().replace("-", "_")
            == reward_mode_filter
        ]

    sections = []
    if reward_mode_filter:
        sections.append(f"## Reward mode: {reward_mode_filter}")
        sections.append(build_table(rows, metrics))
    else:
        reward_modes = sorted(
            {r["reward_mode"].strip().lower().replace("-", "_") for r in rows}
        )
        for mode in reward_modes:
            mode_rows = [
                r
                for r in rows
                if r["reward_mode"].strip().lower().replace("-", "_") == mode
            ]
            sections.append(f"## Reward mode: {mode}")
            sections.append(build_table(mode_rows, metrics))

    with open(args.out_path, "w") as f:
        f.write("\n\n".join(sections))

    print(f"Wrote report to {args.out_path}")


if __name__ == "__main__":
    main()
