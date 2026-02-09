import argparse
import csv
import math
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot benchmark summary results."
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
        default="experiments/results/benchmark_plot.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--metric-x",
        dest="metric_x",
        default="max_drawdown_mean",
        help="X-axis metric column.",
    )
    parser.add_argument(
        "--metric-y",
        dest="metric_y",
        default="sharpe_mean",
        help="Y-axis metric column.",
    )
    parser.add_argument(
        "--layout",
        dest="layout",
        default="small_multiples",
        choices=["small_multiples", "single"],
        help="Plot layout style.",
    )
    parser.add_argument(
        "--kind",
        dest="kind",
        default="bar",
        choices=["bar", "scatter"],
        help="Plot kind.",
    )
    parser.add_argument(
        "--split-modes",
        dest="split_modes",
        action="store_true",
        help="Generate separate plots per reward_mode.",
    )
    return parser.parse_args()


def load_rows(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def plot_rows(rows, args, suffix=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from exc

    agents = sorted({r["agent"].strip() for r in rows})
    reward_modes = sorted(
        {
            r["reward_mode"].strip().lower().replace("-", "_")
            for r in rows
        }
    )
    scenarios = sorted({r["scenario"] for r in rows})

    marker_map = {
        "raw": "o",
        "risk_adjusted": "s",
    }

    if args.layout == "single":
        fig, ax = plt.subplots(figsize=(7, 5))
        axes = [ax]
        scenario_groups = {"All": rows}
    else:
        n = len(scenarios)
        cols = 2 if n >= 2 else n
        rows_count = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows_count, cols, figsize=(5.2 * cols, 3.8 * rows_count)
        )
        if hasattr(axes, "ravel"):
            axes = list(axes.ravel())
        elif not isinstance(axes, (list, tuple)):
            axes = [axes]
        scenario_groups = defaultdict(list)
        for r in rows:
            scenario_groups[r["scenario"]].append(r)

    if args.kind == "scatter":
        color_map = {
            agent: plt.cm.tab10(i % 10) for i, agent in enumerate(agents)
        }
        for ax, (scenario, group) in zip(axes, scenario_groups.items()):
            for row in group:
                x = to_float(row[args.metric_x])
                y = to_float(row[args.metric_y])
                if math.isnan(x) or math.isnan(y):
                    continue
                agent = row["agent"].strip()
                mode = row["reward_mode"].strip().lower().replace("-", "_")
                marker = marker_map.get(mode, "o")
                ax.scatter(
                    x,
                    y,
                    alpha=0.85,
                    marker=marker,
                    s=50,
                    facecolors=color_map[agent],
                    edgecolors="black",
                    linewidths=0.4,
                )
            ax.set_title(scenario)
            ax.set_xlabel(args.metric_x)
            ax.set_ylabel(args.metric_y)
            ax.grid(True, alpha=0.3)
    else:
        from matplotlib.patches import Patch

        mode_colors = {
            "raw": "#4C78A8",
            "risk_adjusted": "#F58518",
        }
        reward_modes = [m for m in reward_modes if m in mode_colors]
        n_modes = max(1, len(reward_modes))
        width = 0.8 / n_modes
        for ax, (scenario, group) in zip(axes, scenario_groups.items()):
            values = {
                (
                    r["agent"].strip(),
                    r["reward_mode"].strip().lower().replace("-", "_"),
                ): to_float(r[args.metric_y])
                for r in group
            }
            x_positions = list(range(len(agents)))
            for idx, mode in enumerate(reward_modes):
                offset = (idx - (n_modes - 1) / 2) * width
                ys = []
                for agent in agents:
                    ys.append(values.get((agent, mode), math.nan))
                ax.bar(
                    [x + offset for x in x_positions],
                    ys,
                    width=width,
                    label=mode,
                    color=mode_colors.get(mode, "#999999"),
                    alpha=0.85,
                )
            ax.set_title(scenario)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(agents, rotation=0)
            ax.set_ylabel(args.metric_y)
            ax.grid(True, axis="y", alpha=0.3)
            if len(scenario_groups) == 1:
                # Add value labels for single-plot cases (skip near-zero)
                for i, y in enumerate(ys):
                    if math.isnan(y) or abs(y) < 1e-6:
                        continue
                    ax.text(
                        x_positions[i] + offset,
                        y + (0.01 if y >= 0 else -0.01),
                        f"{y:.3f}",
                        ha="center",
                        va="bottom" if y >= 0 else "top",
                        fontsize=8,
                    )

    if args.layout != "single":
        used = len(scenario_groups)
        for ax in axes[used:]:
            ax.axis("off")

    if args.kind == "scatter":
        agent_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=agent,
                markerfacecolor=color_map[agent],
                markeredgecolor="black",
                markersize=8,
            )
            for agent in agents
        ]
        fig.legend(
            handles=agent_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=max(1, len(agent_handles)),
            frameon=False,
            title="Agent (color)",
        )
    else:
        mode_handles = [
            Patch(color=mode_colors[mode], label=mode)
            for mode in reward_modes
        ]
        if mode_handles:
            if len(scenario_groups) == 1:
                fig.legend(
                    handles=mode_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    ncol=max(1, len(mode_handles)),
                    frameon=False,
                    title="Reward mode",
                )
            else:
                fig.legend(
                    handles=mode_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.985),
                    ncol=max(1, len(mode_handles)),
                    frameon=False,
                    title="Reward mode (color)",
                )

    top = 1.0
    bottom = 0.14
    save_kwargs = {}
    if args.kind == "bar" and len(scenario_groups) == 1:
        top = 0.86
        bottom = 0.12
        save_kwargs["bbox_inches"] = "tight"
    fig.subplots_adjust(
        top=top, bottom=bottom, left=0.08, right=0.98, wspace=0.25, hspace=0.35
    )
    out_path = args.out_path
    if suffix:
        base, ext = os.path.splitext(args.out_path)
        out_path = f"{base}_{suffix}{ext}"
    plt.savefig(out_path, dpi=200, **save_kwargs)
    print(f"Wrote plot to {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(args.in_path)
    if not rows:
        raise SystemExit("No rows found in summary CSV.")

    ensure_parent_dir(args.out_path)

    if args.split_modes:
        mode_groups = defaultdict(list)
        for r in rows:
            mode = r["reward_mode"].strip().lower().replace("-", "_")
            mode_groups[mode].append(r)
        for mode, group in mode_groups.items():
            plot_rows(group, args, suffix=mode)
    else:
        plot_rows(rows, args)


if __name__ == "__main__":
    main()
