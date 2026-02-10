import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark + summary + plots for paper results."
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/results",
        help="Output directory for CSV/plots.",
    )
    parser.add_argument(
        "--agents",
        default="random,rule_based,buy_and_hold,ppo",
        help="Comma-separated agents.",
    )
    parser.add_argument(
        "--reward-modes",
        default="raw,risk_adjusted",
        help="Comma-separated reward modes.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Episodes per non-PPO agent.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="PPO training timesteps.",
    )
    parser.add_argument(
        "--ppo-repeats",
        type=int,
        default=3,
        help="PPO repeats per setting.",
    )
    parser.add_argument(
        "--trade-size",
        type=int,
        default=50,
        help="Units per trade.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.02,
        help="PPO entropy coefficient.",
    )
    parser.add_argument(
        "--inactivity-penalty",
        type=float,
        default=0.05,
        help="Penalty for not trading.",
    )
    parser.add_argument(
        "--drawdown-coeff",
        type=float,
        default=0.1,
        help="Drawdown penalty coefficient.",
    )
    parser.add_argument(
        "--volatility-coeff",
        type=float,
        default=0.1,
        help="Volatility penalty coefficient.",
    )
    parser.add_argument(
        "--prices-csv",
        default="",
        help="Optional prices CSV for real-world validation.",
    )
    parser.add_argument(
        "--price-col",
        default="Close",
        help="Price column in CSV.",
    )
    parser.add_argument(
        "--date-col",
        default="Date",
        help="Date column for sorting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Train split ratio when using prices CSV.",
    )
    parser.add_argument(
        "--error-bars",
        action="store_true",
        help="Add std error bars to plot.",
    )
    parser.add_argument(
        "--ppo-progress",
        action="store_true",
        help="Print PPO progress logs.",
    )
    parser.add_argument(
        "--ppo-log-every",
        type=int,
        default=2000,
        help="Steps between PPO progress logs.",
    )
    return parser.parse_args()


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = "realdata" if args.prices_csv else "benchmark"
    results = os.path.join(args.out_dir, f"{prefix}_results.csv")
    summary = os.path.join(args.out_dir, f"{prefix}_summary.csv")
    plot = os.path.join(args.out_dir, f"{prefix}_plot.png")
    table = os.path.join(args.out_dir, f"{prefix}_table.md")

    cmd = [
        sys.executable,
        "-m",
        "backend.simulations.run_benchmark",
        "--agents",
        args.agents,
        "--reward-modes",
        args.reward_modes,
        "--episodes",
        str(args.episodes),
        "--timesteps",
        str(args.timesteps),
        "--ppo-repeats",
        str(args.ppo_repeats),
        "--trade-size",
        str(args.trade_size),
        "--entropy-coef",
        str(args.entropy_coef),
        "--inactivity-penalty",
        str(args.inactivity_penalty),
        "--drawdown-coeff",
        str(args.drawdown_coeff),
        "--volatility-coeff",
        str(args.volatility_coeff),
        "--out",
        results,
    ]
    if args.prices_csv:
        cmd.extend(
            [
                "--prices-csv",
                args.prices_csv,
                "--price-col",
                args.price_col,
                "--date-col",
                args.date_col,
                "--train-ratio",
                str(args.train_ratio),
            ]
        )
    if args.ppo_progress:
        cmd.extend(
            ["--ppo-progress", "--ppo-log-every", str(args.ppo_log_every)]
        )

    run(cmd)
    run(
        [
            sys.executable,
            "-m",
            "backend.simulations.summarize_results",
            "--in",
            results,
            "--out",
            summary,
        ]
    )

    plot_cmd = [
        sys.executable,
        "-m",
        "backend.simulations.plot_results",
        "--in",
        summary,
        "--out",
        plot,
        "--kind",
        "bar",
        "--metric-y",
        "total_reward_mean",
    ]
    if args.error_bars:
        plot_cmd.append("--error-bars")
    run(plot_cmd)

    run(
        [
            sys.executable,
            "-m",
            "backend.simulations.report_results",
            "--in",
            summary,
            "--out",
            table,
        ]
    )


if __name__ == "__main__":
    main()
