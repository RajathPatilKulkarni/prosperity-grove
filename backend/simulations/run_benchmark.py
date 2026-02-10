import argparse
import csv
import os

from backend.simulations.data_loader import load_prices_csv
from backend.simulations.run_simulation import (
    run_experiment,
    run_ppo_episode,
    run_ppo_train_eval,
)
from experiments.market_scenarios import SCENARIOS, regime_schedule


DEFAULT_SCENARIOS = [
    "bull",
    "bear",
    "sideways",
    "volatile",
    "regime_shift_short",
    "regime_shift_long",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark sweeps and export CSV results."
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="random,rule_based,ppo",
        help="Comma-separated list of agents.",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated list of scenarios.",
    )
    parser.add_argument(
        "--reward-modes",
        type=str,
        default="raw,risk_adjusted",
        help="Comma-separated list of reward modes.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per run for non-PPO agents.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Training timesteps for PPO.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Seed offset for non-PPO episodes.",
    )
    parser.add_argument(
        "--ppo-repeats",
        type=int,
        default=3,
        help="Number of PPO runs per setting.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="",
        help="Custom regime schedule, e.g. bull,volatile,bear",
    )
    parser.add_argument(
        "--schedule-length",
        type=int,
        default=20,
        help="Per-regime length for custom schedule.",
    )
    parser.add_argument(
        "--drawdown-coeff",
        type=float,
        default=0.02,
        help="Drawdown penalty coefficient for risk-adjusted reward.",
    )
    parser.add_argument(
        "--volatility-coeff",
        type=float,
        default=0.02,
        help="Volatility penalty coefficient for risk-adjusted reward.",
    )
    parser.add_argument(
        "--trade-penalty-coeff",
        type=float,
        default=0.0,
        help="Trade penalty coefficient for risk-adjusted reward.",
    )
    parser.add_argument(
        "--invalid-action-penalty",
        type=float,
        default=0.5,
        help="Penalty for invalid actions (e.g., sell with no holdings).",
    )
    parser.add_argument(
        "--inactivity-penalty",
        type=float,
        default=0.02,
        help="Penalty for not executing a trade in a step.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.05,
        help="PPO entropy coefficient (encourages exploration).",
    )
    parser.add_argument(
        "--trade-size",
        type=int,
        default=1,
        help="Units per trade (increases action impact).",
    )
    parser.add_argument(
        "--ppo-progress",
        action="store_true",
        help="Print PPO training progress.",
    )
    parser.add_argument(
        "--ppo-log-every",
        type=int,
        default=5000,
        help="Steps between PPO progress logs.",
    )
    parser.add_argument(
        "--prices-csv",
        type=str,
        default="",
        help="CSV file with historical prices (overrides scenarios).",
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="Close",
        help="Column name for prices in CSV.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="",
        help="Optional date column for sorting CSV.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Train split ratio for PPO when using prices CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional max rows to load from CSV (0 = all).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/results/benchmark_results.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def scenario_prices(scenario, schedule, schedule_length):
    if schedule:
        return regime_schedule(schedule, length=schedule_length), "custom"
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown market scenario: {scenario}")
    return SCENARIOS[scenario](), scenario


def write_row(writer, base, metrics):
    row = {**base, **metrics}
    writer.writerow(row)


def run_benchmark():
    args = parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    reward_modes = [
        r.strip() for r in args.reward_modes.split(",") if r.strip()
    ]
    schedule = [s.strip() for s in args.schedule.split(",") if s.strip()]

    out_path = args.out
    ensure_parent_dir(out_path)

    scenario_groups = []
    if args.prices_csv:
        max_rows = args.max_rows or None
        date_col = args.date_col.strip() or None
        prices = load_prices_csv(
            args.prices_csv,
            price_col=args.price_col,
            date_col=date_col,
            max_rows=max_rows,
        )
        if len(prices) < 2:
            raise ValueError("Need at least 2 prices for evaluation.")
        train_ratio = args.train_ratio
        if 0 < train_ratio < 1:
            split_idx = max(2, int(len(prices) * train_ratio))
            split_idx = min(split_idx, len(prices) - 2)
            train_prices = prices[:split_idx]
            eval_prices = prices[split_idx:]
        else:
            train_prices = prices
            eval_prices = prices
        label = os.path.splitext(os.path.basename(args.prices_csv))[0]
        scenario_groups.append((f"real_{label}", train_prices, eval_prices))
    else:
        for scenario in scenarios:
            prices, scenario_label = scenario_prices(
                scenario, schedule, args.schedule_length
            )
            scenario_groups.append((scenario_label, prices, prices))

    fieldnames = [
        "scenario",
        "reward_mode",
        "agent",
        "run_id",
        "final_value",
        "total_reward",
        "max_drawdown",
        "volatility",
        "sharpe",
        "turnover",
        "action_hold_ratio",
        "action_buy_ratio",
        "action_sell_ratio",
        "executed_trade_ratio",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario_label, train_prices, eval_prices in scenario_groups:
            for reward_mode in reward_modes:
                for agent in agents:
                    if agent == "ppo":
                        for i in range(args.ppo_repeats):
                            if train_prices is eval_prices:
                                result = run_ppo_episode(
                                    train_prices,
                                    timesteps=args.timesteps,
                                    reward_mode=reward_mode,
                                    drawdown_coeff=args.drawdown_coeff,
                                    volatility_coeff=args.volatility_coeff,
                                    trade_penalty_coeff=args.trade_penalty_coeff,
                                    invalid_action_penalty=args.invalid_action_penalty,
                                    inactivity_penalty=args.inactivity_penalty,
                                    entropy_coef=args.entropy_coef,
                                    trade_size=args.trade_size,
                                    progress=args.ppo_progress,
                                    log_every=args.ppo_log_every,
                                    progress_label=f"PPO {reward_mode} {i+1}/{args.ppo_repeats}",
                                )
                            else:
                                result = run_ppo_train_eval(
                                    train_prices,
                                    eval_prices,
                                    timesteps=args.timesteps,
                                    reward_mode=reward_mode,
                                    drawdown_coeff=args.drawdown_coeff,
                                    volatility_coeff=args.volatility_coeff,
                                    trade_penalty_coeff=args.trade_penalty_coeff,
                                    invalid_action_penalty=args.invalid_action_penalty,
                                    inactivity_penalty=args.inactivity_penalty,
                                    entropy_coef=args.entropy_coef,
                                    trade_size=args.trade_size,
                                    progress=args.ppo_progress,
                                    log_every=args.ppo_log_every,
                                    progress_label=f"PPO {reward_mode} {i+1}/{args.ppo_repeats}",
                                )
                            metrics = result["metrics"]
                            base = {
                                "scenario": scenario_label,
                                "reward_mode": reward_mode,
                                "agent": agent,
                                "run_id": i,
                            }
                            write_row(writer, base, metrics)
                    else:
                        result = run_experiment(
                            eval_prices,
                            agent_type=agent,
                            n_episodes=args.episodes,
                            seed_start=args.seed_start,
                            reward_mode=reward_mode,
                            drawdown_coeff=args.drawdown_coeff,
                            volatility_coeff=args.volatility_coeff,
                            trade_penalty_coeff=args.trade_penalty_coeff,
                            invalid_action_penalty=args.invalid_action_penalty,
                            inactivity_penalty=args.inactivity_penalty,
                            trade_size=args.trade_size,
                        )
                        for i, episode in enumerate(result["episodes"]):
                            metrics = episode["metrics"]
                            base = {
                                "scenario": scenario_label,
                                "reward_mode": reward_mode,
                                "agent": agent,
                                "run_id": i,
                            }
                            write_row(writer, base, metrics)

    print(f"Wrote benchmark results to {out_path}")


if __name__ == "__main__":
    run_benchmark()
