import argparse
import csv
import os

from backend.simulations.run_simulation import (
    run_experiment,
    run_ppo_episode,
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
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in scenarios:
            prices, scenario_label = scenario_prices(
                scenario, schedule, args.schedule_length
            )

            for reward_mode in reward_modes:
                for agent in agents:
                    if agent == "ppo":
                        for i in range(args.ppo_repeats):
                            result = run_ppo_episode(
                                prices,
                                timesteps=args.timesteps,
                                reward_mode=reward_mode,
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
                            prices,
                            agent_type=agent,
                            n_episodes=args.episodes,
                            seed_start=args.seed_start,
                            reward_mode=reward_mode,
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
