import argparse
import csv
import os

from backend.simulations.run_simulation import run_ppo_episode
from experiments.market_scenarios import SCENARIOS, regime_schedule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PPO trained with raw vs risk-adjusted reward."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="regime_shift_short",
        help="Scenario name.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Training timesteps per run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of runs per reward mode.",
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
        help="Drawdown penalty coefficient.",
    )
    parser.add_argument(
        "--volatility-coeff",
        type=float,
        default=0.02,
        help="Volatility penalty coefficient.",
    )
    parser.add_argument(
        "--trade-penalty-coeff",
        type=float,
        default=0.0,
        help="Trade penalty coefficient.",
    )
    parser.add_argument(
        "--invalid-action-penalty",
        type=float,
        default=0.0,
        help="Penalty for invalid actions (e.g., sell with no holdings).",
    )
    parser.add_argument(
        "--inactivity-penalty",
        type=float,
        default=0.0,
        help="Penalty for not executing a trade in a step.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="PPO entropy coefficient (encourages exploration).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print PPO training progress.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Steps between PPO progress logs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/results/ppo_compare.csv",
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


def main():
    args = parse_args()
    schedule = [s.strip() for s in args.schedule.split(",") if s.strip()]
    prices, scenario_label = scenario_prices(
        args.scenario, schedule, args.schedule_length
    )

    ensure_parent_dir(args.out)
    fieldnames = [
        "scenario",
        "agent",
        "reward_mode",
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

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for mode in ["raw", "risk_adjusted"]:
            for i in range(args.repeats):
                label = f"PPO {mode} {i+1}/{args.repeats}"
                result = run_ppo_episode(
                    prices,
                    timesteps=args.timesteps,
                    reward_mode=mode,
                    drawdown_coeff=args.drawdown_coeff,
                    volatility_coeff=args.volatility_coeff,
                    trade_penalty_coeff=args.trade_penalty_coeff,
                    invalid_action_penalty=args.invalid_action_penalty,
                    inactivity_penalty=args.inactivity_penalty,
                    entropy_coef=args.entropy_coef,
                    progress=args.progress,
                    log_every=args.log_every,
                    progress_label=label,
                )
                metrics = result["metrics"]
                writer.writerow(
                    {
                        "scenario": scenario_label,
                        "agent": "ppo",
                        "reward_mode": mode,
                        "run_id": i,
                        **metrics,
                    }
                )

    print(f"Wrote PPO comparison results to {args.out}")


if __name__ == "__main__":
    main()
