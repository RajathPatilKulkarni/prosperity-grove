import numpy as np

from backend.env.market_env import MarketEnvironment
from backend.agents.random_agent import RandomAgent
from backend.agents.rule_based_agent import RuleBasedAgent
from backend.agents.ppo_agent import train_ppo
from backend.env.rl_env import RLMarketEnv
from experiments.market_scenarios import SCENARIOS, regime_schedule
from experiments.experiment_config import ExperimentConfig
from backend.simulations.metrics import (
    compute_returns,
    max_drawdown,
    volatility,
    sharpe_ratio,
    turnover,
)


# JSON-safe serializer (CRITICAL)
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# Single episode (random / rule)
def run_episode(
    prices,
    agent_type="random",
    seed=None,
    reward_mode="raw",
    drawdown_coeff=0.01,
    volatility_coeff=0.01,
    trade_penalty_coeff=0.0,
    invalid_action_penalty=0.0,
    inactivity_penalty=0.0,
):
    env = MarketEnvironment(
        prices,
        reward_mode=reward_mode,
        drawdown_coeff=drawdown_coeff,
        volatility_coeff=volatility_coeff,
        trade_penalty_coeff=trade_penalty_coeff,
        invalid_action_penalty=invalid_action_penalty,
        inactivity_penalty=inactivity_penalty,
    )

    if agent_type == "random":
        agent = RandomAgent(seed=seed)
    elif agent_type == "rule_based":
        agent = RuleBasedAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    state = env.reset()
    done = False

    total_reward = 0.0
    history = []
    actions = []

    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)

        total_reward += reward
        history.append(state["portfolio_value"])
        actions.append(action)

    returns = compute_returns(history)
    total_actions = max(1, len(actions))
    action_counts = {
        "action_hold_ratio": actions.count(0) / total_actions,
        "action_buy_ratio": actions.count(1) / total_actions,
        "action_sell_ratio": actions.count(2) / total_actions,
        "executed_trade_ratio": env.executed_trades / total_actions,
    }
    metrics = {
        "final_value": history[-1],
        "total_reward": total_reward,
        "max_drawdown": max_drawdown(history),
        "volatility": volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "turnover": turnover(env.executed_trades, len(actions)),
    }
    metrics.update(action_counts)

    return make_json_serializable({
        "metrics": metrics,
        "trajectory": history,
        "actions": actions,
    })


# Multi-episode experiment
def run_experiment(
    prices,
    agent_type="random",
    n_episodes=10,
    seed_start=0,
    reward_mode="raw",
    drawdown_coeff=0.01,
    volatility_coeff=0.01,
    trade_penalty_coeff=0.0,
    invalid_action_penalty=0.0,
    inactivity_penalty=0.0,
):
    results = []

    for i in range(n_episodes):
        metrics = run_episode(
            prices,
            agent_type=agent_type,
            seed=seed_start + i,
            reward_mode=reward_mode,
            drawdown_coeff=drawdown_coeff,
            volatility_coeff=volatility_coeff,
            trade_penalty_coeff=trade_penalty_coeff,
            invalid_action_penalty=invalid_action_penalty,
            inactivity_penalty=inactivity_penalty,
        )
        results.append(metrics)

    final_values = [r["metrics"]["final_value"] for r in results]
    rewards = [r["metrics"]["total_reward"] for r in results]
    drawdowns = [r["metrics"]["max_drawdown"] for r in results]
    vols = [r["metrics"]["volatility"] for r in results]
    sharpes = [r["metrics"]["sharpe"] for r in results]
    turnovers = [r["metrics"]["turnover"] for r in results]

    summary = {
        "agent": agent_type,
        "episodes": n_episodes,
        "final_value_mean": sum(final_values) / len(final_values),
        "reward_mean": sum(rewards) / len(rewards),
        "best_final_value": max(final_values),
        "worst_final_value": min(final_values),
        "max_drawdown_mean": sum(drawdowns) / len(drawdowns),
        "volatility_mean": sum(vols) / len(vols),
        "sharpe_mean": sum(sharpes) / len(sharpes),
        "turnover_mean": sum(turnovers) / len(turnovers),
        "reward_mode": reward_mode,
    }

    return make_json_serializable({
        "summary": summary,
        "episodes": results,
    })


# PPO episode
def run_ppo_episode(
    prices,
    timesteps=10_000,
    reward_mode="raw",
    drawdown_coeff=0.01,
    volatility_coeff=0.01,
    trade_penalty_coeff=0.0,
    invalid_action_penalty=0.0,
    inactivity_penalty=0.0,
    entropy_coef=0.0,
    progress=False,
    log_every=5000,
    progress_label="PPO",
):
    model = train_ppo(
        prices,
        timesteps=timesteps,
        reward_mode=reward_mode,
        drawdown_coeff=drawdown_coeff,
        volatility_coeff=volatility_coeff,
        trade_penalty_coeff=trade_penalty_coeff,
        invalid_action_penalty=invalid_action_penalty,
        inactivity_penalty=inactivity_penalty,
        entropy_coef=entropy_coef,
        progress=progress,
        log_every=log_every,
        progress_label=progress_label,
    )
    env = RLMarketEnv(
        prices,
        reward_mode=reward_mode,
        drawdown_coeff=drawdown_coeff,
        volatility_coeff=volatility_coeff,
        trade_penalty_coeff=trade_penalty_coeff,
        invalid_action_penalty=invalid_action_penalty,
        inactivity_penalty=inactivity_penalty,
    )

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    history = []
    actions = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        total_reward += reward
        history.append(obs[3])  # portfolio value
        actions.append(int(action))

    returns = compute_returns(history)
    total_actions = max(1, len(actions))
    action_counts = {
        "action_hold_ratio": actions.count(0) / total_actions,
        "action_buy_ratio": actions.count(1) / total_actions,
        "action_sell_ratio": actions.count(2) / total_actions,
        "executed_trade_ratio": env.env.executed_trades / total_actions,
    }
    metrics = {
        "final_value": history[-1],
        "total_reward": total_reward,
        "max_drawdown": max_drawdown(history),
        "volatility": volatility(returns),
        "sharpe": sharpe_ratio(returns),
        "turnover": turnover(env.env.executed_trades, len(actions)),
    }
    metrics.update(action_counts)

    return make_json_serializable({
        "metrics": metrics,
        "trajectory": history,
        "actions": actions,
    })


# Config-driven dispatcher
def run_configured_experiment(config: ExperimentConfig):
    if config.schedule:
        prices = regime_schedule(
            config.schedule, length=config.schedule_length
        )
    else:
        if config.scenario not in SCENARIOS:
            raise ValueError(f"Unknown market scenario: {config.scenario}")
        prices = SCENARIOS[config.scenario]()

    if config.agent_type == "ppo":
        return run_ppo_episode(
            prices,
            timesteps=config.timesteps,
            reward_mode=config.reward_mode,
            drawdown_coeff=config.drawdown_coeff,
            volatility_coeff=config.volatility_coeff,
            trade_penalty_coeff=config.trade_penalty_coeff,
            invalid_action_penalty=config.invalid_action_penalty,
            inactivity_penalty=config.inactivity_penalty,
        )

    return run_experiment(
        prices,
        agent_type=config.agent_type,
        n_episodes=config.episodes,
        reward_mode=config.reward_mode,
        drawdown_coeff=config.drawdown_coeff,
        volatility_coeff=config.volatility_coeff,
        trade_penalty_coeff=config.trade_penalty_coeff,
        invalid_action_penalty=config.invalid_action_penalty,
        inactivity_penalty=config.inactivity_penalty,
    )


# Local test runner
if __name__ == "__main__":
    configs = [
        ExperimentConfig("bull", "random", episodes=20),
        ExperimentConfig("bull", "rule_based", episodes=1),
        ExperimentConfig("bull", "ppo", timesteps=15_000),

        ExperimentConfig("bear", "random", episodes=20),
        ExperimentConfig("bear", "ppo", timesteps=15_000),

        ExperimentConfig("volatile", "ppo", timesteps=15_000),
        ExperimentConfig(
            "regime_shift_short",
            "ppo",
            timesteps=15_000,
            reward_mode="risk_adjusted",
        ),
    ]

    for cfg in configs:
        print("\nRunning:", cfg)
        result = run_configured_experiment(cfg)
        print(result)
