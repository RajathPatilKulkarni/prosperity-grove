import numpy as np

from backend.env.market_env import MarketEnvironment
from backend.agents.random_agent import RandomAgent
from backend.agents.rule_based_agent import RuleBasedAgent
from backend.agents.ppo_agent import train_ppo
from backend.env.rl_env import RLMarketEnv
from experiments.market_scenarios import SCENARIOS
from experiments.experiment_config import ExperimentConfig


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
def run_episode(prices, agent_type="random", seed=None):
    env = MarketEnvironment(prices)

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

    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)

        total_reward += reward
        history.append(state["portfolio_value"])

    return make_json_serializable({
        "final_value": history[-1],
        "total_reward": total_reward,
        "trajectory": history,
    })


# Multi-episode experiment
def run_experiment(prices, agent_type="random", n_episodes=10, seed_start=0):
    results = []

    for i in range(n_episodes):
        metrics = run_episode(
            prices,
            agent_type=agent_type,
            seed=seed_start + i,
        )
        results.append(metrics)

    final_values = [r["final_value"] for r in results]
    rewards = [r["total_reward"] for r in results]

    summary = {
        "agent": agent_type,
        "episodes": n_episodes,
        "final_value_mean": sum(final_values) / len(final_values),
        "reward_mean": sum(rewards) / len(rewards),
        "best_final_value": max(final_values),
        "worst_final_value": min(final_values),
    }

    return make_json_serializable({
        "summary": summary,
        "episodes": results,
    })


# PPO episode
def run_ppo_episode(prices, timesteps=10_000):
    model = train_ppo(prices, timesteps=timesteps)
    env = RLMarketEnv(prices)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    history = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        total_reward += reward
        history.append(obs[3])  # portfolio value

    return make_json_serializable({
        "final_value": history[-1],
        "total_reward": total_reward,
        "trajectory": history,
    })


# Config-driven dispatcher
def run_configured_experiment(config: ExperimentConfig):
    if config.scenario not in SCENARIOS:
        raise ValueError(f"Unknown market scenario: {config.scenario}")

    prices = SCENARIOS[config.scenario]()

    if config.agent_type == "ppo":
        return run_ppo_episode(prices, timesteps=config.timesteps)

    return run_experiment(
        prices,
        agent_type=config.agent_type,
        n_episodes=config.episodes,
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
    ]

    for cfg in configs:
        print("\nRunning:", cfg)
        result = run_configured_experiment(cfg)
        print(result)