from backend.env.market_env import MarketEnvironment
from backend.agents.random_agent import RandomAgent
import statistics


def run_episode(prices, seed=None):
    env = MarketEnvironment(prices)
    agent = RandomAgent(seed=seed)

    state = env.reset()
    done = False

    total_reward = 0.0
    history = []

    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)

        total_reward += reward
        history.append(state["portfolio_value"])

    episode_metrics = {
        "steps": len(history),
        "initial_value": history[0],
        "final_value": history[-1],
        "total_reward": total_reward,
        "max_value": max(history),
        "min_value": min(history),
    }

    return episode_metrics


def run_experiment(prices, n_episodes=10, seed_start=0):
    results = []

    for i in range(n_episodes):
        metrics = run_episode(prices, seed=seed_start + i)
        results.append(metrics)

    final_values = [r["final_value"] for r in results]
    rewards = [r["total_reward"] for r in results]

    summary = {
        "episodes": n_episodes,
        "final_value_mean": statistics.mean(final_values),
        "final_value_std": statistics.pstdev(final_values),
        "reward_mean": statistics.mean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "best_final_value": max(final_values),
        "worst_final_value": min(final_values),
    }

    return {
        "summary": summary,
        "episodes": results,
    }


if __name__ == "__main__":
    prices = [100, 102, 101, 105, 107, 106, 108]

    experiment = run_experiment(prices, n_episodes=20)

    print("Experiment summary:")
    for k, v in experiment["summary"].items():
        print(f"{k}: {v}")