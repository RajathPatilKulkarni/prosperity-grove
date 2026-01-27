from backend.env.market_env import MarketEnvironment
from backend.agents.random_agent import RandomAgent


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

    return {
        "metrics": episode_metrics,
        "trajectory": history,
    }


if __name__ == "__main__":
    prices = [100, 102, 101, 105, 107, 106, 108]

    result = run_episode(prices, seed=42)

    print("Episode metrics:")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")