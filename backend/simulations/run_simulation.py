from backend.env.market_env import MarketEnvironment
from backend.agents.random_agent import RandomAgent
from backend.agents.rule_based_agent import RuleBasedAgent
import statistics



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

    return {
        "final_value": history[-1],
        "total_reward": total_reward,
        "trajectory": history,
    }


def run_experiment(prices, agent_type="random", n_episodes=10, seed_start=0):
    results = []

    for i in range(n_episodes):
        metrics = run_episode(
            prices,
            agent_type=agent_type,
            seed=seed_start + i
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

    return {
        "summary": summary,
        "episodes": results,
    }


if __name__ == "__main__":
    prices = [100, 102, 101, 105, 107, 106, 108]

    random_exp = run_experiment(prices, agent_type="random", n_episodes=20)
    rule_exp = run_experiment(prices, agent_type="rule_based", n_episodes=1)

    print("Random agent summary:", random_exp["summary"])
    print("Rule-based agent summary:", rule_exp["summary"])