import random

class RandomAgent:
    """
    Baseline agent that selects actions uniformly at random.
    """

    def __init__(self, action_space=(0, 1, 2), seed=None):
        self.action_space = action_space
        if seed is not None:
            random.seed(seed)

    def act(self, state):
        return random.choice(self.action_space)