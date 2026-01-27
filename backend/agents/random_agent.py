import random

class RandomAgent:
    """
    Baseline agent that selects actions uniformly at random.
    """

    def __init__(self, action_space=(0, 1, 2)):
        self.action_space = action_space

    def act(self, state):
        return random.choice(self.action_space)