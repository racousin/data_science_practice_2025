class Agent:
    def __init__(self, env):
        self.env = env
        # Policy indices correspond to observations 0..63 on FrozenLake 8x8.
        self.policy = [
            3, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 2, 2, 1,
            3, 3, 0, 0, 2, 3, 2, 1,
            3, 3, 3, 1, 0, 0, 2, 2,
            0, 3, 0, 0, 2, 1, 3, 2,
            0, 0, 0, 1, 3, 0, 0, 2,
            0, 0, 1, 0, 0, 0, 0, 2,
            0, 1, 0, 0, 1, 2, 1, 0,
        ]

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        return self.policy[int(observation)]