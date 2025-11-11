import numpy as np

class Agent:
    def __init__(
        self,
        env,
    ):
        self.env = env
        self.Q = np.load("Q.npy")

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info = None):
        return np.argmax(self.Q[observation])