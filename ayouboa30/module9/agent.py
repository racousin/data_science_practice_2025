import random
import numpy as np
from collections import defaultdict
import gymnasium as gym

class Agent:
    def __init__(self, env: gym.Env, q_table: dict = None):
        self.env = env
        
        if q_table is None:
            self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        else:
            self.Q = q_table
            
    def choose_action(self, observation: tuple, reward=0.0, terminated=False, truncated=False, info=None):
        s = observation

        row = self.Q[s]
        
        # Si la ligne est entièrement à zéro ou l'état n'a pas été exploré
        if np.all(row == 0):
             return self.env.action_space.sample()

        m = np.max(row)
        
        # On utilise une petite tolérance pour les comparaisons de flottants
        meilleures = [a for a, q in enumerate(row) if abs(q - m) < 1e-6]
        
        if meilleures:
            return random.choice(meilleures)
        else:
            return self.env.action_space.sample()
