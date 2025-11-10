import gymnasium as gym
import numpy as np
import random
import time
from collections import deque
import matplotlib.pyplot as plt

# --- DÉFINITION DE LA CLASSE AGENT (INCLUS POUR AUTONOMIE) ---
class Agent:
    # Paramètres Q-Learning par défaut (Attributs de Classe)
    DEFAULT_LR = 0.1
    DEFAULT_GAMMA = 0.99
    DEFAULT_INITIAL_EPSILON = 1.0
    DEFAULT_EPSILON_DECAY = 0.00005
    DEFAULT_FINAL_EPSILON = 0.1

    # Signature mise à jour pour respecter la contrainte
    def __init__(self, env, player_name=None):
        self.env = env
        self.player_name = player_name  # Optionnel pour les systèmes multi-agents

        # Initialisation des paramètres de l'instance
        self.lr = self.DEFAULT_LR
        self.gamma = self.DEFAULT_GAMMA
        self.epsilon = self.DEFAULT_INITIAL_EPSILON
        self.epsilon_decay = self.DEFAULT_EPSILON_DECAY
        self.final_epsilon = self.DEFAULT_FINAL_EPSILON
        
        # Initialisation de la Q-table (états x actions).
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        # Stratégie epsilon-gloutonne:
        if random.random() < self.epsilon:
            action = self.env.action_space.sample() # Exploration
        else:
            action = np.argmax(self.Q[observation]) # Exploitation
            
        return action

    def learn(self, obs, action, reward, next_obs, terminated):
        max_future_q = np.max(self.Q[next_obs]) if not terminated else 0
        
        td_target = reward + self.gamma * max_future_q
        td_error = td_target - self.Q[obs, action]
        
        self.Q[obs, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        return self.epsilon