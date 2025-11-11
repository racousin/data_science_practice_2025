import numpy as np
from typing import Optional

class Agent:
    """
    Q-Learning agent for FrozenLake-v1 8x8 (stochastic, is_slippery=True).
    - Entraîne dans __init__ avec décroissance epsilon/alpha raccourcie.
    - Politique greedy à l'évaluation (tie-breaker RIGHT > DOWN > LEFT > UP, évite l'inverse immédiat).
    """

    def __init__(self, env):
        self.env = env
        self.nS = int(getattr(env.observation_space, "n"))
        self.nA = int(getattr(env.action_space, "n"))

        # Q-table
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float32)

        # Hyperparams
        self.gamma = 0.99

        # Exploration / apprentissage - décays plus courts
        self.eps_start = 1.0
        self.eps_end   = 0.10
        self.eps_decay_episodes = 30000

        self.alpha_start = 1.0
        self.alpha_end   = 0.10
        self.alpha_decay_episodes = 30000

        # Entraînement plus court pour éviter le time limit
        self.max_steps_per_ep = 100
        self.train_episodes   = 35000

        # État pour la politique d'éval
        self._prev_action: Optional[int] = None

        self._train_q_learning()

        # Reset propre
        try:
            self.env.reset()
        except TypeError:
            _ = self.env.reset()
        self._prev_action = None

        # cache inverse actions (0=LEFT,1=DOWN,2=RIGHT,3=UP)
        self._inverse = {0: 2, 2: 0, 1: 3, 3: 1}

    @staticmethod
    def _lin(start, end, step, total):
        if total <= 0:
            return end
        if step >= total:
            return end
        # linéaire rapide
        return start + (end - start) * (step / float(total))

    def _train_q_learning(self):
        Q = self.Q  # alias local (un peu plus rapide)
        gamma = self.gamma
        env = self.env
        max_steps = self.max_steps_per_ep

        for ep in range(self.train_episodes):
            # reset (gymnasium renvoie (obs, info))
            out = env.reset()
            state = int(out[0] if isinstance(out, tuple) else out)

            eps   = self._lin(self.eps_start,   self.eps_end,   ep, self.eps_decay_episodes)
            alpha = self._lin(self.alpha_start, self.alpha_end, ep, self.alpha_decay_episodes)

            for _ in range(max_steps):
                # ε-greedy
                if np.random.random() < eps:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(Q[state]))

                step_out = env.step(action)
                # gymnasium: (obs, reward, terminated, truncated, info)
                next_state = int(step_out[0])
                reward     = float(step_out[1])
                terminated = bool(step_out[2])
                truncated  = bool(step_out[3])

                # Q-learning update
                best_next = float(np.max(Q[next_state]))
                td_target = reward + gamma * best_next * (0.0 if terminated else 1.0)
                Q[state, action] += alpha * (td_target - float(Q[state, action]))

                state = next_state
                if terminated or truncated:
                    break

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        """
        Évaluation rapide et stable (pas d’update en ligne).
        - Évite l'action inverse immédiate pour casser des cycles.
        - Tie-breaker déterministe: RIGHT > DOWN > LEFT > UP parmi les meilleures.
        """
        state = int(observation)
        if state == 0 or terminated or truncated:
            self._prev_action = None

        row = self.Q[state]
        m = np.max(row)
        # candidates des meilleures actions
        # ordre de préférence fixe pour éviter np.random.choice (plus rapide/déterministe)
        pref_order = (2, 1, 0, 3)  # RIGHT, DOWN, LEFT, UP

        # exclure l'inverse si possible
        candidates = [a for a in range(4) if row[a] == m]
        if self._prev_action is not None and len(candidates) > 1:
            inv = self._inverse[self._prev_action]
            if inv in candidates and len(candidates) > 1:
                candidates = [a for a in candidates if a != inv] or candidates

        # applique l'ordre de préférence
        for a in pref_order:
            if a in candidates:
                action = int(a)
                break

        self._prev_action = action
        return action
