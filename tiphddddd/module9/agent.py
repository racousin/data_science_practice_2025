import numpy as np

class Agent:
    def __init__(self, env):
        self.env = env
        gamma = 0.99
        tol = 1e-10
        max_iter = 10000

        nS = env.observation_space.n
        nA = env.action_space.n
        P = getattr(getattr(env, "unwrapped", env), "P", None)

        if P is not None:
            V = np.zeros(nS, dtype=float)

            def q_of(s, a, Vvec):
                total = 0.0
                for (p, ns, r, done) in P[s][a]:
                    total += p * (r + (0.0 if done else gamma * Vvec[ns]))
                return total

            for _ in range(max_iter):
                delta = 0.0
                for s in range(nS):
                    qs = [q_of(s, a, V) for a in range(nA)]
                    best = max(qs)
                    delta = max(delta, abs(best - V[s]))
                    V[s] = best
                if delta < tol:
                    break

            policy = np.zeros(nS, dtype=int)
            for s in range(nS):
                qs = [q_of(s, a, V) for a in range(nA)]
                policy[s] = int(np.argmax(qs))
            self.policy = policy.tolist()
        else:
            # fallback：快速 Q-learning 预训练
            Q = np.zeros((nS, nA), dtype=float)
            eps, eps_end, eps_decay = 1.0, 0.05, 0.999
            alpha = 0.6

            for _ in range(8000):
                s, _ = env.reset()
                done = False
                while not done:
                    if np.random.rand() < eps:
                        a = env.action_space.sample()
                    else:
                        a = int(np.argmax(Q[s]))
                    s2, r, terminated, truncated, _ = env.step(a)
                    done = terminated or truncated
                    Q[s, a] += alpha * (r + (0.0 if done else gamma * np.max(Q[s2])) - Q[s, a])
                    s = s2
                eps = max(eps_end, eps * eps_decay)

            self.policy = np.argmax(Q, axis=1).tolist()

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        return int(self.policy[observation])
