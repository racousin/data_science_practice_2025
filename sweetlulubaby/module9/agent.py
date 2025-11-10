# agent.py
import numpy as np

class Agent:
    """
    FrozenLake-v1 (8x8, is_slippery=True, success_rate=1/3) agent.
    不读取 env.P；用官方默认 8x8 地图与滑动概率自建转移，初始化时用 Value Iteration 求策略。
    评测时 choose_action 直接查表，单步 O(1)。
    """

    def __init__(self, env):
        # 官方默认 8x8 地图（Gymnasium 文档）
        # 0:LEFT, 1:DOWN, 2:RIGHT, 3:UP
        desc = [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
        self.nrow, self.ncol = len(desc), len(desc[0])
        self.nS, self.nA = self.nrow * self.ncol, 4
        self.desc = np.asarray([[c for c in row] for row in desc], dtype="U1")

        # 滑动参数（与文档一致）
        success_rate = 1.0 / 3.0
        slip_side = (1.0 - success_rate) / 2.0

        # 方向与“左/右侧向”关系
        LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
        left_of = {LEFT: UP, DOWN: LEFT, RIGHT: DOWN, UP: RIGHT}
        right_of = {LEFT: DOWN, DOWN: RIGHT, RIGHT: UP, UP: LEFT}
        moves = {
            LEFT:  (0, -1),
            DOWN:  (1,  0),
            RIGHT: (0,  1),
            UP:    (-1, 0),
        }

        def to_s(r, c):  # (row, col) -> state id
            return r * self.ncol + c

        def step_from(r, c, a):
            dr, dc = moves[a]
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= self.nrow or nc < 0 or nc >= self.ncol:
                nr, nc = r, c  # 撞墙留在原地
            tile = self.desc[nr, nc]
            if tile == "G":
                return to_s(nr, nc), 1.0, True
            if tile == "H":
                return to_s(nr, nc), 0.0, True
            return to_s(nr, nc), 0.0, False

        # 构建转移：P[s][a] = list of (prob, ns, r, done)
        P = [[[] for _ in range(self.nA)] for __ in range(self.nS)]
        for r in range(self.nrow):
            for c in range(self.ncol):
                s = to_s(r, c)
                tile = self.desc[r, c]
                if tile in ("H", "G"):  # 终止状态：价值为0，策略任意
                    for a in range(self.nA):
                        P[s][a].append((1.0, s, 0.0, True))
                    continue
                for a in range(self.nA):
                    for prob, aa in (
                        (success_rate, a),
                        (slip_side, left_of[a]),
                        (slip_side, right_of[a]),
                    ):
                        ns, rwd, done = step_from(r, c, aa)
                        P[s][a].append((prob, ns, rwd, done))

        # -------- Value Iteration ----------
        gamma, theta, max_iters = 0.99, 1e-8, 10000
        V = np.zeros(self.nS, dtype=np.float64)

        for _ in range(max_iters):
            delta = 0.0
            for s in range(self.nS):
                # 终止格子的价值恒为0
                if self.desc[s // self.ncol, s % self.ncol] in ("H", "G"):
                    continue
                q_best = -1e9
                for a in range(self.nA):
                    q = 0.0
                    for prob, ns, rwd, done in P[s][a]:
                        q += prob * (rwd + (0.0 if done else gamma * V[ns]))
                    if q > q_best:
                        q_best = q
                delta = max(delta, abs(q_best - V[s]))
                V[s] = q_best
            if delta < theta:
                break

        # 提取确定性策略；平局打破偏好：RIGHT > DOWN > LEFT > UP
        pref = [RIGHT, DOWN, LEFT, UP]
        policy = np.zeros(self.nS, dtype=np.int64)
        for s in range(self.nS):
            q = np.zeros(self.nA)
            for a in range(self.nA):
                for prob, ns, rwd, done in P[s][a]:
                    q[a] += prob * (rwd + (0.0 if done else gamma * V[ns]))
            best_val = q.max()
            best_as = {a for a in range(self.nA) if q[a] == best_val}
            for a in pref:
                if a in best_as:
                    policy[s] = a
                    break

        self.policy = policy

    def choose_action(
        self,
        observation,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info=None,
    ):
        return int(self.policy[int(observation)])

