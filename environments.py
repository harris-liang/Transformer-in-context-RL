"""Define the MDP environments and feature maps used in the experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def action_probs(task, state, theta, epsilon=None):
    """Epsilon-greedy SARSA or epsilon-mixed softmax AC probabilities.

    SARSA ties are split uniformly, using the original experiments' ``isclose``
    tolerance. For AC, ``epsilon=0`` gives the underlying softmax policy.
    ``theta`` contains w for SARSA and concatenated (lambda, w) for AC.
    """
    epsilon = task.epsilon if epsilon is None else epsilon
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    features = task.policy_features(state).astype(np.float64)
    values = features @ (theta[:task.m] if task.algorithm == "ac" else theta)
    if task.algorithm == "sarsa":
        best = np.isclose(values, values.max())
        probs = best.astype(np.float64) / best.sum()
    elif task.algorithm == "ac":
        probs = np.exp(values - values.max())
        probs /= probs.sum()
    else:
        raise ValueError(f"Unknown algorithm: {task.algorithm}")
    probs = (1 - epsilon) * probs + epsilon / task.nA
    return probs / probs.sum()


@dataclass
class TabularTask:
    """Random finite MDP; P[s,a,s'] and rewards R[a,s']."""

    algorithm: str
    P0: np.ndarray
    P: np.ndarray
    R: np.ndarray
    phi: np.ndarray
    x: np.ndarray | None
    gamma: float = 0.5
    epsilon: float = 0.1

    @property
    def nS(self):
        return self.P.shape[0]

    @property
    def nA(self):
        return self.P.shape[1]

    @property
    def d(self):
        return self.phi.shape[-1] if self.algorithm == "sarsa" else self.x.shape[-1]

    @property
    def m(self):
        return 0 if self.algorithm == "sarsa" else self.phi.shape[-1]

    def policy_features(self, state):
        return self.phi[state]

    def value_features(self, state):
        if self.x is None:
            raise ValueError("SARSA uses state-action features, not value features")
        return self.x[state]

    def reset(self, rng):
        return int(rng.choice(self.nS, p=self.P0))

    def step(self, state, action, rng):
        next_state = int(rng.choice(self.nS, p=self.P[state, action]))
        return next_state, float(self.R[action, next_state])


def random_task(rng, algorithm="sarsa", nS=9, nA=4, gamma=0.5, epsilon=0.1):
    """Draw a fresh MDP and independent uniform feature tables on every call."""
    if algorithm not in ("sarsa", "ac"):
        raise ValueError("algorithm must be 'sarsa' or 'ac'")
    P0 = rng.dirichlet(np.ones(nS))
    P = rng.dirichlet(np.ones(nS), size=(nS, nA))
    # Keep probabilities in float64: float32 rounding can break rng.choice.
    P0 /= P0.sum()
    P /= P.sum(axis=-1, keepdims=True)
    R = rng.uniform(-1, 1, size=(nA, nS))
    phi = rng.uniform(-1, 1, size=(nS, nA, nS * nA)).astype(np.float32)
    x = rng.uniform(-1, 1, size=(nS, nS)).astype(np.float32) if algorithm == "ac" else None
    return TabularTask(algorithm, P0, P, R, phi, x, gamma, epsilon)


class MountainCarTask:
    """Continuous MountainCar with independent 4x3 RBF bases per action.

    The 36-dimensional feature is e_a tensor psi_a(s), with unit-norm psi_a.
    Goal transitions pay the goal reward and restart from the task's P0.
    ``nS=None`` denotes the continuous state space.
    """

    algorithm = "sarsa"
    nS, nA, d, m = None, 3, 36, 0
    gamma, epsilon = 0.99, 0.1
    x_min, x_max, v_max = -1.2, 0.6, 0.07

    def __init__(self, rng, rotation=False):
        base = np.array([((j + 0.5) / 4, (k + 0.5) / 3)
                         for j in range(4) for k in range(3)])
        self.sigmas = np.empty(3)
        self.centers = np.empty((3, 12, 2))
        self.rotations = []
        for action in range(self.nA):
            self.sigmas[action] = rng.uniform(0.15, 0.35)
            self.centers[action] = base + rng.uniform(-0.08, 0.08, size=base.shape)
            self.rotations.append(np.linalg.qr(rng.standard_normal((12, 12)))[0]
                                  if rotation else None)
        self.F, self.G = rng.uniform(0.0015, 0.003, size=2)
        self.x_goal = rng.uniform(0.4, 0.55)
        center = rng.uniform(-0.9, self.x_goal - 0.1)
        half_width = rng.uniform(0.1, 0.5)
        self.x0_lo = max(center - half_width, self.x_min)
        self.x0_hi = min(center + half_width, self.x_goal - 0.02)
        self.step_cost = rng.uniform(-0.1, -0.02)
        self.goal_reward = rng.uniform(1, 2)

    def policy_features(self, state):
        x, v = state
        z = np.array([(x - self.x_min) / (self.x_max - self.x_min),
                      (v + self.v_max) / (2 * self.v_max)])
        squared_distance = ((self.centers - z) ** 2).sum(axis=-1)
        psi = np.exp(-squared_distance / (2 * self.sigmas[:, None] ** 2))
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
        phi = np.zeros((self.nA, self.d), dtype=np.float32)
        for action in range(self.nA):
            value = psi[action] if self.rotations[action] is None else self.rotations[action] @ psi[action]
            phi[action, 12 * action:12 * (action + 1)] = value
        return phi

    def reset(self, rng):
        return float(rng.uniform(self.x0_lo, self.x0_hi)), float(rng.uniform(-0.02, 0.02))

    def step(self, state, action, rng):
        x, v = state
        v = float(np.clip(v + (action - 1) * self.F - self.G * np.cos(3 * x),
                          -self.v_max, self.v_max))
        x = float(np.clip(x + v, self.x_min, self.x_max))
        if x <= self.x_min:
            v = 0.0
        if x >= self.x_goal:
            return self.reset(rng), float(self.goal_reward)
        return (x, v), float(self.step_cost)


class CliffWalkTask:
    """Fixed 3x3 grid, row-major states; actions up, down, left, right.

    ``move`` includes slip but returns the landed cell before resets. ``step``
    returns the continuing state after a cliff or goal transition resets to 0.
    """

    algorithm = "sarsa"
    nS, nA, d, m = 9, 4, 36, 0
    gamma, epsilon = 1.0, 0.1
    start, cliff, goal = 0, 4, 8
    slip, step_cost, cliff_penalty, goal_reward = 0.1, -0.125, -1.0, 1.0
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, rng):
        self.phi = rng.uniform(-1, 1, size=(self.nS, self.nA, self.d)).astype(np.float32)

    def policy_features(self, state):
        return self.phi[state]

    def reset(self, rng):
        return self.start

    def _land(self, state, action):
        row, col = divmod(state, 3)
        dr, dc = self.moves[action]
        return 3 * min(max(row + dr, 0), 2) + min(max(col + dc, 0), 2)

    def move(self, state, action, rng):
        if rng.random() < self.slip:
            action = int(rng.integers(self.nA))
        return self._land(state, action)

    def step(self, state, action, rng):
        landed = self.move(state, action, rng)
        if landed == self.cliff:
            return self.start, self.cliff_penalty
        if landed == self.goal:
            return self.start, self.goal_reward
        return landed, self.step_cost

    def hitting_time_transitions(self):
        """Return P[s,a,s'] with an absorbing goal and cliff resets to start.

        For optimal hitting times iterate t[s] = 1 + min_a P[s,a] @ t,
        then set t[goal] = 0. A uniform policy uses the mean over actions.
        This kernel is for evaluation; training also resets after the goal.
        """
        P = np.zeros((self.nS, self.nA, self.nS), dtype=np.float64)
        for state in range(self.nS):
            for action in range(self.nA):
                if state == self.goal:
                    P[state, action, self.goal] = 1.0
                    continue
                for executed in range(self.nA):
                    probability = self.slip / self.nA + (1 - self.slip) * (executed == action)
                    landed = self._land(state, executed)
                    next_state = self.start if landed == self.cliff else landed
                    P[state, action, next_state] += probability
        return P
