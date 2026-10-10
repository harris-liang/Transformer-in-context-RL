"""Generate tasks and prompts following the paper's experimental setup."""

import numpy as np
import torch

from environments import MountainCarTask, TabularTask

PROTOCOL = "paper-v1"


class PaperTasks:
    def __init__(self, config, seed=None, feature_seed=None, value_seed=None):
        self.config = config
        seed = config["seed"] if seed is None else seed
        feature_seed = seed if feature_seed is None else feature_seed
        value_seed = feature_seed + 1 if value_seed is None else value_seed
        self.rng = np.random.RandomState(seed)
        self.features = np.random.default_rng(feature_seed)
        self.values = np.random.default_rng(value_seed)

    def draw(self, random_init=True):
        c, rng = self.config, self.rng
        if c["family"] == "mountaincar":
            task = MountainCarTask(self.features, rotation=c.get("feature_rotation", False))
        else:
            ns, na = c["nS"], c["nA"]
            phi = self.features.uniform(-1, 1, (ns, na, ns * na)).astype(np.float32)
            x = (self.values.uniform(-1, 1, (ns, ns)).astype(np.float32)
                 if c["algorithm"] == "ac" else None)
            p0 = rng.dirichlet(np.ones(ns)).astype(np.float32)
            p = np.empty((ns, na, ns), np.float32)
            for s in range(ns):
                for a in range(na):
                    p[s, a] = rng.dirichlet(np.ones(ns)).astype(np.float32)
            reward = rng.uniform(-1, 1, (na, ns)).astype(np.float32)
            task = TabularTask(c["algorithm"], p0, p, reward, phi, x,
                               c["gamma"], c["epsilon"])
            if task.algorithm == "sarsa" and random_init:
                # The original MDP constructor and prompt each draw w0.
                torch.rand((task.d, 1))
        task.reset(rng)  # Original MDP constructor samples an initial state.
        task.paper_theta = np.zeros(task.d + task.m, dtype=np.float32)
        return task


def feature(task, state, action):
    if task.nS is not None:
        return task.phi[state, action].copy()
    z = np.array([(state[0] - task.x_min) / (task.x_max - task.x_min),
                  (state[1] + task.v_max) / (2 * task.v_max)])
    psi = np.exp(-((task.centers[action] - z) ** 2).sum(1)
                 / (2 * task.sigmas[action] ** 2))
    psi = psi / np.linalg.norm(psi)
    if task.rotations[action] is not None:
        psi = task.rotations[action] @ psi
    psi = psi.astype(np.float32)
    phi = np.zeros(task.d, np.float32)
    phi[action * 12:(action + 1) * 12] = psi
    return phi


def behavior_probs(task, state, theta):
    logits = (task.phi[state] @ theta[:task.m].reshape(-1, 1)).reshape(-1)
    soft = np.exp(logits - np.max(logits))
    soft = soft / np.sum(soft)
    return ((1 - task.epsilon) * soft + task.epsilon / task.nA).astype(np.float32)


def sample_action(task, state, theta, rng):
    if task.algorithm == "ac":
        return int(rng.choice(task.nA, p=behavior_probs(task, state, theta)))
    if rng.rand() < task.epsilon:
        return int(rng.randint(task.nA))
    if task.nS is None:
        q = np.array([theta[a * 12:(a + 1) * 12] @
                      feature(task, state, a)[a * 12:(a + 1) * 12]
                      for a in range(task.nA)])
    else:
        q = (task.phi[state] @ theta.reshape(-1, 1)).reshape(-1)
    return int(rng.choice(np.flatnonzero(np.isclose(q, q.max()))))


class PaperPrompt:
    def __init__(self, task, config, rng, random_init=True):
        self.task, self.config, self.rng = task, config, rng
        self.theta = (2 * torch.rand((task.d + task.m, 1)) - 1 if random_init
                      else torch.zeros((task.d + task.m, 1))).flatten()
        self.t = 0
        self.state = task.reset(rng)
        if task.algorithm == "sarsa":
            task.paper_theta = self.theta.numpy().copy()
        self.action = sample_action(task, self.state, task.paper_theta, rng)
        self.H, self.teacher_H = self._window()
        # The original AC driver sets its actor after prompt.reset().
        task.paper_theta = self.theta.numpy().copy()

    def _window(self):
        task, c, rng = self.task, self.config, self.rng
        n, d, m = c["window"], task.d, task.m
        states, actions, rewards = [self.state], [self.action], []
        for _ in range(n):
            state, reward = task.step(states[-1], actions[-1], rng)
            self.t += 1
            action = (sample_action(task, state, task.paper_theta, rng)
                      if self.t < c["horizon"] else 0)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        self.state, self.action = states[-1], actions[-1]
        r = 2 * d + m + 1
        H = torch.zeros((3 * d + 2 * m + 2, n + 1), dtype=torch.float32)
        if task.algorithm == "sarsa":
            phi = torch.from_numpy(np.array([feature(task, s, a)
                                            for s, a in zip(states, actions)]))
            H[:d, :n], H[d:2 * d, :n] = phi[:-1].T, task.gamma * phi[1:].T
        else:
            H[:d, :n] = torch.from_numpy(np.stack([task.x[s] for s in states[:-1]], axis=1))
            H[d:2 * d, :n] = torch.from_numpy(task.gamma * np.stack([task.x[s] for s in states[1:]], axis=1))
            score = [task.gamma ** i * (task.phi[s, a] -
                     behavior_probs(task, s, task.paper_theta) @ task.phi[s])
                     for i, (s, a) in enumerate(zip(states[:-1], actions[:-1]))]
            H[2 * d + 1:r, :n] = torch.from_numpy(np.stack(score, axis=1))
        H[2 * d, :n] = torch.tensor(rewards, dtype=torch.float32)
        H[r, -1] = 1
        H[r + 1:, -1] = self.theta
        if not c["raw"]:
            return H, H
        raw = H.clone()
        raw[2 * d + 1:r, :n] = torch.from_numpy(np.stack(
            [task.phi[s, a] for s, a in zip(states[:-1], actions[:-1])], axis=1))
        return raw, H

    def step(self, theta):
        self.theta = theta.detach().clone().float().flatten()
        self.task.paper_theta = self.theta.numpy().copy()
        self.H, self.teacher_H = self._window()
        return self.H
