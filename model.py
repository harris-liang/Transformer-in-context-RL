"""Define prompts, analytical teachers, and attention layer."""

import numpy as np
import torch
from torch import nn

from environments import action_probs


def collect(task, theta, n, rng, state=None, raw=False):
    """Collect one window under theta; return input, engineered prompt, state.

    For AC, theta = [lambda, w]. The score is that of the paper's pure
    softmax policy; epsilon exploration affects sampling only. Every window
    starts by sampling an action under its current parameter.
    """
    if n < 1:
        raise ValueError("A context window needs at least one transition")
    if raw and task.algorithm != "ac":
        raise ValueError("The raw-score-row ablation is defined only for AC")
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    d, m = task.d, task.m
    if theta.size != d + m or not np.isfinite(theta).all():
        raise ValueError("Invalid policy/value parameter")
    state = task.reset(rng) if state is None else state
    r = 2 * d + m + 1
    H = np.zeros((3 * d + 2 * m + 2, n + 1), dtype=np.float32)
    H[r, -1] = 1
    H[r + 1:, -1] = theta
    raw_features = []
    action = int(rng.choice(task.nA, p=action_probs(task, state, theta)))
    for i in range(n):
        next_state, reward = task.step(state, action, rng)
        next_action = int(rng.choice(task.nA, p=action_probs(task, next_state, theta)))
        if task.algorithm == "sarsa":
            H[:d, i] = task.policy_features(state)[action]
            H[d:2 * d, i] = task.gamma * task.policy_features(next_state)[next_action]
        else:
            phi = task.policy_features(state)
            soft = action_probs(task, state, theta, epsilon=0.0)
            H[:d, i] = task.value_features(state)
            H[d:2 * d, i] = task.gamma * task.value_features(next_state)
            H[2 * d + 1:r, i] = task.gamma ** i * (phi[action] - soft @ phi)
            raw_features.append(phi[action])
        H[2 * d, i] = reward
        state, action = next_state, next_action
    teacher_H = torch.from_numpy(H)
    if not raw:
        return teacher_H, teacher_H, state
    input_H = teacher_H.clone()
    input_H[2 * d + 1:r, :n] = torch.as_tensor(np.asarray(raw_features).T)
    return input_H, teacher_H, state


def teacher_update(H, algorithm, d, m=0, alpha=0.2, beta=0.8):
    """Batch SARSA or AC update; discounted next features are already in H."""
    n = H.shape[1] - 1
    x, xp, rewards = H[:d, :n], H[d:2 * d, :n], H[2 * d:2 * d + 1, :n]
    w = H[-d:, -1:]
    delta = rewards + (w.T @ xp) - (w.T @ x)
    if algorithm == "sarsa":
        return (w + (alpha / n) * (x @ delta.T)).flatten()
    if algorithm != "ac":
        raise ValueError("algorithm must be sarsa or ac")
    lam = H[-d - m:-d, -1:]
    g = H[2 * d + 1:2 * d + 1 + m, :n]
    return torch.cat((lam + (alpha / n) * (g @ delta.T),
                      w + (beta / n) * (x @ delta.T))).flatten()


def theoretical_matrices(algorithm, d, m=0, alpha=0.2, beta=0.8):
    """Choose zero for all unconstrained entries in the paper's P*, V*."""
    D, r = 3 * d + 2 * m + 2, 2 * d + m + 1
    P, V = torch.zeros(D, D), torch.zeros(D, D)
    P[:d, -d:] = -torch.eye(d)
    P[d:2 * d, -d:] = torch.eye(d)
    P[2 * d, r] = 1
    V[-d:, :d] = (alpha if algorithm == "sarsa" else beta) * torch.eye(d)
    if algorithm == "ac":
        V[r + 1:r + 1 + m, 2 * d + 1:r] = alpha * torch.eye(m)
    return P, V


class Transformer(nn.Module):
    """One full residual attention layer; optional 1/2-head score ablations.

    Only initialization imposes the paper's zeros. There is no structural
    projection during training. Softmax normalizes over keys (matrix rows).
    """

    def __init__(self, algorithm, d, m=0, activation="linear", heads=1):
        super().__init__()
        if algorithm not in {"sarsa", "ac"} or d < 1 or heads < 1:
            raise ValueError("Invalid transformer dimensions or algorithm")
        if (algorithm == "sarsa" and m != 0) or (algorithm == "ac" and m < 1):
            raise ValueError("AC needs actor features; SARSA uses m=0")
        if activation not in {"linear", "relu", "softmax"}:
            raise ValueError("Unknown attention activation")
        self.algorithm, self.d, self.m = algorithm, d, m
        self.activation = activation
        self.D, self.r = 3 * d + 2 * m + 2, 2 * d + m + 1
        self.P, self.V = nn.ParameterList(), nn.ParameterList()
        for _ in range(heads):
            P, V = torch.empty(self.D, self.D), torch.empty(self.D, self.D)
            nn.init.xavier_normal_(P, gain=0.1)
            nn.init.xavier_normal_(V, gain=0.1)
            P[:, :self.r] = 0
            P[self.r:, self.r:] = 0
            V[:self.r] = 0
            V[self.r:, self.r:] = 0
            self.P.append(nn.Parameter(P))
            self.V.append(nn.Parameter(V))

    def forward(self, H):
        n = H.shape[1] - 1
        if n < 1 or H.shape[0] != self.D:
            raise ValueError("Prompt dimensions do not match the model")
        update = torch.zeros_like(H)
        for P, V in zip(self.P, self.V):
            scores = H.T @ (P @ H)
            if self.activation == "relu":
                scores = scores.relu()
            elif self.activation == "softmax":
                scores = scores.softmax(dim=0)
            update = update + (V @ H) @ scores
        return H + update / n

    def predict(self, H):
        return self(H)[-(self.d + self.m):, -1]

    def cosine_similarity(self, alpha=0.2, beta=0.8):
        """Signed effective-block cosine, for the single linear head only."""
        if len(self.P) != 1 or self.activation != "linear":
            return {}
        P, V = theoretical_matrices(self.algorithm, self.d, self.m, alpha, beta)
        pairs = ((self.P[0][:self.r, self.r:], P[:self.r, self.r:]),
                 (self.V[0][self.r + 1:, :self.r], V[self.r + 1:, :self.r]))
        result = {}
        for name, (learned, target) in zip(("P12", "V21"), pairs):
            result[name] = float(nn.functional.cosine_similarity(
                learned.detach().cpu().flatten(), target.flatten(), dim=0))
        return result
