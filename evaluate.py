"""Closed-loop control and cliff-walk evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from environments import CliffWalkTask, action_probs
from model import collect, teacher_update
from train import load_checkpoint


def value_iteration(task):
    reward = np.einsum("san,an->sa", task.P, task.R)
    Q = np.zeros_like(reward)
    for _ in range(10000):
        next_Q = reward + task.gamma * (task.P @ Q.max(axis=1))
        if np.max(np.abs(next_Q - Q)) < 1e-10:
            return next_Q
        Q = next_Q
    raise RuntimeError("Value iteration did not converge")


def rollout_return(task, probabilities, rng, episodes, horizon):
    """Vectorized discounted Monte Carlo returns with an explicit RNG."""
    states = rng.choice(task.nS, size=episodes, p=task.P0)
    returns = np.zeros(episodes)
    for t in range(horizon):
        cdf = probabilities[states].cumsum(axis=1)
        cdf[:, -1] = 1.0
        actions = (rng.random(episodes)[:, None] > cdf).sum(axis=1)
        cdf = task.P[states, actions].cumsum(axis=1)
        cdf[:, -1] = 1.0
        states = (rng.random(episodes)[:, None] > cdf).sum(axis=1)
        returns += task.gamma ** t * task.R[actions, states]
    return float(returns.mean())


def advance(task, theta, model, config, rng, state):
    H, teacher_H, state = collect(task, theta, config["window"], rng, state, config["raw"])
    if model is None:
        updated = teacher_update(teacher_H, task.algorithm, task.d, task.m,
                                 config["alpha"], config["beta"])
    else:
        updated = model.predict(H)
    if not torch.isfinite(updated).all() or float(updated.norm()) > 1e6:
        raise RuntimeError("Closed-loop parameters diverged; no truncated score was substituted")
    return updated.detach().cpu().numpy(), state


def paper_rollout(task, parameter, rng, episodes, horizon, baseline=None):
    """The original vectorized rollout, including its random draw order."""
    p = task.P.astype(np.float64)
    p /= p.sum(axis=2, keepdims=True)
    p0 = task.P0.astype(np.float64)
    p0 /= p0.sum()
    states = rng.choice(task.nS, size=episodes, p=p0)
    returns, discount = np.zeros(episodes), 1.0
    if baseline == "oracle":
        reward = np.einsum("san,an->sa", p, task.R.astype(np.float64))
        q = np.zeros_like(reward)
        for _ in range(1000):
            new = reward + task.gamma * np.einsum("san,n->sa", p, q.max(axis=1))
            if np.max(np.abs(new - q)) < 1e-9:
                break
            q = new
        q = new
    elif parameter is not None:
        q = task.phi.astype(np.float64) @ parameter[:task.m if task.m else task.d].astype(np.float64)
    else:
        q = np.zeros((task.nS, task.nA))
    if task.algorithm == "ac":
        if baseline == "oracle":
            policy = np.eye(task.nA)[q.argmax(axis=1)]
        elif baseline == "random":
            policy = np.full((task.nS, task.nA), 1 / task.nA)
        else:
            policy = np.exp(q - q.max(axis=1, keepdims=True))
            policy /= policy.sum(axis=1, keepdims=True)
    for _ in range(horizon):
        if task.algorithm == "ac":
            actions = (policy[states].cumsum(axis=1) > rng.random(episodes)[:, None]).argmax(axis=1)
        else:
            actions = q[states].argmax(axis=1)
            if baseline == "random":
                mask = rng.random(episodes) < 1.0
                actions = np.where(mask, rng.integers(task.nA, size=episodes), actions)
        states = (p[states, actions].cumsum(axis=1) > rng.random(episodes)[:, None]).argmax(axis=1)
        returns += discount * task.R[actions, states].astype(np.float64)
        discount *= task.gamma
    return float(returns.mean())


@torch.no_grad()
def evaluate(model, config, num_tasks=100, updates=None, eval_every=10,
             episodes=32, eval_horizon=50, seed=0):
    """Replay the paper evaluation, with a deterministic post-training RNG state.

    Historical AC evaluates consecutive, independently initialized transformer
    and teacher prompts on the same MDP; it does not pair their initial weights.
    """
    from paper_protocol import PaperTasks, PaperPrompt
    from train import make_model
    if config["family"] != "random":
        raise ValueError("MountainCar reports training and parameter recovery only")
    updates = (3000 if config["algorithm"] == "ac" else 100) if updates is None else updates
    if min(num_tasks, eval_every, episodes, eval_horizon) < 1 or updates < 0:
        raise ValueError("Invalid evaluation counts")
    model.eval()
    checkpoints = sorted(set(range(0, updates + 1, eval_every)) | {updates})
    results = {key: np.zeros((num_tasks, len(checkpoints)))
               for key in ("tf_returns", "teacher_returns")}
    results.update(oracle_returns=np.zeros(num_tasks), random_returns=np.zeros(num_tasks),
                   checkpoint_steps=np.asarray(checkpoints))
    master = np.random.default_rng(seed)
    task_seeds = [int(master.integers(0, 2**31 - 1)) for _ in range(num_tasks)]
    c = dict(config, horizon=(updates + 2) * config["window"])
    with torch.random.fork_rng(devices=[]):
        # Recreate the state reached by training then evaluating in one notebook.
        torch.manual_seed(config["seed"])
        make_model(config)
        if config["algorithm"] == "ac":
            for _ in range(config["tasks"]):
                torch.rand((config["m"] + config["d"], 1))
        for i, task_seed in enumerate(task_seeds):
            stream = PaperTasks(c, seed=task_seed, feature_seed=config["seed"] + i,
                                value_seed=config["seed"] + i)
            task = stream.draw(random_init=False)
            for baseline in ("oracle", "random"):
                results[baseline + "_returns"][i] = paper_rollout(
                    task, None, np.random.default_rng(task_seed + 1), episodes, eval_horizon, baseline)
            for key, predictor in (("tf_returns", model), ("teacher_returns", None)):
                rng = np.random.RandomState(task_seed + 100)
                prompt = PaperPrompt(task, c, rng, random_init=task.algorithm == "ac")
                for t in range(updates + 1):
                    if t in checkpoints:
                        j = checkpoints.index(t)
                        results[key][i, j] = paper_rollout(
                            task, prompt.theta.numpy(), np.random.default_rng(task_seed + 2 + 1000 * t),
                            episodes, eval_horizon)
                    if t < updates:
                        theta = (predictor.predict(prompt.H) if predictor is not None else
                                 teacher_update(prompt.teacher_H, task.algorithm, task.d, task.m,
                                                c["alpha"], c["beta"]))
                        if not torch.isfinite(theta).all() or float(theta.norm()) > 1e6:
                            raise RuntimeError(f"{key}: diverged on test MDP {i}, update {t + 1}")
                        prompt.step(theta)
            if (i + 1) % 10 == 0:
                print(f"{config['experiment']}: evaluated {i + 1}/{num_tasks} MDPs", flush=True)
    results["task_seeds"] = np.asarray(task_seeds, dtype=np.int64)
    results["algorithm"] = np.asarray(config["algorithm"])
    return results

def hitting_time(task, theta, rng, episodes, cap):
    lengths, successes = [], 0
    for _ in range(episodes):
        state = task.start
        for t in range(1, cap + 1):
            probabilities = (np.full(task.nA, 1 / task.nA) if theta is None else
                             action_probs(task, state, theta, epsilon=0.0))
            action = int(rng.choice(task.nA, p=probabilities))
            landed = task.move(state, action, rng)
            if landed == task.goal:
                successes += 1
                break
            state = task.start if landed == task.cliff else landed
        lengths.append(t)
    return dict(mean_steps=float(np.mean(lengths)), success_rate=successes / episodes)


@torch.no_grad()
def evaluate_cliff(model, config, updates=100, episodes=32, cap=100, seed=0):
    if config["algorithm"] != "sarsa" or config["d"] != 36 or config["family"] != "random":
        raise ValueError("Cliff walk needs a SARSA checkpoint trained on the random 9-state family")
    if updates < 0 or min(episodes, cap) < 1:
        raise ValueError("Invalid cliff-walk evaluation counts")
    model.eval()
    task = CliffWalkTask(np.random.default_rng(seed))
    result = {}
    for name, policy_model in (("transformer", model), ("teacher", None)):
        theta, state = np.zeros(task.d, dtype=np.float32), None
        collection_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
        for _ in range(updates):
            theta, state = advance(task, theta, policy_model, config, collection_rng, state)
        result[name] = hitting_time(task, theta, np.random.default_rng(seed + 1), episodes, cap)
    result["random"] = hitting_time(task, None, np.random.default_rng(seed + 1), episodes, cap)
    transitions = task.hitting_time_transitions()
    times = np.zeros(task.nS)
    for _ in range(10000):
        updated = 1 + (transitions @ times).min(axis=1)
        updated[task.goal] = 0
        if np.max(np.abs(updated - times)) < 1e-10:
            break
        times = updated
    else:
        raise RuntimeError("Optimal hitting-time iteration did not converge")
    result["optimal_expected_steps"] = float(updated[task.start])
    return result


def plot_control(results, path=None, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    algorithm = str(results["algorithm"])
    label = "Actor-Critic" if algorithm == "ac" else "SARSA"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    steps = results["checkpoint_steps"]
    for key, text, marker in (("tf_returns", "transformer", "o"),
                               ("teacher_returns", "teacher AC" if algorithm == "ac" else "teacher SARSA", "s")):
        values = results[key]
        ax.plot(steps, values.mean(axis=0), marker=marker, label=text)
        low, high = np.percentile(values, [25, 75], axis=0)
        ax.fill_between(steps, low, high, alpha=.2)
    ax.axhline(results["oracle_returns"].mean(), color="k", linestyle="--", label="oracle")
    ax.axhline(results["random_returns"].mean(), color="gray", linestyle=":", label="random")
    ax.set_xlabel("In-context update step", fontsize=18)
    ax.set_ylabel("Mean discounted return", fontsize=18)
    ax.set_title(f"{label} Transformer Control performance", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=14)
    fig.tight_layout()
    if path is not None:
        path = Path(path)
        fig.savefig(path, dpi=350, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--eval-horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cliff", action="store_true")
    parser.add_argument("--cap", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for checkpoint in args.checkpoints:
        model, config = load_checkpoint(checkpoint)
        out = args.out or checkpoint.parent
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{config['experiment']}_{checkpoint.stem}_{'cliff' if args.cliff else 'control'}"
        metadata = dict(checkpoint=str(checkpoint), config=config,
                        evaluation={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                                    if k not in {"checkpoints", "out"}})
        if args.cliff:
            metadata["results"] = evaluate_cliff(model, config,
                updates=100 if args.updates is None else args.updates,
                episodes=args.episodes, cap=args.cap, seed=args.seed)
        else:
            result = evaluate(model, config, args.tasks, args.updates, args.eval_every,
                              args.episodes, args.eval_horizon, args.seed)
            np.savez_compressed(out / f"{stem}.npz", **result)
            metadata["results"] = {key: float(result[key][:, -1].mean()) for key in ("tf_returns", "teacher_returns")}
            metadata["results"].update({key: float(result[key].mean()) for key in ("oracle_returns", "random_returns")})
            plot_control(result, out / f"{stem}.pdf")
        (out / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata["results"], indent=2))


if __name__ == "__main__":
    main()
