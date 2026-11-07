"""Monte Carlo check of the SARSA richness conditions in the paper appendix.

Paper run: python verify_assumptions.py
Smoke run: python verify_assumptions.py --samples 80 --directions 1 \
    --samples-per-direction 20 --skip-rho

The fixed-weight sweep approximates the universal quantifier in kappa_R.
The rho estimate uses full empirical moments, an exact constrained eigenvalue
at each scale, and the paper's one-dimensional numerical scale search.
Random draws use the shared training sampler and explicit NumPy generators;
they do not reproduce the legacy mixed NumPy/PyTorch random stream bitwise.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

from environments import random_task
from model import collect, teacher_update


NS, NA, D = 9, 4, 36
GAMMA, EPSILON, ALPHA = 0.5, 0.1, 0.2
WINDOW, HORIZON = 20, 1000
N_WINDOWS = HORIZON // WINDOW - 1  # Final horizon-reaching window is not trained.
DX, DW = 2 * D + 1, D + 1
DIM_U, DIM_W = DX * DW, D * DX


def log(message: str) -> None:
    print(message, flush=True)


def structural_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Canonical effective SARSA blocks P_12* and Vbar_21*."""
    p = np.zeros((DX, DW))
    p[:D, 1:] = -np.eye(D)
    p[D : 2 * D, 1:] = np.eye(D)
    p[-1, 0] = 1.0
    v = np.zeros((D, DX))
    v[:, :D] = ALPHA * np.eye(D)
    return p, v


def prompt_statistics(h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wtilde, R=mean(phi x^T), and b=mean(x delta)."""
    h = h.detach().cpu().numpy().astype(np.float64)
    x, wtilde = h[:DX, :-1], h[DX:, -1]
    phi, gamma_phi_next, reward = x[:D], x[D : 2 * D], x[-1]
    delta = reward + wtilde[1:] @ (gamma_phi_next - phi)
    return wtilde, phi @ x.T / x.shape[1], x @ delta / x.shape[1]


@dataclass
class Moments:
    wtilde: np.ndarray
    b: np.ndarray
    a: np.ndarray | None
    cross: np.ndarray | None


def sample_moments(n_samples: int, seed: int, batch_size: int, rho: bool) -> tuple[Moments, dict]:
    """Sample training windows and accumulate moments in a single pass."""
    rng = np.random.default_rng(seed)
    time_rng = np.random.default_rng(seed + 10_000)
    moments = Moments(np.zeros((DW, DW)), np.zeros((DX, DX)), None, None)
    if rho:
        moments.a = np.zeros((DIM_U, DIM_U))
        moments.cross = np.zeros((DIM_U, DIM_W))
    times, norms = [], []
    p_star, v_star = structural_matrices()

    for start in range(0, n_samples, batch_size):
        count = min(batch_size, n_samples - start)
        weights = np.empty((count, DW))
        regressors = np.empty((count, D, DX))
        targets = np.empty((count, DX))
        for j in range(count):
            task = random_task(rng, algorithm="sarsa", nS=NS, nA=NA,
                               gamma=GAMMA, epsilon=EPSILON)
            theta = rng.uniform(-1.0, 1.0, D).astype(np.float32)
            t = int(time_rng.integers(N_WINDOWS))
            state = None
            for window in range(t + 1):
                _, h, state = collect(task, theta, WINDOW, rng, state=state)
                if window < t:
                    theta = teacher_update(h, algorithm="sarsa", d=D, m=0,
                                           alpha=ALPHA, beta=0.8)
            weights[j], regressors[j], targets[j] = prompt_statistics(h)
            if start + j < 3:
                # Independent block identities catch gamma/scaling mistakes.
                np.testing.assert_allclose(
                    ALPHA * regressors[j] @ p_star @ weights[j],
                    v_star @ targets[j], atol=2e-10, rtol=2e-10,
                )
            times.append(t)
            norms.append(float(np.linalg.norm(weights[j, 1:])))

        moments.wtilde += weights.T @ weights
        moments.b += targets.T @ targets
        if rho:
            # Row-major vectorization: design[j] rvec(U) = R_j U wtilde_j.
            design = np.einsum("nki,na->nkia", regressors, weights,
                               optimize=True).reshape(count, D, DIM_U)
            flat = design.reshape(count * D, DIM_U)
            moments.a += flat.T @ flat
            for k in range(D):
                moments.cross[:, k * DX : (k + 1) * DX] += design[:, k].T @ targets
        log(f"Mixture samples: {start + count}/{n_samples}")

    for matrix in (moments.wtilde, moments.b, moments.a, moments.cross):
        if matrix is not None:
            matrix /= n_samples
            if not np.isfinite(matrix).all():
                raise FloatingPointError("Non-finite empirical moment")
    return moments, {
        "N": n_samples,
        "window_index_support": [0, N_WINDOWS - 1],
        "sampled_time_min": min(times),
        "sampled_time_max": max(times),
        "sampled_time_mean": float(np.mean(times)),
        "w_norm_mean": float(np.mean(norms)),
        "w_norm_max": max(norms),
    }


def smallest_eigenvalue(matrix: np.ndarray) -> float:
    return float(linalg.eigh((matrix + matrix.T) / 2,
                            subset_by_index=[0, 0], eigvals_only=True)[0])


def estimate_conditional_kappa_r(directions: int, samples: int, seed: int) -> dict:
    """Sweep {0, +/-u_1,...,+/-u_k,5u_1}, with fresh tasks at each weight."""
    rng = np.random.default_rng(seed)
    direction_rng = np.random.default_rng(seed + 20_000)
    weights = [("zero", np.zeros(D))]
    for j in range(directions):
        u = direction_rng.standard_normal(D)
        u /= np.linalg.norm(u)
        weights.extend(((f"unit_{j + 1}", u), (f"negative_unit_{j + 1}", -u)))
    if directions:
        weights.append(("five_times_unit_1", 5 * weights[1][1]))
    estimates = {}
    for label, theta in weights:
        moment = np.zeros((DX, DX))
        for _ in range(samples):
            task = random_task(rng, algorithm="sarsa", nS=NS, nA=NA,
                               gamma=GAMMA, epsilon=EPSILON)
            _, h, _ = collect(task, theta.astype(np.float32), WINDOW, rng)
            _, regressor, _ = prompt_statistics(h)
            moment += regressor.T @ regressor
        estimates[label] = smallest_eigenvalue(moment / samples)
        log(f"kappa_R[{label}] = {estimates[label]:.8g}")
    worst = min(estimates, key=estimates.get)
    return {
        "samples_per_weight": samples,
        "total_sweep_mdps": len(weights) * samples,
        "by_weight": estimates,
        "worst_weight": worst,
        "kappa_R_hat": estimates[worst],
    }


def constrained_minimum(eigenvalues: np.ndarray, weights: np.ndarray) -> float:
    """Smallest eigenvalue restricted to a hyperplane, via its secular root.

    eigenvalues are ordered; weights are squared projections of the unit normal
    onto that eigenbasis. Repeated zero eigenvalues may be combined in one entry.
    """
    low, high = float(eigenvalues[0]), float(eigenvalues[1])
    gap = high - low
    if weights[0] < 1e-20 or gap < 1e-12:
        return low

    def secular(mu: float) -> float:
        return float(np.sum(weights / (eigenvalues - mu)))

    left, right = np.nextafter(low, high), np.nextafter(high, low)
    if secular(left) >= 0:
        return low  # Root is within machine precision of the lower pole.
    if secular(right) <= 0:
        return high  # An eigenvector at the upper endpoint remains feasible.
    return float(optimize.brentq(secular, left, right, xtol=1e-13, rtol=1e-13))


def estimate_rho(moments: Moments) -> dict:
    """Whiten the full moments, then search the paper's normal-space scales."""
    if moments.a is None or moments.cross is None:
        raise ValueError("rho requires the full empirical moments")
    try:
        chol_a = linalg.cholesky(moments.a, lower=True)
        chol_b = linalg.cholesky(moments.b, lower=True)
    except linalg.LinAlgError as error:
        raise ValueError(
            "Empirical moments are not positive definite; increase --samples "
            "or use --skip-rho for a smoke test. No diagonal regularization is applied."
        ) from error
    cross = linalg.solve_triangular(chol_a, moments.cross, lower=True)
    for k in range(D):
        block = slice(k * DX, (k + 1) * DX)
        cross[:, block] = linalg.solve_triangular(chol_b, cross[:, block].T,
                                                lower=True).T
    left, singular, right_t = linalg.svd(cross, full_matrices=False)
    p_star, v_star = structural_matrices()
    normal_u = linalg.solve_triangular(chol_a, p_star.ravel(), lower=True)
    normal_w = linalg.solve_triangular(chol_b, -v_star.T, lower=True).T.ravel()
    # The sign of the W block is from n(c)=[c n_U; -c^{-1} n_W].
    pu, pw = left.T @ normal_u, right_t @ normal_w
    norm_u2, norm_w2 = normal_u @ normal_u, normal_w @ normal_w
    null_u2 = max(0.0, float(norm_u2 - pu @ pu))
    null_w2 = max(0.0, float(norm_w2 - pw @ pw))
    eigenvalues = np.concatenate((-singular, [0.0], singular[::-1]))

    def rho_for_normal(u_scale: float, w_scale: float) -> float:
        u, w = u_scale * pu, w_scale * pw
        norm2 = u_scale**2 * norm_u2 + w_scale**2 * norm_w2
        negative = 0.5 * (u - w)**2
        positive = 0.5 * (u + w)**2
        zero = u_scale**2 * null_u2 + w_scale**2 * null_w2
        weights = np.concatenate((negative, [zero], positive[::-1])) / norm2
        return -constrained_minimum(eigenvalues, weights)

    def rho_at(log_s: float) -> float:
        return rho_for_normal(float(np.exp(-log_s / 2)), float(np.exp(log_s / 2)))

    grid = np.linspace(-24.0, 24.0, 97)
    values = np.asarray([rho_at(s) for s in grid])
    best = int(np.argmax(values))
    best_rho, best_log_s = float(values[best]), float(grid[best])
    for i in np.argsort(values)[-5:]:
        result = optimize.minimize_scalar(
            lambda s: -rho_at(s), method="bounded",
            bounds=(grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]),
            options={"xatol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"Scale refinement failed: {result.message}")
        if -result.fun > best_rho:
            best_rho, best_log_s = float(-result.fun), float(result.x)
    limit_infinity, limit_zero = rho_for_normal(1.0, 0.0), rho_for_normal(0.0, 1.0)
    worst_c: float | str = float(np.exp(-best_log_s / 2))
    if limit_infinity > best_rho:
        best_rho, worst_c = limit_infinity, "infinity"
    if limit_zero > best_rho:
        best_rho, worst_c = limit_zero, "zero"
    return {
        "rho_hat": best_rho,
        "worst_c": worst_c,
        "rho_limit_c_to_infinity": limit_infinity,
        "rho_limit_c_to_zero": limit_zero,
        "largest_unconstrained_correlation": float(singular[0]),
        "second_unconstrained_correlation": float(singular[1]),
        "log_scale_grid": [-24.0, 24.0, 97],
        "refined_neighborhoods": 5,
        "diagonal_regularization": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--samples-per-direction", type=int, default=800)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--kappa-r-seed", type=int, default=5101)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--skip-rho", action="store_true",
                        help="Skip large moment matrices and the expensive SVD (smoke tests).")
    parser.add_argument("--output", type=Path, default=Path("runs/assumptions.json"))
    args = parser.parse_args()
    if min(args.samples, args.samples_per_direction, args.batch_size) <= 0:
        parser.error("Sample counts and batch size must be positive")
    if args.directions < 0:
        parser.error("--directions must be nonnegative")
    if not args.skip_rho and (args.samples < DX or args.samples * min(D, WINDOW) < DIM_U):
        parser.error("Too few samples for nonsingular rho moments; use --skip-rho for a smoke test")
    started = time.perf_counter()
    moments, diagnostics = sample_moments(args.samples, args.seed, args.batch_size,
                                          rho=not args.skip_rho)
    results = {
        "setup": {
            "n_states": NS, "n_actions": NA, "dimension": D, "gamma": GAMMA,
            "epsilon": EPSILON, "alpha": ALPHA, "window": WINDOW,
            "horizon": HORIZON, "initial_weight": "uniform(-1,1)",
            "seed": args.seed, "kappa_R_seed": args.kappa_r_seed,
            "moment_batch_size": args.batch_size,
        },
        "sampling": diagnostics,
        "paper_sample_counts": (args.samples == 10_000 and args.directions == 8
                                and args.samples_per_direction == 800),
        "kappa_wtilde_hat": smallest_eigenvalue(moments.wtilde),
        "kappa_q_hat": smallest_eigenvalue(moments.b),
        "interpretation": "Empirical diagnostics, not population confidence bounds or a proof.",
    }
    log(f"kappa_wtilde={results['kappa_wtilde_hat']:.8g}; kappa_q={results['kappa_q_hat']:.8g}")
    if args.skip_rho:
        results["rho"] = {"skipped": True}
    else:
        log("Whitening and solving constrained canonical correlation")
        results["rho"] = estimate_rho(moments)
        log(f"rho = {results['rho']['rho_hat']:.8g}")
    del moments
    results["conditional_kappa_R"] = estimate_conditional_kappa_r(
        args.directions, args.samples_per_direction, args.kappa_r_seed,
    )
    results["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    log(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    # Some Apple Accelerate builds report floating-point flags on finite BLAS
    # products. Scope suppression to this CLI; explicit finiteness checks above
    # and SciPy's input validation still reject genuinely invalid moments.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        main()
