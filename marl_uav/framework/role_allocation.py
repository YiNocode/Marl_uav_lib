"""Transport-based role allocation (fixed-budget entropic optimal transport)."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def sinkhorn_transport_plan(
    cost: np.ndarray,
    *,
    epsilon: float,
    num_iters: int,
) -> np.ndarray:
    """
    Uniform-marginal entropic OT plan via log-domain Sinkhorn.

    Parameters
    ----------
    cost : (n, n) non-negative transport costs C_ij.
    epsilon : entropic temperature (same units as cost).
    num_iters : Sinkhorn iterations (fixed budget).
    """
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"cost must be square 2D, got shape {c.shape}")
    n = int(c.shape[0])
    if n == 0:
        return c.copy()

    eps = max(float(epsilon), 1e-12)
    iters = max(int(num_iters), 1)
    log_k = -c / eps
    log_k -= np.max(log_k)
    k = np.exp(log_k)

    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(n, dtype=np.float64)
    log_a = -np.log(n)
    log_b = -np.log(n)

    for _ in range(iters):
        log_kv = log_k + log_v[None, :]
        log_u = log_a - np.logaddexp.reduce(log_kv, axis=1)
        log_ku = log_k + log_u[:, None]
        log_v = log_b - np.logaddexp.reduce(log_ku, axis=0)

    log_p = log_u[:, None] + log_k + log_v[None, :]
    p = np.exp(log_p)
    total = float(np.sum(p))
    if total > 0.0:
        p /= total
    return p.astype(np.float64)


def _hard_assignment_from_plan(plan: np.ndarray) -> np.ndarray:
    """Maximum-weight one-to-one matching on transport plan (exact for small n)."""
    p = np.asarray(plan, dtype=np.float64)
    n = int(p.shape[0])
    best_perm = tuple(range(n))
    best_score = -np.inf
    for perm in permutations(range(n)):
        score = float(sum(p[i, perm[i]] for i in range(n)))
        if score > best_score:
            best_score = score
            best_perm = perm
    return np.asarray(best_perm, dtype=np.int64)


def entropic_ot_assignment(
    cost: np.ndarray,
    *,
    epsilon: float,
    num_iters: int,
    prev_assignment: np.ndarray | None = None,
    inertia_margin: float = 0.0,
) -> np.ndarray:
    """
    Hard UAV→slot assignment from an entropic OT plan, with optional inertia.

    Returns ``assignment[i] = slot index`` for pursuer ``i``.
    """
    c = np.asarray(cost, dtype=np.float64)
    plan = sinkhorn_transport_plan(c, epsilon=epsilon, num_iters=num_iters)
    new_assignment = _hard_assignment_from_plan(plan)

    if prev_assignment is None:
        return new_assignment

    prev = np.asarray(prev_assignment, dtype=np.int64).reshape(-1)
    if prev.shape[0] != c.shape[0] or len(np.unique(prev)) != c.shape[0]:
        return new_assignment

    n = int(c.shape[0])
    new_cost = float(sum(c[i, int(new_assignment[i])] for i in range(n)))
    old_cost = float(sum(c[i, int(prev[i])] for i in range(n)))
    if new_cost < old_cost - float(inertia_margin):
        return new_assignment
    return prev.copy()


def default_ot_epsilon(cost: np.ndarray, epsilon: float, epsilon_scale: float | None) -> float:
    """Resolve OT temperature from absolute epsilon and/or scale × mean(cost)."""
    c = np.asarray(cost, dtype=np.float64)
    base = max(float(epsilon), 1e-9)
    if epsilon_scale is None:
        return base
    scale = max(float(epsilon_scale), 0.0)
    mean_c = float(np.mean(c)) if c.size else 0.0
    return max(base, scale * max(mean_c, 1e-9))
