# -*- coding: utf-8 -*-
"""Controlled simulation (article experiment E7, design in Section 2.7).
ZERO LLM calls.

A synthetic judge with fully controlled properties ranks N objects whose
hidden strengths are drawn from N(0, 1); the true ranking is the strength
order. Judge parameters:

  competence beta : P(the STRONGER object wins | decisive verdict)
                    = sigmoid(beta * |s_i - s_j|); beta from ~random (0.5)
                    to near-perfect (4+).
  tie model       : "const"     — every pair ties with probability tau;
                    "threshold" — P(tie | pair) = exp(-|s_i - s_j| / delta),
                    with delta calibrated (scale-free bisection) so the MEAN
                    tie probability over this realization's pairs equals tau.

Input order of the data file is varied separately: "random" permutation,
"pos" — positively correlated with the true order (target Spearman
`order_rho`, noisy-key construction), "neg" — the exact mirror of "pos"
(rho = -order_rho).

Per realization ONE full verdict matrix is generated and consumed by all
nine algorithms — the same protocol as for the real tournaments, where
every algorithm replays one verdict matrix. Aggregators run on the dense
matrix with vectorized math equal to the sparse implementations in
ranklib.aggregators (equivalence is asserted by the authors' test suite);
sorts run through ranklib.sorts on the ids in input order.
"""
import itertools
import time

import numpy as np
import pandas as pd

from .aggregators import rank_centrality_dense, _ranks_from_scores
from .metrics import (nontransitivity_exact_dense, ranks_from_order, spearman,
                      topbot_overlap)
from .sorts import SORTS, run_sort

ALGORITHMS = ("copeland", "winrate", "bt", "rank_centrality",
              "heapsort", "quicksort", "bubble", "merge", "insertion")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------- competence in interpretable units -------------------------
# beta alone is NOT identifiable: the error probability depends on the product
# beta * |s_i - s_j|, so doubling the spread of true quality in the pool is
# equivalent to doubling beta. The scale-free quantity is the ERROR RATE — the
# share of decisive verdicts in which the judge prefers the objectively weaker
# object. That is what practitioners can measure, so thresholds are reported
# in those units. With s ~ N(0,1) the pair difference is N(0,2), hence
#     error_rate(beta) = E[ sigmoid(-beta * |d|) ],  |d| ~ half-normal(sqrt 2).
# Quadrature + bisection are done by hand: the project deliberately carries no
# scipy dependency (see metrics.spearman).

_ERR_NODES = 20001          # trapezoid nodes on [0, 12] standard deviations
_ERR_UPPER = 12.0


def error_rate(beta):
    """Share of DECISIVE verdicts the judge gets wrong, for competence `beta`,
    on a pool with unit-variance true qualities (ties excluded). beta <= 0
    gives 0.5 (coin flip)."""
    if beta <= 0:
        return 0.5
    u = np.linspace(0.0, _ERR_UPPER, _ERR_NODES)          # |d| / sqrt(2)
    pdf = 2.0 * np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    z = np.clip(beta * np.sqrt(2.0) * u, -700.0, 700.0)
    y = pdf / (1.0 + np.exp(z))
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(u)))


def beta_for_error(err, lo=1e-9, hi=1e4, tol=1e-12):
    """Inverse of error_rate: competence beta giving the requested error rate
    (monotone decreasing, so plain bisection). `err` must lie in (0, 0.5)."""
    if not 0.0 < err < 0.5:
        raise ValueError(f"error rate must be in (0, 0.5), got {err!r}")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if error_rate(mid) > err:                          # too many errors
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _perm_spearman(pos_a, pos_b):
    """Spearman rho of two permutation-position arrays (no ties -> exact
    closed formula)."""
    n = len(pos_a)
    d = pos_a.astype(float) - pos_b.astype(float)
    return 1.0 - 6.0 * float(d @ d) / (n * (n * n - 1))


def calibrate_delta(absd, tau, iters=100):
    """delta such that mean(exp(-absd/delta)) == tau (monotone in delta;
    geometric bisection, scale-free)."""
    if tau <= 0.0:
        return 0.0
    lo, hi = 1e-9, 1e9
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        if float(np.exp(-absd / mid).mean()) < tau:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def gen_verdicts(s, beta, tie_model, tau, rng):
    """Full antisymmetric verdict matrix V (int8): +1 = row beats column,
    -1 = loses, 0 = tie. Returns (V, empirical_tie_rate)."""
    n = len(s)
    iu, ju = np.triu_indices(n, 1)
    d = s[iu] - s[ju]
    absd = np.abs(d)
    if tie_model == "const":
        p_tie = np.full(absd.shape, float(tau))
    elif tie_model == "threshold":
        delta = calibrate_delta(absd, tau)
        p_tie = np.exp(-absd / delta) if delta > 0 else np.zeros_like(absd)
    else:
        raise ValueError(f"unknown tie model {tie_model!r}")
    tie = rng.random(len(d)) < p_tie
    i_wins = rng.random(len(d)) < _sigmoid(beta * d)
    v = np.where(tie, 0, np.where(i_wins, 1, -1)).astype(np.int8)
    V = np.zeros((n, n), np.int8)
    V[iu, ju] = v
    V[ju, iu] = -v
    return V, float(tie.mean())


def gen_input_order(s, kind, order_rho, rng, tol=0.005):
    """Input-file order of ids. "random": permutation. "pos": sort by
    (s + lam*noise) with lam bisected to hit Spearman(order, true) ==
    order_rho. "neg": the mirror of "pos" (exactly -order_rho).
    Returns (ids_in_input_order, empirical_rho_vs_truth)."""
    n = len(s)
    true_pos = np.empty(n, int)
    true_pos[np.argsort(-s, kind="stable")] = np.arange(n)
    if kind == "random":
        ids = rng.permutation(n)
        pos = np.empty(n, int)
        pos[ids] = np.arange(n)
        return list(ids), _perm_spearman(pos, true_pos)
    if kind not in ("pos", "neg"):
        raise ValueError(f"unknown input-order kind {kind!r}")
    eps = rng.standard_normal(n)

    def order_pos(lam):
        ids = np.argsort(-(s + lam * eps), kind="stable")
        pos = np.empty(n, int)
        pos[ids] = np.arange(n)
        return ids, pos

    lo, hi = 0.0, 1e3          # rho decreases monotonically in lam
    for _ in range(60):
        mid = (lo + hi) / 2
        _ids, pos = order_pos(mid)
        r = _perm_spearman(pos, true_pos)
        if abs(r - order_rho) < tol:
            break
        if r > order_rho:
            lo = mid
        else:
            hi = mid
    ids, pos = order_pos((lo + hi) / 2 if abs(r - order_rho) >= tol else mid)
    rho = _perm_spearman(pos, true_pos)
    ids = list(ids)
    if kind == "neg":
        ids = ids[::-1]
        rho = -rho
    return ids, rho


# ----------------------- dense aggregators (vectorized) ----------------------

def copeland_scores_dense(V):
    return ((V > 0).sum(1) - (V < 0).sum(1)).astype(float)


def winrate_scores_dense(V):
    wins = (V > 0).sum(1).astype(float)
    dec = wins + (V < 0).sum(1)
    out = np.full(len(V), 0.5)
    nz = dec > 0
    out[nz] = wins[nz] / dec[nz]
    return out


def bt_theta_dense(V, alpha=1.0, max_iter=1000, tol=1e-9):
    """Vectorized regularized MM (same math as aggregators.bradley_terry_
    strengths on a COMPLETE one-game-per-pair matrix): tie = half a win,
    2*alpha anchor games at strength 1, geometric-mean normalization."""
    n = len(V)
    wins = (V > 0).sum(1) + 0.5 * ((V == 0).sum(1) - 1)   # diagonal excluded
    pi = np.ones(n)
    logpi = np.zeros(n)
    for _ in range(max_iter):
        S = 1.0 / (pi[:, None] + pi[None, :])
        denom = 2.0 * alpha / (pi + 1.0) + S.sum(1) - 1.0 / (2.0 * pi)
        new = (wins + alpha) / denom
        lognew = np.log(new)
        lognew -= lognew.mean()                            # geometric mean 1
        delta = float(np.abs(lognew - logpi).max())
        logpi = lognew
        pi = np.exp(logpi)
        if delta < tol:
            break
    return logpi


def rc_scores_dense(V, damping=0.99):
    """Rank Centrality on a complete verdict matrix. The stationary vector is
    obtained by a direct linear solve of the same teleported chain the ported
    power iteration converges to (equivalence asserted by the authors' tests)."""
    n = len(V)
    wins = (V > 0).astype(float)
    games = np.ones((n, n)) - np.eye(n)
    frac = np.zeros((n, n))
    mask = games > 0
    frac[mask] = wins.T[mask] / games[mask]
    np.fill_diagonal(frac, 0.0)
    dmax = frac.sum(axis=1).max()
    if not dmax > 0:
        dmax = 1.0
    P = frac / dmax
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(P, 1.0 - P.sum(axis=1))
    P = damping * P + (1.0 - damping) / n
    A = P.T - np.eye(n)
    A[-1, :] = 1.0                                        # sum(pi) = 1
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    return np.abs(pi)


DENSE_AGGREGATORS = {
    "copeland": copeland_scores_dense,
    "winrate": winrate_scores_dense,
    "bt": bt_theta_dense,
    "rank_centrality": rc_scores_dense,
}


# ------------------------------- the runner -----------------------------------

def run_realization(n, beta, tie_model, tau, order_kind, order_rho,
                    cond_id, rep, master_seed, alpha=1.0, topk=50):
    """One realization -> one row per algorithm (list of dicts)."""
    rng = np.random.default_rng([master_seed, cond_id, rep])
    s = rng.standard_normal(n)
    true_ids = list(np.argsort(-s, kind="stable"))
    true_ranks = ranks_from_order(true_ids)
    V, tie_emp = gen_verdicts(s, beta, tie_model, tau, rng)
    input_ids, order_rho_emp = gen_input_order(s, order_kind, order_rho, rng)
    input_ranks = {rid: pos + 1 for pos, rid in enumerate(input_ids)}
    nt_rate, _, _ = nontransitivity_exact_dense((V > 0).astype(float))

    VL = V.tolist()                                       # fast scalar lookup
    win = lambda i, j: VL[i][j]                           # noqa: E731
    n_pairs = n * (n - 1) // 2

    base = {"n": n, "beta": beta, "tie_model": tie_model, "tau": tau,
            "order_kind": order_kind, "order_rho_target": order_rho,
            "order_rho_emp": order_rho_emp, "rep": rep,
            "tie_rate_emp": tie_emp, "nontransitivity": nt_rate}
    rows = []
    for algo in ALGORITHMS:
        if algo in DENSE_AGGREGATORS:
            scores_vec = DENSE_AGGREGATORS[algo](V) if algo != "bt" \
                else bt_theta_dense(V, alpha=alpha)
            scores = {i: float(scores_vec[i]) for i in range(n)}
            ranks, ordered = _ranks_from_scores(scores, list(range(n)))
            cost, kind = n_pairs, "agg"
        else:
            ordered, cost = run_sort(algo, win, input_ids, seed=rep)
            ranks = ranks_from_order(ordered)
            kind = "sort"
        t, b = topbot_overlap(ordered, true_ids, k=topk)
        rows.append({**base, "algorithm": algo, "kind": kind,
                     "rho_truth": spearman(ranks, true_ranks),
                     "rho_input": spearman(ranks, input_ranks),
                     "top": t, "bot": b, "comparisons": cost})
    return rows


def run_simulation(taus=tuple(round(0.1 * k, 1) for k in range(9)),
                   betas=(0.5, 1.5, 4.0),
                   tie_models=("const", "threshold"),
                   order_kinds=("random", "pos", "neg"),
                   reps=30, n=400, order_rho=0.5, alpha=1.0,
                   master_seed=20260705, progress=True):
    """Full grid -> long DataFrame (one row per realization x algorithm)."""
    conds = list(itertools.product(tie_models, taus, betas, order_kinds))
    rows = []
    t0 = time.time()
    total = len(conds) * reps
    done = 0
    for cond_id, (tie_model, tau, beta, order_kind) in enumerate(conds):
        for rep in range(reps):
            rows.extend(run_realization(
                n, beta, tie_model, tau, order_kind, order_rho,
                cond_id, rep, master_seed, alpha=alpha))
            done += 1
            if progress and done % 100 == 0:
                el = time.time() - t0
                print(f"  {done}/{total} realizations "
                      f"({el:.0f}s elapsed, ~{el / done * (total - done):.0f}s left)",
                      flush=True)
    df = pd.DataFrame(rows)
    if progress:
        print(f"  done: {total} realizations, {len(df)} rows, "
              f"{time.time() - t0:.0f}s")
    return df


def suitability_table(df, rho_thresholds=(0.90, 0.95)):
    """Suitability table: per (algorithm, tie_model, beta) the maximal tau at
    which the WORST-case-over-input-order mean rho_truth still clears each
    threshold. -1 means the algorithm fails already at tau=0."""
    g = (df.groupby(["algorithm", "tie_model", "beta", "tau", "order_kind"])
           ["rho_truth"].mean().reset_index())
    worst = (g.groupby(["algorithm", "tie_model", "beta", "tau"])["rho_truth"]
              .min().reset_index())
    out = []
    for (algo, tm, beta), grp in worst.groupby(["algorithm", "tie_model", "beta"]):
        grp = grp.sort_values("tau")
        row = {"algorithm": algo, "tie_model": tm, "beta": beta}
        for thr in rho_thresholds:
            ok = grp[grp.rho_truth >= thr]["tau"]
            # highest tau such that ALL taus up to it clear the threshold
            tau_max = -1.0
            for tau_ in grp.tau:
                if grp[grp.tau == tau_]["rho_truth"].iloc[0] >= thr:
                    tau_max = tau_
                else:
                    break
            row[f"tau_max_rho{thr:g}"] = tau_max
        out.append(row)
    return pd.DataFrame(out)
