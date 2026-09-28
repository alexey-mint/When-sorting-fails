# -*- coding: utf-8 -*-
"""The applicability threshold of an algorithm as an ESTIMATED quantity,
not a grid point.

The task is of the same nature as dose-response analysis: find the dose (the
tie rate tau) at which the response (agreement with the truth, rho) crosses a
given level. Hence:

  * interpolation is done on the Fisher scale z = artanh(rho), the standard
    variance-stabilizing transform for correlation coefficients, rather than
    on rho itself, where the curve is visibly non-linear;
  * the confidence interval of the threshold is obtained by bootstrapping over
    the simulation repetitions, with no assumption about the shape of the
    distribution;
  * a threshold is reported TOGETHER with the accuracy criterion: without a
    named criterion it is undefined.

The judge's competence is expressed as an error rate rather than beta: beta
itself is not identifiable (see `simulation.error_rate`).

No LLM calls; scipy is not required.
"""
import numpy as np
import pandas as pd

from .simulation import error_rate

RHO_CLIP = 0.999999


def fisher_z(r):
    """artanh, clipped so that it stays finite at the boundaries."""
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -RHO_CLIP, RHO_CLIP))


def threshold_from_curve(taus, rhos, criterion):
    """First tau at which the curve drops below the criterion.

    Returns (tau*, status):
      ("ok")     — the crossing is found inside the grid;
      (">max")   — the curve stays above the criterion on the whole grid
                   (tau* = the top of the grid, a lower bound);
      ("none")   — the criterion is not met even at the smallest tau; there is
                   no threshold.
    """
    taus = np.asarray(taus, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    order = np.argsort(taus)
    taus, rhos = taus[order], rhos[order]
    if rhos[0] < criterion:
        return np.nan, "none"
    zc = fisher_z(criterion)
    z = fisher_z(rhos)
    for i in range(len(taus) - 1):
        if z[i] >= zc > z[i + 1]:
            frac = (z[i] - zc) / (z[i] - z[i + 1])
            return float(taus[i] + frac * (taus[i + 1] - taus[i])), "ok"
    return float(taus[-1]), ">max"


def _boot_thresholds(mat, taus, criterion, n_boot, rng):
    """mat: (n_tau x n_reps) values of rho. Bootstrap over the repetitions."""
    n_tau, n_reps = mat.shape
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_reps, size=(n_tau, n_reps))
        means = np.take_along_axis(mat, idx, axis=1).mean(axis=1)
        t, st = threshold_from_curve(taus, means, criterion)
        if st == "ok":
            out.append(t)
    return np.array(out)


def estimate_thresholds(long_df, criteria=(0.95, 0.90, 0.85), n_boot=2000,
                        seed=20260918, tie_model="threshold"):
    """Thresholds for every combination (algorithm, competence, input order).

    `long_df` is simulation_long.csv of any simulation run. Returns a long
    table with tau*, its confidence interval and a status."""
    d = long_df[long_df.tie_model == tie_model]
    rng = np.random.default_rng(seed)
    rows = []
    keys = ["algorithm", "beta", "order_kind"]
    for (algo, beta, order), g in d.groupby(keys, sort=True):
        piv = g.pivot_table(index="tau", columns="rep", values="rho_truth")
        taus = piv.index.values
        mat = piv.values
        means = mat.mean(axis=1)
        for crit in criteria:
            t, status = threshold_from_curve(taus, means, crit)
            lo = hi = np.nan
            if status == "ok" and n_boot:
                b = _boot_thresholds(mat, taus, crit, n_boot, rng)
                if len(b) >= max(20, 0.5 * n_boot):
                    lo, hi = np.percentile(b, [2.5, 97.5])
            rows.append({
                "algorithm": algo, "beta": float(beta),
                "error_rate": error_rate(float(beta)),
                "order_kind": order, "criterion": crit,
                "tau_star": t, "ci_lo": lo, "ci_hi": hi, "status": status,
                "rho_at_tau_min": float(means[0]),
            })
    return pd.DataFrame(rows)


def critical_error_rate(long_df, criteria=(0.95, 0.90, 0.85),
                        tie_model="threshold", order_kind="random",
                        sorts=("heapsort", "quicksort", "bubble", "merge",
                               "insertion")):
    """The error rate at which the BEST sort fails the criterion even at a zero
    tie rate. Fisher-scale interpolation along the error-rate axis."""
    d = long_df[(long_df.tie_model == tie_model)
                & (long_df.order_kind == order_kind)
                & (long_df.algorithm.isin(sorts))]
    tau_min = d.tau.min()
    g = (d[d.tau == tau_min].groupby(["beta", "algorithm"]).rho_truth.mean()
         .reset_index())
    best = g.groupby("beta").rho_truth.max().reset_index()
    best["error_rate"] = [error_rate(b) for b in best.beta]
    best = best.sort_values("error_rate")
    e = best.error_rate.values
    z = fisher_z(best.rho_truth.values)
    out = []
    for crit in criteria:
        zc = fisher_z(crit)
        val, status = np.nan, "out of range"
        if z[0] < zc:
            status = "not met even by the best judge on the grid"
        elif z[-1] >= zc:
            status = "met even by the worst judge on the grid"
        else:
            for i in range(len(e) - 1):
                if z[i] >= zc > z[i + 1]:
                    frac = (z[i] - zc) / (z[i] - z[i + 1])
                    val = float(e[i] + frac * (e[i + 1] - e[i]))
                    status = "ok"
                    break
        out.append({"criterion": crit, "tau": float(tau_min),
                    "critical_error_rate": val, "status": status})
    return pd.DataFrame(out)
