# -*- coding: utf-8 -*-
"""Connectivity of a random comparison graph: Erdős–Rényi vs Monte Carlo (nb11).

Backs the budget-scaling statements of the article (Sections 3.4, 3.5, 4, 5).
When only a random subset of pairs is compared, the Bradley–Terry model and
Rank Centrality need the comparison graph to be connected. With d comparisons
per object on average the graph is connected with probability about
exp(-N e^{-d}) (Erdős and Rényi, 1959), so a target probability P needs
d(P) = ln N - ln(-ln P) comparisons per object; ln N + 3 gives P ≈ 0.95.

The script checks the formula by Monte Carlo for N = 399 (our corpus) and
N = 200 (P-DD) and writes results/<time>_nb11_connectivity/ with
connectivity.csv, thresholds.csv, report.txt and config.json, plus a record in
results/runs_log.jsonl. The published run is results/20260926_134640_nb11_connectivity.

    python -X utf8 run_connectivity.py

Takes a few seconds. ZERO LLM calls.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ranklib.runlog import Run                                   # noqa: E402


def connected(n, a, b):
    """Is the undirected graph on n vertices with edges (a, b) connected?

    Min-label propagation with pointer jumping; labels never leave their
    component, so the graph is connected iff a single label survives."""
    lab = np.arange(n)
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        if np.array_equal(new, lab):
            return bool((lab == lab[0]).all())
        lab = new


def run_connectivity():
    """Connectivity of a random comparison graph: Erdős–Rényi vs Monte Carlo.

    Pairs are drawn uniformly without replacement, as in the budget curves.
    With d comparisons per object on average (d = 2k/N for k pairs), the graph
    is connected with probability ~ exp(-N e^{-d}); Erdős and Rényi (1959):
    k = (N/2)(ln N + c) pairs give P(connected) -> exp(-e^{-c}). Hence the
    per-object requirement for a target probability P is ln N - ln(-ln P)."""
    SIZES = (399, 200)
    FRACS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.10)
    TARGETS = (float(np.exp(-1.0)), 0.95, 0.99)
    DRAWS, SEED = 1000, 20260926
    run = Run("nb11_connectivity",
              {"purpose": "connectivity threshold of a random comparison graph, "
                          "for rescaling the budget to another dataset size",
               "formula": "P(connected) ~ exp(-N*exp(-d)), d = 2k/N; "
                          "d(P) = ln N - ln(-ln P)",
               "source": "Erdős, Rényi (1959), On random graphs I",
               "sampling": "k pairs uniformly without replacement (G(N, M))",
               "sizes": SIZES, "fractions": FRACS, "draws": DRAWS,
               "seed": SEED})
    rng = np.random.default_rng(SEED)
    rows, thr = [], []
    for N in SIZES:
        iu, ju = np.triu_indices(N, 1)
        TOT = len(iu)
        for frac in FRACS:
            k = int(round(TOT * frac))
            d = 2 * k / N
            conn, iso = 0, 0
            for _ in range(DRAWS):
                sel = rng.choice(TOT, size=k, replace=False)
                a, b = iu[sel], ju[sel]
                conn += connected(N, a, b)
                iso += int((np.bincount(np.concatenate([a, b]),
                                        minlength=N) == 0).sum())
            rows.append({"n_objects": N, "fraction_of_pairs": frac,
                         "n_pairs": k, "comparisons_per_object": d,
                         "c": d - np.log(N),
                         "p_connected_formula": float(np.exp(-N * np.exp(-d))),
                         "p_connected_observed": conn / DRAWS,
                         "isolated_expected": float(N * np.exp(-d)),
                         "isolated_observed_mean": iso / DRAWS,
                         "n_draws": DRAWS})
        for P in TARGETS:
            c = -np.log(-np.log(P))
            d = np.log(N) + c
            thr.append({"n_objects": N, "target_p_connected": P, "c": c,
                        "comparisons_per_object": d,
                        "fraction_of_pairs": d / (N - 1),
                        "n_pairs": N * d / 2})
    run.save_table(pd.DataFrame(rows), "connectivity.csv")
    run.save_table(pd.DataFrame(thr), "thresholds.csv")
    run.log("CONNECTIVITY OF A RANDOM COMPARISON GRAPH: FORMULA AND MONTE CARLO")
    run.log(pd.DataFrame(rows)[["n_objects", "fraction_of_pairs",
                                "comparisons_per_object", "p_connected_formula",
                                "p_connected_observed", "isolated_expected",
                                "isolated_observed_mean"]]
            .to_string(index=False, float_format="%.3f"))
    run.log("\nREQUIRED NUMBER OF COMPARISONS PER OBJECT d = ln N + c")
    run.log(pd.DataFrame(thr).to_string(index=False, float_format="%.4f"))
    worst = max(abs(r["p_connected_formula"] - r["p_connected_observed"])
                for r in rows)
    run.finish({"max_abs_formula_minus_observed": round(worst, 3),
                "d95": {str(t["n_objects"]): round(t["comparisons_per_object"], 2)
                        for t in thr if t["target_p_connected"] == 0.95}})


if __name__ == "__main__":
    run_connectivity()
