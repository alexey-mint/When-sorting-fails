# -*- coding: utf-8 -*-
"""Quality metrics for extracted rankings. ZERO LLM calls.

spearman / nontransitivity_sampled are verbatim ports from the authors' LLM
collector; ranks_from_order and topbot_overlap from its offline analysis
script. nontransitivity_exact is new: the full-enumeration cycle count over
ALL triples via matrix powers. It replaces the sampled estimate, whose
2000-triple samples gave 1.9 % vs 2.1 % on the same matrix (llama-v2).
"""
import random

import numpy as np
import pandas as pd


def ranks_from_order(ordered):
    """Strict 1..n ranks from a total order (1 = first = strongest)."""
    return {rid: pos + 1 for pos, rid in enumerate(ordered)}


def topbot_overlap(o1, o2, k=50):
    """(top-k, bottom-k) intersection sizes of two orderings."""
    return len(set(o1[:k]) & set(o2[:k])), len(set(o1[-k:]) & set(o2[-k:]))


def spearman(ranks_a: dict, ranks_b: dict) -> float:
    """Spearman ρ between two {id: rank} dicts over their common ids
    (via pandas; no scipy dependency)."""
    common = [k for k in ranks_a if k in ranks_b]
    if len(common) < 2:
        return float("nan")
    sa = pd.Series({k: ranks_a[k] for k in common})
    sb = pd.Series({k: ranks_b[k] for k in common})
    return float(sa.corr(sb, method="spearman"))


def input_order_ranks(ids_in_order):
    """{id: position+1} — 'rank' induced by the input sequence itself."""
    return {rid: pos + 1 for pos, rid in enumerate(ids_in_order)}


def nontransitivity_sampled(cache, row_ids, sample=2000, seed=0):
    """Fraction of sampled all-decisive triples (a,b,c) that form a cycle.
    Kept for comparability with the collector's reports (same RNG usage).
    Returns (rate, n_decisive_triples_in_sample)."""
    rnd = random.Random(seed)
    rids = list(row_ids)
    if len(rids) < 3:
        return 0.0, 0

    def winner(i, j):
        rec = cache.get(i, j)
        return rec.get("winner") if rec else None

    cyc = 0
    tot = 0
    for _ in range(sample):
        a, b, c = rnd.sample(rids, 3)
        wins = {a: 0, b: 0, c: 0}
        ok = True
        for x, y in ((a, b), (a, c), (b, c)):
            w = winner(x, y)
            if w is None:
                ok = False
                break
            wins[w] += 1
        if not ok:
            continue
        tot += 1
        if sorted(wins.values()) == [1, 1, 1]:   # cycle: each wins once
            cyc += 1
    return (cyc / tot if tot else 0.0), tot


def win_matrix(cache, row_ids):
    """Dense decisive-win adjacency W (float64): W[a, b] = 1 iff a beat b."""
    idx = {r: k for k, r in enumerate(row_ids)}
    rset = set(row_ids)
    n = len(row_ids)
    W = np.zeros((n, n))
    for rec in cache._data.values():
        a, b = rec["key"].split("_")
        a, b = int(a), int(b)
        if a not in rset or b not in rset:
            continue
        w = rec.get("winner")
        if w == a:
            W[idx[a], idx[b]] = 1.0
        elif w == b:
            W[idx[b], idx[a]] = 1.0
    return W


def nontransitivity_exact_dense(W):
    """Exact cycle rate over ALL triples of a decisive-win matrix W.
    #cyclic = trace(W^3)/3; #all-decisive triples = trace(D^3)/6, D = W + W^T.
    Returns (rate, n_decisive_triples, n_cycles)."""
    cycles = float(np.trace(W @ W @ W)) / 3.0
    D = W + W.T
    decisive = float(np.trace(D @ D @ D)) / 6.0
    rate = cycles / decisive if decisive else 0.0
    return rate, int(round(decisive)), int(round(cycles))


def nontransitivity_exact(cache, row_ids):
    """Exact cycle rate for a cache (full enumeration)."""
    return nontransitivity_exact_dense(win_matrix(cache, row_ids))
