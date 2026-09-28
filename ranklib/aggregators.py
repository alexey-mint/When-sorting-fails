# -*- coding: utf-8 -*-
"""Statistical aggregators over a comparison matrix. ZERO LLM calls.

Ported from the authors' LLM collector (math and iteration/summation order
preserved, so the collector's own reports are reproduced bit-for-bit):
  - copeland_ranking / winrate_ranking / bradley_terry_strengths / bt_ranking /
    bt_strengths_se: from the collector module;
  - rank_centrality: from the collector's offline analysis script (the dense
    core is factored into rank_centrality_dense with the SAME operation
    sequence, reused by the simulation).

Every ranking function takes (cache, row_ids) — `cache` is anything with a
`._data` dict of records — and returns (scores, ranks, ordered_ids), ordered
strongest-first.
"""
import math
import random

import numpy as np


def _wins_losses(cache, row_ids):
    """Per-item (#decisive wins, #decisive losses) over cached pairs. Ties
    contribute to neither."""
    rset = set(row_ids)
    win = {r: 0 for r in row_ids}
    loss = {r: 0 for r in row_ids}
    for rec in cache._data.values():
        a, b = rec["key"].split("_")
        a, b = int(a), int(b)
        if a not in rset or b not in rset:
            continue
        w = rec.get("winner")
        if w is None:
            continue
        if w == a:
            win[a] += 1; loss[b] += 1
        else:
            win[b] += 1; loss[a] += 1
    return win, loss


def _ranks_from_scores(scores, row_ids):
    """Competition ranking (1 = highest score; ties share rank, next jumps).
    Deterministic tie-break by row id for a stable ordered list."""
    ordered = sorted(row_ids, key=lambda r: (-scores[r], r))
    ranks = {ordered[0]: 1}
    cur = 1
    for k in range(1, len(ordered)):
        if scores[ordered[k]] == scores[ordered[k - 1]]:
            ranks[ordered[k]] = cur
        else:
            cur = k + 1
            ranks[ordered[k]] = cur
    return ranks, ordered


def copeland_ranking(cache, row_ids):
    """Copeland score = #wins − #losses over all cached opponents. Exact on a
    complete round-robin. Returns (scores, ranks, ordered_ids)."""
    win, loss = _wins_losses(cache, row_ids)
    scores = {r: win[r] - loss[r] for r in row_ids}
    ranks, ordered = _ranks_from_scores(scores, row_ids)
    return scores, ranks, ordered


def winrate_ranking(cache, row_ids):
    """Win-rate = wins / (wins + losses); 0.5 when no decisive comparisons."""
    win, loss = _wins_losses(cache, row_ids)
    scores = {}
    for r in row_ids:
        dec = win[r] + loss[r]
        scores[r] = win[r] / dec if dec > 0 else 0.5
    ranks, ordered = _ranks_from_scores(scores, row_ids)
    return scores, ranks, ordered


def bradley_terry_strengths(cache, row_ids, alpha=1.0, max_iter=1000, tol=1e-9):
    """Bradley-Terry MLE strengths pi_i via regularized MM iteration (Hunter
    2004). P(i beats j) = pi_i / (pi_i + pi_j). Each cached pair counts as one
    game; a tie (winner=None) is scored as half a win to each side. `alpha`
    adds a weak prior — 2*alpha virtual games per item against an anchor of
    strength 1, winning alpha of them — which guarantees finite, connected
    estimates even on sparse/incomplete caches and fixes the overall scale.
    Returns {id: pi} normalized to geometric mean 1."""
    rset = set(row_ids)
    wins = {r: 0.0 for r in row_ids}           # decisive + 0.5*ties
    games = {r: {} for r in row_ids}           # opponent id -> #comparisons
    for rec in cache._data.values():
        a, b = rec["key"].split("_")
        a, b = int(a), int(b)
        if a not in rset or b not in rset:
            continue
        games[a][b] = games[a].get(b, 0) + 1
        games[b][a] = games[b].get(a, 0) + 1
        w = rec.get("winner")
        if w == a:
            wins[a] += 1.0
        elif w == b:
            wins[b] += 1.0
        else:
            wins[a] += 0.5
            wins[b] += 0.5

    pi = {r: 1.0 for r in row_ids}
    for _ in range(max_iter):
        new = {}
        for r in row_ids:
            denom = 2.0 * alpha / (pi[r] + 1.0)          # anchor games
            for opp, ng in games[r].items():
                denom += ng / (pi[r] + pi[opp])
            new[r] = (wins[r] + alpha) / denom if denom > 0 else pi[r]
        logmean = sum(math.log(v) for v in new.values()) / len(new)
        scale = math.exp(logmean)
        max_delta = 0.0
        for r in row_ids:
            new[r] /= scale
            d = abs(math.log(new[r]) - math.log(pi[r]))
            if d > max_delta:
                max_delta = d
        pi = new
        if max_delta < tol:
            break
    return pi


def bt_ranking(cache, row_ids, alpha=1.0):
    """Bradley-Terry ranking. score = theta = log(strength). Aggregates ALL
    cached verdicts statistically, independent of any traversal order."""
    pi = bradley_terry_strengths(cache, row_ids, alpha=alpha)
    scores = {r: math.log(pi[r]) for r in row_ids}
    ranks, ordered = _ranks_from_scores(scores, row_ids)
    return scores, ranks, ordered


def bt_strengths_se(cache, row_ids, pi=None, alpha=1.0):
    """Asymptotic standard errors of theta_i = log(pi_i) from the regularized
    Bradley-Terry Fisher information (weighted graph Laplacian + anchor term,
    inverted via numpy). Returns {id: se}."""
    if pi is None:
        pi = bradley_terry_strengths(cache, row_ids, alpha=alpha)
    theta = {r: math.log(pi[r]) for r in row_ids}
    idx = {r: k for k, r in enumerate(row_ids)}
    n = len(row_ids)
    H = np.zeros((n, n))
    rset = set(row_ids)
    for rec in cache._data.values():
        a, b = rec["key"].split("_")
        a, b = int(a), int(b)
        if a not in rset or b not in rset:
            continue
        p = 1.0 / (1.0 + math.exp(-(theta[a] - theta[b])))
        w = p * (1.0 - p)                       # one game per cached pair
        ia, ib = idx[a], idx[b]
        H[ia, ia] += w
        H[ib, ib] += w
        H[ia, ib] -= w
        H[ib, ia] -= w
    for r in row_ids:                           # anchor (2*alpha games vs theta0=0)
        p0 = 1.0 / (1.0 + math.exp(-theta[r]))
        H[idx[r], idx[r]] += 2.0 * alpha * p0 * (1.0 - p0)
    try:
        cov = np.linalg.inv(H)
        return {r: float(np.sqrt(max(cov[idx[r], idx[r]], 0.0))) for r in row_ids}
    except np.linalg.LinAlgError:
        return {r: float("nan") for r in row_ids}


# ------------------------------ rank centrality ------------------------------

def rank_centrality_dense(wins, games, damping=0.99, max_iter=5000, tol=1e-12):
    """Dense core of Rank Centrality (Negahban-Oh-Shah 2017 flavor, PageRank
    style): stationary distribution of a random walk that moves from i to j in
    proportion to how often j beat i; light teleport (1-damping) keeps the
    chain ergodic. `wins[a, b]` = #times a beat b; `games` = symmetric
    #comparisons. Operation sequence identical to the collector's version.
    Returns the stationary vector pi (numpy, length n)."""
    n = wins.shape[0]
    frac = np.zeros((n, n))                      # frac[i, j] = rate j beat i
    mask = games > 0
    frac[mask] = wins.T[mask] / games[mask]
    np.fill_diagonal(frac, 0.0)
    dmax = frac.sum(axis=1).max()
    if not dmax > 0:
        dmax = 1.0
    P = frac / dmax
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(P, 1.0 - P.sum(axis=1))     # row-stochastic
    P = damping * P + (1.0 - damping) / n        # teleport -> ergodic

    pi = np.full(n, 1.0 / n)
    Pt = P.T
    for _ in range(max_iter):
        nxt = Pt @ pi
        s = nxt.sum()
        if s > 0:
            nxt /= s
        if np.abs(nxt - pi).sum() < tol:
            pi = nxt
            break
        pi = nxt
    return np.abs(pi)


def rank_centrality(cache, row_ids, damping=0.99, max_iter=5000, tol=1e-12):
    """Spectral ranking on the win-graph. Returns (scores, ranks, ordered)."""
    idx = {r: k for k, r in enumerate(row_ids)}
    rset = set(row_ids)
    n = len(row_ids)
    wins = np.zeros((n, n))     # wins[a, b] = #times a beat b
    games = np.zeros((n, n))    # symmetric #comparisons between a and b
    for rec in cache._data.values():
        a, b = rec["key"].split("_")
        a, b = int(a), int(b)
        if a not in rset or b not in rset:
            continue
        ia, ib = idx[a], idx[b]
        games[ia, ib] += 1; games[ib, ia] += 1
        w = rec.get("winner")
        if w == a:
            wins[ia, ib] += 1
        elif w == b:
            wins[ib, ia] += 1
    pi = rank_centrality_dense(wins, games, damping=damping,
                               max_iter=max_iter, tol=tol)
    scores = {row_ids[k]: float(pi[k]) for k in range(n)}
    ranks, ordered = _ranks_from_scores(scores, row_ids)
    return scores, ranks, ordered
