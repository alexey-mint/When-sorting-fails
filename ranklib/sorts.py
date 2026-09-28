# -*- coding: utf-8 -*-
"""Comparison-based sorts driven by a judge comparator. ZERO LLM calls.

Every sort takes a winner lookup `win(i, j) -> +1 / -1 / 0` (+1: i beats j,
0: tie or unknown) and an id list IN INPUT ORDER, and returns the ordering
strongest-first. Tie handling is deterministic "do not swap" everywhere —
the property the article studies.

Provenance (semantics preserved verbatim):
  - bubble / insertion / merge: the offline analysis script of the authors'
    LLM collector (the sorts behind its first algorithm benchmark, E1);
  - heapsort / quicksort: the collector module (production algorithms;
    verdict tests "i"/"EQ" translated to win() sign tests; quicksort keeps the
    pivot RNG `random.Random(seed + 1)` and the in-place three-way partition,
    so a replay with the same seed reproduces the production trajectory).

Unlike the production code, these accept an ARBITRARY input order — required
by experiment E4 (input-order permutation control).
"""
import random


def bubble_sort(win, ids, seed=None):
    a = list(ids)
    n = len(a)
    for i in range(n - 1):
        swapped = False
        for k in range(n - 1 - i):
            if win(a[k], a[k + 1]) < 0:        # a[k] loses -> bubble the winner up
                a[k], a[k + 1] = a[k + 1], a[k]
                swapped = True
        if not swapped:                         # already ordered
            break
    return a


def insertion_sort(win, ids, seed=None):
    a = list(ids)
    for i in range(1, len(a)):
        cur = a[i]
        k = i - 1
        while k >= 0 and win(a[k], cur) < 0:    # left loses to cur -> shift right
            a[k + 1] = a[k]
            k -= 1
        a[k + 1] = cur
    return a


def merge_sort(win, ids, seed=None):
    a = list(ids)
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left = merge_sort(win, a[:mid])
    right = merge_sort(win, a[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if win(left[i], right[j]) >= 0:         # win or tie -> left first (stable)
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


def heapsort(win, ids, seed=None):
    """Ascending heapsort on 'relevance', returned reversed (strongest-first).
    In siftdown a tie counts as '>=': no exchange happens on ties."""
    a = list(ids)
    n = len(a)

    def siftdown(root, end):
        while True:
            l = 2 * root + 1
            r = 2 * root + 2
            if l >= end:
                return
            if r < end:
                bigger = l if win(a[l], a[r]) >= 0 else r   # "i"/"EQ" -> left
            else:
                bigger = l
            if win(a[bigger], a[root]) > 0:                 # "i" -> exchange
                a[root], a[bigger] = a[bigger], a[root]
                root = bigger
            else:
                return

    for i in range(n // 2 - 1, -1, -1):
        siftdown(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        siftdown(0, end)
    return a[::-1]


def quicksort(win, ids, seed=0):
    """Three-way quicksort (less / equal / more), random pivot from
    random.Random(seed + 1) — the exact production RNG. The 'equal' middle
    keeps segment order and is never re-processed."""
    a = list(ids)
    qrng = random.Random(seed + 1)

    def sort(lo, hi):
        if hi - lo <= 1:
            return
        pivot_pos = qrng.randrange(lo, hi)
        pivot = a[pivot_pos]
        candidate_ids = [a[k] for k in range(lo, hi) if k != pivot_pos]
        less, eq, more = [], [pivot], []
        for cid in candidate_ids:               # verdict = compare(cid, pivot)
            v = win(cid, pivot)
            if v > 0:
                more.append(cid)
            elif v < 0:
                less.append(cid)
            else:
                eq.append(cid)
        a[lo:hi] = less + eq + more
        sort(lo, lo + len(less))
        sort(lo + len(less) + len(eq), hi)

    sort(0, len(a))
    return a[::-1]


SORTS = {
    "bubble": bubble_sort,
    "insertion": insertion_sort,
    "merge": merge_sort,
    "heapsort": heapsort,
    "quicksort": quicksort,
}


def run_sort(name, win, ids, seed=0, record_pairs=False):
    """Run a sort with comparison counting. Returns (ordered, n_comparisons)
    or (ordered, n_comparisons, requested_keys) with record_pairs=True —
    the trajectory of unique canonical pairs the sort asked about (E6)."""
    cnt = {"n": 0}
    seen = dict() if record_pairs else None     # ordered unique canonical keys

    def counting(i, j):
        cnt["n"] += 1
        if record_pairs:
            seen[f"{min(i, j)}_{max(i, j)}"] = None
        return win(i, j)

    ordered = SORTS[name](counting, list(ids), seed=seed)
    if record_pairs:
        return ordered, cnt["n"], list(seen.keys())
    return ordered, cnt["n"]
