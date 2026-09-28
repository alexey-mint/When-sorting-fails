# -*- coding: utf-8 -*-
"""Figure builders — the single source of every plot in this package.

Each function takes already-computed DataFrame(s) and returns a matplotlib
Figure; it does no computation and no saving. Both the notebook runners
(ranklib.experiments) and the offline regenerator (regen_figures.py) call
these, so figure labels live in exactly one place.

All axis labels/titles are English (article requirement).
"""
import numpy as np

# Line-style groups (names only; figures.py must not import the aggregator
# functions themselves — it only needs to know which methods are sorts).
SORT_NAMES = ("heapsort", "quicksort", "bubble", "merge", "insertion")
AGG_NAMES = ("copeland", "winrate", "bt", "rank_centrality")

# Draw the series key (direct labels with leader lines) on fig2_degradation.
# Set to False to get clean, label-free plots — e.g. when the article caption
# already lists the series. Per-call override: fig2_degradation(..., labels=False).
SHOW_SERIES_LABELS = True


def get_pyplot():
    """pyplot with a headless-safe backend outside Jupyter."""
    import os
    import matplotlib
    if not os.environ.get("JPY_PARENT_PID"):      # headless -> no GUI backend
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ------------------------------- nb02 ----------------------------------------

def fig_methods_vs_tie_rate(long_df):
    """rho vs tie rate per algorithm (needs columns method, matrix, tie_rate,
    rho_vs_gold)."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    stat = (long_df.groupby(["method", "matrix", "tie_rate"])["rho_vs_gold"]
                   .agg(["mean", "min", "max"]).reset_index())
    for method, g in stat.groupby("method"):
        g = g.sort_values("tie_rate")
        style = "-o" if method in SORT_NAMES else "--s"
        ax.plot(g.tie_rate * 100, g["mean"], style, label=method, alpha=0.8)
    ax.set_xlabel("Tie rate, %")
    ax.set_ylabel("Spearman's ρ vs. matrix reference")
    ax.axhline(0, color="gray", lw=0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Algorithm accuracy by tie-rate level (real matrices)")
    return fig


# ------------------------------- nb03 ----------------------------------------

def fig3_input_order(summary_df):
    """(A) rho_input vs tie rate for sorts; (B) rho_gold spread under
    permutations vs the actual file order."""
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for algo, g in summary_df[summary_df.kind == "sort"].groupby("algorithm"):
        g = g.sort_values("tie_rate")
        ax.errorbar(g.tie_rate * 100, g.rho_input_mean, yerr=g.rho_input_std,
                    marker="o", capsize=3, label=algo)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Tie rate, %")
    ax.set_ylabel("ρ(extracted ranking, input order)")
    ax.set_title("A. Input-order reproduction\n"
                 "(mean ± σ over permutations)")
    ax.legend(fontsize=8)
    ax = axes[1]
    sorts = summary_df[summary_df.kind == "sort"]
    x = np.arange(len(sorts))
    ax.errorbar(x, sorts.rho_gold_mean, yerr=sorts.rho_gold_std, fmt="o",
                capsize=3, label="permutations (mean ± σ)")
    ax.scatter(x, sorts.rho_gold_actual, marker="x", color="red", zorder=3,
               label="actual file order")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.algorithm}\n{r.matrix}" for r in
                        sorts.itertuples()], fontsize=6, rotation=90)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("ρ(extracted ranking, reference)")
    ax.set_title("B. Accuracy under random input orders")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------- nb04 ----------------------------------------

def fig1_budget(summary_df, topk=50):
    """rho and bottom-K overlap vs budget share (needs matrix, rate, rho_mean,
    rho_min, rho_max, bot_mean, tie_rate)."""
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, g in summary_df.groupby("matrix"):
        g = g.sort_values("rate")
        tie = g.tie_rate.iloc[0]
        lbl = f"{label} ({tie:.0%} ties)"
        axes[0].errorbar(g.rate * 100, g.rho_mean,
                         yerr=[g.rho_mean - g.rho_min, g.rho_max - g.rho_mean],
                         marker="o", capsize=3, label=lbl)
        axes[1].plot(g.rate * 100, g.bot_mean, "-o", label=lbl)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Share of full-matrix pairs, % (log scale)")
    axes[0].set_ylabel("ρ vs. reference (mean and range over 3 runs)")
    axes[0].axhline(0.95, color="gray", ls=":", lw=1)
    axes[0].set_title("A. Random-pair accuracy + Bradley-Terry")
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Share of full-matrix pairs, % (log scale)")
    axes[1].set_ylabel(f"Overlap of bottom {topk} positions (out of {topk})")
    axes[1].axhline(40, color="gray", ls=":", lw=1)
    axes[1].set_title("B. Bottom-ranking convergence")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------- nb05 ----------------------------------------

def _sim_tie_model(summary_df):
    """The tie model to plot: 'threshold' if present, else the first."""
    models = list(summary_df.tie_model.unique())
    return "threshold" if "threshold" in models else models[0]


def _sim_random_subset(summary_df):
    """Rows behind the degradation panels: the plotted tie model, random input
    order. Shared by fig2_degradation and data_fig2_degradation so the drawn
    lines and the dumped tables come from one selection."""
    tm = _sim_tie_model(summary_df)
    return summary_df[(summary_df.tie_model == tm)
                      & (summary_df.order_kind == "random")]


def _algo_order(index):
    """Canonical algorithm order for data dumps (aggregators first, then sorts;
    anything unknown appended). Readability only — never used for plotting."""
    present = list(index)
    ordered = [a for a in (AGG_NAMES + SORT_NAMES) if a in present]
    ordered += [a for a in present if a not in ordered]
    return ordered


def _label_lines_directly(ax, series, gap=0.062, pad=0.055, room=0.28,
                          fontsize=8):
    """Direct labels with leader lines at the right end of each curve, instead
    of a legend box. `series` = [(name, color, x_end, y_end)].

    Labels are spread vertically (greedy top-down, minimum `gap` between them)
    because several series end within ~0.005 of each other — the aggregators
    all sit at ρ≈0.99, and bubble/insertion coincide exactly. The leader line
    ties each spread-out label back to its true endpoint, so nothing is
    misrepresented. `room` reserves x-space for the text on the right."""
    if not series:
        return
    series = sorted(series, key=lambda s: -s[3])        # top curve first
    x_end = max(s[2] for s in series)
    x_text = x_end + pad
    y_labels = []
    for k, (_name, _color, _x, y) in enumerate(series):
        y_labels.append(y if k == 0 else min(y, y_labels[-1] - gap))
    for (name, color, x, y), y_lab in zip(series, y_labels):
        ax.annotate(name, xy=(x, y), xytext=(x_text, y_lab),
                    color=color, fontsize=fontsize, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.6,
                                    shrinkA=0, shrinkB=1))
    # Widen only to make room for the text, keeping the ticks inside the data
    # range — otherwise matplotlib re-ticks and invents τ values (1.0, 1.2)
    # that were never simulated.
    lo, hi = ax.get_xlim()
    keep = [t for t in ax.get_xticks() if lo <= t <= x_end + 1e-9]
    ax.set_xlim(lo, max(hi, x_text + room))
    if keep:
        ax.set_xticks(keep)


def fig2_degradation(summary_df, field_df=None, labels=None):
    """Degradation curves, ONE STANDALONE FIGURE PER competence level beta
    (three panels in a row came out too small on the article page).

    Returns {suffix: Figure}, e.g. {"beta0.5": fig, "beta1.5": fig,
    "beta4": fig}; callers save each as <fig_name>_<suffix>.

    Series identification: direct labels with leader lines at the right end of
    every curve, drawn on EACH figure (they are placed independently in the
    article, so each must stand on its own). Set `labels=False` — or flip the
    module constant SHOW_SERIES_LABELS above — to drop them. A legend box was
    worse here: neighbouring series share similar colors, the aggregator
    curves lie on top of each other, and legend(loc="best") drifted onto data.

    All figures share one y-range, computed over every beta, so the three
    standalone plots stay visually comparable.

    `field_df` is accepted for signature symmetry with data_fig2_degradation
    but is NOT drawn: the real-matrix overlay (black stars + labels) cluttered
    the panel and its annotations were unreadable. The field points are still
    exported as a section of the companion <fig>.txt.
    """
    if labels is None:
        labels = SHOW_SERIES_LABELS
    plt = get_pyplot()
    tm = _sim_tie_model(summary_df)
    sub = _sim_random_subset(summary_df)
    betas = sorted(sub.beta.unique())

    # One y-range for all betas: separate figures must not autoscale apart.
    std = sub.rho_truth_std.fillna(0.0)
    y_lo = float((sub.rho_truth_mean - std).min())
    y_hi = float((sub.rho_truth_mean + std).max())
    margin = 0.05 * max(y_hi - y_lo, 1e-6)

    out = {}
    for beta in betas:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        endpoints = []                  # (name, color, x_end, y_end)
        for algo, g in sub[sub.beta == beta].groupby("algorithm"):
            g = g.sort_values("tau")
            style = "--s" if algo in AGG_NAMES else "-o"
            cont = ax.errorbar(g.tau, g.rho_truth_mean, yerr=g.rho_truth_std,
                               fmt=style, ms=3, capsize=2, label=algo,
                               alpha=0.85)
            if len(g):
                endpoints.append((algo, cont.lines[0].get_color(),
                                  float(g.tau.iloc[-1]),
                                  float(g.rho_truth_mean.iloc[-1])))
        ax.set_ylim(y_lo - margin, y_hi + margin)
        ax.set_xlabel("Tie rate τ")
        ax.set_ylabel("ρ vs. true order (mean ± σ)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"Degradation curves, competence β = {beta:g}\n"
                     f"(random input order, tie model: {tm})")
        if labels:
            _label_lines_directly(ax, endpoints)
        fig.tight_layout()
        out[f"beta{beta:g}"] = fig
    return out


def fig2b_input_leak(summary_df):
    """Mechanism: rho to the INPUT order under the '+corr.' ('pos') ordering,
    beta = middle level. Returns None if no 'pos' order in the data."""
    tm = _sim_tie_model(summary_df)
    if "pos" not in set(summary_df.order_kind.unique()):
        return None
    betas = sorted(summary_df[(summary_df.tie_model == tm)
                              & (summary_df.order_kind == "random")].beta.unique())
    if not betas:
        betas = sorted(summary_df.beta.unique())
    beta_mid = betas[len(betas) // 2]
    sub2 = summary_df[(summary_df.tie_model == tm)
                      & (summary_df.beta == beta_mid)
                      & (summary_df.order_kind == "pos")
                      & (summary_df.kind == "sort")]
    if sub2.empty:
        return None
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for algo, g in sub2.groupby("algorithm"):
        g = g.sort_values("tau")
        ax.errorbar(g.tau, g.rho_input_mean, yerr=g.rho_input_std,
                    fmt="-o", ms=3, capsize=2, label=algo)
    ax.set_xlabel("Tie rate τ")
    ax.set_ylabel("ρ(extracted ranking, input order)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"Mechanism: input-order reproduction "
                 f"(β = {beta_mid:g}, '+corr.' order)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig2c_suitability(summary_df):
    """Suitability heatmap: worst-over-input-order mean rho vs truth, beta =
    middle level."""
    plt = get_pyplot()
    tm = _sim_tie_model(summary_df)
    beta_mid = sorted(summary_df.beta.unique())[
        len(summary_df.beta.unique()) // 2]
    worst = (summary_df[(summary_df.tie_model == tm)
                        & (summary_df.beta == beta_mid)]
             .groupby(["algorithm", "tau"])["rho_truth_mean"].min()
             .reset_index())
    piv = worst.pivot(index="algorithm", columns="tau",
                      values="rho_truth_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8)
    for (yy, xx), val in np.ndenumerate(piv.values):
        ax.text(xx, yy, f"{val:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, label="ρ vs. truth (worst input order)")
    ax.set_xlabel("Tie rate τ")
    ax.set_title(f"Suitability map (β = {beta_mid:g}, model: {tm})")
    fig.tight_layout()
    return fig


# ======================= data behind each figure =============================
# Each data_<fig>() returns a list of (section_title, DataFrame) — exactly the
# numbers the matching builder plots. Used by _regen_figures.py to write a
# <fig>.txt next to the image. The selection/pivot expressions mirror the
# builders (shared helpers where the selection is non-trivial), so the dumped
# tables equal what is drawn.

def data_fig_methods_vs_tie_rate(long_df):
    stat = (long_df.groupby(["method", "matrix", "tie_rate"])["rho_vs_gold"]
                   .agg(["mean", "min", "max"]).reset_index()
                   .sort_values(["method", "tie_rate"]))
    return [("rho_vs_gold per method x matrix (mean/min/max); "
             "each plotted line = one method across matrices", stat)]


def data_fig3_input_order(summary_df):
    cols = ["matrix", "tie_rate", "algorithm", "rho_input_mean", "rho_input_std",
            "rho_gold_mean", "rho_gold_std", "rho_input_actual",
            "rho_gold_actual"]
    sorts = (summary_df[summary_df.kind == "sort"][cols]
             .sort_values(["algorithm", "tie_rate"]))
    return [("sorts: input-order reproduction (panel A: rho_input_*) and "
             "accuracy under permutations (panel B: rho_gold_*)", sorts)]


def data_fig1_budget(summary_df, topk=50):
    cols = [c for c in ["matrix", "tie_rate", "rate", "n_pairs", "rho_mean",
                        "rho_min", "rho_max", "top_mean", "bot_mean"]
            if c in summary_df.columns]
    curve = summary_df[cols].sort_values(["matrix", "rate"])
    return [(f"budget curve: rho vs. reference (rho_mean/min/max) and bottom-"
             f"{topk} overlap (bot_mean) per matrix x pair-share", curve)]


def data_fig2_degradation(summary_df, field_df=None):
    """Per-beta pivots (algorithm x tau) of the plotted mean rho-to-truth and
    its sigma (error bars), for the plotted tie model and random input order.

    Keyed by the same suffix fig2_degradation uses, so every standalone figure
    gets its own companion table file: {"beta0.5": sections, ...}."""
    sub = _sim_random_subset(summary_df)
    tm = _sim_tie_model(summary_df)
    out = {}
    for beta in sorted(sub.beta.unique()):
        g = sub[sub.beta == beta]
        sections = []
        for value, tag in (("rho_truth_mean", "mean rho to true order (line)"),
                           ("rho_truth_std", "sigma over repetitions "
                                             "(error bars)")):
            piv = g.pivot(index="algorithm", columns="tau", values=value)
            piv = piv.reindex(_algo_order(piv.index))
            sections.append((f"beta = {beta:g}, tie model {tm}, random input "
                             f"order — {tag}", piv))
        if field_df is not None and len(field_df):
            sections.append(("field points for comparison with the curves "
                             "above (real matrices; not drawn on the figure): "
                             "tie_rate vs rho_vs_gold per sort x matrix",
                             field_df))
        out[f"beta{beta:g}"] = sections
    return out


def data_fig2b_input_leak(summary_df):
    if "pos" not in set(summary_df.order_kind.unique()):
        return None
    tm = _sim_tie_model(summary_df)
    betas = sorted(_sim_random_subset(summary_df).beta.unique())
    if not betas:
        betas = sorted(summary_df.beta.unique())
    beta_mid = betas[len(betas) // 2]
    sub2 = summary_df[(summary_df.tie_model == tm)
                      & (summary_df.beta == beta_mid)
                      & (summary_df.order_kind == "pos")
                      & (summary_df.kind == "sort")]
    if sub2.empty:
        return None
    sections = []
    for value, tag in (("rho_input_mean", "mean rho to input order (line)"),
                       ("rho_input_std", "sigma (error bars)")):
        piv = sub2.pivot(index="algorithm", columns="tau", values=value)
        piv = piv.reindex(_algo_order(piv.index))
        sections.append((f"beta = {beta_mid:g}, tie model {tm}, '+corr.' input "
                         f"order, sorts — {tag}", piv))
    return sections


def data_fig2c_suitability(summary_df):
    tm = _sim_tie_model(summary_df)
    beta_mid = sorted(summary_df.beta.unique())[
        len(summary_df.beta.unique()) // 2]
    worst = (summary_df[(summary_df.tie_model == tm)
                        & (summary_df.beta == beta_mid)]
             .groupby(["algorithm", "tau"])["rho_truth_mean"].min()
             .reset_index())
    piv = worst.pivot(index="algorithm", columns="tau",
                      values="rho_truth_mean")
    piv = piv.reindex(_algo_order(piv.index))
    return [(f"worst-over-input-order mean rho to truth (beta = {beta_mid:g}, "
             f"tie model {tm}) — the heatmap cells", piv)]


# --------------------- thresholds in practitioner units ----------------------

CRITERION_STYLE = {0.95: ("-o", "#1f77b4"), 0.90: ("-s", "#d62728"),
                   0.85: ("-^", "#2ca02c")}


def _threshold_series(th_df, z_df, algorithm, order_kind, criterion):
    """(x, y, lo, hi) for one criterion: error rate -> max tolerable tie rate.
    The curve is closed at the critical error rate z, where the algorithm only
    just reaches the criterion with NO ties at all, so the tolerable tie rate
    there is exactly zero. That is what makes the curve end on the axis."""
    g = th_df[(th_df.algorithm == algorithm)
              & (th_df.order_kind == order_kind)
              & (th_df.criterion == criterion)
              & (th_df.status == "ok")].sort_values("error_rate")
    x = list(g.error_rate.values * 100.0)
    y = list(g.tau_star.values)
    lo = list(g.ci_lo.values)
    hi = list(g.ci_hi.values)
    zrow = z_df[(z_df.algorithm == algorithm) & (z_df.criterion == criterion)]
    if len(zrow) and np.isfinite(zrow.critical_error_rate.values[0]):
        z = float(zrow.critical_error_rate.values[0]) * 100.0
        if not x or z > x[-1]:
            x.append(z); y.append(0.0); lo.append(np.nan); hi.append(np.nan)
    return np.array(x), np.array(y), np.array(lo), np.array(hi)


def threshold_axis_limits(th_df, z_df, algorithms, order_kind="random",
                          criteria=(0.95, 0.90, 0.85), pad=0.05):
    """Common (xlim, ylim) spanning several algorithms.

    Each algorithm is drawn as its own standalone figure, so without shared
    limits matplotlib autoscales them apart and heapsort — whose thresholds are
    the smallest — would look as roomy as quicksort. Pass the result to
    fig_threshold_curves to keep the set visually comparable."""
    xs, ys = [], []
    for algo in algorithms:
        for crit in criteria:
            x, y, lo, hi = _threshold_series(th_df, z_df, algo, order_kind,
                                             crit)
            if not len(x):
                continue
            xs.extend(x.tolist())
            ys.extend(y.tolist())
            ys.extend(hi[np.isfinite(hi)].tolist())
    if not xs:
        return None, None
    xlo, xhi = min(xs), max(xs)
    yhi = max(ys)
    dx = pad * max(xhi - xlo, 1e-9)
    return (xlo - dx, xhi + dx), (0.0, yhi * (1.0 + pad))


def fig_threshold_curves(th_df, z_df, algorithm="quicksort",
                         order_kind="random", criteria=(0.95, 0.90, 0.85),
                         xlim=None, ylim=None):
    """Maximum tolerable tie rate versus judge error rate, one curve per
    accuracy criterion. A threshold is undefined without naming the criterion,
    so the criterion is a curve, not a caption.

    `xlim` / `ylim` fix the axes so figures for different algorithms can be
    compared side by side; None autoscales each figure on its own data
    (see threshold_axis_limits)."""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for crit in criteria:
        style, color = CRITERION_STYLE.get(crit, ("-o", None))
        x, y, lo, hi = _threshold_series(th_df, z_df, algorithm, order_kind,
                                         crit)
        if not len(x):
            continue
        ax.plot(x, y, style, color=color, ms=4, lw=1.8,
                label=f"ρ ≥ {crit:g}")
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=0.18,
                            linewidth=0)
        ax.plot([x[-1]], [y[-1]], "o", color=color, ms=7, mfc="white",
                mew=1.6, zorder=5)
    ax.set_xlabel("Judge error rate ε, %")
    ax.set_ylabel("Maximum tolerable tie rate τ")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", title="Accuracy criterion", fontsize=9,
              title_fontsize=9)
    ax.set_title("\n".join([
        f"Where {algorithm} stops being usable",
        f"(simulation, {order_kind} input order; open circle = the error rate",
        "at which the criterion fails even with no ties)"]))
    fig.tight_layout()
    return fig


def data_threshold_curves(th_df, z_df, algorithm="quicksort",
                          order_kind="random", criteria=(0.95, 0.90, 0.85)):
    sections = []
    for crit in criteria:
        g = th_df[(th_df.algorithm == algorithm)
                  & (th_df.order_kind == order_kind)
                  & (th_df.criterion == crit)].sort_values("error_rate").copy()
        g["errors_per_100"] = (g.error_rate * 100).round(1)
        cols = ["errors_per_100", "tau_star", "ci_lo", "ci_hi", "status",
                "rho_at_tau_min"]
        sections.append((f"{algorithm}, {order_kind} input order, "
                         f"criterion rho >= {crit:g}: max tolerable tie rate",
                         g[cols]))
    z = z_df[z_df.algorithm == algorithm].copy()
    z["critical_errors_per_100"] = (z.critical_error_rate * 100).round(2)
    sections.append((f"{algorithm}: critical error rate — the criterion fails "
                     f"even at zero tie rate",
                     z[["criterion", "critical_errors_per_100", "status"]]))
    return sections
