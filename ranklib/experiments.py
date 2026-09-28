# -*- coding: utf-8 -*-
"""Experiment runners behind the simulation notebooks. ZERO LLM calls.

Each runner takes the notebook's PARAMS dict, computes, saves artifacts via
ranklib.runlog (tables CSV + figures PNG/SVG + report.txt + a runs_log.jsonl
record) and returns the key DataFrames for inline display.

Mapping to the article:
  run_simulation_experiment  nb05  experiment E7: Tables 4-6, Figures 2 and 5;
                             nb07  the competence x tie-rate sweep (NB_ID
                                   "nb07_competence_sweep")
  run_threshold_analysis     nb07  thresholds: Figure 3, Figures A1-A2,
                                   Tables A1-A3
"""
import os

import pandas as pd

from .runlog import Run
from . import figures
from .simulation import run_simulation, suitability_table


def _plt():
    return figures.get_pyplot()


# ================================ nb05 =======================================

def field_points_from_results(stats_csv, bench_csv):
    """Field overlay points for the simulation figures: (tie_rate, rho) per
    sort algorithm per matrix, read from nb01/nb02 output tables."""
    st = pd.read_csv(stats_csv)
    be = pd.read_csv(bench_csv)
    be = be[be.kind == "sort"]
    g = (be.groupby(["matrix", "method"])["rho_vs_gold"].mean().reset_index()
           .merge(st[["label", "tie_rate"]], left_on="matrix",
                  right_on="label"))
    return g[["matrix", "method", "tie_rate", "rho_vs_gold"]]


def run_simulation_experiment(params):
    """E7: the controlled simulation over the grid of Section 2.7. Returns
    (long_df, summary_df, suitability_df, run)."""
    quick = params.get("QUICK", False)
    kw = dict(
        taus=tuple(params.get("TAUS", tuple(round(0.1 * k, 1)
                                            for k in range(9)))),
        betas=tuple(params.get("BETAS", (0.5, 1.5, 4.0))),
        tie_models=tuple(params.get("TIE_MODELS", ("const", "threshold"))),
        order_kinds=tuple(params.get("ORDER_KINDS", ("random", "pos", "neg"))),
        reps=params.get("REPS", 30),
        n=params.get("N", 400),
        order_rho=params.get("ORDER_RHO", 0.5),
        alpha=params.get("ALPHA", 1.0),
        master_seed=params.get("MASTER_SEED", 20260705),
    )
    if quick:
        kw.update(taus=(0.0, 0.2, 0.4, 0.6, 0.8), betas=(1.5,),
                  tie_models=("threshold",), reps=params.get("REPS_QUICK", 3))
    run = Run(params.get("NB_ID", "nb05_simulation"),
              {**params, "effective_grid": kw})
    run.log(f"grid: {kw}")
    df = run_simulation(**kw)
    run.save_table(df, "simulation_long.csv")

    summary = (df.groupby(["tie_model", "beta", "order_kind", "tau",
                           "algorithm", "kind"])
               .agg(rho_truth_mean=("rho_truth", "mean"),
                    rho_truth_std=("rho_truth", "std"),
                    rho_input_mean=("rho_input", "mean"),
                    rho_input_std=("rho_input", "std"),
                    top_mean=("top", "mean"), bot_mean=("bot", "mean"),
                    tie_emp_mean=("tie_rate_emp", "mean"),
                    nontrans_mean=("nontransitivity", "mean"),
                    comparisons_mean=("comparisons", "mean"))
               .reset_index())
    run.save_table(summary, "simulation_summary.csv")
    suit = suitability_table(df, rho_thresholds=tuple(
        params.get("RHO_THRESHOLDS", (0.90, 0.95))))
    run.save_table(suit, "suitability.csv")
    run.log("\nSuitability thresholds (max τ under the worst input order):")
    run.log(suit.to_string(index=False))

    field = None
    if params.get("FIELD_STATS_CSV") and params.get("FIELD_BENCH_CSV"):
        field = field_points_from_results(params["FIELD_STATS_CSV"],
                                          params["FIELD_BENCH_CSV"])
        run.save_table(field, "field_points.csv")

    plt = _plt()

    # figure 2: degradation curves — one standalone figure per beta
    built = figures.fig2_degradation(summary, field_df=field)
    for suffix, fig in (built.items() if isinstance(built, dict)
                        else [(None, built)]):
        run.save_fig(fig, f"fig2_degradation_{suffix}" if suffix
                     else "fig2_degradation")
        plt.close(fig)

    # figure 2b: the mechanism — rho to the INPUT order under 'pos' ordering
    fig = figures.fig2b_input_leak(summary)
    if fig is not None:
        run.save_fig(fig, "fig2b_input_leak")
        plt.close(fig)

    # figure 2c: suitability heatmap (worst-case mean rho), beta mid
    fig = figures.fig2c_suitability(summary)
    run.save_fig(fig, "fig2c_suitability")
    plt.close(fig)

    run.finish({"rows": len(df), "grid": {k: str(v) for k, v in kw.items()},
                "quick": quick})
    return df, summary, suit, run


# ================================ nb07 =======================================

def run_threshold_analysis(params):
    """Applicability threshold of the sorts in practical units.

    Input: simulation_long.csv of the "judge competence x tie rate" sweep.
    Output: a table of thresholds with confidence intervals, the critical
    error rates, and the figure "judge error rate -> maximum tolerable tie
    rate" for a family of accuracy criteria. No LLM calls.
    """
    from .thresholds import critical_error_rate, estimate_thresholds

    sim_dir = params["SIM_RUN"]
    criteria = tuple(params.get("CRITERIA", (0.95, 0.90, 0.85)))
    algorithms = tuple(params.get("ALGORITHMS",
                                  ("quicksort", "merge", "heapsort")))
    order_kind = params.get("ORDER_KIND", "random")
    n_boot = params.get("N_BOOT", 2000)
    seed = params.get("BOOT_SEED", 20260918)

    long_path = os.path.join(sim_dir, "simulation_long.csv")
    run = Run("nb07_thresholds", params, inputs=[long_path])
    long_df = pd.read_csv(long_path)
    run.log(f"source: {os.path.basename(sim_dir)}, rows {len(long_df)}")

    th = estimate_thresholds(long_df, criteria=criteria, n_boot=n_boot,
                             seed=seed)
    th["errors_per_100"] = (th.error_rate * 100).round(2)
    run.save_table(th, "thresholds.csv")

    zrows = []
    for algo in sorted(set(th.algorithm)):
        z = critical_error_rate(long_df, criteria=criteria,
                                order_kind=order_kind, sorts=(algo,))
        z["algorithm"] = algo
        zrows.append(z)
    zdf = pd.concat(zrows, ignore_index=True)
    zdf["critical_errors_per_100"] = (zdf.critical_error_rate * 100).round(2)
    run.save_table(zdf, "critical_error_rates.csv")

    run.log("")
    run.log("Critical error rate (the criterion is not met even without ties):")
    piv = zdf[zdf.algorithm.isin(algorithms)].pivot(
        index="algorithm", columns="criterion",
        values="critical_errors_per_100")
    run.log(piv.to_string())

    # Shared axis limits: the per-algorithm figures are standalone, and
    # without them matplotlib scales each one to its own data, so heapsort
    # with the smallest thresholds would look as roomy as quicksort.
    xlim = ylim = None
    if params.get("SHARED_AXES", True):
        xlim, ylim = figures.threshold_axis_limits(
            th, zdf, algorithms, order_kind=order_kind, criteria=criteria)
        run.log(f"shared axes: x={tuple(round(v, 2) for v in xlim)}, "
                f"y={tuple(round(v, 3) for v in ylim)}")

    for algo in algorithms:
        run.log("")
        run.log(f"Maximum tolerable tie rate, {algo}, order {order_kind}:")
        sub = th[(th.algorithm == algo) & (th.order_kind == order_kind)]
        view = sub.pivot(index="errors_per_100", columns="criterion",
                         values="tau_star")
        run.log(view.round(3).to_string())
        fig = figures.fig_threshold_curves(th, zdf, algorithm=algo,
                                           order_kind=order_kind,
                                           criteria=criteria,
                                           xlim=xlim, ylim=ylim)
        run.save_fig(fig, f"fig_thresholds_{algo}")
        _plt().close(fig)

    run.finish({"algorithms": list(algorithms),
                "criteria": list(criteria),
                "critical_error_rates": {
                    a: {str(c): float(v) for c, v in row.items()}
                    for a, row in piv.iterrows()}})
    return th, zdf, run
