# -*- coding: utf-8 -*-
"""Regenerate the article's figures from the saved result tables — no
experiment is re-run.

Every figure is fully determined by CSV files already saved in a
results/<time>_<nb_id>/ folder. For each figure the script takes the FRESHEST
such folder, reads its table(s) (read-only), rebuilds the figure with
ranklib.figures (the same builders the notebooks use, so labels never
diverge) and writes PNG (300 dpi) + SVG + a text file with the plotted
numbers into a NEW folder figures/<YYYYMMDD_HHMMSS>/. Nothing existing is
overwritten.

    python -X utf8 regen_figures.py           # rebuild all figures
    python -X utf8 regen_figures.py --list    # show the source folders only

Figures of the article: 1 fig_methods_vs_tie_rate, 2 fig2_degradation_beta*,
3 fig_thresholds_quicksort, 4 fig3_input_order, 5 fig2b_input_leak,
6 fig1_budget, A1 fig_thresholds_merge, A2 fig_thresholds_heapsort.
fig2c_suitability is an extra view of the simulation that is not in the
article. ZERO LLM calls.
"""
import datetime
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ranklib import figures                     # noqa: E402
from ranklib.runlog import save_figure          # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# figure name -> (nb_id, required CSV, builder, data_fn). The builder returns a
# figure; data_fn returns [(section_title, DataFrame)] with the exact numbers
# plotted, written next to the image as <fig>.txt. Extra inputs (topk / field
# points) are resolved once in prepare_inputs() and passed to BOTH.
SPEC = [
    ("fig_methods_vs_tie_rate", "nb02_algorithms_bench", "methods_long.csv",
     figures.fig_methods_vs_tie_rate, figures.data_fig_methods_vs_tie_rate),
    ("fig3_input_order", "nb03_input_order", "input_order_summary.csv",
     figures.fig3_input_order, figures.data_fig3_input_order),
    ("fig1_budget", "nb04_budget_curves", "budget_summary.csv",
     figures.fig1_budget, figures.data_fig1_budget),
    ("fig2_degradation", "nb05_simulation", "simulation_summary.csv",
     figures.fig2_degradation, figures.data_fig2_degradation),
    ("fig2b_input_leak", "nb05_simulation", "simulation_summary.csv",
     figures.fig2b_input_leak, figures.data_fig2b_input_leak),
    ("fig2c_suitability", "nb05_simulation", "simulation_summary.csv",
     figures.fig2c_suitability, figures.data_fig2c_suitability),
]
# The threshold figures need two tables and the run's config (algorithms,
# criteria, input order, shared axes).
THRESHOLDS = ("nb07_thresholds", "thresholds.csv", "critical_error_rates.csv")

FLOAT_FMT = "{:.3f}".format


def latest_run_dir(nb_id, required_csv):
    """Freshest results/<time>_<nb_id>/ containing required_csv (run_id starts
    with the timestamp, so lexicographic max == most recent). None if absent."""
    if not os.path.isdir(RESULTS):
        return None
    cands = []
    for name in os.listdir(RESULTS):
        d = os.path.join(RESULTS, name)
        if (os.path.isdir(d) and name.endswith("_" + nb_id)
                and os.path.exists(os.path.join(d, required_csv))):
            cands.append(name)
    if not cands:
        return None
    return os.path.join(RESULTS, sorted(cands)[-1])


def _read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare_inputs(fig_name, csv_name, src_dir):
    """Load the source table and resolve the figure's extra kwargs (shared by
    the builder and the data dumper). Returns (df, kwargs)."""
    df = _read_csv(os.path.join(src_dir, csv_name))
    kwargs = {}
    if fig_name == "fig1_budget":
        topk = 50
        cfg = os.path.join(src_dir, "config.json")
        if os.path.exists(cfg):
            with open(cfg, encoding="utf-8") as f:
                topk = json.load(f).get("TOPK", 50) or 50
        kwargs["topk"] = topk
    elif fig_name == "fig2_degradation":
        fp = os.path.join(src_dir, "field_points.csv")
        kwargs["field_df"] = _read_csv(fp) if os.path.exists(fp) else None
    return df, kwargs


def write_data_txt(path, fig_name, src_dir, csv_name, sections):
    """Write the plotted numbers as a readable text file (one section per
    table). `sections` = [(title, DataFrame)]. The row index is printed only
    for pivots (a real index); plain frames keep their default RangeIndex
    hidden."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Data behind {fig_name}\n")
        f.write(f"source: results/{os.path.basename(src_dir)}/{csv_name}\n")
        f.write(f"generated: "
                f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
        for title, table in sections:
            f.write(f"\n=== {title} ===\n")
            f.write(table.to_string(
                index=not isinstance(table.index, pd.RangeIndex),
                float_format=FLOAT_FMT))
            f.write("\n")


def threshold_figures(src, out_dir, plt, manifest):
    """fig_thresholds_<algorithm> for every algorithm of the nb07 run."""
    nb_id, th_csv, z_csv = THRESHOLDS
    th = _read_csv(os.path.join(src, th_csv))
    zdf = _read_csv(os.path.join(src, z_csv))
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    algorithms = tuple(cfg.get("ALGORITHMS", ("quicksort", "merge", "heapsort")))
    criteria = tuple(cfg.get("CRITERIA", (0.95, 0.90, 0.85)))
    order_kind = cfg.get("ORDER_KIND", "random")
    xlim = ylim = None
    if cfg.get("SHARED_AXES", True):
        xlim, ylim = figures.threshold_axis_limits(
            th, zdf, algorithms, order_kind=order_kind, criteria=criteria)
    n = 0
    for algo in algorithms:
        name = f"fig_thresholds_{algo}"
        fig = figures.fig_threshold_curves(th, zdf, algorithm=algo,
                                           order_kind=order_kind,
                                           criteria=criteria,
                                           xlim=xlim, ylim=ylim)
        save_figure(fig, out_dir, name)
        plt.close(fig)
        sections = figures.data_threshold_curves(th, zdf, algorithm=algo,
                                                 order_kind=order_kind,
                                                 criteria=criteria)
        write_data_txt(os.path.join(out_dir, f"{name}.txt"), name, src,
                       f"{th_csv} + {z_csv}", sections)
        manifest.append(f"{name}\t.png/.svg/.txt\t<- "
                        f"{os.path.basename(src)}/{th_csv} + {z_csv}")
        print(f"  ok   {name}.png/.svg/.txt  <- {os.path.basename(src)}")
        n += 1
    return n


def main():
    list_only = "--list" in sys.argv
    plan = []
    for fig_name, nb_id, csv_name, builder, data_fn in SPEC:
        src = latest_run_dir(nb_id, csv_name)
        plan.append((fig_name, nb_id, csv_name, builder, data_fn, src))
    th_src = latest_run_dir(THRESHOLDS[0], THRESHOLDS[1])

    print(f"{'figure':<24}{'source run folder':<40}")
    print("-" * 64)
    for fig_name, nb_id, csv_name, _b, _d, src in plan:
        print(f"{fig_name:<24}"
              + (os.path.basename(src) if src else f"MISSING ({nb_id}/{csv_name})"))
    print(f"{'fig_thresholds_*':<24}"
          + (os.path.basename(th_src) if th_src
             else f"MISSING ({THRESHOLDS[0]}/{THRESHOLDS[1]})"))
    if list_only:
        return

    out_dir = os.path.join(
        FIGURES, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    plt = figures.get_pyplot()
    manifest = []
    n_ok = 0
    for fig_name, nb_id, csv_name, builder, data_fn, src in plan:
        if src is None:
            manifest.append(f"{fig_name}\tSKIPPED (no source: {nb_id}/{csv_name})")
            print(f"  skip {fig_name}: source not found")
            continue
        df, kwargs = prepare_inputs(fig_name, csv_name, src)
        built = builder(df, **kwargs)
        if built is None:
            manifest.append(f"{fig_name}\tSKIPPED (builder opted out) "
                            f"<- {os.path.basename(src)}/{csv_name}")
            print(f"  skip {fig_name}: builder returned None "
                  f"(e.g. no '+corr.' input order)")
            continue
        data = data_fn(df, **kwargs)
        # A builder returns one Figure, or {suffix: Figure} when it splits into
        # several standalone figures (fig2_degradation: one per beta).
        items = built.items() if isinstance(built, dict) else [(None, built)]
        for suffix, fig in items:
            name = f"{fig_name}_{suffix}" if suffix else fig_name
            save_figure(fig, out_dir, name)
            plt.close(fig)
            sections = data.get(suffix) if isinstance(data, dict) else data
            write_data_txt(os.path.join(out_dir, f"{name}.txt"),
                           name, src, csv_name, sections or [])
            manifest.append(f"{name}\t.png/.svg/.txt\t<- "
                            f"{os.path.basename(src)}/{csv_name}")
            print(f"  ok   {name}.png/.svg/.txt  <- {os.path.basename(src)}")
            n_ok += 1
    if th_src is None:
        manifest.append("fig_thresholds_*\tSKIPPED (no source: "
                        f"{THRESHOLDS[0]}/{THRESHOLDS[1]})")
        print("  skip fig_thresholds_*: source not found")
    else:
        n_ok += threshold_figures(th_src, out_dir, plt, manifest)

    with open(os.path.join(out_dir, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("Regenerated figures + their source-data tables (<fig>.txt)\n")
        f.write(f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write("\n".join(manifest) + "\n")
    print(f"\n{n_ok} figures (+ .txt data) -> {out_dir}")


if __name__ == "__main__":
    main()
