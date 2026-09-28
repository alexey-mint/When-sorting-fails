# Scripts and notebooks

Run everything from the package root. Only `tournament.ipynb` (through
`collector.py`) calls a language model; everything else works on stored data.
Each simulation run writes a NEW folder `results/<YYYYMMDD_HHMMSS>_<id>/` and
appends one line to `results/runs_log.jsonl`; nothing that already exists is
overwritten. The published runs stay next to your own runs, so the two can be
compared file by file.

| File | What it does | Article | Runtime | Output |
|---|---|---|---|---|
| `tournament.ipynb` + `collector.py` | Collects a complete tournament with an LLM judge (Phase A) and analyses a matrix without the model (Phase B) | Sections 2.2, 2.3, 2.8 (all seven matrices) | Phase A: hours; Phase B: ~1 min | `tournaments/` |
| `nb05_simulation.ipynb` | Controlled simulation, experiment E7 | Tables 4–6, Figures 2 and 5 | ~25 min (full grid) | `results/<time>_nb05_simulation/` |
| `nb07_thresholds.ipynb` | Judge competence × tie rate sweep; applicability thresholds of the sorts | Figure 3, Figures A1–A2, Tables A1–A3 | ~30 s; the sweep itself ~2.5 h | `results/<time>_nb07_competence_sweep/`, `results/<time>_nb07_thresholds/` |
| `run_connectivity.py` | Connectivity of a random comparison graph (Erdős–Rényi bound checked by Monte Carlo) | Sections 3.4, 3.5, 4 | seconds | `results/<time>_nb11_connectivity/` |
| `regen_figures.py` | Rebuilds the figures from the saved tables | Figures 1–6, A1, A2 | ~1 min | `figures/<time>/` |
| `prepare_pdd.py` | Converts the P-DD release file to the collector's input schema | Section 4 | seconds | `data/pdd_context_200.csv` |

## Environment

Python 3.10 with the packages in `requirements.txt` (numpy 2.2.2, pandas
2.2.3, matplotlib 3.10.0; Jupyter for the notebooks). These are the versions
of the published runs. With them, on Windows, a rerun reproduces the published
tables byte for byte (see "Verification" below). With other versions small
differences in the last digits are possible.

## tournament.ipynb and collector.py — collecting a tournament

The notebook the authors used to collect the seven matrices, cleaned up; the
logic lives in `collector.py`. All settings are in the first code cell.

- `PRESET` — one of the seven matrices. Each preset fixes:
  - the instruction (`prompts/`);
  - the presentation: both orders, or one call per pair in a random order;
  - the note for top-level comments;
  - the input file and the judge.
- `COLLECT` — run Phase A (model calls) or skip it.
- `ANALYZE` — analyse the matrix just collected (`"collected"`) or the
  published one in `data/matrices/` (`"published"`).
- `N_ROWS` — `None` for all objects, or a small number for a first test.
- `SERVER_URL` / `OLLAMA_URL` — the model servers: llama-server with its
  OpenAI-compatible endpoint for Gemma4:26, and Ollama for llama3.1:8b.

**Phase A** puts every pair to the judge and appends each verdict to
`tournaments/<input>_REF_<backend>_<model>_<instruction>_<order>.cache.jsonl`,
in the format of the published matrices.

- A status file and `tournaments/runs_log.jsonl` make the collection
  resumable: after an interruption, run the cell again.
- Requests use temperature 0, seed 0 and at most 8 answer tokens; requests to
  llama-server ask for at least 16.
- For llama-server the request also disables "thinking" on reasoning chat
  templates.
- Failed calls and answers without a letter count as "equal" and are listed in
  `tournaments/failures.log`.

The Russian original of the corpus is not published, so GA2, GA1, GR1 and LR2
cannot be collected again. The English presets (GA2_en_ru, GA2_en) and PDD
can, with the same model served locally.

**Phase B** needs no model. It builds the reference rankings of the full
matrix: Copeland, win rate and Bradley–Terry, with standard errors. It replays
heapsort and quicksort (pivot seeds 0, 1, 2) on the matrix, where every
comparison is a lookup. It reports Spearman's ρ to the Copeland reference and
the number of comparisons each method used. It also computes the
non-transitivity rate, both sampled and exact. The report and the reference
ranking are saved to `tournaments/`. The last cell compares a new collection
with the published matrix: agreement of the verdicts, tie rates and ρ between
the two references.

`collector.py` is the part of the authors' collection code (pairwise_core
2.5) needed for complete tournaments. The instructions come from
`prompts/registry.json`, which also sets the language of the note for
top-level comments; only the two backends used for the published matrices are
kept. With a stub judge it sends the same prompts and stores the same records
as the original for all seven configurations.

## nb05_simulation.ipynb — controlled simulation (E7)

A synthetic judge ranks 400 objects with hidden N(0, 1) strengths. The judge
is described by its competence β and its tie rate τ, with two tie models
(constant and threshold). The input order of the objects is random, positively
correlated with the truth, or the mirror of the latter. Every realization
generates one full verdict matrix and runs all nine algorithms on it. The
design is described in Section 2.7 of the article.

All parameters are in the first code cell (`PARAMS`):

- `QUICK=True` is a smoke test (one β, five τ levels, three repetitions).
  `QUICK=False` is the full grid of the article, 9 τ × 2 tie models × 3 β ×
  3 input orders × 30 repetitions = 4,860 realizations.
- `MASTER_SEED=20260705` is the seed of the published run. Every realization
  draws its own random generator from the master seed, the index of its
  combination in the grid and the repetition number, so the same grid with the
  same seed reproduces the same numbers.
- `FIELD_STATS_CSV` / `FIELD_BENCH_CSV` point to the published tables of the
  real tournaments (runs nb01 and nb02). Their tie rates and accuracies are
  written to `field_points.csv` for comparison with the simulated curves.

Outputs:

- `simulation_long.csv` — one row per realization × algorithm.
- `simulation_summary.csv` — means and standard deviations per combination.
  Tables 4–6 of the article are taken from this file: threshold tie model,
  random input order, β = 4 / 1.5 / 0.5.
- `suitability.csv` — the largest τ at which an algorithm still clears ρ ≥ 0.90
  or 0.95 under the worst input order.
- Figures and `report.txt`.

## nb07_thresholds.ipynb — thresholds in practical units

β is not identifiable on its own, so this notebook expresses the judge's
competence as its error rate ε, the share of decisive verdicts given to the
weaker object. `ranklib.simulation.error_rate` and `beta_for_error` convert
between the two.

**Step 1, the sweep.** The simulation of nb05 on a finer grid: 11 error rates
(2–18 %) × 9 tie rates (0–0.4) × 3 input orders × 30 repetitions, threshold
tie model, seed 20260918. The seed differs from nb05, so the sweep is also an
independent replication of the main run. By default the notebook reads the
published sweep `results/20260918_120914_nb07_competence_sweep`; set
`RUN_SWEEP = True` to recompute it (about 2.5 hours).

**Step 2, the thresholds.** For every algorithm, error rate and input order
the notebook finds the largest tie rate at which the mean Spearman correlation
with the truth still meets the criterion (ρ ≥ 0.95, 0.90, 0.85). The crossing
is interpolated on the Fisher scale, with a bootstrap 95 % confidence interval
over the repetitions (2,000 resamples). It also finds the critical error rate,
at which the criterion fails even without ties.

Outputs:

- `thresholds.csv` — Tables A1–A3.
- `critical_error_rates.csv`.
- `fig_thresholds_<algorithm>` — Figure 3 (quicksort), Figures A1 (merge) and
  A2 (heapsort).
- `report.txt`.

## run_connectivity.py — how many random pairs are enough

```bash
python -X utf8 run_connectivity.py
```

With k random pairs out of N objects each object takes part in d = 2k/N
comparisons on average. The comparison graph is connected with probability
about exp(−N·e^(−d)) (Erdős and Rényi, 1959), so a target probability P needs
d = ln N − ln(−ln P) comparisons per object; ln N + 3 gives P ≈ 0.95. The
script checks this by Monte Carlo for N = 399 and N = 200: 1,000 draws per
point, fractions of pairs from 1 % to 10 %, seed 20260926.

Outputs:

- `connectivity.csv` — the formula against the observed share of connected
  graphs, and the expected against the observed number of isolated objects.
- `thresholds.csv` — the required comparisons per object for P = 1/e, 0.95
  and 0.99.
- `report.txt`.

This is the basis for rescaling the budget of Section 3.4 to a set of another
size.

## regen_figures.py — figures from the saved tables

```bash
python -X utf8 regen_figures.py          # rebuild all figures
python -X utf8 regen_figures.py --list   # only show which run feeds which figure
```

For every figure the script takes the freshest run folder that contains the
required table, rebuilds the figure with `ranklib.figures` (the same builders
the notebooks use) and writes PNG (300 dpi), SVG and a `.txt` file with the
plotted numbers into `figures/<time>/`. The figures of the article:

| Article | File |
|---|---|
| Figure 1 | `fig_methods_vs_tie_rate` |
| Figure 2 | `fig2_degradation_beta0.5`, `_beta1.5`, `_beta4` |
| Figure 3 | `fig_thresholds_quicksort` |
| Figure 4 | `fig3_input_order` |
| Figure 5 | `fig2b_input_leak` |
| Figure 6 | `fig1_budget` |
| Figure A1 | `fig_thresholds_merge` |
| Figure A2 | `fig_thresholds_heapsort` |

`fig2c_suitability` is an extra view of the simulation that is not in the
article. The folder shipped in `figures/` was produced by this script from the
published tables.

## prepare_pdd.py — the external data set

The HolisticEval / P-DD set (Pang et al., 2020) is not redistributed here.
Download `context_data_release.csv` into `data/external/`:

```bash
curl -L --create-dirs -o data/external/context_data_release.csv https://raw.githubusercontent.com/alexzhou907/dialogue_evaluation/master/context_data_release.csv
```

Then run the script:

```bash
python -X utf8 prepare_pdd.py
```

The script compares the SHA256 of your download with the file we used. It then
writes `data/pdd_context_200.csv` in the collector's schema: a header is added,
the columns are renamed, and the mean and SD of the ten ratings are added.
Texts and row order are unchanged, so object k of `data/matrices/PDD.jsonl` is
row k (0-based) of the release file. The script reports whether the result is
identical to the file the PDD matrix was collected on. `--check` only reports.

## ranklib — the library behind the notebooks

| Module | Contents |
|---|---|
| `simulation.py` | The synthetic judge: verdict generation, tie models, input orders, dense aggregators, the grid runner, `error_rate` / `beta_for_error` |
| `thresholds.py` | Threshold estimation on the Fisher scale with bootstrap CIs; critical error rate |
| `aggregators.py` | Copeland, win rate, Bradley–Terry (MM with regularization, standard errors), Rank Centrality |
| `sorts.py` | Bubble, insertion, merge, heapsort, quicksort with a comparator; any input order; comparison counting |
| `metrics.py` | Spearman ρ, top/bottom overlaps, non-transitivity (sampled and exact) |
| `figures.py` | All figure builders and the functions that export the plotted numbers |
| `experiments.py` | The runners called by the notebooks |
| `runlog.py` | Run folders and `runs_log.jsonl` |

`ranklib` is the part of the authors' offline experiment stand that does not
depend on the comment corpus; its version (1.4.2) is the version the published
results were computed with. The code that produced the other published runs
(nb01–nb04, nb06, nb08–nb10: analysis of the verdict matrices) is available
from the authors on request.

## Run folders and the run log

Every run folder contains `config.json` (the parameters exactly as passed),
the output tables (CSV, UTF-8 with BOM), `report.txt`, and the figures for the
runs that draw them. Each run also appends one JSON line to
`results/runs_log.jsonl`:

- `run_id`, `nb_id`, `status`, `started_at`, `duration_seconds`;
- `ranklib_version`, `git_commit` (the authors' repository), `python`;
- `config`;
- `inputs_sha256_12` — the first 12 hex digits of the SHA256 of every input
  file;
- `artifacts` and key summary numbers.

For the published runs the input hashes can be checked against
`SHA256SUMS.txt`: the matrices in `data/matrices/` are byte-identical to the
inputs the runs read.

## Verification

Reruns from this package were compared with the published files:

- `nb05_simulation.ipynb` with `QUICK=False`: `simulation_long.csv`,
  `simulation_summary.csv`, `suitability.csv`, `field_points.csv`,
  `report.txt` and `config.json` are identical to
  `results/20260705_223837_nb05_simulation`;
- `nb07_thresholds` (from the published sweep): `thresholds.csv`,
  `critical_error_rates.csv`, `report.txt` and `config.json` are identical;
  the three threshold figures are byte-identical to Figures 3, A1 and A2 of
  the article;
- `run_connectivity.py`: all four files are identical;
- `prepare_pdd.py`: the derived file is identical to the one the PDD matrix
  was collected on;
- `regen_figures.py`: the figures are pixel-identical to the authors' figure
  set of 20 July 2026;
- `collector.py`: with a stub judge, the same prompts, stored records and
  counters as the original collection code for all seven configurations;
- `tournament.ipynb`, Phase B on the published PDD matrix: the same report as
  the authors' original run.
