# When Sorting Fails: data and code

Data and code for the article **"When Sorting Fails: Ranking from LLM Pairwise
Comparisons Under High Tie Rates"** [PLACEHOLDER: journal reference and DOI of
the article].

The article studies what happens when a ranking is extracted from the pairwise
verdicts of a large language model (LLM) that often answers "equal". It
compares nine algorithms: four aggregators (Copeland, win rate, Bradley–Terry,
Rank Centrality) and five comparison sorts (quicksort, merge, heapsort,
insertion, bubble). The comparison runs on complete tournaments collected with
two judges, and on a controlled simulation with a synthetic judge. This
package contains:

- the verdict matrices of all seven complete tournaments;
- the English translation of the comment corpus;
- the judge instructions and the code that collected the tournaments;
- the simulation code;
- the result tables behind every table and figure of the article.

## Contents

| Path | What it is |
|---|---|
| `data/corpus_400-2_en.csv` | English translation of the corpus: 399 "parent – reply" pairs of YouTube comments |
| `data/README.txt` | Description of the corpus, the same as in its Zenodo record |
| `data/matrices/` | The seven verdict matrices (`GA2.jsonl` … `PDD.jsonl`) and their metadata `matrices.json` |
| `prompts/` | The judge instructions, verbatim, with `registry.json` and a description |
| `collector.py`, `tournament.ipynb` | Collection of a complete tournament with an LLM judge and its offline analysis |
| `ranklib/`, `nb05_simulation.ipynb`, `nb07_thresholds.ipynb` | Controlled simulation (experiment E7) and applicability thresholds; no model calls |
| `run_connectivity.py` | Connectivity of a random comparison graph (budget scaling) |
| `prepare_pdd.py` | Converts the external HolisticEval / P-DD set to the collector's schema |
| `regen_figures.py`, `figures/` | The figures of the article, rebuilt from the tables in `results/` |
| `results/` | The runs behind the tables and figures of the article, with `runs_log.jsonl` |
| `SCRIPTS.md` | What every script and notebook does and how to run it |
| `LICENSE`, `LICENSE-DATA` | MIT for the code, CC BY-NC 4.0 for the data |
| `SHA256SUMS.txt` | Checksums of all files |

## Quick start

```bash
pip install -r requirements.txt
python -X utf8 regen_figures.py          # the figures of the article from results/
python -X utf8 run_connectivity.py       # a few seconds
jupyter notebook nb05_simulation.ipynb   # QUICK=True: a smoke run in a minute or two
```

`tournament.ipynb` with `COLLECT = False` and `ANALYZE = "published"` analyses
any published matrix without a model: reference rankings plus replays of
heapsort and quicksort. Details are in [SCRIPTS.md](SCRIPTS.md).

## The corpus

The corpus is also published on Zenodo as a standalone data set,
https://doi.org/10.5281/zenodo.23014057; `data/corpus_400-2_en.csv` is the
same file. Please cite the Zenodo record when you use the corpus.

Comments under one YouTube video; the enriched subsample (internal label
`400-2`) of Section 2.1 of the article. In line with the YouTube data policy
the original comments are not published. The file holds their English
translation, made by the GPT6-Astra model. The model was instructed to render
the meaning rather than the letter, including the deliberate spelling
distortions, character substitutions and allusions with which the authors
evade moderation.

| Column | Content |
|---|---|
| `id` | Object number, 0–398; equal to the object index in the verdict matrices |
| `parent_text_clean` | The parent: the description of the video for a top-level comment, otherwise the comment the reply answers |
| `textOriginal_clean` | The reply |
| `_parent_method` | How the parent was determined (see below) |

The values of `_parent_method`:

- `top_level` (112 rows) — a top-level comment; its parent is the video
  description.
- `second_level` (77) — a reply that does not start with an @-mention; its
  parent is the root comment of the thread.
- `mention_backtrack` (210) — a reply that starts with an @-mention; its parent
  is the comment of the mentioned user, found by walking back through the
  thread. If that user is not in the thread, the root comment of the thread is
  used instead.

The rows are in the order of the original subsample. This order is not random
with respect to the judging criterion: extreme cases were added by hand
(Section 2.1). That matters for the input-order experiments.

The published file is anonymized. In the authors' working data, user names
were replaced by pseudonyms such as user1 and user2. The pseudonyms were used
to link replies to their parent comments; the file contains neither the names
nor the pseudonyms. Also removed are comment identifiers, avatar and channel
links, likes, dates and the Russian originals. The texts contain no @-mentions
or links.

## The verdict matrices

One JSON line per pair of objects, all n(n−1)/2 pairs (79,401 for the corpus,
19,900 for P-DD):

```json
{"key": "0_3", "winner": 3, "raw_ab": "A", "raw_ba": "B", "confidence": "strong", "ts": 1781375368.1498132}
```

- `key` — `"i_j"` with i < j, the object ids.
- `winner` — the id of the winner, or `null` for a tie. A tie is a mutual
  "equal" or a contradiction between the two presentation orders; the share of
  `null` is the tie rate of the article.
- `raw_ab` — the judge's letter when object j was shown first (as pair A).
- `raw_ba` — the letter when object i was shown first. The letters refer to
  positions in the prompt, not to objects.
- `confidence`, for the matrices with both orders:
  - `strong` — the two orders agree;
  - `weak` — one order is decisive and the other says "equal";
  - `disagreement` — the orders contradict each other, which counts as a tie.
- `confidence`, for the matrices with one call per pair: `single`. There the
  presentation order was drawn at random and not stored (`raw_ba` = `null`).
  For decisive verdicts it follows from `winner` and `raw_ab`; for ties it is
  unknown.
- `ts` — time of the call (Unix seconds).

Keep the files as they are: the budget curves of the article sample pairs
from the key order of the file.

| Matrix | Judge | Instruction | Presentation | Objects | Model calls | Tie rate |
|---|---|---|---|---|---|---|
| GA2 | Gemma4:26 | authenticity (ru) | both orders | corpus (Russian) | 158,802 | 18.35 % |
| GA1 | Gemma4:26 | authenticity (ru) | one call, random order | corpus (Russian) | 79,401 | 21.09 % |
| GR1 | Gemma4:26 | relevance (ru) | one call, random order | corpus (Russian) | 79,401 | 31.12 % |
| LR2 | llama3.1:8b | relevance (ru) | both orders | corpus (Russian) | 158,802 | 56.27 % |
| GA2_en_ru | Gemma4:26 | authenticity (ru) | both orders | corpus (English) | 158,802 | 16.85 % |
| GA2_en | Gemma4:26 | authenticity (en) | both orders | corpus (English) | 158,802 | 16.30 % |
| PDD | Gemma4:26 | coherence (en) | both orders | P-DD | 39,800 | 28.57 % |

- Gemma4:26 is gemma-4-26B-A4B-it-GGUF:IQ4_XS, served by llama.cpp's
  llama-server; llama3.1:8b is llama3.1:8b-instruct-q4_K_M in Ollama.
- All runs used temperature 0 and one pair per call.
- The note for top-level comments was on for the six corpus matrices and off
  for P-DD.
- The ids of all six corpus matrices refer to the rows of
  `corpus_400-2_en.csv`; the translation keeps the row order of the Russian
  original.

`matrices.json` records all of this per matrix, together with the collection
period, the counts and the SHA256 of each file.

## The external data set (P-DD)

The HolisticEval / P-DD set (Pang et al., 2020) is not redistributed. Download
its release file and run `prepare_pdd.py` to rebuild
`data/pdd_context_200.csv`, the input of the PDD matrix; object k of
`PDD.jsonl` is row k of the release file. See [SCRIPTS.md](SCRIPTS.md).

## Results and the article

Each folder `results/<time>_<id>/` holds the parameters (`config.json`), the
tables (CSV), and a log (`report.txt`) of one run. `results/runs_log.jsonl` has
one record per run: code version, parameters, and the SHA256 prefixes of the
inputs.

| Article | Run folder | Main files |
|---|---|---|
| Table 1 | `20260705_222729_nb01_matrix_stats` | `table1_view.csv`, `matrix_stats.csv` |
| Tables 2, 3; Figure 1 | `20260705_222911_nb02_algorithms_bench` | `methods_long.csv`, `table3_view.csv` |
| Tables 4–6; Figures 2, 5 | `20260705_223837_nb05_simulation` | `simulation_summary.csv`, `simulation_long.csv` |
| Tables 7, 8; Figure 4 | `20260705_223451_nb03_input_order` | `input_order_summary.csv`, `input_order_long.csv` |
| Tables 9, 10; Figure 6 | `20260705_223100_nb04_budget_curves` | `budget_summary.csv`, `qsort_trajectory_bt.csv` |
| Table 11 (E8) | `20260706_232020_nb06_batch_test` | `shares.csv`, `summary.json`, `report.txt` |
| Figure 3; Figures A1, A2; Tables A1–A3 | `20260918_210724_nb07_thresholds`, from the sweep `20260918_120914_nb07_competence_sweep` | `thresholds.csv`, `critical_error_rates.csv` |
| Budget scaling (Sections 3.4, 3.5, 4) | `20260926_134640_nb11_connectivity` | `connectivity.csv`, `thresholds.csv` |
| Translated corpus (Section 4) | `20260925_214500_nb08_english_corpus` | `descriptors.csv`, `methods.csv`, `reference_agreement.csv` |
| Repeated runs (Section 4) | `20260926_141948_nb09_repeat_runs` | `repeat_agreement.csv`, `pivot_seed_agreement.csv`, `pivot_seed_runs.csv` |
| External set P-DD (Section 4) | `20261002_125400_nb10_external_corpus` | `descriptors.csv`, `methods.csv`, `human_agreement.csv`, `budget.csv` |

Notes on the result tables:

- They use the labels of the authors' stand in the `matrix` column: `llama-v2`
  = LR2, `Gemma4-v2` = GR1, `Gemma4-v3-uni` = GA1, `Gemma4-v3-bidir` = GA2.
- The logs of the stand were written in Russian. Their labels and headings
  were translated into English for this package; the numbers are unchanged.
- The authenticity instruction is called `auth_v3` throughout the package; the
  authors' original logs name it `relevance_v3` (see `prompts/README.md`).
- Inputs that are not published carry neutral names in `config.json`,
  `timing.csv` and `runs_log.jsonl`: `corpus_400-2_ru.csv`, the E8 files
  `E8_*`, and the observational runs `obs_*`.

## What is not included

- The Russian original of the corpus.
- The verdicts of the batched-submission test E8 and of the observational
  phase.
- The analysis code behind the runs nb01–nb04, nb06 and nb08–nb10.

These are available from the authors on request. The simulation (E7), the
thresholds, the connectivity check and the collection of tournaments can be
run from this package.

## Verification

Reruns from this package were compared with the published files:

- **Simulation (nb05).** The full grid of `nb05_simulation.ipynb` reproduces
  `simulation_long.csv`, `simulation_summary.csv`, `suitability.csv`,
  `field_points.csv`, `report.txt` and `config.json` byte for byte.
- **Thresholds (nb07).** Computed from the published sweep, `thresholds.csv`,
  `critical_error_rates.csv` and `report.txt` are identical. The three
  threshold figures are byte-identical to Figures 3, A1 and A2 of the article.
- **Connectivity.** All tables of `run_connectivity.py` are identical.
- **P-DD.** `prepare_pdd.py` reproduces the file the PDD matrix was collected
  on.
- **Collector.** `collector.py` was compared with the original collection code
  using a stub judge, for all seven configurations. The prompts sent and the
  records stored are identical.
- **Tournament analysis.** Phase B of `tournament.ipynb` on the published PDD
  matrix reproduces the authors' original report.
- **Figures.** The figures in `figures/` are pixel-identical to the authors'
  figure set of 20 July 2026.

The environment was Python 3.10 with numpy 2.2.2, pandas 2.2.3 and
matplotlib 3.10.0, on Windows.

## License

- The code (`*.py`, `*.ipynb`, `ranklib/`) is under the MIT License, see
  [LICENSE](LICENSE).
- The data (`data/`, `prompts/`, `results/`, `figures/`) is under CC BY-NC 4.0,
  see [LICENSE-DATA](LICENSE-DATA).
- The corpus consists of translations of publicly posted comments; the rights
  to the original comments remain with their authors. Please do not attempt to
  identify them.

## Citation

The article: [PLACEHOLDER: citation of the article.]

The corpus: Mints, O. (2026). YouTube comment dialogues: 399 parent comment –
reply pairs from a socio-political discussion, translated into English
(Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23014057
