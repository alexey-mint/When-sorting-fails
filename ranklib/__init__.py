# -*- coding: utf-8 -*-
"""ranklib (simulation subset) — code of the controlled simulation for the
article "When Sorting Fails: Ranking from LLM Pairwise Comparisons Under High
Tie Rates".

This is the part of the authors' offline experiment stand that does not depend
on the comment corpus: the synthetic judge (simulation), threshold estimation
(thresholds), the nine ranking algorithms (aggregators, sorts), the quality
metrics, the figure builders and run logging. ZERO LLM calls anywhere.

The version is the stand version the published results were computed with;
every run record in results/runs_log.jsonl carries it.
"""

__version__ = "1.4.2"

from .aggregators import (                        # noqa: F401
    copeland_ranking, winrate_ranking, bradley_terry_strengths,
    bt_ranking, bt_strengths_se, rank_centrality,
)
from .sorts import SORTS, run_sort                # noqa: F401
from .metrics import (                            # noqa: F401
    spearman, ranks_from_order, topbot_overlap,
    nontransitivity_sampled, nontransitivity_exact,
)
