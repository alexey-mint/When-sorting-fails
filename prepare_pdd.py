# -*- coding: utf-8 -*-
"""Convert the HolisticEval / P-DD release file into the collector's schema.

The P-DD set (Pang et al., 2020) is not redistributed here. Download
`context_data_release.csv` from the authors' repository

    https://raw.githubusercontent.com/alexzhou907/dialogue_evaluation/master/context_data_release.csv

into data/external/ and run

    python -X utf8 prepare_pdd.py            # write data/pdd_context_200.csv
    python -X utf8 prepare_pdd.py --check    # report only, write nothing

The source file is left untouched; its SHA256 is compared with the file we
used. Only the wrapper changes: a header is added and the columns are renamed
to the collector's schema, and the mean and standard deviation of the ten
ratings are added. Texts are copied verbatim (no cleaning, no normalization,
no truncation) and the row order is preserved, so object k of
data/matrices/PDD.jsonl is row k (0-based) of the release file.

ZERO LLM calls.
"""
import argparse
import csv
import hashlib
import io
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "external", "context_data_release.csv")
DST = os.path.join(HERE, "data", "pdd_context_200.csv")
URL = ("https://raw.githubusercontent.com/alexzhou907/dialogue_evaluation/"
       "master/context_data_release.csv")
# SHA256 of the release file we downloaded and of the derived file the PDD
# matrix was collected on.
SRC_SHA256 = "05456ad4832238c7f60b248b6485af41b5184735c996406fdf40635b688d9cde"
DST_SHA256 = "fe3466d4cb36bd7cb0370d5d2f663bd66bdbf68c7ac70619ac3051e3fd46a6b0"
N_RATINGS = 10
COLUMNS = (["id", "parent_text_clean", "textOriginal_clean",
            "human_mean", "human_sd"]
           + [f"rating_{k:02d}" for k in range(1, N_RATINGS + 1)])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_source(path):
    """The release file has NO header row: reading it as if it had one would
    silently drop the first pair."""
    with io.open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    bad = [i for i, r in enumerate(rows, 1) if len(r) != 3 + N_RATINGS]
    if bad:
        raise ValueError(f"rows with an unexpected number of fields: {bad[:5]}")
    out = []
    for r in rows:
        ratings = [int(x) for x in r[3:]]
        if not all(1 <= v <= 5 for v in ratings):
            raise ValueError(f"rating outside 1..5 in the row with id={r[0]}")
        out.append({
            "id": r[0],
            "parent_text_clean": r[1],
            "textOriginal_clean": r[2],
            "human_mean": round(statistics.fmean(ratings), 4),
            "human_sd": round(statistics.stdev(ratings), 4),
            **{f"rating_{k:02d}": v for k, v in enumerate(ratings, 1)},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report only, write nothing")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        print(f"source file not found: {os.path.relpath(SRC, HERE)}\n"
              f"download it from\n  {URL}\ninto data/external/ and run again")
        sys.exit(1)
    rows = read_source(SRC)
    ids = [r["id"] for r in rows]
    parents = [r["parent_text_clean"] for r in rows]
    replies = [r["textOriginal_clean"] for r in rows]

    src_hash = sha256(SRC)
    print(f"source: {os.path.relpath(SRC, HERE)}")
    print(f"  SHA256 {src_hash}"
          + ("  (the file we used)" if src_hash == SRC_SHA256
             else "  (DIFFERS from the file we used)"))
    print(f"  rows {len(rows)}, unique ids {len(set(ids))}, "
          f"unique contexts {len(set(parents))}, "
          f"unique responses {len(set(replies))}")
    empty = sum(1 for t in parents + replies if not t.strip())
    print(f"  empty texts {empty}")
    n = len(rows)
    print(f"  full tournament: {n * (n - 1) // 2} pairs, "
          f"{n * (n - 1)} calls with bidirectional collection")

    if args.check:
        print("\n--check: the derived file is not written")
        return

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with io.open(DST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    dst_hash = sha256(DST)
    print(f"\nwritten: {os.path.relpath(DST, HERE)}")
    print(f"  SHA256 {dst_hash}"
          + ("  (identical to the file the PDD matrix was collected on)"
             if dst_hash == DST_SHA256 else "  (DIFFERS from our file)"))
    print(f"  columns: {', '.join(COLUMNS[:5])}, rating_01…rating_{N_RATINGS:02d}")

    # the texts must match the source character for character
    back = read_source(SRC)
    with io.open(DST, encoding="utf-8", newline="") as f:
        got = list(csv.DictReader(f))
    assert len(got) == len(back), "the number of rows differs"
    for a, b in zip(back, got):
        assert a["parent_text_clean"] == b["parent_text_clean"], a["id"]
        assert a["textOriginal_clean"] == b["textOriginal_clean"], a["id"]
        assert a["id"] == b["id"], a["id"]
    print("  check: texts, row order and ids match the source")


if __name__ == "__main__":
    main()
