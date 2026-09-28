# -*- coding: utf-8 -*-
"""Run logging: per-run artifact folders + an append-only runs_log.jsonl.

Every notebook run creates results/<YYYYMMDD_HHMMSS>_<nb_id>/ holding
config.json, output tables (CSV, utf-8-sig for Excel/Orange), figures
(PNG 300 dpi + SVG) and report.txt, then appends ONE json line to
runs_log.jsonl at the project root: version, git commit, config, input
hashes, artifact list, key summary numbers — enough to re-trace any number
in the article to code + data.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

from . import __version__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RUNS_LOG = os.path.join(ROOT, "runs_log.jsonl")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def save_figure(fig, out_dir, name, dpi=300):
    """Write a figure as <name>.png (dpi) + <name>.svg into out_dir. The single
    figure-saving path — used by Run.save_fig and by _regen_figures.py."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=dpi,
                    bbox_inches="tight")
    return os.path.join(out_dir, f"{name}.png")


class Run:
    """Context of one experiment run. Use start() / finish()."""

    def __init__(self, nb_id, config, inputs=(), results_dir=None,
                 runs_log=None):
        self.nb_id = nb_id
        self.config = config
        self.runs_log = runs_log or RUNS_LOG
        self.started = datetime.datetime.now()
        self.t0 = time.time()
        self.run_id = self.started.strftime("%Y%m%d_%H%M%S") + "_" + nb_id
        self.dir = os.path.join(results_dir or RESULTS, self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.inputs = {os.path.basename(p): sha256_file(p)[:12] for p in inputs}
        self._report = []
        with open(os.path.join(self.dir, "config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)
        print(f"[{self.nb_id}] run {self.run_id} -> {self.dir}")

    # ---- artifacts ----
    def path(self, name):
        return os.path.join(self.dir, name)

    def save_table(self, df, name):
        p = self.path(name)
        df.to_csv(p, index=False, encoding="utf-8-sig")
        return p

    def save_json(self, obj, name):
        p = self.path(name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        return p

    def save_fig(self, fig, name, dpi=300):
        return save_figure(fig, self.dir, name, dpi=dpi)

    def log(self, text=""):
        print(text)
        self._report.append(str(text))

    # ---- finalization ----
    def finish(self, summary=None, status="completed"):
        if self._report:
            with open(self.path("report.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(self._report))
        rec = {
            "run_id": self.run_id,
            "nb_id": self.nb_id,
            "status": status,
            "ranklib_version": __version__,
            "git_commit": git_commit(),
            "python": sys.version.split()[0],
            "started_at": self.started.isoformat(timespec="seconds"),
            "duration_seconds": round(time.time() - self.t0, 1),
            "config": self.config,
            "inputs_sha256_12": self.inputs,
            "artifacts": sorted(os.listdir(self.dir)),
            "summary": summary or {},
        }
        with open(self.runs_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"[{self.nb_id}] finished in {rec['duration_seconds']}s; "
              f"logged to runs_log.jsonl")
        return rec
