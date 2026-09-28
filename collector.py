# -*- coding: utf-8 -*-
"""Collector of full pairwise tournaments with an LLM judge.

This is the part of the authors' collection code (pairwise_core 2.5) that
produced the verdict matrices in data/matrices/. Every pair of objects is put
to the judge, once in a random presentation order or in both orders, and each
verdict is appended to a crash-safe JSONL store with one record per pair (the
format is described in README.md). An interrupted collection resumes where it
stopped. Driven by tournament.ipynb.

Differences from the original module:
  * the judge instructions are read from prompts/ through prompts/registry.json
    instead of being hard-coded; the registry states each instruction's
    language, and that language selects the language of the note for
    top-level comments (the original took it from the name suffix "_eng_v3");
  * only the two backends used for the published matrices are kept: a local
    Ollama server (llama3.1:8b) and an OpenAI-compatible local server such as
    llama.cpp's llama-server (gemma-4-26B);
  * one pair per call (the batched submission tested in experiment E8 and the
    sorting-driven, pointwise and anchor modes of the original are left out).

The rendered prompts and the stored records are identical to the original
collector for every configuration of the published matrices; this was checked
with a stub backend.

This is the only module of the package that calls a language model.
"""
import datetime
import functools
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

try:
    from tqdm.auto import tqdm
    TQDM = True
except Exception:
    TQDM = False

COLLECTOR_VERSION = "1.0 (from pairwise_core 2.5)"
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


# ============================================================ pure helpers
def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def normalize(s) -> str:
    if isinstance(s, str):
        return " ".join(s.strip().split())
    return ""


def atomic_write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, obj: dict) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def load_api_keys(path: str, verbose: bool = True) -> dict:
    """Read KEY=value lines from a plain text file. Lines starting with # are
    ignored; empty values are skipped; quotes are tolerated. Missing file ->
    {} (a local server usually needs no key)."""
    keys = {}
    if not os.path.exists(path):
        if verbose:
            print(f"[api_keys] '{path}' not found — no keys loaded.")
        return keys
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'").strip()
            if k and v:
                keys[k] = v
    if verbose:
        print(f"[api_keys] loaded {len(keys)} key(s) from '{path}': "
              f"{', '.join(sorted(keys)) or '(none)'}")
    return keys


# ============================================================ instructions
@functools.lru_cache(maxsize=None)
def load_registry(prompts_dir: str = PROMPTS_DIR) -> dict:
    """prompts/registry.json: instructions (file, criterion, language, the
    matrices collected with it) and the top-level notes per language."""
    with open(os.path.join(prompts_dir, "registry.json"), encoding="utf-8") as f:
        return json.load(f)


def _read_template(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read().replace("\r\n", "\n")
    return text[:-1] if text.endswith("\n") else text


@functools.lru_cache(maxsize=None)
def get_instruction(name: str, prompts_dir: str = PROMPTS_DIR) -> dict:
    """Registry entry of one instruction plus its template text."""
    reg = load_registry(prompts_dir)
    entry = reg["instructions"].get(name)
    if entry is None:
        raise ValueError(f"Unknown instruction {name!r}. "
                         f"Available: {sorted(reg['instructions'])}")
    return {**entry, "name": name,
            "template": _read_template(os.path.join(prompts_dir, entry["file"]))}


def list_instructions(prompts_dir: str = PROMPTS_DIR) -> pd.DataFrame:
    reg = load_registry(prompts_dir)
    return pd.DataFrame([{"instruction": k, "criterion": v["criterion"],
                          "language": v["language"],
                          "matrices": ", ".join(v["matrices"])}
                         for k, v in reg["instructions"].items()])


def build_root_note(is_root_a: bool, is_root_b: bool, root_clause: bool,
                    lang: str = "ru", prompts_dir: str = PROMPTS_DIR) -> str:
    """Per-side note for top-level comments: their 'parent' is the video
    description, so the judge should assess the relation to the topic of the
    video rather than literal correspondence. Returns '' or the note text +
    two newlines."""
    if not root_clause:
        return ""
    notes = load_registry(prompts_dir)["root_notes"][lang]
    if is_root_a and is_root_b:
        return notes["both"] + "\n\n"
    if is_root_a or is_root_b:
        return notes["one"].format(side="A" if is_root_a else "B") + "\n\n"
    return ""


def build_pairwise_prompt(parent_a, reply_a, parent_b, reply_b,
                          prompt_version, root_clause=False,
                          is_root_a=False, is_root_b=False) -> str:
    ins = get_instruction(prompt_version)
    return ins["template"].format(
        parent_A=normalize(parent_a), reply_A=normalize(reply_a),
        parent_B=normalize(parent_b), reply_B=normalize(reply_b),
        root_note=build_root_note(is_root_a, is_root_b, root_clause,
                                  lang=ins["language"]),
    )


VERDICT_RE = re.compile(r"\b([ABE])\b")
# Cyrillic homoglyphs/phonetic equivalents some models emit: А→A, В→B,
# Б→B, Е→E. Applied (after .upper()) before the regex.
_CYRILLIC_VERDICT_MAP = str.maketrans({"А": "A", "В": "B", "Б": "B", "Е": "E"})


def parse_verdict_strict(text):
    """Return 'A'/'B'/'E', or None if no verdict letter found."""
    if not text:
        return None
    t = text.upper().translate(_CYRILLIC_VERDICT_MAP)
    m = VERDICT_RE.search(t)
    return m.group(1) if m else None


# ============================================================ rate limiter
class RateLimiter:
    """Thread-safe sliding-window limiter (requests per minute)."""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.window = deque()
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            while self.window and now - self.window[0] > 60.0:
                self.window.popleft()
            if len(self.window) >= self.rpm:
                sleep_for = 60.0 - (now - self.window[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self.window.append(time.time())


# ============================================================ config objects
@dataclass
class GlobalSettings:
    """Infrastructure shared by all runs of a session."""
    # Ollama (local)
    ollama_url: str = "http://localhost:11434/api/generate"
    keep_alive: str = "5m"
    # OpenAI-compatible LOCAL server (llama-server / vLLM / LM Studio / TGI).
    # The server is launched by the user; cfg.model is just a label.
    openai_local_url: str = "http://localhost:8080/v1/chat/completions"
    openai_local_api_key: str = ""   # only if the server was started with --api-key
    openai_local_rpm: int = 0        # 0 = no throttle (local)
    # HTTP / retries
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.8
    # output
    flatten_output: bool = True      # single-line text cells in output CSVs
    # persistence
    checkpoint_every: int = 50
    heartbeat_seconds: int = 60
    runs_log_path: str = "tournaments/runs_log.jsonl"
    fail_log: str = "tournaments/failures.log"
    # provenance
    script_version: str = ""         # set by the notebook that drives the run


@dataclass
class ExperimentConfig:
    """Per-run knobs. The fields are those of the original collector, so the
    configuration hash (used to resume an interrupted run) is computed in the
    same way; algorithm, hybrid_k_passes and warmstart_csv are not used by a
    tournament and keep their defaults."""
    backend: str                       # "ollama" | "openai_local"
    model: str                         # model name (Ollama) or label (local server)
    prompt_version: str = "relevance_v2"
    algorithm: str = "heapsort_scratch"
    order_mode: str = "bidirectional"
    # secondary
    comparison_batch_size: int = 1
    hybrid_k_passes: int = 3
    root_clause: bool = True
    comment_level_filter: Optional[str] = None   # None|"top_level"|"reply"
    disable_thinking: bool = True
    # sampling
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    num_predict: int = 8
    # data
    input_csv: str = ""
    text_col: str = "textOriginal_clean"
    parent_col: str = "parent_text_clean"
    level_col: str = "_parent_method"
    root_level_value: str = "top_level"
    n_rows: Optional[int] = None
    warmstart_csv: Optional[str] = None
    # identity / outputs (derived by make_config)
    exp_id: str = ""
    output_csv: str = ""
    cache_path: str = ""
    status_path: str = ""
    # run control
    force_rerun: bool = False


def _slug(s: str) -> str:
    """Filesystem-safe short slug."""
    s = str(s)
    s = s.replace("relevance_v", "v").replace("_scratch", "")
    s = s.replace("bidirectional", "bidir").replace("unidirectional_random", "uni")
    return re.sub(r"[^0-9A-Za-z._-]+", "-", s).strip("-")


def rederive_paths(cfg: "ExperimentConfig", out_dir=None) -> "ExperimentConfig":
    """(Re)derive output/cache/status paths from cfg.input_csv + cfg.exp_id,
    in out_dir or next to the input file."""
    in_dir = out_dir or (os.path.dirname(cfg.input_csv) or ".")
    stem = os.path.splitext(os.path.basename(cfg.input_csv))[0]
    base = os.path.join(in_dir, f"{stem}_{cfg.exp_id}")
    cfg.output_csv = base + ".csv"
    cfg.cache_path = base + ".cache.jsonl"
    cfg.status_path = base + ".status.json"
    return cfg


def flatten_text_cells(df):
    """Single-line every text cell: replace embedded CR/LF with a space and
    collapse whitespace runs, so output CSV records never span physical lines.
    Numeric columns are untouched. Returns a copy."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: re.sub(r"\s{2,}", " ",
                                 v.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")).strip()
                if isinstance(v, str) else v)
    return out


def make_config(backend, model, prompt_version="relevance_v2",
                algorithm="heapsort_scratch", order_mode="bidirectional",
                *, input_csv, out_dir=None, exp_id=None, **overrides) -> ExperimentConfig:
    """Build a full ExperimentConfig from a compact spec. Derives exp_id and
    output/cache/status paths (each configuration gets its own)."""
    if exp_id is None:
        parts = [_slug(backend), _slug(model), _slug(prompt_version),
                 _slug(algorithm), _slug(order_mode)]
        if overrides.get("comment_level_filter"):
            parts.append(_slug(overrides["comment_level_filter"]))
        if overrides.get("root_clause") is False:
            parts.append("noroot")
        exp_id = "_".join(parts)

    cfg = ExperimentConfig(
        backend=backend, model=model, prompt_version=prompt_version,
        algorithm=algorithm, order_mode=order_mode,
        input_csv=input_csv, exp_id=exp_id,
    )
    rederive_paths(cfg, out_dir=out_dir)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown ExperimentConfig field: {k!r}")
        setattr(cfg, k, v)
    return cfg


def make_tournament_config(backend, model, prompt_version, order_mode, *,
                           input_csv, out_dir=None, **overrides) -> ExperimentConfig:
    """Configuration of a full tournament, with the naming scheme of the
    authors' runs: exp_id = "REF_<backend>_<model>_<instruction>_<order>[_noroot]"."""
    cfg = make_config(backend, model, prompt_version, "heapsort_scratch",
                      order_mode, input_csv=input_csv, out_dir=out_dir,
                      **overrides)
    cfg.exp_id = "REF_" + cfg.exp_id.replace("_heapsort", "")
    return rederive_paths(cfg, out_dir=out_dir)


def compute_config_hash(cfg: ExperimentConfig) -> str:
    """Full-config hash: identifies a run for resume/skip."""
    payload = {
        "backend": cfg.backend, "model": cfg.model,
        "prompt_version": cfg.prompt_version, "algorithm": cfg.algorithm,
        "order_mode": cfg.order_mode,
        "comparison_batch_size": cfg.comparison_batch_size,
        "hybrid_k_passes": cfg.hybrid_k_passes,
        "root_clause": cfg.root_clause,
        "comment_level_filter": cfg.comment_level_filter,
        "disable_thinking": (cfg.disable_thinking
                             if cfg.backend in ("google", "openrouter") else None),
        "temperature": cfg.temperature, "top_p": cfg.top_p, "seed": cfg.seed,
        "input_csv": os.path.abspath(cfg.input_csv),
        "text_col": cfg.text_col, "parent_col": cfg.parent_col,
        "level_col": cfg.level_col, "root_level_value": cfg.root_level_value,
        "n_rows": cfg.n_rows,
        "warmstart_csv": (os.path.abspath(cfg.warmstart_csv)
                          if cfg.warmstart_csv else None),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


# ============================================================ cache & status
class ComparisonCache:
    """Crash-safe append-only JSONL store. Key = "{min}_{max}"."""
    def __init__(self, path: str, checkpoint_every=50, heartbeat_seconds=60):
        self.path = path
        self.checkpoint_every = checkpoint_every
        self.heartbeat_seconds = heartbeat_seconds
        ensure_parent_dir(path)
        self._data: dict = {}
        self._buffer: list = []
        self._last_flush_ts: float = time.time()
        if os.path.exists(path):
            self._replay()

    @staticmethod
    def _key(i, j) -> str:
        a, b = (i, j) if i <= j else (j, i)
        return f"{a}_{b}"

    def _replay(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = rec.get("key")
                if k is not None:
                    self._data[k] = rec

    def get(self, i, j):
        return self._data.get(self._key(i, j))

    def set(self, i, j, winner_kind, raw_ab, raw_ba, confidence) -> None:
        if winner_kind == "i":
            winner_id = i
        elif winner_kind == "j":
            winner_id = j
        else:
            winner_id = None
        rec = {"key": self._key(i, j), "winner": winner_id,
               "raw_ab": raw_ab, "raw_ba": raw_ba,
               "confidence": confidence, "ts": time.time()}
        self._data[rec["key"]] = rec
        self._buffer.append(rec)

    def maybe_flush(self) -> None:
        if (len(self._buffer) >= self.checkpoint_every
                or (time.time() - self._last_flush_ts) >= self.heartbeat_seconds):
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            self._last_flush_ts = time.time()
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        self._buffer.clear()
        self._last_flush_ts = time.time()

    def stats(self) -> dict:
        n = len(self._data)
        ties = sum(1 for r in self._data.values() if r.get("winner") is None)
        dis = sum(1 for r in self._data.values()
                  if r.get("confidence") == "disagreement")
        return {"n_unique_comparisons": n, "n_ties": ties,
                "n_disagreements": dis}


def load_matrix(path: str) -> ComparisonCache:
    """Open a verdict matrix (a published one from data/matrices/ or a store
    written by a tournament) for reading. Nothing is written unless set() and
    flush() are called."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return ComparisonCache(path)


def matrix_objects(cache) -> list:
    """Object ids 0..n-1 of a complete matrix."""
    top = max(max(int(x) for x in k.split("_")) for k in cache._data)
    return list(range(top + 1))


def win_function(cache):
    """win(i, j) for ranklib.sorts: +1 if i won, -1 if j won, 0 for a tie or a
    missing pair."""
    def win(i, j):
        rec = cache.get(i, j)
        if rec is None or rec.get("winner") is None:
            return 0
        return 1 if rec["winner"] == i else -1
    return win


class RunStatus:
    def __init__(self, path: str, config_hash: str):
        self.path = path
        self.config_hash = config_hash
        ensure_parent_dir(path)
        self.state: dict = {}

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                try:
                    self.state = json.load(f)
                except json.JSONDecodeError:
                    self.state = {}
        return self.state

    def write(self, status: str, **fields):
        rec = dict(self.state)
        rec.update({"status": status, "config_hash": self.config_hash,
                    "last_heartbeat": now_iso()})
        rec.update(fields)
        if status == "completed":
            rec["finished_at"] = now_iso()
        atomic_write_json(self.path, rec)
        self.state = rec

    def matches(self) -> bool:
        return self.state.get("config_hash") == self.config_hash


# ============================================================ backends
class OllamaBackend:
    """Local Ollama server (native /api/generate). Used for LR2."""
    def __init__(self, cfg: ExperimentConfig, settings: GlobalSettings):
        self.cfg = cfg
        self.s = settings
        self.session = requests.Session()
        self.url = settings.ollama_url
        self.headers = {}
        self.limiter = None

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        cfg, s = self.cfg, self.s
        if max_tokens is None:
            max_tokens = cfg.num_predict
        payload = {
            "model": cfg.model, "prompt": prompt, "stream": True,
            "keep_alive": s.keep_alive,
            "options": {
                "temperature": cfg.temperature, "top_p": cfg.top_p,
                "seed": cfg.seed, "num_predict": max_tokens,
                # stop sequences as in the original runs ("Пара" = "Pair")
                "stop": ["\n\n", "Пара", "###"],
            },
        }
        wait = 0.2
        for attempt in range(1, s.max_retries + 1):
            if self.limiter is not None:
                self.limiter.wait()
            try:
                with self.session.post(self.url, json=payload,
                                       headers=self.headers,
                                       timeout=s.request_timeout, stream=True) as resp:
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    chunks = []
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        j = json.loads(line.decode("utf-8"))
                        piece = j.get("response", "")
                        if piece:
                            chunks.append(piece)
                    return "".join(chunks)
            except Exception:
                if attempt >= s.max_retries:
                    raise
                time.sleep(wait)
                wait *= s.retry_backoff
        return ""


class OpenAILocalBackend:
    """Client for an OpenAI-compatible LOCAL server (`/v1/chat/completions`),
    e.g. llama.cpp's llama-server (`llama-server -m model.gguf --port 8080`);
    also works with vLLM / LM Studio / TGI by changing `openai_local_url`.
    `cfg.model` is a LABEL (the real model is whatever the server runs). Auth
    is optional (only if the server was started with --api-key). Used for the
    Gemma matrices."""
    def __init__(self, cfg: ExperimentConfig, settings: GlobalSettings):
        self.cfg = cfg
        self.s = settings
        self.session = requests.Session()
        self._headers = {"Content-Type": "application/json"}
        if settings.openai_local_api_key:
            self._headers["Authorization"] = f"Bearer {settings.openai_local_api_key}"
        self.limiter = (RateLimiter(settings.openai_local_rpm)
                        if settings.openai_local_rpm else None)

    def generate(self, prompt: str, max_tokens: int = None) -> str:
        cfg, s = self.cfg, self.s
        if max_tokens is None:
            max_tokens = cfg.num_predict
        body = {"model": cfg.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": cfg.temperature, "top_p": cfg.top_p,
                "max_tokens": max(16, max_tokens), "seed": cfg.seed}
        if cfg.disable_thinking:
            # Suppress chain-of-thought on reasoning models (Gemma/Qwen3-style
            # chat templates). Without this such a model spends the whole token
            # budget in `reasoning_content` and returns EMPTY `content` with
            # finish_reason='length', so every verdict falls back to "E".
            # Harmless on non-reasoning templates (the kwarg is simply unused).
            body["chat_template_kwargs"] = {"enable_thinking": False}
        wait = 0.25
        last_err = None
        for attempt in range(1, s.max_retries + 1):
            if self.limiter is not None:
                self.limiter.wait()
            try:
                resp = self.session.post(s.openai_local_url, json=body,
                                         headers=self._headers,
                                         timeout=s.request_timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices in response: {str(data)[:300]}")
                msg = choices[0].get("message") or {}
                text = msg.get("content") or ""
                if not text:
                    fr = choices[0].get("finish_reason", "")
                    hint = (" — reasoning model: emitted reasoning_content but no "
                            "content. Keep disable_thinking=True (sends "
                            "chat_template_kwargs.enable_thinking=false) or raise "
                            "num_predict." if msg.get("reasoning_content") else "")
                    raise RuntimeError(f"Empty content (finish_reason={fr!r}){hint}")
                return text
            except Exception as e:
                last_err = e
                if attempt >= s.max_retries:
                    break
                time.sleep(wait)
                wait *= s.retry_backoff
        raise RuntimeError(f"openai_local backend failed after {s.max_retries} "
                           f"attempts: {last_err}")


def make_backend(cfg: ExperimentConfig, settings: GlobalSettings):
    if cfg.backend == "ollama":
        return OllamaBackend(cfg, settings)
    if cfg.backend == "openai_local":
        return OpenAILocalBackend(cfg, settings)
    raise ValueError(f"Unknown backend: {cfg.backend!r} "
                     f"(supported: 'ollama', 'openai_local')")


# ============================================================ dataset
def load_dataset(cfg: ExperimentConfig) -> pd.DataFrame:
    """Read + clean + level-filter + head + reset index. Rows with an empty
    reply or parent are dropped, and the rest are numbered 0..N-1: that number
    is the object id in the verdict store."""
    df = pd.read_csv(cfg.input_csv)
    if cfg.text_col not in df.columns or cfg.parent_col not in df.columns:
        raise ValueError(f"Missing columns {cfg.text_col} or {cfg.parent_col} "
                         f"in {cfg.input_csv}")
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str).str.strip()
    df[cfg.parent_col] = df[cfg.parent_col].fillna("").astype(str).str.strip()
    df = df[(df[cfg.text_col] != "") & (df[cfg.parent_col] != "")].copy()

    has_level = cfg.level_col in df.columns
    if cfg.comment_level_filter in ("top_level", "reply"):
        if not has_level:
            raise ValueError(f"comment_level_filter={cfg.comment_level_filter!r} "
                             f"requires column {cfg.level_col!r}.")
        if cfg.comment_level_filter == "top_level":
            df = df[df[cfg.level_col] == cfg.root_level_value].copy()
        else:
            df = df[df[cfg.level_col] != cfg.root_level_value].copy()

    if isinstance(cfg.n_rows, int) and cfg.n_rows > 0:
        df = df.head(cfg.n_rows).copy()
    return df.reset_index(drop=True)


# ============================================================ the tournament
class Tournament:
    """One full tournament: every pair of objects put to the judge once (the
    presentation order drawn at random) or in both orders."""

    def __init__(self, cfg: ExperimentConfig, settings: GlobalSettings,
                 df: pd.DataFrame = None):
        if cfg.comparison_batch_size != 1:
            raise ValueError("this collector submits one pair per call "
                             "(comparison_batch_size must be 1)")
        self.cfg = cfg
        self.s = settings
        self.df = load_dataset(cfg) if df is None else df.reset_index(drop=True)
        self.row_ids = list(self.df.index)
        self.n_rows = len(self.row_ids)

        has_level = cfg.level_col in self.df.columns
        if has_level:
            self.is_root_by_id = {rid: (self.df.at[rid, cfg.level_col]
                                        == cfg.root_level_value)
                                  for rid in self.row_ids}
        else:
            self.is_root_by_id = {rid: False for rid in self.row_ids}
        self.n_root_rows = sum(1 for v in self.is_root_by_id.values() if v)

        self.counters = {"n_llm_calls": 0, "n_cache_hits": 0,
                         "n_eq_verdicts": 0, "n_disagreements": 0,
                         "n_backend_failures": 0, "n_unparsed_verdicts": 0,
                         "n_single_verdicts": 0}
        self._counter_lock = threading.Lock()
        self._bidir_pool = ThreadPoolExecutor(max_workers=2,
                                              thread_name_prefix="bidir")
        self.rng = random.Random(cfg.seed)
        self._hb = [time.time()]
        self.config_hash = compute_config_hash(cfg)
        self.cache = ComparisonCache(cfg.cache_path, settings.checkpoint_every,
                                     settings.heartbeat_seconds)
        self.status = RunStatus(cfg.status_path, self.config_hash)
        self.backend = None        # built lazily in collect_all_pairs()
        self.record = None
        self.run_id = None
        self.started_at = None

    # ---- counters ----
    def _bump(self, key, by=1):
        with self._counter_lock:
            self.counters[key] += by

    # ---- LLM call ----
    def _single_llm_call(self, parent_a, reply_a, parent_b, reply_b,
                         is_root_a=False, is_root_b=False) -> str:
        cfg = self.cfg
        prompt = build_pairwise_prompt(parent_a, reply_a, parent_b, reply_b,
                                       cfg.prompt_version, cfg.root_clause,
                                       is_root_a, is_root_b)
        try:
            text = self.backend.generate(prompt, max_tokens=cfg.num_predict)
        except Exception as e:
            with open(self.s.fail_log, "a", encoding="utf-8") as log:
                log.write(f"[{now_iso()}] [{cfg.exp_id}] backend failure: {e}\n{prompt}\n\n")
            self._bump("n_backend_failures")
            return "E"
        self._bump("n_llm_calls")
        verdict = parse_verdict_strict(text)
        if verdict is None:
            with open(self.s.fail_log, "a", encoding="utf-8") as log:
                log.write(f"[{now_iso()}] [{cfg.exp_id}] unparsed verdict: {text[:200]!r}\n")
            self._bump("n_unparsed_verdicts")
            return "E"
        return verdict

    @staticmethod
    def _aggregate_bidirectional(v_ab, v_ba):
        fwd = {"A": "i", "B": "j", "E": "EQ"}[v_ab]
        rev = {"A": "j", "B": "i", "E": "EQ"}[v_ba]
        if fwd == rev:
            return fwd, "strong"
        if fwd == "EQ":
            return rev, "weak"
        if rev == "EQ":
            return fwd, "weak"
        return "EQ", "disagreement"

    def _heartbeat(self):
        if (time.time() - self._hb[0]) >= self.s.heartbeat_seconds:
            self._write_status("running")
            self._hb[0] = time.time()

    def _warn_stale_cache(self):
        """If the store already has records written by a DIFFERENT
        configuration (config hash mismatch, or no status file), those verdicts
        would be silently reused. Say so."""
        if not self.cache._data:
            return
        if self.status.state.get("config_hash") == self.config_hash:
            return
        print(f"[{self.cfg.exp_id}] WARNING: existing store "
              f"({len(self.cache._data)} entries) was written by a DIFFERENT "
              f"config (config_hash mismatch or missing status). Stored verdicts "
              f"will be reused. For a clean run delete the .cache.jsonl/.status.json "
              f"or set a custom exp_id.")

    def _write_status(self, status):
        self.status.write(status, run_id=self.run_id, started_at=self.started_at,
                          exp_id=self.cfg.exp_id,
                          input_csv=os.path.abspath(self.cfg.input_csv),
                          n_rows=self.n_rows, algorithm=self.cfg.algorithm,
                          order_mode=self.cfg.order_mode,
                          model_backend=self.cfg.backend, model_name=self.cfg.model,
                          n_llm_calls_so_far=self.counters["n_llm_calls"])

    # ---- compare primitive ----
    def compare(self, i, j) -> str:
        if i == j:
            return "EQ"
        cached = self.cache.get(i, j)
        if cached is not None:
            self._bump("n_cache_hits")
            w = cached.get("winner")
            return "EQ" if w is None else ("i" if w == i else "j")

        cfg = self.cfg
        parent_i = self.df.at[i, cfg.parent_col]; reply_i = self.df.at[i, cfg.text_col]
        parent_j = self.df.at[j, cfg.parent_col]; reply_j = self.df.at[j, cfg.text_col]
        root_i = self.is_root_by_id.get(i, False)
        root_j = self.is_root_by_id.get(j, False)

        if cfg.order_mode == "bidirectional":
            f_ab = self._bidir_pool.submit(self._single_llm_call,
                                           parent_i, reply_i, parent_j, reply_j,
                                           root_i, root_j)
            f_ba = self._bidir_pool.submit(self._single_llm_call,
                                           parent_j, reply_j, parent_i, reply_i,
                                           root_j, root_i)
            v_ab = f_ab.result(); v_ba = f_ba.result()
            winner_kind, confidence = self._aggregate_bidirectional(v_ab, v_ba)
            self.cache.set(i, j, winner_kind, raw_ab=v_ab, raw_ba=v_ba,
                           confidence=confidence)
            if winner_kind == "EQ":
                self._bump("n_eq_verdicts")
            if confidence == "disagreement":
                self._bump("n_disagreements")
        else:  # any other order_mode: one call, presentation order drawn at random
            flip = self.rng.random() < 0.5
            if flip:
                v_ab = self._single_llm_call(parent_j, reply_j, parent_i, reply_i,
                                             root_j, root_i)
                mapping = {"A": "j", "B": "i", "E": "EQ"}
            else:
                v_ab = self._single_llm_call(parent_i, reply_i, parent_j, reply_j,
                                             root_i, root_j)
                mapping = {"A": "i", "B": "j", "E": "EQ"}
            winner_kind = mapping[v_ab]
            self.cache.set(i, j, winner_kind, raw_ab=v_ab, raw_ba=None,
                           confidence="single")
            self._bump("n_single_verdicts")
            if winner_kind == "EQ":
                self._bump("n_eq_verdicts")

        self.cache.maybe_flush()
        self._heartbeat()
        rec = self.cache.get(i, j)
        w = rec.get("winner") if rec else None
        return "EQ" if w is None else ("i" if w == i else "j")

    def compare_many_against_pivot(self, pivot, candidate_ids, on_done=None) -> dict:
        results = {}
        uncached = []
        for cid in candidate_ids:
            if cid == pivot:
                results[cid] = "EQ"
                if on_done:
                    on_done(1)
                continue
            rec = self.cache.get(cid, pivot)
            if rec is not None:
                self._bump("n_cache_hits")
                w = rec.get("winner")
                results[cid] = "EQ" if w is None else ("i" if w == cid else "j")
                if on_done:
                    on_done(1)
            else:
                uncached.append(cid)
        for cid in uncached:                      # one pair per call
            results[cid] = self.compare(cid, pivot)
            if on_done:
                on_done(1)
        return results

    # ---- the full tournament ----
    def collect_all_pairs(self, progress: bool = True) -> dict:
        """Fill the store with ALL n(n-1)/2 unique pairwise comparisons for
        this configuration. Resumable: stored pairs are instant hits. Writes a
        run record to settings.runs_log_path. Returns that record."""
        cfg, s = self.cfg, self.s
        n = self.n_rows
        total_pairs = n * (n - 1) // 2

        self.status.load()
        prev = dict(self.status.state)
        self._warn_stale_cache()
        if (prev and prev.get("status") == "completed" and self.status.matches()
                and not cfg.force_rerun):
            print(f"[{cfg.exp_id}] all-pairs already collected "
                  f"({len(self.cache._data)} pairs). force_rerun=True to redo.")
            return {"exp_id": cfg.exp_id, "status": "skipped",
                    "mode": "all_pairs", "n_unique_comparisons": len(self.cache._data),
                    "cache_path": os.path.abspath(cfg.cache_path),
                    "config_hash": self.config_hash}

        resume = bool(prev and prev.get("status") == "running" and self.status.matches())
        self.run_id = prev.get("run_id") if resume else now_iso()
        self.started_at = prev.get("started_at") if resume else now_iso()
        if resume:
            print(f"[{cfg.exp_id}] resuming all-pairs (store: "
                  f"{len(self.cache._data)}/{total_pairs} pairs).")

        self.backend = make_backend(cfg, s)
        self._write_status("running")

        pbar = tqdm(total=total_pairs, desc=f"all-pairs[{cfg.exp_id}]",
                    disable=not (progress and TQDM)) if TQDM else None

        def _on_done(k):
            if pbar is not None:
                pbar.update(k)
                pbar.set_postfix(llm=self.counters["n_llm_calls"],
                                 cache=self.counters["n_cache_hits"], refresh=False)

        t0 = time.time()
        try:
            for idx, i in enumerate(self.row_ids):
                cands = self.row_ids[idx + 1:]      # only j > i (canonical)
                if cands:
                    self.compare_many_against_pivot(i, cands, on_done=_on_done)
        finally:
            self.cache.flush()
            if pbar is not None:
                pbar.close()
            self._write_status("running")
        duration_s = time.time() - t0

        cstats = self.cache.stats()
        c = self.counters
        record = {
            "core_version": COLLECTOR_VERSION, "script_version": s.script_version,
            "exp_id": cfg.exp_id, "run_id": self.run_id,
            "started_at": self.started_at, "finished_at": now_iso(),
            "duration_seconds": round(duration_s, 3), "mode": "all_pairs",
            "input_csv": os.path.abspath(cfg.input_csv),
            "cache_path": os.path.abspath(cfg.cache_path),
            "n_rows_processed": n, "n_pairs_total": total_pairs,
            "backend": cfg.backend, "model": cfg.model,
            "prompt_version": cfg.prompt_version, "order_mode": cfg.order_mode,
            "comparison_batch_size": cfg.comparison_batch_size,
            "root_clause": cfg.root_clause,
            "comment_level_filter": cfg.comment_level_filter,
            "n_root_rows": self.n_root_rows,
            "n_llm_calls": c["n_llm_calls"],
            "n_unique_comparisons": cstats["n_unique_comparisons"],
            "n_cache_hits": c["n_cache_hits"],
            "n_eq_verdicts": c["n_eq_verdicts"],
            "eq_rate": c["n_eq_verdicts"] / max(1, cstats["n_unique_comparisons"]),
            "n_disagreements": c["n_disagreements"],
            "disagreement_rate": c["n_disagreements"] / max(1, cstats["n_unique_comparisons"]),
            "n_backend_failures": c["n_backend_failures"],
            "n_unparsed_verdicts": c["n_unparsed_verdicts"],
            "config_hash": self.config_hash, "status": "completed",
        }
        append_jsonl(s.runs_log_path, record)
        self._write_status("completed")
        self._bidir_pool.shutdown(wait=False)
        self.record = record
        covered = cstats["n_unique_comparisons"]
        print(f"[{cfg.exp_id}] all-pairs done: {duration_s:.1f}s, "
              f"{covered}/{total_pairs} pairs stored, {c['n_llm_calls']} LLM calls, "
              f"eq {record['eq_rate']:.1%}, disagreement {record['disagreement_rate']:.1%}")
        if covered < total_pairs:
            print(f"  WARNING: matrix incomplete ({covered}/{total_pairs}) — "
                  f"some pairs missing (backend failures?).")
        return record
