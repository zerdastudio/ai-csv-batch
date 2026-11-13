"""The batch loop: read rows, ask the providers, save progress, log every call.

Kept free of any provider SDK, so the whole loop can be tested offline with MockProvider.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from providers import FatalError, RetryableError

COST_LOG_FIELDS = ["timestamp", "row_id", "provider", "model", "attempt", "outcome",
                   "input_tokens", "output_tokens", "cost_usd", "latency_ms", "error"]

# Shapes of API keys that should never reach a log, even if a provider echoes one back in an error message.
_KEY_SHAPES = re.compile(r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,})")
RETRY_AFTER_CAP = 300.0   # never wait longer than this on a provider's retry-after


def build_schema(fields: list[dict]) -> dict:
    """JSON schema for one answer: every field required, nothing else allowed."""
    props = {}
    for f in fields:
        if f.get("choices"):
            props[f["name"]] = {"type": "string", "enum": list(f["choices"])}
        else:
            props[f["name"]] = {"type": "string"}
            if f.get("description"):
                props[f["name"]]["description"] = f["description"]
    return {"type": "object", "properties": props,
            "required": [f["name"] for f in fields], "additionalProperties": False}


def build_system(instructions: str, fields: list[dict]) -> str:
    lines = [instructions.strip(), "", "Answer with a JSON object with these fields:"]
    for f in fields:
        if f.get("choices"):
            lines.append(f"- {f['name']}: one of {', '.join(f['choices'])}")
        else:
            lines.append(f"- {f['name']}: {f.get('description', 'text')}")
    return "\n".join(lines)


def check_answer(data: dict, fields: list[dict]) -> str | None:
    """Return a problem description, or None if the answer fits the fields."""
    for f in fields:
        value = data.get(f["name"])
        if not isinstance(value, str) or not value.strip():
            return f"field '{f['name']}' missing or empty"
        if f.get("choices") and value not in f["choices"]:
            return f"field '{f['name']}' has '{value}', not one of the choices"
    extra = set(data) - {f["name"] for f in fields}
    if extra:
        return f"unexpected fields: {', '.join(sorted(extra))}"
    return None


def mask_secrets(text: str, secrets: list[str]) -> str:
    for s in secrets:
        text = text.replace(s, "***")
    return _KEY_SHAPES.sub("***", text)


@dataclass
class Price:
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000


@dataclass
class Stats:
    done: int = 0
    skipped: int = 0                                   # answered in an earlier run, unchanged since
    changed: int = 0                                   # answered earlier, but the text or the task changed: redone
    empty: list = field(default_factory=list)          # rows with no text: never sent
    failed: list = field(default_factory=list)
    cost_usd: float = 0.0
    calls: dict = field(default_factory=dict)          # (provider, model, outcome) -> count


class Runner:
    def __init__(self, providers: list, prices: list[Price], fields: list[dict], instructions: str,
                 out_dir: Path, max_attempts: int = 4, base_delay: float = 1.0, max_delay: float = 30.0,
                 sleep=time.sleep, clock=time.monotonic, secrets: list[str] | None = None):
        if not providers:
            raise ValueError("at least one provider is needed")
        self.providers = providers
        self.prices = prices
        self.fields = fields
        self.schema = build_schema(fields)
        self.system = build_system(instructions, fields)
        self.out_dir = Path(out_dir)
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.sleep = sleep
        self.clock = clock
        self.secrets = secrets or []
        self.disabled: set[int] = set()   # positions of providers switched off for the rest of the run
        self.progress_path = self.out_dir / "progress.jsonl"
        self.cost_log_path = self.out_dir / "cost_log.csv"

    # ---------- progress (resume) ----------

    def row_key(self, text: str) -> str:
        """Fingerprint of what produced an answer: the row's text, the instructions and the fields.
        If any of them changes, the saved answer no longer belongs to the row and the row is done again."""
        blob = json.dumps([text, self.system, self.schema], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def load_progress(self) -> dict:
        """row_id -> {"key", "answer"}, the last record of each row winning."""
        done = {}
        if self.progress_path.exists():
            # errors="replace": a line cut inside a multi-byte character must not stop every later resume
            with self.progress_path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict) or not isinstance(rec.get("answer"), dict):
                            continue
                        done[str(rec["row_id"])] = {"key": rec.get("key"), "answer": rec["answer"]}
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue   # a line cut by a crash, or edited by hand: that row is simply done again
        return done

    def current_answers(self, rows: list[dict], id_column: str, text_column: str) -> dict:
        """row_id -> answer, only for saved answers that still match the row's text and the task."""
        saved = self.load_progress()
        out = {}
        for row in rows:
            row_id = str(row[id_column])
            rec = saved.get(row_id)
            text = row.get(text_column)
            if rec and text and rec["key"] == self.row_key(text):
                out[row_id] = rec["answer"]
        return out

    def _start_progress(self) -> None:
        """If a crash left the last line without its end, close it, so the next record starts on a line of its own."""
        if self.progress_path.exists() and self.progress_path.stat().st_size:
            with self.progress_path.open("rb") as fh:
                fh.seek(-1, 2)
                last = fh.read(1)
            if last != b"\n":
                with self.progress_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n")

    def _save_progress(self, row_id: str, key: str, answer: dict) -> None:
        with self.progress_path.open("a", encoding="utf-8") as fh:
            # ASCII only (accents become \u escapes): a crash cannot cut the file inside a character
            fh.write(json.dumps({"row_id": row_id, "key": key, "answer": answer}) + "\n")
            fh.flush()

    # ---------- cost log ----------

    def _log(self, stats: Stats, row_id, provider, price, attempt, outcome, tin=0, tout=0, ms=0, error=""):
        cost = price.cost(tin, tout)
        new = not self.cost_log_path.exists()
        with self.cost_log_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(COST_LOG_FIELDS)
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), row_id, provider.name,
                        provider.model, attempt, outcome, tin, tout, f"{cost:.6f}", ms,
                        mask_secrets(error, self.secrets)[:300]])
        key = (provider.name, provider.model, outcome)
        stats.calls[key] = stats.calls.get(key, 0) + 1
        stats.cost_usd += cost

    # ---------- one row ----------

    def process_row(self, row_id: str, text: str, stats: Stats) -> dict | None:
        """Try each provider in order; return the answer, or None if every provider failed."""
        for index, (provider, price) in enumerate(zip(self.providers, self.prices)):
            if index in self.disabled:
                continue
            for attempt in range(1, self.max_attempts + 1):
                t0 = self.clock()
                try:
                    reply = provider.complete(self.system, text, self.schema)
                except FatalError as e:
                    ms = int((self.clock() - t0) * 1000)
                    outcome = "refused" if str(e).startswith("refusal") else "fatal_error"
                    self._log(stats, row_id, provider, price, attempt, outcome, e.input_tokens,
                              e.output_tokens, ms, str(e))
                    if e.disable:
                        self.disabled.add(index)   # a bad key or model fails every row: stop using it
                    break                          # this row goes to the next provider
                except RetryableError as e:
                    ms = int((self.clock() - t0) * 1000)
                    outcome = "bad_output" if e.bad_output else "retryable_error"
                    self._log(stats, row_id, provider, price, attempt, outcome, e.input_tokens,
                              e.output_tokens, ms, str(e))
                    if attempt < self.max_attempts:
                        self.sleep(self._delay(attempt, e.retry_after))
                    continue
                ms = int((self.clock() - t0) * 1000)
                problem = check_answer(reply.data, self.fields)
                if problem:
                    self._log(stats, row_id, provider, price, attempt, "bad_output", reply.input_tokens,
                              reply.output_tokens, ms, problem)
                    if attempt < self.max_attempts:
                        self.sleep(self._delay(attempt))
                    continue
                self._log(stats, row_id, provider, price, attempt, "ok", reply.input_tokens, reply.output_tokens, ms)
                return reply.data
        return None

    def _delay(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, RETRY_AFTER_CAP)
        return min(self.base_delay * 2 ** (attempt - 1) + random.uniform(0, self.base_delay), self.max_delay)

    # ---------- the whole file ----------

    def run(self, rows: list[dict], id_column: str, text_column: str, limit: int | None = None) -> Stats:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._start_progress()
        saved = self.load_progress()
        stats = Stats()
        sent = 0
        for row in rows:
            row_id = str(row[id_column])
            text = row.get(text_column)
            if text is None or not text.strip():
                stats.empty.append(row_id)
                continue
            key = self.row_key(text)
            rec = saved.get(row_id)
            if rec and rec["key"] == key:
                stats.skipped += 1
                continue
            if limit is not None and sent >= limit:
                break
            if rec:
                stats.changed += 1
            sent += 1
            if len(self.disabled) == len(self.providers):
                stats.failed.append(row_id)
                continue
            answer = self.process_row(row_id, text, stats)
            if answer is None:
                stats.failed.append(row_id)
            else:
                self._save_progress(row_id, key, answer)
                stats.done += 1
        return stats


def read_rows(path: Path, id_column: str, text_column: str) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if rows:
        missing = [c for c in (id_column, text_column) if c not in rows[0]]
        if missing:
            raise ValueError(f"column(s) not found in {path}: {', '.join(missing)}")
    ids = [r[id_column] for r in rows]
    if any(i is None or not str(i).strip() for i in ids):
        raise ValueError(f"some rows have no value in the id column '{id_column}'")
    if len(ids) != len(set(ids)):
        raise ValueError(f"the id column '{id_column}' has repeated values: resume needs unique ids")
    return rows


def check_columns(rows: list[dict], fields: list[dict]) -> None:
    """Refuse output field names that already exist as input columns (they would be overwritten)."""
    columns = list(rows[0].keys()) if rows else []
    clash = [f["name"] for f in fields if f["name"] in columns]
    if clash:
        raise ValueError(f"output field(s) already exist as input columns: {', '.join(clash)}")


def write_output(path: Path, rows: list[dict], id_column: str, fields: list[dict], answers: dict) -> int:
    """Write the input rows plus one column per field; rows without an answer get empty cells.
    `answers` is row_id -> answer (see Runner.current_answers). Returns how many rows have an answer."""
    check_columns(rows, fields)
    names = [f["name"] for f in fields]
    columns = [c for c in (rows[0].keys() if rows else []) if c is not None]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    answered = 0
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns + names, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            answer = answers.get(str(row[id_column]))
            out = {c: row.get(c) for c in columns}
            out.update({n: (answer or {}).get(n, "") for n in names})
            answered += answer is not None
            w.writerow(out)
    tmp.replace(path)
    return answered
