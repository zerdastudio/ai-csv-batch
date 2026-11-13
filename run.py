"""Run a CSV through an AI model, row by row.

    python run.py --config config.toml
    python run.py --config config.toml --limit 10      # try the first 10 rows not done yet
    python run.py --config config.toml --dry-run       # offline, with simulated answers

Running the same command again resumes: rows already answered are not sent again.
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

import providers as prov
from batch import Price, Runner, check_columns, read_rows, write_output


def load_dotenv(path: Path) -> None:
    """Load KEY=value lines from a .env file into the environment, without overwriting what is already set.
    Accepts `export KEY=value`, quoted values, comments after an unquoted value, and a file saved with a BOM."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] in ('"', "'"):
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            hash_at = value.find(" #")
            if hash_at >= 0:
                value = value[:hash_at].rstrip()
        if key and value and key not in os.environ:
            os.environ[key] = value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a CSV through an AI model, row by row.")
    ap.add_argument("--config", default="config.toml", help="settings file (see config.example.toml)")
    ap.add_argument("--limit", type=int, default=None, help="process at most this many new rows")
    ap.add_argument("--dry-run", action="store_true", help="use simulated answers: no network, no cost")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}. Copy config.example.toml to config.toml and edit it.")
        return 2
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    load_dotenv(base / ".env")

    task, run = cfg["task"], cfg.get("run", {})
    fields = cfg["fields"]
    specs = [{"name": "mock"}] if args.dry_run else cfg["providers"]
    out_dir = base / run.get("output_dir", "out")
    if args.dry_run:
        out_dir = out_dir / "dry_run"

    for s in specs:
        if not prov.key_present(s["name"]):
            print(f"Note: no API key for '{s['name']}' in the environment or .env.")

    try:
        rows = read_rows(base / task["input"], task["id_column"], task["text_column"])
        check_columns(rows, fields)
        providers = [prov.build(s) for s in specs]
    except Exception as e:   # bad input, a missing key, a wrong name: stop before the first row, not halfway
        print(f"Cannot start: {e}")
        return 2
    prices = [Price(s.get("price_input_per_mtok", 0.0), s.get("price_output_per_mtok", 0.0)) for s in specs]
    runner = Runner(providers, prices, fields, task["instructions"], out_dir,
                    max_attempts=run.get("max_attempts", 4), base_delay=run.get("base_delay", 1.0),
                    max_delay=run.get("max_delay", 30.0), secrets=prov.secrets_in_environment())

    output = out_dir / task.get("output_name", "output.csv")
    try:
        stats = runner.run(rows, task["id_column"], task["text_column"], limit=args.limit)
    finally:   # whatever happens, the answers saved so far reach the output file
        answered = write_output(output, rows, task["id_column"], fields,
                                runner.current_answers(rows, task["id_column"], task["text_column"]))

    print(f"Rows answered this run: {stats.done}   already done before: {stats.skipped}   "
          f"failed: {len(stats.failed)}")
    if stats.changed:
        print(f"Rows answered before but changed since (text, instructions or fields), done again: {stats.changed}")
    if stats.empty:
        print(f"Rows with no text, not sent: {len(stats.empty)} ({', '.join(stats.empty[:10])}"
              + (" ..." if len(stats.empty) > 10 else "") + ")")
    print(f"Answered in total: {answered} of {len(rows)}   ->  {output}")
    print(f"Cost of this run: ${stats.cost_usd:.4f}   (every call is in {runner.cost_log_path})")
    for (name, model, outcome), n in sorted(stats.calls.items()):
        print(f"  {name:10} {model:24} {outcome:16} {n}")
    if stats.failed:
        print(f"Failed rows (run again to retry them): {', '.join(stats.failed[:20])}"
              + (" ..." if len(stats.failed) > 20 else ""))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
