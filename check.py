"""Compare the output with the answers you expect for a few rows.

    python check.py out/reviews_labeled.csv data/expected.csv

`expected.csv` has the id column and any of the output fields; a cell left empty is not checked
(useful for free-text fields such as a summary). Exit code 0 only if every checked cell matches.
"""
from __future__ import annotations

import csv
import sys


def compare(output_path: str, expected_path: str, id_column: str = "id") -> tuple[int, int, list[str]]:
    with open(output_path, newline="", encoding="utf-8-sig") as fh:
        out = {r[id_column]: r for r in csv.DictReader(fh)}
    checked, matched, problems = 0, 0, []
    with open(expected_path, newline="", encoding="utf-8-sig") as fh:
        for exp in csv.DictReader(fh):
            row_id = exp[id_column]
            got = out.get(row_id)
            for col, want in exp.items():
                if col == id_column or not want.strip():
                    continue
                checked += 1
                have = (got or {}).get(col, "")
                if got is None:
                    problems.append(f"row {row_id}: not in the output")
                elif have.strip().lower() == want.strip().lower():
                    matched += 1
                else:
                    problems.append(f"row {row_id}, {col}: expected '{want}', got '{have or '(empty)'}'")
    return checked, matched, problems


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    checked, matched, problems = compare(argv[0], argv[1], *(argv[2:3]))
    print(f"Checked cells: {checked}   matching: {matched}")
    for p in problems:
        print("  " + p)
    return 0 if checked and matched == checked else 1


if __name__ == "__main__":
    sys.exit(main())
