"""Offline tests: no network, no keys, no cost.

    python -m unittest discover -s tests -v
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import check                                   # noqa: E402
import run as cli                              # noqa: E402
from batch import Price, Runner, build_schema, check_answer, read_rows, write_output   # noqa: E402
from providers import FatalError, MockProvider, Reply   # noqa: E402

FIELDS = [{"name": "sentiment", "choices": ["positive", "neutral", "negative"]},
          {"name": "summary", "description": "one sentence"}]
ROWS = [{"id": f"R{i}", "review": f"review number {i}"} for i in range(1, 6)]


def make_runner(providers, out_dir, prices=None, max_attempts=3):
    prices = prices or [Price(1.0, 2.0) for _ in providers]
    return Runner(providers, prices, FIELDS, "Classify.", Path(out_dir), max_attempts=max_attempts,
                  base_delay=0.01, max_delay=0.01, sleep=lambda s: None)


def cost_log(out_dir):
    with (Path(out_dir) / "cost_log.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Answers(unittest.TestCase):
    def test_schema_requires_every_field_and_nothing_else(self):
        s = build_schema(FIELDS)
        self.assertEqual(s["required"], ["sentiment", "summary"])
        self.assertFalse(s["additionalProperties"])
        self.assertEqual(s["properties"]["sentiment"]["enum"], ["positive", "neutral", "negative"])

    def test_check_answer(self):
        self.assertIsNone(check_answer({"sentiment": "positive", "summary": "ok"}, FIELDS))
        self.assertIn("not one of", check_answer({"sentiment": "great", "summary": "ok"}, FIELDS))
        self.assertIn("missing", check_answer({"sentiment": "positive"}, FIELDS))
        self.assertIn("missing", check_answer({"sentiment": "positive", "summary": "  "}, FIELDS))
        self.assertIn("unexpected", check_answer({"sentiment": "positive", "summary": "x", "x": "y"}, FIELDS))


class Retries(unittest.TestCase):
    def test_retry_then_success(self):
        with tempfile.TemporaryDirectory() as d:
            p = MockProvider(script=["retry", "retry", "ok"])
            stats = make_runner([p], d).run(ROWS[:1], "id", "review")
            self.assertEqual(stats.done, 1)
            self.assertEqual([r["outcome"] for r in cost_log(d)], ["retryable_error", "retryable_error", "ok"])

    def test_bad_json_is_retried_and_logged_as_bad_output(self):
        with tempfile.TemporaryDirectory() as d:
            p = MockProvider(script=["bad", "ok"])
            stats = make_runner([p], d).run(ROWS[:1], "id", "review")
            self.assertEqual(stats.done, 1)
            self.assertEqual([r["outcome"] for r in cost_log(d)], ["bad_output", "ok"])

    def test_answer_outside_the_choices_is_retried_and_its_cost_counted(self):
        class Wrong:
            name, model, calls = "wrong", "m", 0

            def complete(self, system, text, schema):
                self.calls += 1
                if self.calls == 1:
                    return Reply({"sentiment": "great", "summary": "x"}, 100, 10)
                return Reply({"sentiment": "positive", "summary": "x"}, 100, 10)
        with tempfile.TemporaryDirectory() as d:
            stats = make_runner([Wrong()], d).run(ROWS[:1], "id", "review")
            log = cost_log(d)
            self.assertEqual([r["outcome"] for r in log], ["bad_output", "ok"])
            self.assertGreater(float(log[0]["cost_usd"]), 0)       # a bad answer is still paid for
            self.assertAlmostEqual(stats.cost_usd, 2 * (100 * 1.0 + 10 * 2.0) / 1e6)


class Fallback(unittest.TestCase):
    def test_second_provider_takes_over_after_retries_run_out(self):
        with tempfile.TemporaryDirectory() as d:
            first = MockProvider(name="first", script=["retry"])
            second = MockProvider(name="second", script=["ok"])
            stats = make_runner([first, second], d, max_attempts=3).run(ROWS[:2], "id", "review")
            self.assertEqual(stats.done, 2)
            self.assertEqual(first.calls, 6)     # 3 tries per row, both rows
            self.assertEqual(second.calls, 2)

    def test_fatal_error_switches_the_provider_off_for_the_rest_of_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            first = MockProvider(name="first", script=["fatal"])
            second = MockProvider(name="second", script=["ok"])
            stats = make_runner([first, second], d).run(ROWS, "id", "review")
            self.assertEqual(stats.done, 5)
            self.assertEqual(first.calls, 1)     # a bad key is tried once, not once per row

    def test_two_providers_with_the_same_name_are_kept_apart(self):
        with tempfile.TemporaryDirectory() as d:
            first = MockProvider(name="anthropic", script=["fatal"])
            second = MockProvider(name="anthropic", script=["ok"])
            stats = make_runner([first, second], d).run(ROWS[:2], "id", "review")
            self.assertEqual(stats.done, 2)

    def test_all_providers_off_fails_rows_without_calling(self):
        with tempfile.TemporaryDirectory() as d:
            p = MockProvider(script=["fatal"])
            stats = make_runner([p], d).run(ROWS, "id", "review")
            self.assertEqual(stats.done, 0)
            self.assertEqual(len(stats.failed), 5)
            self.assertEqual(p.calls, 1)

    def test_refusal_does_not_switch_the_provider_off(self):
        class Refuses:
            name, model, calls = "r", "m", 0

            def complete(self, system, text, schema):
                self.calls += 1
                if self.calls == 1:
                    raise FatalError("refusal: the model declined this row", disable=False)
                return Reply({"sentiment": "neutral", "summary": "x"}, 1, 1)
        with tempfile.TemporaryDirectory() as d:
            p = Refuses()
            stats = make_runner([p], d).run(ROWS[:2], "id", "review")
            self.assertEqual(stats.failed, ["R1"])
            self.assertEqual(stats.done, 1)


class Resume(unittest.TestCase):
    def test_second_run_does_not_call_again_for_done_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = MockProvider()
            make_runner([p1], d).run(ROWS, "id", "review", limit=3)
            self.assertEqual(p1.calls, 3)
            p2 = MockProvider()
            stats = make_runner([p2], d).run(ROWS, "id", "review")
            self.assertEqual(p2.calls, 2)
            self.assertEqual((stats.done, stats.skipped), (2, 3))

    def test_failed_rows_are_retried_on_the_next_run(self):
        with tempfile.TemporaryDirectory() as d:
            make_runner([MockProvider(script=["fatal"])], d).run(ROWS[:2], "id", "review")
            stats = make_runner([MockProvider()], d).run(ROWS[:2], "id", "review")
            self.assertEqual(stats.done, 2)

    def test_a_line_cut_by_a_crash_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            r = make_runner([MockProvider()], d)
            r.run(ROWS[:2], "id", "review")
            with r.progress_path.open("a", encoding="utf-8") as fh:
                fh.write('{"row_id": "R3", "ans')
            self.assertEqual(set(r.load_progress()), {"R1", "R2"})
            stats = make_runner([MockProvider()], d).run(ROWS[:3], "id", "review")
            self.assertEqual(stats.done, 1)                     # R3, after the cut line
            again = MockProvider()
            make_runner([again], d).run(ROWS[:3], "id", "review")
            self.assertEqual(again.calls, 0)                    # and R3 is not paid a second time

    def test_output_has_answers_and_empty_cells_for_rows_not_done(self):
        with tempfile.TemporaryDirectory() as d:
            r = make_runner([MockProvider()], d)
            r.run(ROWS, "id", "review", limit=2)
            out = Path(d) / "out.csv"
            answered = write_output(out, ROWS, "id", FIELDS, r.current_answers(ROWS, "id", "review"))
            with out.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(answered, 2)
            self.assertEqual(rows[0]["sentiment"], "positive")
            self.assertEqual(rows[4]["sentiment"], "")
            self.assertEqual(list(rows[0].keys()), ["id", "review", "sentiment", "summary"])


class Inputs(unittest.TestCase):
    def test_repeated_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "in.csv"
            f.write_text("id,review\n1,a\n1,b\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(f, "id", "review")

    def test_missing_column_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "in.csv"
            f.write_text("id,text\n1,a\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(f, "id", "review")

    def test_output_field_clashing_with_an_input_column_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [{"id": "1", "review": "a", "summary": "old"}]
            with self.assertRaises(ValueError):
                write_output(Path(d) / "o.csv", rows, "id", FIELDS, {})


class EndToEnd(unittest.TestCase):
    def test_dry_run_on_the_sample_data(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = (ROOT / "config.example.toml").read_text(encoding="utf-8")
            cfg = cfg.replace('input = "data/reviews.csv"', f'input = "{(ROOT / "data" / "reviews.csv").as_posix()}"')
            (Path(d) / "config.toml").write_text(cfg, encoding="utf-8")
            code = cli.main(["--config", str(Path(d) / "config.toml"), "--dry-run", "--limit", "10"])
            self.assertEqual(code, 0)
            out = Path(d) / "out" / "dry_run" / "reviews_labeled.csv"
            with out.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 100)
            self.assertEqual(sum(1 for r in rows if r["sentiment"]), 10)

    def test_check_counts_matches_and_skips_empty_cells(self):
        with tempfile.TemporaryDirectory() as d:
            out, exp = Path(d) / "o.csv", Path(d) / "e.csv"
            out.write_text("id,sentiment,summary\nR1,positive,x\nR2,neutral,y\n", encoding="utf-8")
            exp.write_text("id,sentiment,summary\nR1,Positive,\nR2,negative,\nR9,neutral,\n", encoding="utf-8")
            checked, matched, problems = check.compare(str(out), str(exp))
            self.assertEqual((checked, matched), (3, 1))
            self.assertEqual(len(problems), 2)


if __name__ == "__main__":
    unittest.main()
