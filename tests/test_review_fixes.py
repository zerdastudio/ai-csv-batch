"""Offline tests for the cases found in review: one class per case, no network, no keys.

    python -m unittest discover -s tests -v
"""
import csv
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic                      # noqa: E402
import httpx2                         # noqa: E402
from google.genai import errors as genai_errors, types as gt   # noqa: E402

import providers as prov              # noqa: E402
import run as cli                     # noqa: E402
from batch import Price, Runner, mask_secrets, write_output   # noqa: E402
from providers import FatalError, MockProvider, Reply   # noqa: E402

FIELDS = [{"name": "label", "choices": ["a", "b"]}]
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["a", "b"]}},
          "required": ["label"], "additionalProperties": False}
ROWS = [{"id": f"R{i}", "review": f"review number {i}"} for i in range(1, 6)]


def runner(providers, d, max_attempts=2, prices=None):
    prices = prices or [Price(4.0, 20.0) for _ in providers]
    return Runner(providers, prices, FIELDS, "Classify.", Path(d), max_attempts=max_attempts,
                  base_delay=0.0, max_delay=0.0, sleep=lambda s: None)


def log(d):
    with (Path(d) / "cost_log.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def stub(fn):
    return SimpleNamespace(messages=SimpleNamespace(create=fn),
                           chat=SimpleNamespace(completions=SimpleNamespace(create=fn)),
                           models=SimpleNamespace(generate_content=fn))


class OneBadRowDoesNotStopTheFile(unittest.TestCase):          # review D1
    def test_a_400_on_one_row_fails_that_row_only(self):
        req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
        p = prov.AnthropicProvider("claude-opus-5-5")
        sent = []

        def fn(**kw):
            text = kw["messages"][0]["content"]
            sent.append(text)
            if text == "review number 2":
                raise anthropic.BadRequestError("prompt is too long", response=httpx2.Response(400, request=req),
                                                body=None)
            return anthropic.types.Message.model_construct(
                id="m", type="message", role="assistant", model="m", stop_reason="end_turn", stop_sequence=None,
                content=[anthropic.types.TextBlock.model_construct(type="text", text='{"label": "a"}')],
                usage=anthropic.types.Usage.model_construct(input_tokens=10, output_tokens=5))
        p.client = stub(fn)
        with tempfile.TemporaryDirectory() as d:
            stats = runner([p], d).run(ROWS, "id", "review")
            self.assertEqual(stats.failed, ["R2"])
            self.assertEqual(stats.done, 4)

    def test_status_codes_that_switch_a_provider_off(self):
        for code, disable in [(401, True), (403, True), (404, True), (400, False), (413, False), (422, False)]:
            with self.subTest(code):
                err = prov._status_error(code, "x")
                self.assertIsInstance(err, FatalError)
                self.assertEqual(err.disable, disable)

    def test_gemini_wrong_key_400_switches_off(self):
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        p = prov.GeminiProvider("m")

        def raise_(**kw):
            raise genai_errors.APIError(400, {"error": {"message": "API key not valid. Please pass a valid API key.",
                                                        "status": "INVALID_ARGUMENT"}})
        p.client = stub(raise_)
        with self.assertRaises(FatalError) as cm:
            p.complete("s", "t", SCHEMA)
        self.assertTrue(cm.exception.disable)

    def test_mock_reject_keeps_the_provider(self):
        with tempfile.TemporaryDirectory() as d:
            p = MockProvider(script=["reject", "ok"])
            stats = runner([p], d).run(ROWS[:4], "id", "review")
            self.assertEqual((stats.done, len(stats.failed)), (2, 2))


class GeminiNetworkErrorsAreRetried(unittest.TestCase):       # review D2
    def test_refused_connection_and_read_timeout_fall_back(self):
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        import httpx
        for exc in (httpx.ConnectError("refused"), httpx.ReadTimeout("timed out"), httpx2.ConnectError("refused")):
            with self.subTest(type(exc).__name__), tempfile.TemporaryDirectory() as d:
                g = prov.GeminiProvider("m")

                def raise_(**kw):
                    raise exc
                g.client = stub(raise_)
                stats = runner([g, MockProvider()], d).run(ROWS[:2], "id", "review")
                self.assertEqual(stats.done, 2)
                self.assertIn("retryable_error", [r["outcome"] for r in log(d)])

    def test_real_sdk_against_a_closed_port(self):
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()                         # nothing listens there now
        from google import genai
        g = prov.GeminiProvider("m")
        g.client = genai.Client(http_options={"base_url": f"http://127.0.0.1:{port}"})
        with self.assertRaises(prov.RetryableError):
            g.complete("s", "t", SCHEMA)


class PaidAnswersThatAreThrownAwayAreCounted(unittest.TestCase):   # review D3
    def test_anthropic_cut_answer_carries_its_tokens(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
        p = prov.AnthropicProvider("m")
        p.client = stub(lambda **k: anthropic.types.Message.model_construct(
            id="m", type="message", role="assistant", model="m", stop_reason="max_tokens", stop_sequence=None,
            content=[anthropic.types.TextBlock.model_construct(type="text", text='{"lab')],
            usage=anthropic.types.Usage.model_construct(input_tokens=120, output_tokens=2000)))
        with tempfile.TemporaryDirectory() as d:
            runner([p], d, max_attempts=1).run(ROWS[:1], "id", "review")
            row = log(d)[0]
            self.assertEqual((row["outcome"], row["input_tokens"], row["output_tokens"]), ("bad_output", "120", "2000"))
            self.assertAlmostEqual(float(row["cost_usd"]), (120 * 4 + 2000 * 20) / 1e6)

    def test_not_json_carries_its_tokens_on_every_provider(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage
        o = prov.OpenAIProvider("m")
        o.client = stub(lambda **k: ChatCompletion.model_construct(
            id="c", object="chat.completion", created=0, model="m",
            choices=[Choice.model_construct(index=0, finish_reason="stop", message=ChatCompletionMessage.model_construct(
                role="assistant", content="not json", refusal=None))],
            usage=CompletionUsage.model_construct(prompt_tokens=30, completion_tokens=7, total_tokens=37)))
        g = prov.GeminiProvider("m")
        g.client = stub(lambda **k: gt.GenerateContentResponse(
            candidates=[gt.Candidate(finish_reason=gt.FinishReason.MAX_TOKENS,
                                     content=gt.Content(role="model", parts=[gt.Part(text='{"la')]))],
            usage_metadata=gt.GenerateContentResponseUsageMetadata(prompt_token_count=40, candidates_token_count=1,
                                                                   thoughts_token_count=3999)))
        for p, expected in [(o, (30, 7)), (g, (40, 4000))]:
            with self.subTest(p.name):
                with self.assertRaises(prov.RetryableError) as cm:
                    p.complete("s", "t", SCHEMA)
                self.assertEqual((cm.exception.input_tokens, cm.exception.output_tokens), expected)


class ResumeAfterACutLine(unittest.TestCase):                  # review D4
    def test_record_after_a_cut_line_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            r = runner([MockProvider()], d)
            r.run(ROWS[:2], "id", "review")
            with r.progress_path.open("a", encoding="utf-8") as fh:
                fh.write('{"row_id": "R3", "ans')
            runner([MockProvider()], d).run(ROWS[:3], "id", "review")
            third = MockProvider()
            stats = runner([third], d).run(ROWS[:3], "id", "review")
            self.assertEqual((third.calls, stats.skipped), (0, 3))


class NoKeyForAnthropic(unittest.TestCase):                    # review D5
    def test_no_credentials_switches_off_and_falls_back(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
                os.environ.pop(v, None)
            p = prov.AnthropicProvider("m")

            def fn(**kw):
                raise TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token"')
            p.client = stub(fn)
            with tempfile.TemporaryDirectory() as d:
                stats = runner([p, MockProvider()], d).run(ROWS[:3], "id", "review")
                self.assertEqual(stats.done, 3)
                self.assertEqual([r["outcome"] for r in log(d)].count("fatal_error"), 1)

    def test_another_type_error_is_not_hidden(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
        p = prov.AnthropicProvider("m")

        def fn(**kw):
            raise TypeError("unexpected keyword argument")
        p.client = stub(fn)
        with self.assertRaises(TypeError):
            p.complete("s", "t", SCHEMA)


class ContentBlocksAreRefusals(unittest.TestCase):             # review D6
    def test_one_call_for_each_kind_of_block(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        o = prov.OpenAIProvider("m")
        o.client = stub(lambda **k: ChatCompletion.model_construct(
            id="c", object="chat.completion", created=0, model="m", usage=None,
            choices=[Choice.model_construct(index=0, finish_reason="content_filter",
                                            message=ChatCompletionMessage.model_construct(role="assistant",
                                                                                          content=None, refusal=None))]))
        g1 = prov.GeminiProvider("m")
        g1.client = stub(lambda **k: gt.GenerateContentResponse(
            candidates=[gt.Candidate(finish_reason=gt.FinishReason.SAFETY)]))
        g2 = prov.GeminiProvider("m")
        g2.client = stub(lambda **k: gt.GenerateContentResponse(
            prompt_feedback=gt.GenerateContentResponsePromptFeedback(block_reason=gt.BlockedReason.SAFETY)))
        for p in (o, g1, g2):
            with self.subTest(p.name), tempfile.TemporaryDirectory() as d:
                stats = runner([p], d, max_attempts=4).run(ROWS[:2], "id", "review")
                outcomes = [r["outcome"] for r in log(d)]
                self.assertEqual(outcomes, ["refused", "refused"])     # one call per row, provider still on
                self.assertEqual(len(stats.failed), 2)


class ChangedInputIsDoneAgain(unittest.TestCase):              # review D8
    def test_same_ids_new_text(self):
        with tempfile.TemporaryDirectory() as d:
            runner([MockProvider()], d).run(ROWS[:2], "id", "review")
            new_rows = [{"id": "R1", "review": "a different text"}, ROWS[1]]
            p = MockProvider()
            stats = runner([p], d).run(new_rows, "id", "review")
            self.assertEqual((p.calls, stats.changed, stats.skipped), (1, 1, 1))

    def test_a_new_field_redoes_every_row(self):
        with tempfile.TemporaryDirectory() as d:
            runner([MockProvider()], d).run(ROWS[:2], "id", "review")
            more = FIELDS + [{"name": "note", "description": "text"}]
            r = Runner([MockProvider()], [Price()], more, "Classify.", Path(d), sleep=lambda s: None)
            stats = r.run(ROWS[:2], "id", "review")
            self.assertEqual((stats.done, stats.changed), (2, 2))
            out = Path(d) / "o.csv"
            write_output(out, ROWS[:2], "id", more, r.current_answers(ROWS[:2], "id", "review"))
            with out.open(newline="", encoding="utf-8") as fh:
                self.assertTrue(all(row["note"] for row in csv.DictReader(fh)))

    def test_old_answers_never_reach_the_output_of_a_changed_row(self):
        with tempfile.TemporaryDirectory() as d:
            r = runner([MockProvider()], d)
            r.run(ROWS[:1], "id", "review")
            changed = [{"id": "R1", "review": "a different text"}]
            self.assertEqual(r.current_answers(changed, "id", "review"), {})


class EmptyOrMissingText(unittest.TestCase):                   # review D9
    def test_dry_run_with_a_missing_cell(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "in.csv").write_text("id,review\nR1,good\nR2\nR3,  \n", encoding="utf-8")
            cfg = (ROOT / "config.example.toml").read_text(encoding="utf-8")
            cfg = cfg.replace('input = "data/reviews.csv"', 'input = "in.csv"')
            (Path(d) / "config.toml").write_text(cfg, encoding="utf-8")
            code = cli.main(["--config", str(Path(d) / "config.toml"), "--dry-run"])
            self.assertEqual(code, 0)
            with (Path(d) / "out" / "dry_run" / "reviews_labeled.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([bool(r["sentiment"]) for r in rows], [True, False, False])


class DotEnvForms(unittest.TestCase):                          # review D10
    def test_common_forms(self):
        names = ["GIGT_A", "GIGT_B", "GIGT_C", "GIGT_D", "GIGT_E", "GIGT_F"]
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}, clear=False):
            for n in names:
                os.environ.pop(n, None)
            os.environ["GIGT_F"] = "from-environment"
            f = Path(d) / ".env"
            f.write_text("GIGT_A=first\n"
                         "export GIGT_B=exported\n"
                         "GIGT_C=val   # a comment\n"
                         'GIGT_D="quoted # not a comment"\n'
                         "GIGT_E=a=b\n"
                         "GIGT_F=from-file\n", encoding="utf-8-sig")       # with a BOM, like Windows Notepad
            cli.load_dotenv(f)
            self.assertEqual([os.environ.get(n) for n in names],
                             ["first", "exported", "val", "quoted # not a comment", "a=b", "from-environment"])


class KeysNeverReachTheLog(unittest.TestCase):                 # review point 4
    def test_masking(self):
        self.assertEqual(mask_secrets("bad key sk-proj-abcdef123456", []), "bad key ***")
        self.assertEqual(mask_secrets("key AIzaSyA1234567890abcdefghijk invalid", []), "key *** invalid")
        self.assertEqual(mask_secrets("echo my-secret-value back", ["my-secret-value"]), "echo *** back")

    def test_a_provider_echoing_the_key_is_masked_in_the_log(self):
        class Echo:
            name, model = "echo", "m"

            def complete(self, system, text, schema):
                raise FatalError("invalid key my-secret-value-123", disable=True)
        with tempfile.TemporaryDirectory() as d:
            r = Runner([Echo()], [Price()], FIELDS, "x", Path(d), sleep=lambda s: None,
                       secrets=["my-secret-value-123"])
            r.run(ROWS[:1], "id", "review")
            self.assertNotIn("my-secret-value-123", (Path(d) / "cost_log.csv").read_text(encoding="utf-8"))


class RetryAfterIsHonoured(unittest.TestCase):                 # review, opinion on 429
    def test_header_sets_the_pause(self):
        req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        err = prov._status_error(429, "x", httpx2.Response(429, request=req, headers={"retry-after": "7"}))
        self.assertEqual(err.retry_after, 7.0)
        pauses = []

        class Limited:
            name, model, calls = "l", "m", 0

            def complete(self, system, text, schema):
                self.calls += 1
                if self.calls == 1:
                    raise prov.RetryableError("429", retry_after=7.0)
                return Reply({"label": "a"}, 1, 1)
        with tempfile.TemporaryDirectory() as d:
            Runner([Limited()], [Price()], FIELDS, "x", Path(d), sleep=pauses.append).run(ROWS[:1], "id", "review")
        self.assertEqual(pauses, [7.0])


class StartupErrorsAreMessages(unittest.TestCase):             # review, opinion on run.py
    def test_repeated_id_is_a_message_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "in.csv").write_text("id,review\n1,a\n1,b\n", encoding="utf-8")
            cfg = (ROOT / "config.example.toml").read_text(encoding="utf-8").replace(
                'input = "data/reviews.csv"', 'input = "in.csv"')
            (Path(d) / "config.toml").write_text(cfg, encoding="utf-8")
            self.assertEqual(cli.main(["--config", str(Path(d) / "config.toml"), "--dry-run"]), 2)


class OutputIsWrittenEvenIfTheRunBreaks(unittest.TestCase):
    def test_answers_saved_before_an_unexpected_error_reach_the_output(self):
        class Breaks:
            name, model, calls = "b", "m", 0

            def complete(self, system, text, schema):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("something nobody expected")
                return Reply({"sentiment": "positive", "topic": "quality", "summary": "x"}, 1, 1)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "in.csv").write_text("id,review\n1,a\n2,b\n3,c\n", encoding="utf-8")
            cfg = (ROOT / "config.example.toml").read_text(encoding="utf-8").replace(
                'input = "data/reviews.csv"', 'input = "in.csv"')
            (Path(d) / "config.toml").write_text(cfg, encoding="utf-8")
            with mock.patch.object(prov, "build", lambda spec: Breaks()), mock.patch.object(
                    prov, "key_present", lambda name: True):
                with self.assertRaises(RuntimeError):
                    cli.main(["--config", str(Path(d) / "config.toml")])
            with (Path(d) / "out" / "reviews_labeled.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["sentiment"] for r in rows], ["positive", "positive", ""])



class ResumeSurvivesACutInsideACharacter(unittest.TestCase):    # review round 2, G2-1
    def test_cut_inside_an_accented_letter(self):
        with tempfile.TemporaryDirectory() as d:
            r = runner([MockProvider()], d)
            r._save_progress("R0", "k", {"label": "café"})
            text = r.progress_path.read_text(encoding="utf-8")
            self.assertTrue(text.isascii())                      # written as \u escapes: nothing to cut inside
            with r.progress_path.open("ab") as fh:              # an older or foreign file, cut inside "é"
                fh.write(b'{"row_id": "R9", "key": "k", "answer": {"label": "caf\xc3')
            self.assertEqual(r.load_progress()["R0"]["answer"], {"label": "café"})
            stats = runner([MockProvider()], d).run(ROWS[:2], "id", "review")
            self.assertEqual(stats.done, 2)


class ProgressLinesThatAreNotObjects(unittest.TestCase):      # review round 2, G2-4
    def test_lines_edited_by_hand(self):
        with tempfile.TemporaryDirectory() as d:
            r = runner([MockProvider()], d)
            r.run(ROWS[:1], "id", "review")
            with r.progress_path.open("a", encoding="utf-8") as fh:
                fh.write('[1, 2]\nnull\n"x"\n7\n{"row_id": "R2", "answer": "not an object"}\n')
            self.assertEqual(set(r.load_progress()), {"R1"})
            stats = runner([MockProvider()], d).run(ROWS[:2], "id", "review")
            self.assertEqual((stats.skipped, stats.done), (1, 1))


class BodiesThatAreNotTheProvidersJson(unittest.TestCase):    # review round 2, G2-2
    def test_a_str_body_is_retried_and_falls_back(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
        os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
        for p in (prov.AnthropicProvider("m"), prov.OpenAIProvider("m")):
            with self.subTest(p.name), tempfile.TemporaryDirectory() as d:
                p.client = stub(lambda **k: "<html>proxy login</html>")
                stats = runner([p, MockProvider()], d).run(ROWS[:1], "id", "review")
                self.assertEqual(stats.done, 1)
                self.assertEqual([r["outcome"] for r in log(d)][:2], ["retryable_error", "retryable_error"])

    def test_gemini_json_decode_error_is_retried(self):
        import json
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        g = prov.GeminiProvider("m")

        def fn(**kw):
            json.loads("<html>")
        g.client = stub(fn)
        with self.assertRaises(prov.RetryableError):
            g.complete("s", "t", SCHEMA)


class GeminiPauseIsRead(unittest.TestCase):                   # review round 2, G2-3
    def test_retry_info_in_the_body_and_header(self):
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        g = prov.GeminiProvider("m")
        body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                          "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}}

        def fn(**kw):
            raise genai_errors.APIError(429, body)
        g.client = stub(fn)
        with self.assertRaises(prov.RetryableError) as cm:
            g.complete("s", "t", SCHEMA)
        self.assertEqual(cm.exception.retry_after, 17.0)
        resp = httpx2.Response(429, request=httpx2.Request("POST", "https://x"), headers={"retry-after": "7"})

        def fn2(**kw):
            raise genai_errors.APIError(429, {"error": {"message": "quota"}}, response=resp)
        g.client = stub(fn2)
        with self.assertRaises(prov.RetryableError) as cm:
            g.complete("s", "t", SCHEMA)
        self.assertEqual(cm.exception.retry_after, 7.0)


class OutOfCreditSwitchesOff(unittest.TestCase):              # review round 2, opinion
    def test_known_messages(self):
        a = prov._status_error(400, "Your credit balance is too low to access the Anthropic API.")
        o = prov._status_error(429, "Error code: 429 - {'error': {'code': 'insufficient_quota'}}")
        plain = prov._status_error(429, "rate limited")
        self.assertTrue(isinstance(a, FatalError) and a.disable)
        self.assertTrue(isinstance(o, FatalError) and o.disable)
        self.assertIsInstance(plain, prov.RetryableError)

    def test_mask_has_a_word_boundary(self):
        self.assertEqual(mask_secrets("task-12345678 failed", []), "task-12345678 failed")


if __name__ == "__main__":
    unittest.main()
