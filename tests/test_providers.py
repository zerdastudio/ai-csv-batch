"""Offline tests of the provider adapters, with the SDKs' own response and error types.

No network: the SDK client is replaced by a stub that returns a real SDK object or raises a real
SDK exception. This checks that each adapter reads the fields the SDK actually has, and sorts every
error into "try again" or "give up".
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic                      # noqa: E402
import httpx2                         # noqa: E402  (installed with anthropic)
import openai                         # noqa: E402
from google.genai import errors as genai_errors   # noqa: E402

import providers as prov              # noqa: E402

SCHEMA = {"type": "object", "properties": {"label": {"type": "string"}},
          "required": ["label"], "additionalProperties": False}


def stub(fn):
    """An object with .messages.create / .chat.completions.create / .models.generate_content = fn."""
    return SimpleNamespace(messages=SimpleNamespace(create=fn),
                           chat=SimpleNamespace(completions=SimpleNamespace(create=fn)),
                           models=SimpleNamespace(generate_content=fn))


def raiser(exc):
    def fn(**kwargs):
        raise exc
    return fn


class AnthropicAdapter(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
        self.p = prov.AnthropicProvider("claude-opus-5-5", effort="low")
        self.req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def message(self, text, stop="end_turn"):
        return anthropic.types.Message.model_construct(
            id="msg_1", type="message", role="assistant", model="claude-opus-5-5",
            content=[anthropic.types.TextBlock.model_construct(type="text", text=text)],
            stop_reason=stop, stop_sequence=None,
            usage=anthropic.types.Usage.model_construct(input_tokens=120, output_tokens=30))

    def test_reads_text_and_usage_and_sends_schema_and_effort(self):
        seen = {}

        def fn(**kwargs):
            seen.update(kwargs)
            return self.message('{"label": "positive"}')
        self.p.client = stub(fn)
        r = self.p.complete("sys", "text", SCHEMA)
        self.assertEqual((r.data, r.input_tokens, r.output_tokens), ({"label": "positive"}, 120, 30))
        self.assertEqual(seen["output_config"]["format"], {"type": "json_schema", "schema": SCHEMA})
        self.assertEqual(seen["output_config"]["effort"], "low")
        self.assertEqual(seen["system"], "sys")

    def test_refusal_and_truncation(self):
        self.p.client = stub(lambda **k: self.message("", stop="refusal"))
        with self.assertRaises(prov.FatalError):
            self.p.complete("s", "t", SCHEMA)
        self.p.client = stub(lambda **k: self.message('{"lab', stop="max_tokens"))
        with self.assertRaisesRegex(prov.RetryableError, "bad_output"):
            self.p.complete("s", "t", SCHEMA)

    def test_error_mapping(self):
        def status(cls, code):
            return cls("x", response=httpx2.Response(code, request=self.req), body=None)
        cases = [
            (status(anthropic.RateLimitError, 429), prov.RetryableError),
            (status(anthropic.InternalServerError, 500), prov.RetryableError),
            (status(anthropic.APIStatusError, 529), prov.RetryableError),      # overloaded
            (anthropic.APIConnectionError(request=self.req), prov.RetryableError),
            (anthropic.APITimeoutError(request=self.req), prov.RetryableError),
            (status(anthropic.AuthenticationError, 401), prov.FatalError),
            (status(anthropic.NotFoundError, 404), prov.FatalError),
            (status(anthropic.BadRequestError, 400), prov.FatalError),
        ]
        for exc, expected in cases:
            with self.subTest(type(exc).__name__):
                self.p.client = stub(raiser(exc))
                with self.assertRaises(expected):
                    self.p.complete("s", "t", SCHEMA)


class OpenAIAdapter(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
        self.p = prov.OpenAIProvider("some-model")
        self.req = openai._base_client.httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    def completion(self, content, finish="stop", refusal=None):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage
        msg = ChatCompletionMessage.model_construct(role="assistant", content=content, refusal=refusal)
        return ChatCompletion.model_construct(
            id="c1", object="chat.completion", created=0, model="some-model",
            choices=[Choice.model_construct(index=0, message=msg, finish_reason=finish)],
            usage=CompletionUsage.model_construct(prompt_tokens=80, completion_tokens=12, total_tokens=92))

    def test_reads_content_and_usage(self):
        self.p.client = stub(lambda **k: self.completion('{"label": "negative"}'))
        r = self.p.complete("s", "t", SCHEMA)
        self.assertEqual((r.data, r.input_tokens, r.output_tokens), ({"label": "negative"}, 80, 12))

    def test_refusal_length_and_errors(self):
        self.p.client = stub(lambda **k: self.completion(None, refusal="no"))
        with self.assertRaises(prov.FatalError):
            self.p.complete("s", "t", SCHEMA)
        self.p.client = stub(lambda **k: self.completion('{"la', finish="length"))
        with self.assertRaises(prov.RetryableError):
            self.p.complete("s", "t", SCHEMA)
        resp = openai._base_client.httpx.Response(429, request=self.req)
        self.p.client = stub(raiser(openai.RateLimitError("x", response=resp, body=None)))
        with self.assertRaises(prov.RetryableError):
            self.p.complete("s", "t", SCHEMA)
        resp = openai._base_client.httpx.Response(401, request=self.req)
        self.p.client = stub(raiser(openai.AuthenticationError("x", response=resp, body=None)))
        with self.assertRaises(prov.FatalError):
            self.p.complete("s", "t", SCHEMA)


class GeminiAdapter(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
        self.p = prov.GeminiProvider("some-model")

    def test_reads_text_and_usage_including_thinking(self):
        from google.genai import types
        resp = types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(role="model",
                                                              parts=[types.Part(text='{"label": "neutral"}')]))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=50, candidates_token_count=10, thoughts_token_count=40))
        self.p.client = stub(lambda **k: resp)
        r = self.p.complete("s", "t", SCHEMA)
        self.assertEqual((r.data, r.input_tokens, r.output_tokens), ({"label": "neutral"}, 50, 50))

    def test_error_codes(self):
        for code, expected in [(429, prov.RetryableError), (503, prov.RetryableError),
                               (400, prov.FatalError), (403, prov.FatalError)]:
            with self.subTest(code):
                exc = genai_errors.APIError(code, {"error": {"message": "x", "status": "X"}})
                self.p.client = stub(raiser(exc))
                with self.assertRaises(expected):
                    self.p.complete("s", "t", SCHEMA)


if __name__ == "__main__":
    unittest.main()
