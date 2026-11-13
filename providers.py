"""Thin adapters around each AI provider's official SDK.

Every adapter turns one row of text into one `Reply`, or raises one of two errors:
- `RetryableError`: worth trying again later (rate limit, server error, timeout, network, an answer cut short
  or not valid JSON).
- `FatalError`: trying again with this provider will not help. With `disable=True` the problem is the provider
  itself (bad key, unknown model) and it is switched off for the rest of the run; with `disable=False` the
  problem is this row (the request is rejected, the model declines it) and the next row is tried as usual.

Both errors carry the tokens of the call when the provider billed one, so the cost log counts answers that were
paid for and then thrown away.

The SDKs' own automatic retries are switched off, so that every attempt is visible in the cost log.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


class ProviderError(Exception):
    def __init__(self, message: str, input_tokens: int = 0, output_tokens: int = 0, retry_after: float | None = None):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.retry_after = retry_after


class RetryableError(ProviderError):
    """The call failed, but the same call may succeed later."""

    @property
    def bad_output(self) -> bool:
        return str(self).startswith("bad_output")


class FatalError(ProviderError):
    """The call failed, and repeating it with this provider will not help."""

    def __init__(self, message: str, disable: bool, **kw):
        super().__init__(message, **kw)
        self.disable = disable


@dataclass
class Reply:
    data: dict            # the parsed JSON object
    input_tokens: int
    output_tokens: int    # includes reasoning tokens where the provider bills them as output


def _parse_json(text, usage: tuple[int, int]) -> dict:
    """Parse the model's text as a JSON object, or raise RetryableError carrying the tokens already paid."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise RetryableError(f"bad_output: not valid JSON ({e})", *usage) from None
    if not isinstance(data, dict):
        raise RetryableError("bad_output: JSON is not an object", *usage)
    return data


def _retry_after(response) -> float | None:
    """Seconds asked by the provider's retry-after header, if any."""
    try:
        value = response.headers.get("retry-after")
        return float(value) if value is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


# Messages that mean the account has no credit or quota left: every row would fail the same way.
# (OpenAI: 429 "insufficient_quota"; Anthropic: 400 "credit balance is too low". Known shapes, not seen live here.)
_OUT_OF_CREDIT = ("insufficient_quota", "credit balance is too low")


def _status_error(code: int, message: str, response=None, disable_codes=(401, 403, 404),
                  retry_after: float | None = None) -> ProviderError:
    """Sort an HTTP error: 408/409/429/5xx retry; a bad key, an unknown model or an empty account switches the
    provider off; the rest concerns this row only."""
    if any(m in message.lower() for m in _OUT_OF_CREDIT):
        return FatalError(f"HTTP {code}: no credit or quota left: {message}", disable=True)
    if code >= 500 or code in (408, 409, 429):
        return RetryableError(f"HTTP {code}: {message}",
                              retry_after=retry_after if retry_after is not None else _retry_after(response))
    return FatalError(f"HTTP {code}: {message}", disable=code in disable_codes)


def _gemini_retry_delay(details) -> float | None:
    """The pause asked in the body of a Gemini error (RetryInfo, e.g. "17s"), if any."""
    try:
        for d in details.get("error", {}).get("details", []):
            if str(d.get("@type", "")).endswith("RetryInfo"):
                return float(str(d.get("retryDelay", "")).rstrip("s"))
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def _not_an_sdk_object(r, attribute: str) -> None:
    """A 200 whose body is not the provider's JSON (an HTML page from a proxy, for example) comes back as a str."""
    if not hasattr(r, attribute):
        raise RetryableError(f"unexpected response body ({type(r).__name__}), not the provider's JSON")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, effort: str | None = None, max_tokens: int = 2000):
        import anthropic
        self._sdk = anthropic
        self.client = anthropic.Anthropic(max_retries=0)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, text: str, schema: dict) -> Reply:
        a = self._sdk
        output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        try:
            r = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": text}],
                output_config=output_config,
            )
        except (a.APITimeoutError, a.APIConnectionError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from None
        except a.APIStatusError as e:
            raise _status_error(e.status_code, f"{type(e).__name__}: {e}", e.response) from None
        except TypeError as e:
            if "authentication" in str(e):   # no key anywhere: the SDK finds out only at the first call
                raise FatalError(f"no credentials: {e}", disable=True) from None
            raise
        _not_an_sdk_object(r, "usage")
        usage = (r.usage.input_tokens, r.usage.output_tokens)
        if r.stop_reason == "refusal":
            raise FatalError("refusal: the model declined this row", False, input_tokens=usage[0],
                             output_tokens=usage[1])
        if r.stop_reason == "max_tokens":
            raise RetryableError("bad_output: answer cut at max_tokens", *usage)
        text_out = next((b.text for b in r.content if b.type == "text"), "")
        return Reply(_parse_json(text_out, usage), *usage)


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, max_tokens: int = 2000):
        import openai
        self._sdk = openai
        self.client = openai.OpenAI(max_retries=0)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, text: str, schema: dict) -> Reply:
        o = self._sdk
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "row", "schema": schema, "strict": True}},
            )
        except (o.APITimeoutError, o.APIConnectionError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from None
        except o.APIStatusError as e:
            raise _status_error(e.status_code, f"{type(e).__name__}: {e}", e.response) from None
        _not_an_sdk_object(r, "choices")
        if not r.choices:
            raise RetryableError("unexpected response: no choices")
        choice = r.choices[0]
        usage = (r.usage.prompt_tokens, r.usage.completion_tokens) if r.usage else (0, 0)
        if getattr(choice.message, "refusal", None) or choice.finish_reason == "content_filter":
            raise FatalError("refusal: the model declined this row", False, input_tokens=usage[0],
                             output_tokens=usage[1])
        if choice.finish_reason == "length":
            raise RetryableError("bad_output: answer cut at max_tokens", *usage)
        return Reply(_parse_json(choice.message.content, usage), *usage)


# Gemini stops for content reasons with these finish reasons: they are refusals, not answers to retry.
_GEMINI_BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                   "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, max_tokens: int = 2000):
        from google import genai
        from google.genai import errors, types
        self._errors = errors
        self._types = types
        self._network = _transport_errors()
        self.client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, text: str, schema: dict) -> Reply:
        t = self._types
        try:
            r = self.client.models.generate_content(
                model=self.model,
                contents=text,
                config=t.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
        except self._errors.APIError as e:
            code = getattr(e, "code", 0) or 0
            err = _status_error(code, str(e), getattr(e, "response", None),
                                retry_after=_gemini_retry_delay(getattr(e, "details", None)))
            # Gemini reports a wrong key as 400 INVALID_ARGUMENT, so the code alone does not tell
            if isinstance(err, FatalError) and code == 400 and "api key" in str(e).lower():
                err.disable = True
            raise err from None
        except self._network as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from None
        except json.JSONDecodeError as e:   # a 200 whose body is not JSON (an HTML page from a proxy)
            raise RetryableError(f"unexpected response body, not JSON: {e}") from None
        um = r.usage_metadata
        usage = ((um.prompt_token_count or 0) if um else 0,
                 ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0)
        feedback = getattr(r, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise FatalError(f"refusal: prompt blocked ({feedback.block_reason})", False,
                             input_tokens=usage[0], output_tokens=usage[1])
        finish = r.candidates[0].finish_reason if r.candidates else None
        finish_name = getattr(finish, "name", str(finish or ""))
        if finish_name in _GEMINI_BLOCKED:
            raise FatalError(f"refusal: answer blocked ({finish_name})", False,
                             input_tokens=usage[0], output_tokens=usage[1])
        if finish_name == "MAX_TOKENS":
            raise RetryableError("bad_output: answer cut at max_tokens", *usage)
        if not r.text:
            raise RetryableError("bad_output: empty answer", *usage)
        return Reply(_parse_json(r.text, usage), *usage)


def _transport_errors() -> tuple:
    """Network errors of the HTTP libraries the Gemini SDK may use, plus the standard ones."""
    errors = [ConnectionError, TimeoutError]
    for module in ("httpx", "httpx2"):
        try:
            errors.append(__import__(module).TransportError)
        except (ImportError, AttributeError):
            pass
    return tuple(errors)


class MockProvider:
    """Offline stand-in for tests and dry runs. Never calls the network.

    `script` is a list of outcomes used in order, one per call, then repeated from the start:
    "ok", "retry" (a retryable error), "fatal" (a bad key: the provider is switched off),
    "reject" (this row is rejected) or "bad" (an answer that is not valid JSON, still billed).
    """

    def __init__(self, name: str = "mock", model: str = "mock-model", script: list[str] | None = None):
        self.name = name
        self.model = model
        self.script = script or ["ok"]
        self.calls = 0

    def complete(self, system: str, text: str, schema: dict) -> Reply:
        step = self.script[self.calls % len(self.script)]
        self.calls += 1
        if step == "retry":
            raise RetryableError("mock: simulated rate limit")
        if step == "fatal":
            raise FatalError("mock: simulated bad key", disable=True)
        if step == "reject":
            raise FatalError("mock: simulated request rejected for this row", disable=False)
        if step == "bad":
            return Reply(_parse_json("not json", (10, 2)), 10, 2)
        data = {}
        for field, spec in schema["properties"].items():
            data[field] = spec["enum"][0] if "enum" in spec else f"mock {field}"
        return Reply(data, max(1, len(text) // 4), 20)


def build(spec: dict):
    """Build a provider from one [[providers]] entry of the config."""
    name = spec["name"]
    model = spec.get("model", "")
    if name == "mock":
        return MockProvider(model=model or "mock-model", script=spec.get("script"))
    if not model:
        raise ValueError(f"provider '{name}': set 'model' in the config")
    if name == "anthropic":
        return AnthropicProvider(model, effort=spec.get("effort"), max_tokens=spec.get("max_tokens", 2000))
    if name == "openai":
        return OpenAIProvider(model, max_tokens=spec.get("max_tokens", 2000))
    if name == "gemini":
        return GeminiProvider(model, max_tokens=spec.get("max_tokens", 2000))
    raise ValueError(f"unknown provider '{name}' (use anthropic, openai, gemini or mock)")


KEY_VARIABLES = {"anthropic": ["ANTHROPIC_API_KEY"], "openai": ["OPENAI_API_KEY"],
                 "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "mock": []}


def key_present(name: str) -> bool:
    """True when the environment holds a key for this provider (used to warn early, not to read the key)."""
    env = KEY_VARIABLES.get(name, [])
    return not env or any(os.environ.get(v) for v in env)


def secrets_in_environment() -> list[str]:
    """The key values currently set, so that logs can mask them."""
    names = {v for vs in KEY_VARIABLES.values() for v in vs} | {"ANTHROPIC_AUTH_TOKEN"}
    return [os.environ[n] for n in names if len(os.environ.get(n, "")) >= 8]
