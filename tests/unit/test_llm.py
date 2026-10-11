import json
import unittest

from tests import helpers
from sentinelcrew.llm import AnthropicClient, GeminiClient, LLMError, ScriptedLLM, create_llm, create_llm_safe, extract_json
from sentinelcrew.llm.http import TransportError, post_with_retry
from sentinelcrew.validation import SchemaError, validate

SCHEMA = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def gemini_reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def anthropic_reply(text):
    return {"type": "message", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"a": [1,2]}\n```\nDone'), {"a": [1, 2]})

    def test_embedded_with_braces_in_strings(self):
        self.assertEqual(extract_json('Answer: {"a": "x}y", "b": {"c": 1}} trailing'), {"a": "x}y", "b": {"c": 1}})

    def test_escaped_quotes(self):
        self.assertEqual(extract_json(r'here {"a": "say \"hi\""}'), {"a": 'say "hi"'})

    def test_array(self):
        self.assertEqual(extract_json("result [1, 2]"), [1, 2])

    def test_failures(self):
        for bad in (None, "no json here", "{broken"):
            with self.assertRaises(ValueError):
                extract_json(bad)


class TestValidation(unittest.TestCase):
    def test_happy_path(self):
        schema = {"type": "object", "required": ["n", "tags"], "additionalProperties": False, "properties": {
            "n": {"type": "number", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}, "minItems": 1, "maxItems": 2},
            "s": {"type": ["string", "null"], "maxLength": 3}}}
        validate({"n": 0.5, "tags": ["a"], "s": None}, schema)

    def test_violations(self):
        cases = [
            ({"type": "integer"}, True),
            ({"type": "number"}, "1"),
            ({"type": "number", "minimum": 1}, 0),
            ({"type": "number", "maximum": 1}, 2),
            ({"enum": [1]}, 2),
            ({"type": "string", "maxLength": 1}, "ab"),
            ({"type": "object", "required": ["x"]}, {}),
            ({"type": "object", "additionalProperties": False, "properties": {}}, {"x": 1}),
            ({"type": "array", "minItems": 1}, []),
            ({"type": "array", "maxItems": 0}, [1]),
            ({"type": "array", "items": {"type": "string"}}, [1]),
            ({"type": "weird"}, 1),
        ]
        for schema, data in cases:
            with self.subTest(schema=schema), self.assertRaises(SchemaError):
                validate(data, schema)


class TestGemini(unittest.TestCase):
    def test_requires_key(self):
        with self.assertRaises(LLMError):
            GeminiClient("")

    def test_payload_and_parse(self):
        t = FakeTransport([gemini_reply('{"ok": true}')])
        client = GeminiClient("k", model="gemini-test", transport=t)
        self.assertEqual(client.complete_json("sys", "prompt", SCHEMA), {"ok": True})
        url, headers, payload = t.calls[0]
        self.assertIn("gemini-test:generateContent", url)
        self.assertEqual(headers["x-goog-api-key"], "k")
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "sys")
        self.assertEqual(client.usage.calls, 1)

    def test_plain_text_has_no_json_mime(self):
        t = FakeTransport([gemini_reply("hello")])
        client = GeminiClient("k", transport=t)
        self.assertEqual(client.complete("s", "p"), "hello")
        self.assertNotIn("responseMimeType", t.calls[0][2]["generationConfig"])

    def test_corrective_retry(self):
        t = FakeTransport([gemini_reply("not json"), gemini_reply('{"ok": "yes"}'), gemini_reply('{"ok": false}')])
        client = GeminiClient("k", transport=t, max_retries=2)
        self.assertEqual(client.complete_json("s", "p", SCHEMA), {"ok": False})
        self.assertIn("previous answer was invalid", t.calls[1][2]["contents"][0]["parts"][0]["text"])

    def test_gives_up(self):
        t = FakeTransport([gemini_reply("nope")] * 2)
        with self.assertRaises(LLMError):
            GeminiClient("k", transport=t, max_retries=1).complete_json("s", "p", SCHEMA)

    def test_empty_and_blocked(self):
        with self.assertRaises(LLMError):
            GeminiClient.parse_response({"promptFeedback": {"blockReason": "SAFETY"}})
        with self.assertRaises(LLMError):
            GeminiClient.parse_response({"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]})

    def test_transport_error_wrapped(self):
        t = FakeTransport([TransportError("HTTP 401", status=401)])
        client = GeminiClient("k", transport=t)
        with self.assertRaises(LLMError):
            client.complete("s", "p")
        self.assertEqual(client.usage.failures, 1)


class TestAnthropic(unittest.TestCase):
    def test_requires_key(self):
        with self.assertRaises(LLMError):
            AnthropicClient("")

    def test_payload_and_parse(self):
        t = FakeTransport([anthropic_reply('```json\n{"ok": true}\n```')])
        client = AnthropicClient("k", model="claude-test", transport=t)
        self.assertEqual(client.complete_json("sys", "p", SCHEMA), {"ok": True})
        url, headers, payload = t.calls[0]
        self.assertEqual(url, AnthropicClient.URL)
        self.assertEqual(headers["x-api-key"], "k")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["model"], "claude-test")
        self.assertEqual(payload["system"], "sys")
        self.assertIn("Return only valid JSON", payload["messages"][0]["content"])

    def test_errors(self):
        with self.assertRaises(LLMError):
            AnthropicClient.parse_response({"type": "error", "error": {"type": "overloaded_error"}})
        with self.assertRaises(LLMError):
            AnthropicClient.parse_response({"content": [], "stop_reason": "max_tokens"})
        t = FakeTransport([TransportError("boom")])
        with self.assertRaises(LLMError):
            AnthropicClient("k", transport=t).complete("s", "p")


class TestRetry(unittest.TestCase):
    def test_retries_retryable_then_succeeds(self):
        sleeps = []
        t = FakeTransport([TransportError("429", status=429, retryable=True), {"ok": 1}])
        self.assertEqual(post_with_retry(t, "u", {}, {}, 1, retries=2, sleep=sleeps.append), {"ok": 1})
        self.assertEqual(sleeps, [1.0])

    def test_non_retryable_raises_immediately(self):
        t = FakeTransport([TransportError("400", status=400)])
        with self.assertRaises(TransportError):
            post_with_retry(t, "u", {}, {}, 1, retries=3, sleep=lambda s: None)
        self.assertEqual(len(t.calls), 1)

    def test_exhausts_retries(self):
        err = TransportError("503", status=503, retryable=True)
        t = FakeTransport([err, err, err])
        with self.assertRaises(TransportError):
            post_with_retry(t, "u", {}, {}, 1, retries=2, sleep=lambda s: None)
        self.assertEqual(len(t.calls), 3)


class TestFactory(unittest.TestCase):
    def test_offline(self):
        self.assertIsNone(create_llm(helpers.settings()))
        client, note = create_llm_safe(helpers.settings())
        self.assertIsNone(client)
        self.assertIn("offline", note)

    def test_gemini_and_anthropic_selection(self):
        s = helpers.settings(llm_provider="gemini", gemini_api_key="g")
        self.assertIsInstance(create_llm(s), GeminiClient)
        s = helpers.settings(llm_provider="anthropic", anthropic_api_key="a")
        self.assertIsInstance(create_llm(s), AnthropicClient)

    def test_auto_follows_environment(self):
        s = helpers.settings(llm_provider="auto", app_env="production", anthropic_api_key="a", gemini_api_key="g")
        self.assertIsInstance(create_llm(s), AnthropicClient)
        s = s.with_overrides(app_env="development")
        self.assertIsInstance(create_llm(s), GeminiClient)

    def test_missing_key_degrades_gracefully(self):
        client, note = create_llm_safe(helpers.settings(llm_provider="gemini", gemini_api_key=""))
        self.assertIsNone(client)
        self.assertIn("GEMINI_API_KEY", note)

    def test_scripted(self):
        llm = ScriptedLLM({"Critic": {"x": 1}, "Other": lambda p: "raw:" + p[:3]}, default="d")
        self.assertEqual(json.loads(llm.complete("ROLE: Critic", "p")), {"x": 1})
        self.assertEqual(llm.complete("Other", "abcdef"), "raw:abc")
        self.assertEqual(llm.complete("none", "p"), "d")
        with self.assertRaises(LLMError):
            ScriptedLLM().complete("s", "p")
        with self.assertRaises(LLMError):
            ScriptedLLM(fail=True).complete("s", "p")
        self.assertEqual(llm.describe(), {"provider": "scripted", "model": "scripted-v1"})


if __name__ == "__main__":
    unittest.main()
