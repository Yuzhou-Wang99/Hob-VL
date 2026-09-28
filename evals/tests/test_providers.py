"""Provider contract tests: every HTTP operation is mocked, never billable."""

import base64
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from dataclasses import replace
import io
import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from evals import providers
from evals.providers import ProviderConfig, ProviderError, build_request, call_model


IMAGE = b"\x89PNG\r\n\x1a\nimage-content-for-exact-byte-transport-test"
PROMPT = 'Original prompt: A AND B?\nReturn JSON: {"answer": true}.'
FAKE_KEY = "unit-test-key-never-a-real-credential"


def config(provider="openai", **kwargs):
    if provider in {"openai-compatible", "qwen"}:
        kwargs.setdefault("base_url", "https://compatible.example/v1")
    return ProviderConfig(provider=provider, model="test-vision-model", **kwargs)


def success(provider):
    if provider == "openai":
        return {"id": "resp_test", "status": "completed", "output": [
            {"type": "reasoning", "summary": [{"text": "Ignore thinking"}]},
            {"type": "message", "status": "completed", "content": [
                {"type": "output_text", "text": '{"answer": true}'},
            ]},
        ], "usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}}
    if provider == "anthropic":
        return {"id": "msg_test", "content": [
            {"type": "thinking", "thinking": "Ignore thinking"},
            {"type": "text", "text": '{"answer": true}'},
        ], "stop_reason": "end_turn", "usage": {"input_tokens": 100, "output_tokens": 25}}
    if provider == "gemini":
        return {"responseId": "gem_test", "candidates": [{"content": {"parts": [
            {"text": "Ignore thinking", "thought": True}, {"text": '{"answer": true}'},
        ]}, "finishReason": "STOP"}], "usageMetadata": {
            "promptTokenCount": 100, "candidatesTokenCount": 20,
            "thoughtsTokenCount": 5, "totalTokenCount": 125,
        }}
    return {"id": "chat_test", "choices": [{"message": {"content": '{"answer": true}'},
                                             "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}}


class ConfigTests(unittest.TestCase):
    def test_zai_thinking_mode_is_explicit_and_omission_preserves_legacy_config(self):
        for model in ("glm-4.5v", "glm-4.6v", "glm-4.6v-flash", "glm-4.6v-flashx"):
            cfg = ProviderConfig("zai", model)
            baseline = cfg.public_dict()
            self.assertNotIn("thinking_mode", baseline)
            for mode in ("enabled", "disabled"):
                self.assertEqual(replace(cfg, thinking_mode=mode).public_dict(),
                                 {**baseline, "thinking_mode": mode})
        cfg = replace(cfg, thinking_mode="enabled")
        for changes in ({"thinking_mode": "low"}, {"thinking_mode": True},
                        {"provider": "deepseek"}, {"model": "glm-unknown"},
                        {"reasoning_effort": "none"}, {"thinking_budget": 0},
                        {"thinking_level": "minimal"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes).validate()

    def test_deepinfra_scout_none_records_model_selection_and_rejects_unsupported_controls(self):
        cfg = ProviderConfig("deepinfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                             reasoning_effort="none")
        public = cfg.public_dict()
        self.assertEqual(public["api_key_env"], "DEEPINFRA_API_KEY")
        self.assertEqual(public["base_url"], "https://api.deepinfra.com/v1/openai")
        self.assertEqual(public["reasoning_effort"], "none")
        self.assertIsNone(public["image_detail"])
        for changes in ({"model": "some-other-model"}, {"reasoning_effort": "low"},
                        {"thinking_budget": 0}, {"thinking_level": "minimal"},
                        {"image_detail": "high"}, {"image_detail": "auto"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes).validate()

    def test_hosted_gemma_accepts_only_minimal_or_high_thinking_levels(self):
        for model in ("gemma-4-26b-a4b-it", "gemma-4-31b-it", "models/gemma-4-26b-a4b-it"):
            cfg = ProviderConfig("gemini", model, thinking_level="minimal")
            self.assertEqual(cfg.public_dict()["thinking_level"], "minimal")
            self.assertEqual(cfg.public_dict()["api_key_env"], "GEMINI_API_KEY")
            replace(cfg, thinking_level="high").validate()
            for changes in ({"thinking_level": "low"}, {"thinking_level": "medium"},
                            {"thinking_level": "none"}, {"thinking_budget": 0},
                            {"reasoning_effort": "none"}, {"provider": "openai"}):
                with self.subTest(model=model, changes=changes), self.assertRaises(ValueError):
                    replace(cfg, **changes).validate()
        with self.assertRaisesRegex(ValueError, "supported Gemini 3 or Gemma 4"):
            replace(cfg, model="gemma-4-unknown").validate()

    def test_deepseek_none_uses_current_vision_model_and_native_controls(self):
        cfg = ProviderConfig("deepseek", "deepseek-flash", reasoning_effort="none")
        public = cfg.public_dict()
        self.assertEqual(public["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(public["base_url"], "https://api.deepseek.com")
        self.assertEqual(public["image_detail"], "high")
        for detail in ("auto", "low", "high", "original"):
            replace(cfg, image_detail=detail).validate()
        for changes in ({"model": "deepseek-v4-pro"}, {"model": "deepseek-v4-flash-vision-exp"},
                        {"reasoning_effort": "low"}, {"thinking_budget": 0},
                        {"thinking_level": "minimal"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes).validate()

    def test_qwen_and_zai_none_cannot_select_a_thinking_or_unknown_checkpoint(self):
        qwen = replace(config("qwen"), model="qwen3-vl-8b-instruct", reasoning_effort="none")
        zai = replace(config("zai"), model="glm-4.5v", reasoning_effort="none")
        self.assertEqual(qwen.public_dict()["api_key_env"], "DASHSCOPE_API_KEY")
        self.assertEqual(zai.public_dict()["api_key_env"], "ZAI_API_KEY")
        for cfg in (qwen, zai, replace(zai, model="glm-4.6v-flash"),
                    replace(zai, model="glm-4.6v-flashx")):
            self.assertEqual(cfg.public_dict()["reasoning_effort"], "none")
            for change in ({"reasoning_effort": "low"}, {"thinking_budget": 0},
                           {"thinking_level": "minimal"}, {"image_detail": "high"}):
                with self.subTest(provider=cfg.provider, change=change), self.assertRaises(ValueError):
                    replace(cfg, **change).validate()
        for model in ("qwen3-vl-8b-thinking", "qwen3-vl-plus", "qwen3-vl-flash", "qwen-latest"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "Instruct"):
                replace(qwen, model=model).validate()
        with self.assertRaisesRegex(ValueError, "base_url"):
            replace(qwen, base_url=None).validate()
        with self.assertRaisesRegex(ValueError, "glm-4.5v"):
            replace(zai, model="glm-unknown").validate()

    def test_validation_rejects_bad_configuration(self):
        variants = [
            {"provider": "unknown"}, {"model": ""}, {"model": " bad "},
            {"api_key_env": "secret-value=bad"}, {"max_output_tokens": 0},
            {"max_output_tokens": True}, {"max_output_tokens": 2.5},
            {"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")},
            {"retries": -1}, {"retries": 11}, {"retries": True},
            {"temperature": float("nan")}, {"temperature": -1}, {"temperature": 2.1},
            {"base_url": "http://example.com/v1"},
            {"base_url": "https://user:secret@example.com/v1"},
            {"base_url": "https://example.com/v1?key=secret"},
            {"base_url": "https://example.com/v1#secret"},
            {"base_url": "https://example.com:bad/v1"}, {"base_url": "file:///tmp/key"},
        ]
        for values in variants:
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(config(), **values).validate()
        with self.assertRaises(ValueError):
            ProviderConfig("openai-compatible", "model").validate()
        with self.assertRaises(ValueError):
            config("anthropic", temperature=1.5).validate()

    def test_http_loopback_and_custom_https_are_allowed(self):
        for root in ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1",
                     "http://[::1]:8000/v1", "https://proxy.example/api/v1"]:
            config(base_url=root).validate()

    def test_gemini_thinking_budgets_validate_model_ranges_and_zero(self):
        valid = {
            "gemini-2.5-flash": (-1, 0, 1, 24576),
            "gemini-2.5-flash-lite": (-1, 0, 512, 24576),
            "gemini-2.5-pro": (-1, 128, 32768),
        }
        for model, budgets in valid.items():
            for budget in budgets:
                with self.subTest(model=model, budget=budget):
                    replace(config("gemini"), model=model, thinking_budget=budget).validate()
        for model, budget in (("gemini-2.5-flash", -2), ("gemini-2.5-flash", 24577),
                              ("gemini-2.5-flash", True), ("gemini-2.5-flash", 0.0),
                              ("gemini-2.5-flash", "0"), ("gemini-2.5-flash-lite", 1),
                              ("gemini-2.5-pro", 0), ("gemini-2.5-pro", 32769),
                              ("gemini-3-flash-preview", 0), ("gemini-flash-latest", 0),
                              ("gemini-2.5-flash-image", 0)):
            with self.subTest(model=model, budget=budget), self.assertRaisesRegex(ValueError, "thinking_budget"):
                replace(config("gemini"), model=model, thinking_budget=budget).validate()
        replace(config("gemini"), model="models/gemini-2.5-flash", thinking_budget=0).validate()
        for provider in ("openai", "anthropic", "openai-compatible"):
            with self.subTest(provider=provider), self.assertRaisesRegex(ValueError, "gemini provider"):
                config(provider, thinking_budget=0).validate()

    def test_gemini_thinking_levels_reject_incompatible_models_and_budgets(self):
        cfg = replace(config("gemini"), model="gemini-3.5-flash-lite", thinking_level="minimal")
        cfg.validate()
        replace(cfg, model="models/gemini-3.5-flash-lite").validate()
        replace(cfg, model="gemini-3.8-flash", thinking_level="low").validate()
        for changes in ({"model": "gemini-3.8-flash"}, {"model": "gemini-2.5-flash-lite"},
                        {"model": "gemini-3.1-flash-image"}, {"thinking_level": "none"},
                        {"thinking_level": 0}, {"thinking_level": True}, {"thinking_level": []},
                        {"thinking_budget": 0}, {"provider": "anthropic"}, {"provider": "openai"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes).validate()
        self.assertEqual(cfg.public_dict()["thinking_level"], "minimal")
        self.assertNotIn("thinking_budget", cfg.public_dict())

    def test_unused_thinking_budget_preserves_legacy_model_configuration(self):
        expected = {
            "provider": "openai-compatible", "model": "test-vision-model",
            "api_key_env": "OPENAI_API_KEY", "base_url": "https://compatible.example/v1",
            "max_output_tokens": 4096, "temperature": None, "timeout": 120, "retries": 0,
            "reasoning_effort": None, "image_detail": "high",
        }
        self.assertEqual(config("openai-compatible").public_dict(), expected)
        for provider in providers.PROVIDERS:
            self.assertNotIn("thinking_budget", config(provider).public_dict())
            self.assertNotIn("thinking_level", config(provider).public_dict())
        cfg = replace(config("gemini"), model="gemini-2.5-flash", thinking_budget=0)
        self.assertEqual(cfg.public_dict()["thinking_budget"], 0)

    def test_reasoning_effort_and_image_detail_validation(self):
        for provider in ("openai", "openai-compatible"):
            config(provider, reasoning_effort="low", image_detail="high").validate()
            self.assertEqual(config(provider).image_detail, "high")
        config("openai", image_detail="original").validate()
        self.assertIsNone(config("anthropic").image_detail)
        for provider in ("anthropic", "gemini"):
            for values in [{"reasoning_effort": "low"}, {"image_detail": "high"}]:
                with self.subTest(provider=provider, values=values), self.assertRaises(ValueError):
                    config(provider, **values).validate()
        for values in [{"reasoning_effort": "gentle"}, {"reasoning_effort": True},
                       {"image_detail": "medium"}, {"image_detail": "original"}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                config("openai-compatible", **values).validate()

    def test_anthropic_accepts_only_explicit_none_without_changing_old_configuration(self):
        original = config("anthropic").public_dict()
        explicit = config("anthropic", reasoning_effort="none").public_dict()
        self.assertEqual(explicit, {**original, "reasoning_effort": "none"})
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            with self.subTest(effort=effort), self.assertRaisesRegex(ValueError, "only reasoning_effort"):
                config("anthropic", reasoning_effort=effort).validate()
        with self.assertRaises(ValueError):
            config("gemini", reasoning_effort="none").validate()

    def test_offline_build_and_public_dict_never_read_environment_or_network(self):
        with patch.object(providers.os.environ, "get", side_effect=AssertionError("env read")), \
                patch.object(providers.request, "build_opener", side_effect=AssertionError("network")):
            for provider in providers.PROVIDERS:
                cfg = config(provider)
                payload = build_request(cfg, PROMPT, IMAGE, "image/png")
                public = cfg.public_dict()
                self.assertEqual(public["model"], "test-vision-model")
                self.assertNotIn(FAKE_KEY, json.dumps(public))
                self.assertNotIn("temperature", payload.get("generationConfig", payload))

    def test_missing_key_fails_before_http(self):
        with patch.dict(providers.os.environ, {}, clear=True), \
                patch.object(providers, "_post_json") as post:
            with self.assertRaisesRegex(ProviderError, "Missing API key"):
                call_model(config(), {})
            post.assert_not_called()


class RequestTests(unittest.TestCase):
    def test_zai_native_thinking_switch_preserves_prompt_image_and_token_cap(self):
        cfg = ProviderConfig("zai", "glm-4.6v-flashx", max_output_tokens=4096)
        baseline = build_request(cfg, PROMPT, IMAGE, "image/png")
        for mode in ("enabled", "disabled"):
            payload = build_request(replace(cfg, thinking_mode=mode), PROMPT, IMAGE, "image/png")
            self.assertEqual(payload.pop("thinking"), {"type": mode})
            self.assertEqual(payload, baseline)
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertNotIn("reasoning_effort", payload)

    def test_deepinfra_scout_sends_only_prompt_image_and_supported_generation_settings(self):
        cfg = ProviderConfig("deepinfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                             reasoning_effort="none", max_output_tokens=2048)
        for mime in ("image/png", "image/jpeg", "image/webp"):
            with self.subTest(mime=mime):
                payload = build_request(cfg, PROMPT, IMAGE, mime)
                self.assertEqual(payload, {
                    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime};base64," + base64.b64encode(IMAGE).decode("ascii")}},
                    ]}],
                    "max_tokens": 2048,
                })
        with self.assertRaisesRegex(ValueError, "not GIF"):
            build_request(cfg, PROMPT, IMAGE, "image/gif")

    def test_gemma_minimal_preserves_the_prompt_image_and_existing_gemini_payload(self):
        for model in ("gemma-4-26b-a4b-it", "gemma-4-31b-it"):
            cfg = ProviderConfig("gemini", model, thinking_level="minimal", max_output_tokens=2048)
            payload = build_request(cfg, PROMPT, IMAGE, "image/png")
            self.assertEqual(payload, {
                "contents": [{"role": "user", "parts": [
                    {"text": PROMPT},
                    {"inlineData": {"mimeType": "image/png",
                                    "data": base64.b64encode(IMAGE).decode("ascii")}},
                ]}],
                "generationConfig": {"maxOutputTokens": 2048,
                                     "thinkingConfig": {"thinkingLevel": "minimal"}},
            })
            # Model ID travels in the endpoint URL, as it does for Gemini models.
            original = build_request(replace(cfg, thinking_level=None), PROMPT, IMAGE, "image/png")
            del payload["generationConfig"]["thinkingConfig"]
            self.assertEqual(payload, original)

    def test_deepseek_disabled_request_contains_only_prompt_image_and_settings(self):
        cfg = ProviderConfig("deepseek", "deepseek-flash", reasoning_effort="none",
                             max_output_tokens=2048)
        self.assertEqual(build_request(cfg, PROMPT, IMAGE, "image/png"), {
            "model": "deepseek-flash", "max_tokens": 2048,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(IMAGE).decode("ascii"),
                    "detail": "high"}},
            ]}],
        })
        self.assertNotIn("thinking", build_request(replace(cfg, reasoning_effort=None),
                                                   PROMPT, IMAGE, "image/png"))

    def test_qwen_instruct_and_glm_native_disabled_keep_the_exact_prompt_and_image(self):
        for provider, model in (("qwen", "qwen3-vl-8b-instruct"), ("zai", "glm-4.5v"),
                                ("zai", "glm-4.6v-flash"), ("zai", "glm-4.6v-flashx")):
            with self.subTest(provider=provider, model=model):
                cfg = replace(config(provider), model=model, reasoning_effort="none", max_output_tokens=2048)
                payload = build_request(cfg, PROMPT, IMAGE, "image/png")
                expected = {
                    "model": model, "max_tokens": 2048,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(IMAGE).decode("ascii")}},
                    ]}],
                }
                if provider == "zai":
                    expected["thinking"] = {"type": "disabled"}
                self.assertEqual(payload, expected)
                self.assertNotIn("reasoning_effort", payload)

    def test_gemini_minimal_level_preserves_inputs_and_is_sent_as_native_level(self):
        cfg = replace(config("gemini"), model="gemini-3.5-flash-lite")
        baseline = build_request(cfg, PROMPT, IMAGE, "image/png")
        self.assertNotIn("thinkingConfig", baseline["generationConfig"])
        payload = build_request(replace(cfg, thinking_level="minimal"), PROMPT, IMAGE, "image/png")
        self.assertEqual(payload["generationConfig"].pop("thinkingConfig"), {"thinkingLevel": "minimal"})
        self.assertEqual(payload, baseline)

    def test_anthropic_none_explicitly_disables_thinking_and_preserves_prompt_image(self):
        cfg = replace(config("anthropic"), model="claude-haiku-4-5-20251001")
        baseline = build_request(cfg, PROMPT, IMAGE, "image/png")
        self.assertNotIn("thinking", baseline)
        payload = build_request(replace(cfg, reasoning_effort="none"), PROMPT, IMAGE, "image/png")
        self.assertEqual(payload.pop("thinking"), {"type": "disabled"})
        self.assertEqual(payload, baseline)

    def test_gemini_zero_budget_is_sent_explicitly_without_changing_prompt_or_image(self):
        original = replace(config("gemini"), model="gemini-2.5-flash", temperature=0.2)
        baseline = build_request(original, PROMPT, IMAGE, "image/png")
        self.assertNotIn("thinkingConfig", baseline["generationConfig"])
        for budget in (0, -1, 1024):
            with self.subTest(budget=budget):
                payload = build_request(replace(original, thinking_budget=budget), PROMPT, IMAGE, "image/png")
                self.assertEqual(payload["generationConfig"].pop("thinkingConfig"), {"thinkingBudget": budget})
                self.assertEqual(payload, baseline)

    def test_original_prompt_image_and_token_limits_for_all_providers(self):
        for provider in providers.PROVIDERS:
            with self.subTest(provider=provider):
                payload = build_request(config(provider, max_output_tokens=123, temperature=0.2),
                                        PROMPT, IMAGE, "image/png")
                if provider == "openai":
                    parts = payload["input"][0]["content"]
                    prompt = parts[0]["text"]
                    encoded = parts[1]["image_url"].split(",")[1]
                    self.assertEqual(payload["max_output_tokens"], 123)
                    self.assertIs(payload["store"], False)
                elif provider == "anthropic":
                    parts = payload["messages"][0]["content"]
                    prompt = parts[1]["text"]
                    encoded = parts[0]["source"]["data"]
                    self.assertEqual(parts[0]["source"]["media_type"], "image/png")
                    self.assertEqual(payload["max_tokens"], 123)
                elif provider == "gemini":
                    parts = payload["contents"][0]["parts"]
                    prompt = parts[0]["text"]
                    encoded = parts[1]["inlineData"]["data"]
                    self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
                    self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 123)
                else:
                    parts = payload["messages"][0]["content"]
                    prompt = parts[0]["text"]
                    encoded = parts[1]["image_url"]["url"].split(",")[1]
                    self.assertEqual(payload["max_tokens"], 123)
                self.assertEqual(len(parts), 2)
                self.assertEqual(prompt, PROMPT)
                self.assertEqual(base64.b64decode(encoded, validate=True), IMAGE)
                self.assertEqual(payload.get("generationConfig", payload)["temperature"], 0.2)

    def test_reasoning_effort_and_image_detail_reach_the_payload(self):
        payload = build_request(config("openai", reasoning_effort="low", image_detail="high"),
                                PROMPT, IMAGE, "image/png")
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["input"][0]["content"][1]["detail"], "high")
        payload = build_request(config("openai-compatible", reasoning_effort="minimal",
                                       image_detail="low"), PROMPT, IMAGE, "image/png")
        self.assertEqual(payload["reasoning_effort"], "minimal")
        self.assertEqual(payload["messages"][0]["content"][1]["image_url"]["detail"], "low")
        for provider in providers.PROVIDERS:
            payload = build_request(config(provider), PROMPT, IMAGE, "image/png")
            self.assertNotIn("reasoning", payload)
            self.assertNotIn("reasoning_effort", payload)
            parts = (payload["input"][0]["content"] if provider == "openai"
                     else payload["messages"][0]["content"] if provider != "gemini"
                     else payload["contents"][0]["parts"])
            if provider == "openai":
                self.assertEqual(parts[1]["detail"], "high")
            elif provider in {"openai-compatible", "deepseek"}:
                self.assertEqual(parts[1]["image_url"]["detail"], "high")
            else:
                for part in parts:
                    self.assertNotIn("detail", part)

    def test_invalid_request_inputs_fail_offline(self):
        for prompt, image, mime in [("", IMAGE, "image/png"), (PROMPT, b"", "image/png"),
                                     (PROMPT, IMAGE, "application/pdf")]:
            with self.assertRaises(ValueError):
                build_request(config(), prompt, image, mime)

    def test_mocked_http_for_every_provider_serializes_exact_payload_and_auth(self):
        expected_urls = {
            "openai": "https://api.openai.com/v1/responses",
            "anthropic": "https://api.anthropic.com/v1/messages",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/models/test-vision-model:generateContent",
            "openai-compatible": "https://compatible.example/v1/chat/completions",
            "qwen": "https://compatible.example/v1/chat/completions",
            "zai": "https://api.z.ai/api/paas/v4/chat/completions",
            "deepseek": "https://api.deepseek.com/chat/completions",
            "deepinfra": "https://api.deepinfra.com/v1/openai/chat/completions",
        }
        for provider in providers.PROVIDERS:
            with self.subTest(provider=provider):
                cfg = config(provider, api_key_env="TEST_PROVIDER_KEY")
                payload = build_request(cfg, PROMPT, IMAGE, "image/png")
                opener = MagicMock()
                opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(success(provider)).encode()
                with patch.dict(providers.os.environ, {"TEST_PROVIDER_KEY": FAKE_KEY}, clear=True), \
                        patch.object(providers.request, "build_opener", return_value=opener) as build:
                    result = call_model(cfg, payload)
                req = opener.open.call_args.args[0]
                self.assertEqual(req.full_url, expected_urls[provider])
                self.assertEqual(req.get_method(), "POST")
                self.assertEqual(json.loads(req.data), payload)
                self.assertEqual(opener.open.call_args.kwargs["timeout"], 120)
                headers = {k.lower(): v for k, v in req.header_items()}
                auth_name = {"anthropic": "x-api-key", "gemini": "x-goog-api-key"}.get(provider, "authorization")
                self.assertEqual(headers[auth_name], ("Bearer " if auth_name == "authorization" else "") + FAKE_KEY)
                self.assertNotIn(FAKE_KEY, req.full_url)
                self.assertNotIn(FAKE_KEY, req.data.decode())
                self.assertIsInstance(build.call_args.args[0], providers._NoRedirect)
                self.assertEqual(result["text"], '{"answer": true}')
                self.assertTrue(result["complete"])
                for key, value in [("input_tokens", 100), ("output_tokens", 25), ("total_tokens", 125)]:
                    self.assertEqual(result["usage"][key], value)

    def test_gemini_model_prefix_and_escaping(self):
        with patch.dict(providers.os.environ, {"GEMINI_API_KEY": FAKE_KEY}, clear=True), \
                patch.object(providers, "_post_json", return_value=success("gemini")) as post:
            call_model(replace(config("gemini"), model="models/model-with/slash"), {})
        self.assertTrue(post.call_args.args[0].endswith("/models/model-with%2Fslash:generateContent"))


class ResponseTests(unittest.TestCase):
    def test_zai_thinking_audit_never_becomes_a_final_answer_or_invented_usage(self):
        cfg = ProviderConfig("zai", "glm-4.6v-flashx", thinking_mode="enabled")
        for reasoning in (None, "", "Yes, but still working", ["malformed optional field"]):
            for final, finish in (("No", "stop"), (None, "length"), ("Yes", "length")):
                with self.subTest(reasoning=reasoning, final=final, finish=finish):
                    raw = success("zai")
                    raw["choices"] = [{"message": {"content": final, "reasoning_content": reasoning},
                                       "finish_reason": finish}]
                    with patch("evals.providers._api_key", return_value=FAKE_KEY), \
                            patch("evals.providers._post_json", return_value=raw):
                        result = call_model(cfg, {})
                    self.assertEqual(result["text"], final or "")
                    self.assertEqual(result["complete"], finish == "stop")
                    self.assertNotIn("reasoning_tokens", result["usage"])
                    self.assertEqual(result["usage"]["output_tokens"], 25)
                    if isinstance(reasoning, str):
                        self.assertEqual(result["reasoning_content"], reasoning)
                    else:
                        self.assertNotIn("reasoning_content", result)
        with patch("evals.providers._api_key", return_value=FAKE_KEY), \
                patch("evals.providers._post_json", return_value=raw):
            self.assertNotIn("reasoning_content", call_model(replace(cfg, thinking_mode=None), {}))

    def test_hosted_chat_usage_does_not_invent_zero_reasoning(self):
        for provider in ("qwen", "zai", "deepinfra"):
            for tokens in (None, 0, 7):
                with self.subTest(provider=provider, reasoning_tokens=tokens):
                    raw = success(provider)
                    if tokens is not None:
                        raw["usage"]["completion_tokens_details"] = {"reasoning_tokens": tokens}
                    result = self.call_with(provider, raw)
                    self.assertEqual(result["usage"]["input_tokens"], 100)
                    self.assertEqual(result["usage"]["output_tokens"], 25)
                    self.assertEqual(result["usage"]["total_tokens"], 125)
                    if tokens is None:
                        self.assertNotIn("reasoning_tokens", result["usage"])
                    else:
                        self.assertEqual(result["usage"]["reasoning_tokens"], tokens)

    def test_gemini_preserves_zero_positive_and_missing_reasoning_counts(self):
        for thoughts in (0, 5, None):
            raw = success("gemini")
            if thoughts is None:
                raw["usageMetadata"].pop("thoughtsTokenCount")
            else:
                raw["usageMetadata"]["thoughtsTokenCount"] = thoughts
            result = providers._extract_response("gemini", raw)
            self.assertEqual(result["text"], '{"answer": true}')
            if thoughts is None:
                self.assertNotIn("reasoning_tokens", result["usage"])
            else:
                self.assertEqual(result["usage"]["reasoning_tokens"], thoughts)
            self.assertEqual(result["usage"]["output_tokens"], 20 + (thoughts or 0))

    def test_deepseek_preserves_cache_and_reasoning_usage_without_inventing_zero(self):
        response = success("deepseek")
        response["choices"][0]["message"]["reasoning_content"] = "This is not the final answer"
        response["usage"].update(prompt_cache_hit_tokens=40, prompt_cache_miss_tokens=60,
                                 completion_tokens_details={"reasoning_tokens": 0})
        result = self.call_with("deepseek", response)
        self.assertEqual(result["text"], '{"answer": true}')
        self.assertEqual(result["usage"], {
            "input_tokens": 100, "output_tokens": 25, "total_tokens": 125,
            "prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 60,
            "reasoning_tokens": 0,
        })
        for value in (None, True, -1, "0"):
            with self.subTest(value=value):
                response["usage"].update(prompt_cache_hit_tokens=value,
                                         prompt_cache_miss_tokens=value,
                                         completion_tokens_details={"reasoning_tokens": value})
                usage = self.call_with("deepseek", response)["usage"]
                for name in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "reasoning_tokens"):
                    self.assertNotIn(name, usage)

    def test_preserves_reported_cost_without_inventing_missing_cost(self):
        for value in (0, 10_000_000, None, True, -1, "100", 1.5):
            with self.subTest(value=value):
                response = success("openai-compatible")
                response["usage"]["cost_in_usd_ticks"] = value
                result = self.call_with("openai-compatible", response)
                if type(value) is int and value >= 0:
                    self.assertEqual(result["usage"]["cost_in_usd_ticks"], value)
                else:
                    self.assertNotIn("cost_in_usd_ticks", result["usage"])
        result = self.call_with("openai-compatible", success("openai-compatible"))
        self.assertNotIn("cost_in_usd_ticks", result["usage"])

    def call_with(self, provider, response):
        with patch.dict(providers.os.environ, {"TEST_KEY": FAKE_KEY}, clear=True), \
                patch.object(providers, "_post_json", return_value=response):
            return call_model(config(provider, api_key_env="TEST_KEY"), {})

    def test_all_truncated_outputs_remain_incomplete_even_when_json_looks_valid(self):
        for provider in providers.PROVIDERS:
            response = success(provider)
            if provider == "openai":
                response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
            elif provider == "anthropic":
                response["stop_reason"] = "max_tokens"
            elif provider == "gemini":
                response["candidates"][0]["finishReason"] = "MAX_TOKENS"
            else:
                response["choices"][0]["finish_reason"] = "length"
            with self.subTest(provider=provider):
                result = self.call_with(provider, response)
                self.assertFalse(result["complete"])
                self.assertEqual(result["text"], '{"answer": true}')

    def test_refusals_and_gemini_blocked_prompt(self):
        response = success("openai")
        response["output"][-1]["content"].append({"type": "refusal", "refusal": "Cannot answer"})
        result = self.call_with("openai", response)
        self.assertFalse(result["complete"])
        self.assertEqual(result["finish_reason"], "refusal")
        response = success("anthropic")
        response["stop_details"] = {"type": "refusal", "explanation": "refused"}
        self.assertFalse(self.call_with("anthropic", response)["complete"])
        response = success("openai-compatible")
        response["choices"][0]["message"]["refusal"] = "Cannot answer"
        self.assertFalse(self.call_with("openai-compatible", response)["complete"])
        result = self.call_with("gemini", {"promptFeedback": {"blockReason": "SAFETY"}})
        self.assertFalse(result["complete"])
        self.assertEqual(result["finish_reason"], "SAFETY")

    def test_empty_answers_are_not_complete(self):
        for provider in providers.PROVIDERS:
            response = success(provider)
            if provider == "openai":
                response["output"] = []
            elif provider == "anthropic":
                response["content"] = []
            elif provider == "gemini":
                response["candidates"][0]["content"]["parts"] = []
            else:
                response["choices"][0]["message"]["content"] = None
            with self.subTest(provider=provider):
                result = self.call_with(provider, response)
                self.assertFalse(result["complete"])
                self.assertEqual(result["finish_reason"], "empty_output")

    def test_openai_selects_final_answer_ignoring_commentary(self):
        response = success("openai")
        response["output"][-1]["phase"] = "final_answer"
        response["output"].insert(0, {"type": "message", "phase": "commentary", "content": [
            {"type": "output_text", "text": "Analyzing..."},
        ]})
        self.assertEqual(self.call_with("openai", response)["text"], '{"answer": true}')

    def test_anthropic_usage_accounts_for_cached_input(self):
        response = success("anthropic")
        response["usage"].update(cache_creation_input_tokens=20, cache_read_input_tokens=30)
        result = self.call_with("anthropic", response)
        self.assertEqual(result["usage"]["input_tokens"], 150)
        self.assertEqual(result["usage"]["total_tokens"], 175)

    def test_malformed_and_api_error_are_redacted(self):
        for response in [{"output": None}, {"error": {"message": FAKE_KEY}}]:
            with self.assertRaises(ProviderError) as caught:
                self.call_with("openai", response)
            self.assertNotIn(FAKE_KEY, str(caught.exception))


class TransportTests(unittest.TestCase):
    def test_deepinfra_auth_failure_reports_redacted_message_without_retry(self):
        body = io.BytesIO(json.dumps({"error": {"message": f"Invalid key: {FAKE_KEY}"}}).encode())
        failure = HTTPError("https://api.deepinfra.com/v1/openai/chat/completions",
                            401, "Unauthorized", {}, body)
        with patch.dict(providers.os.environ, {"DEEPINFRA_API_KEY": FAKE_KEY}, clear=True), \
                patch.object(providers, "_post_json", side_effect=failure) as post, \
                patch.object(providers.time, "sleep") as sleep:
            with self.assertRaises(ProviderError) as caught:
                call_model(config("deepinfra", retries=3), {})
        self.assertIn("deepinfra request failed with HTTP 401", str(caught.exception))
        self.assertIn("Invalid key: [REDACTED]", str(caught.exception))
        self.assertNotIn(FAKE_KEY, str(caught.exception))
        self.assertFalse(caught.exception.uncertain)
        self.assertFalse(caught.exception.transient)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        self.assertTrue(body.closed)

    def test_compatible_http_error_reports_nested_or_string_message_safely(self):
        message = (f"Request rejected for {FAKE_KEY}; xai-example-secret; "
                   "sk-proj-example-secret.\n\x1b[31m")
        for detail in ({"message": message, "details": "hidden-details"}, message):
            with self.subTest(detail_type=type(detail).__name__):
                body = io.BytesIO(json.dumps({"error": detail, "code": "hidden-code"}).encode())
                failure = HTTPError("https://example.com?secret=" + FAKE_KEY,
                                    400, "hidden-reason", {"private": "hidden-header"}, body)
                with patch.object(providers, "_post_json", side_effect=failure) as post, \
                        patch.object(providers.time, "sleep") as sleep:
                    with self.assertRaises(ProviderError) as caught:
                        call_model(config("openai-compatible", api_key_env="OPENAI_API_KEY",
                                          retries=3), {})
                text = str(caught.exception)
                self.assertIn("HTTP 400", text)
                self.assertIn("Request rejected", text)
                self.assertIn("[REDACTED]", text)
                for secret in (FAKE_KEY, "xai-example-secret", "sk-proj-example-secret",
                               "hidden-details", "hidden-code", "hidden-reason",
                               "hidden-header", "\n", "\x1b"):
                    self.assertNotIn(secret, text)
                self.assertEqual(caught.exception.status_code, 400)
                self.assertFalse(caught.exception.uncertain)
                self.assertFalse(caught.exception.transient)
                self.assertEqual(post.call_count, 1)
                sleep.assert_not_called()
                self.assertTrue(body.closed)

    def test_compatible_unusable_error_body_preserves_http_failure(self):
        for raw in (FAKE_KEY.encode(), b"<html>Bad request</html>",
                    b'{"error": ["unexpected"]}', b'{"error": null}',
                    json.dumps({"error": "x" * 17000}).encode()):
            with self.subTest(raw_size=len(raw)):
                body = io.BytesIO(raw)
                failure = HTTPError("https://example.com", 400, FAKE_KEY, {}, body)
                with patch.object(providers, "_post_json", side_effect=failure):
                    with self.assertRaises(ProviderError) as caught:
                        call_model(config("openai-compatible", api_key_env="OPENAI_API_KEY"), {})
                self.assertEqual(str(caught.exception),
                                 "openai-compatible request failed with HTTP 400")
                self.assertFalse(caught.exception.uncertain)
                self.assertTrue(body.closed)

    def test_hosted_account_errors_are_explained_without_exposing_the_key(self):
        for provider in ("qwen", "zai", "deepseek"):
            with self.subTest(provider=provider):
                body = io.BytesIO(json.dumps({"error": {
                    "message": f"Model access denied for credential {FAKE_KEY}"}}).encode())
                failure = HTTPError("https://example.com", 403, "Forbidden", {}, body)
                cfg = config(provider, api_key_env="OPENAI_API_KEY")
                with patch.object(providers, "_post_json", side_effect=failure) as post:
                    with self.assertRaises(ProviderError) as caught:
                        call_model(cfg, {})
                self.assertIn("Model access denied", str(caught.exception))
                self.assertNotIn(FAKE_KEY, str(caught.exception))
                self.assertIn("[REDACTED]", str(caught.exception))
                self.assertFalse(caught.exception.uncertain)
                self.assertFalse(caught.exception.transient)
                self.assertEqual(post.call_count, 1)
                self.assertTrue(body.closed)

    def setUp(self):
        self.env = patch.dict(providers.os.environ, {"OPENAI_API_KEY": FAKE_KEY}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    @staticmethod
    def http_error(status):
        return HTTPError("https://example.com?secret=" + FAKE_KEY, status, FAKE_KEY,
                         {}, io.BytesIO(FAKE_KEY.encode()))

    def test_http_errors_default_to_one_attempt_and_do_not_disclose_body(self):
        with patch.object(providers, "_post_json", side_effect=self.http_error(429)) as post:
            with self.assertRaisesRegex(ProviderError, "HTTP 429") as caught:
                call_model(config(), {})
        self.assertEqual(post.call_count, 1)
        self.assertNotIn(FAKE_KEY, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_gemini_http_error_explains_rejection_without_retrying_404(self):
        message = "models/gemini-2.5-flash is not found or does not support generateContent."
        body = io.BytesIO(json.dumps({"error": {"code": 404, "status": "NOT_FOUND",
                                               "message": message}}).encode())
        failure = HTTPError("https://example.com", 404, "Not Found", {}, body)
        cfg = replace(config("gemini", api_key_env="OPENAI_API_KEY", retries=3),
                      model="gemini-2.5-flash", thinking_budget=0)
        with patch.object(providers, "_post_json", side_effect=failure) as post:
            with self.assertRaises(ProviderError) as caught:
                call_model(cfg, {})
        self.assertIn(message, str(caught.exception))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(caught.exception.status_code, 404)
        self.assertFalse(caught.exception.uncertain)
        self.assertFalse(caught.exception.transient)
        self.assertTrue(body.closed)

    def test_anthropic_http_error_reports_setup_problem_and_redacts_credentials(self):
        message = f"Credit balance too low. Key {FAKE_KEY}; sk-ant-example-secret"
        body = io.BytesIO(json.dumps({"type": "error", "error": {
            "type": "invalid_request_error", "message": message},
            "request_id": "hidden-request"}).encode())
        failure = HTTPError("https://example.com", 400, FAKE_KEY, {}, body)
        with patch.object(providers, "_post_json", side_effect=failure) as post:
            with self.assertRaises(ProviderError) as caught:
                call_model(config("anthropic", api_key_env="OPENAI_API_KEY", reasoning_effort="none"), {})
        self.assertIn("Credit balance too low", str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))
        for secret in (FAKE_KEY, "sk-ant-example-secret", "hidden-request"):
            self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(post.call_count, 1)
        self.assertFalse(caught.exception.transient)
        self.assertFalse(caught.exception.uncertain)
        self.assertEqual(caught.exception.status_code, 400)
        self.assertTrue(body.closed)

    def test_gemini_http_error_redacts_secrets_before_clipping_and_quotes_controls(self):
        for message in (f"API key {FAKE_KEY} rejected.\n\x1b[31m",
                        "x" * 790 + FAKE_KEY + " trailing text",
                        "Key AIza" + "X" * 35 + " rejected"):
            with self.subTest(message=message):
                body = io.BytesIO(json.dumps({"error": {
                    "message": message, "details": [{"private": "hidden-details"}],
                    "status": "hidden-status"}}).encode())
                failure = HTTPError("https://example.com?key=" + FAKE_KEY, 403,
                                    "hidden-reason", {"private": "hidden-header"}, body)
                with patch.object(providers, "_post_json", side_effect=failure):
                    with self.assertRaises(ProviderError) as caught:
                        call_model(config("gemini", api_key_env="OPENAI_API_KEY"), {})
                text = str(caught.exception)
                self.assertIn("[REDACTED]", text)
                for secret in (FAKE_KEY, FAKE_KEY[:10], "AIza", "hidden-details", "hidden-status",
                               "hidden-reason", "hidden-header", "\n", "\x1b"):
                    self.assertNotIn(secret, text)
                self.assertLess(len(text), 1000)

    def test_gemini_unusable_error_body_preserves_original_http_failure(self):
        bodies = [FAKE_KEY.encode(), b"<html>Not Found</html>", b"[]", b'{"error": null}',
                  b'{"error": {"message": 7}}', b"{", b"\xff",
                  json.dumps({"error": {"message": "x" * 17000}}).encode()]
        for raw in bodies:
            with self.subTest(raw_size=len(raw)):
                body = io.BytesIO(raw)
                failure = HTTPError("https://example.com", 404, FAKE_KEY, {}, body)
                with patch.object(providers, "_post_json", side_effect=failure):
                    with self.assertRaises(ProviderError) as caught:
                        call_model(config("gemini", api_key_env="OPENAI_API_KEY"), {})
                self.assertEqual(str(caught.exception), "gemini request failed with HTTP 404")
                self.assertFalse(caught.exception.uncertain)
                self.assertTrue(body.closed)

        failure = self.http_error(404)
        with patch.object(failure, "read", side_effect=TimeoutError(FAKE_KEY)), \
                patch.object(providers, "_post_json", side_effect=failure):
            with self.assertRaises(ProviderError) as caught:
                call_model(config("gemini", api_key_env="OPENAI_API_KEY"), {})
        self.assertEqual(str(caught.exception), "gemini request failed with HTTP 404")
        self.assertFalse(caught.exception.uncertain)
        self.assertFalse(caught.exception.transient)

    def test_explicit_retries_are_bounded_and_only_transient(self):
        with patch.object(providers, "_post_json", side_effect=[self.http_error(429), success("openai")]) as post, \
                patch.object(providers.time, "sleep") as sleep:
            self.assertTrue(call_model(config(retries=1), {})["complete"])
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1)
        with patch.object(providers, "_post_json", side_effect=self.http_error(401)) as post:
            with self.assertRaisesRegex(ProviderError, "HTTP 401"):
                call_model(config(retries=3), {})
        self.assertEqual(post.call_count, 1)
        with patch.object(providers, "_post_json", side_effect=self.http_error(503)) as post, \
                patch.object(providers.time, "sleep"):
            with self.assertRaisesRegex(ProviderError, "HTTP 503"):
                call_model(config(retries=2), {})
        self.assertEqual(post.call_count, 3)

    def test_network_error_is_redacted(self):
        with patch.object(providers, "_post_json", side_effect=URLError(FAKE_KEY)) as post:
            with self.assertRaisesRegex(ProviderError, "network/timeout") as caught:
                call_model(config(), {})
        self.assertEqual(post.call_count, 1)
        self.assertNotIn(FAKE_KEY, str(caught.exception))

    def test_failed_submissions_classify_billing_uncertainty(self):
        for failure, uncertain in [(TimeoutError(), True), (self.http_error(503), True),
                                   (self.http_error(429), False), (self.http_error(401), False),
                                   (self.http_error(400), False)]:
            with self.subTest(failure=type(failure).__name__, uncertain=uncertain):
                with patch.object(providers, "_post_json", side_effect=failure):
                    with self.assertRaises(ProviderError) as caught:
                        call_model(config(), {})
                self.assertIs(caught.exception.uncertain, uncertain)

    def test_only_transient_failures_allow_continuation(self):
        for status in (302, 400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504, 529):
            with self.subTest(status=status), \
                    patch.object(providers, "_post_json", side_effect=self.http_error(status)):
                with self.assertRaises(ProviderError) as caught:
                    call_model(config(), {})
                self.assertIs(caught.exception.transient,
                              status in (408, 409, 429, 500, 502, 503, 504, 529))
                self.assertEqual(caught.exception.status_code, status)
        for failure in (TimeoutError(FAKE_KEY), URLError(FAKE_KEY)):
            with patch.object(providers, "_post_json", side_effect=failure):
                with self.assertRaises(ProviderError) as caught:
                    call_model(config(), {})
                self.assertTrue(caught.exception.transient)
                self.assertTrue(caught.exception.uncertain)
                self.assertIsNone(caught.exception.status_code)
                self.assertNotIn(FAKE_KEY, str(caught.exception))

    def test_uncertainty_survives_a_later_definite_rejection(self):
        with patch.object(providers, "_post_json", side_effect=[TimeoutError(), self.http_error(429)]), \
                patch.object(providers.time, "sleep"):
            with self.assertRaises(ProviderError) as caught:
                call_model(config(retries=1), {})
        self.assertTrue(caught.exception.uncertain)
        self.assertTrue(caught.exception.transient)

    def test_auth_failure_after_timeout_still_stops_continuation(self):
        with patch.object(providers, "_post_json", side_effect=[TimeoutError(), self.http_error(401)]), \
                patch.object(providers.time, "sleep"):
            with self.assertRaises(ProviderError) as caught:
                call_model(config(retries=1), {})
        self.assertTrue(caught.exception.uncertain)
        self.assertFalse(caught.exception.transient)
        self.assertEqual(caught.exception.status_code, 401)

    def test_redirect_handler_never_follows(self):
        self.assertIsNone(providers._NoRedirect().redirect_request(None, None, 302, "Found", {},
                                                                 "https://untrusted.example"))
        with patch.object(providers, "_post_json", side_effect=self.http_error(302)) as post:
            with self.assertRaisesRegex(ProviderError, "HTTP 302"):
                call_model(config(retries=3), {})
        self.assertEqual(post.call_count, 1)

    def test_bad_api_bodies_are_recoverable_redacted_and_not_implicitly_retried(self):
        for raw in (FAKE_KEY.encode(), b"null", b"[]", b'"unexpected"', b'{"choices": null}'):
            with self.subTest(raw=raw):
                opener = MagicMock()
                opener.open.return_value.__enter__.return_value.read.return_value = raw
                with patch.object(providers.request, "build_opener", return_value=opener):
                    with self.assertRaises(ProviderError) as caught:
                        call_model(config("deepseek", api_key_env="OPENAI_API_KEY"), {})
                self.assertEqual(opener.open.call_count, 1)
                self.assertTrue(caught.exception.transient)
                self.assertTrue(caught.exception.uncertain)
                self.assertNotIn(FAKE_KEY, str(caught.exception))

    def test_explicit_malformed_response_retry_is_bounded(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.side_effect = [
            b"null", json.dumps(success("deepseek")).encode()]
        with patch.object(providers.request, "build_opener", return_value=opener), \
                patch.object(providers.time, "sleep") as sleep:
            result = call_model(config("deepseek", api_key_env="OPENAI_API_KEY", retries=1), {})
        self.assertTrue(result["complete"])
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_retry_after_is_sanitized_and_carried_on_rate_limit_errors(self):
        cases = [("90", 90), ("0", 0), ("-1", None), ("nan", None),
                 ("inf", None), (FAKE_KEY, None)]
        for header, expected in cases:
            failure = self.http_error(429)
            failure.headers["Retry-After"] = header
            with self.subTest(header=header), \
                    patch.object(providers, "_post_json", side_effect=failure):
                with self.assertRaises(ProviderError) as caught:
                    call_model(config(), {})
                self.assertEqual(caught.exception.retry_after_seconds, expected)
                self.assertNotIn(FAKE_KEY, str(caught.exception))
        now = datetime(2026, 9, 15, tzinfo=timezone.utc)
        with patch.object(providers, "datetime") as clock:
            clock.now.return_value = now
            self.assertEqual(providers._retry_after_seconds(format_datetime(now + timedelta(seconds=90))), 90)
        failure = self.http_error(429)
        failure.headers["Retry-After"] = "90"
        with patch.object(providers, "_post_json", side_effect=[failure, success("openai")]), \
                patch.object(providers.time, "sleep") as sleep:
            call_model(config(retries=1), {})
        sleep.assert_called_once_with(90)


if __name__ == "__main__":
    unittest.main()
