"""Small, dependency-free vision API adapters.

Request construction is entirely offline. Only ``call_model`` performs network
I/O, and only it and ``check_api_key`` inspect the configured key environment
variable. There are no retries unless explicitly requested in ProviderConfig.
"""

from __future__ import annotations

import base64
import http.client
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import ipaddress
import json
import math
import os
import re
import time
from typing import Any
from urllib import error, parse, request


PROVIDERS = ("openai", "anthropic", "gemini", "openai-compatible", "qwen", "zai", "deepseek", "deepinfra")
_DEFAULTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
    "openai-compatible": (None, "OPENAI_API_KEY"),
    # Model Studio keys/endpoints are region- and workspace-specific. Copy the
    # OpenAI-compatible API root from the console instead of guessing a host.
    "qwen": (None, "DASHSCOPE_API_KEY"),
    "zai": ("https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
}
_CHAT_PROVIDERS = {"openai-compatible", "qwen", "zai", "deepseek", "deepinfra"}
_DEEPINFRA_SCOUT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
_QWEN3_VL_INSTRUCT_MODELS = {
    f"qwen3-vl-{size}-instruct" for size in ("2b", "4b", "8b", "32b", "30b-a3b", "235b-a22b")
}
_ZAI_THINKING_OFF_MODELS = {"glm-4.5v", "glm-4.6v-flash", "glm-4.6v-flashx"}
_ZAI_THINKING_MODELS = _ZAI_THINKING_OFF_MODELS | {"glm-4.6v"}
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_EFFORT_PROVIDERS = {"openai", "openai-compatible"}
_IMAGE_DETAILS = {"auto", "low", "high", "original"}
_IMAGE_DETAIL_PROVIDERS = {"openai", "openai-compatible", "deepseek"}
# Stable Gemini 2.5 models whose GenerateContent thinking budgets are documented.
# Gemini 3 uses different controls and does not provide a full thinking-off mode.
_GEMINI_THINKING_BUDGETS = {
    "gemini-2.5-flash": (0, 24576, True),
    "gemini-2.5-flash-lite": (512, 24576, True),
    "gemini-2.5-pro": (128, 32768, False),
}
_GEMINI_THINKING_LEVELS = {
    "gemini-3-flash-preview": {"minimal", "low", "medium", "high"},
    "gemini-3.1-flash-lite": {"minimal", "low", "medium", "high"},
    "gemini-3.5-flash-lite": {"minimal", "low", "medium", "high"},
    "gemini-3.5-flash": {"minimal", "low", "medium", "high"},
    "gemini-3.6-flash": {"minimal", "low", "medium", "high"},
    "gemini-3.7-flash": {"low", "medium", "high"},
    "gemini-3.8-flash": {"low", "medium", "high"},
    "gemini-3.1-pro-preview": {"low", "medium", "high"},
}
# On the Gemini API, Gemma 4 uses minimal=disabled and high=enabled.
# Gemini models' minimal level still allows thinking; keep the model distinction.
_GEMMA_THINKING_LEVELS = {
    "gemma-4-26b-a4b-it": {"minimal", "high"},
    "gemma-4-31b-it": {"minimal", "high"},
}


class ProviderError(RuntimeError):
    """An API/transport failure with no credentials or full response body."""

    def __init__(self, message: str, *, uncertain: bool = True,
                 transient: bool = False, status_code: int | None = None,
                 retry_after_seconds: float | None = None):
        super().__init__(message)
        self.uncertain = uncertain
        self.transient = transient
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    max_output_tokens: int = 4096
    temperature: float | None = None
    timeout: float = 120
    retries: int = 0
    reasoning_effort: str | None = None
    image_detail: str | None = None
    thinking_budget: int | None = None
    thinking_level: str | None = None
    thinking_mode: str | None = None

    def __post_init__(self) -> None:
        # Enforce high image detail for providers that expose the knob.
        if self.image_detail is None and self.provider in _IMAGE_DETAIL_PROVIDERS:
            object.__setattr__(self, "image_detail", "high")

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError("provider must be one of: " + ", ".join(PROVIDERS))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if self.model != self.model.strip() or any(ord(c) < 32 for c in self.model):
            raise ValueError("model must not contain control or surrounding whitespace")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if type(self.retries) is not int or not 0 <= self.retries <= 10:
            raise ValueError("retries must be an integer between 0 and 10")
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be a finite positive number")
        if self.temperature is not None:
            upper = 1 if self.provider == "anthropic" else 2
            if (isinstance(self.temperature, bool)
                    or not isinstance(self.temperature, (int, float))
                    or not math.isfinite(self.temperature)
                    or not 0 <= self.temperature <= upper):
                raise ValueError(f"temperature must be between 0 and {upper}")
        if self.reasoning_effort is not None:
            if self.reasoning_effort not in _REASONING_EFFORTS:
                raise ValueError("reasoning_effort must be one of: " + ", ".join(sorted(_REASONING_EFFORTS)))
            if self.provider == "anthropic":
                if self.reasoning_effort != "none":
                    raise ValueError("anthropic supports only reasoning_effort='none' in this runner; "
                                     "it sends thinking.type='disabled'")
            elif self.provider in {"qwen", "zai", "deepseek", "deepinfra"}:
                if self.reasoning_effort != "none":
                    raise ValueError(f"{self.provider} supports only reasoning_effort='none' in this runner")
                if self.provider == "qwen" and self.model not in _QWEN3_VL_INSTRUCT_MODELS:
                    raise ValueError("qwen reasoning_effort='none' requires a released Qwen3-VL "
                                     "Instruct model, such as qwen3-vl-8b-instruct; "
                                     "Thinking checkpoints and hybrid aliases are not interchangeable")
                if self.provider == "zai" and self.model not in _ZAI_THINKING_OFF_MODELS:
                    raise ValueError("zai reasoning_effort='none' requires one of: "
                                     + ", ".join(sorted(_ZAI_THINKING_OFF_MODELS)))
                if self.provider == "deepseek" and self.model != "deepseek-flash":
                    raise ValueError("deepseek reasoning_effort='none' requires deepseek-flash "
                                     "for this vision benchmark; use the current model ID")
                if self.provider == "deepinfra" and self.model != _DEEPINFRA_SCOUT_MODEL:
                    raise ValueError("deepinfra reasoning_effort='none' requires "
                                     + _DEEPINFRA_SCOUT_MODEL + "; the setting records the "
                                     "standard Instruct checkpoint, not a thinking toggle")
            elif self.provider not in _EFFORT_PROVIDERS:
                raise ValueError(f"reasoning_effort is not supported for {self.provider}; "
                                 "that provider's thinking control has different semantics")
        if self.image_detail is not None:
            if self.provider not in _IMAGE_DETAIL_PROVIDERS:
                raise ValueError(f"image_detail is not supported for {self.provider}")
            allowed = (_IMAGE_DETAILS if self.provider in {"openai", "deepseek"}
                       else _IMAGE_DETAILS - {"original"})
            if self.image_detail not in allowed:
                raise ValueError("image_detail must be one of: " + ", ".join(sorted(allowed)))
        if self.thinking_mode is not None:
            if self.provider != "zai" or self.model not in _ZAI_THINKING_MODELS:
                raise ValueError("thinking_mode requires a supported Z.ai vision model: "
                                 + ", ".join(sorted(_ZAI_THINKING_MODELS)))
            if not isinstance(self.thinking_mode, str) or self.thinking_mode not in {"enabled", "disabled"}:
                raise ValueError("thinking_mode must be enabled or disabled")
            if any(value is not None for value in
                   (self.reasoning_effort, self.thinking_budget, self.thinking_level)):
                raise ValueError("thinking_mode cannot be combined with reasoning_effort, "
                                 "thinking_budget, or thinking_level")
        if self.thinking_budget is not None and self.thinking_level is not None:
            raise ValueError("thinking_budget and thinking_level are mutually exclusive")
        if self.thinking_level is not None:
            if self.provider != "gemini":
                raise ValueError("thinking_level is supported only for the gemini provider")
            model_id = self.model.removeprefix("models/")
            allowed = (_GEMINI_THINKING_LEVELS.get(model_id)
                       or _GEMMA_THINKING_LEVELS.get(model_id))
            if allowed is None:
                raise ValueError("thinking_level requires a supported Gemini 3 or Gemma 4 text-output model; "
                                 "use thinking_budget for Gemini 2.5")
            if not isinstance(self.thinking_level, str) or self.thinking_level not in allowed:
                note = ("; minimal disables thinking for Gemma 4" if model_id in _GEMMA_THINKING_LEVELS
                        else "; minimal is not thinking-off")
                raise ValueError(f"thinking_level for {self.model} must be one of: "
                                 + ", ".join(sorted(allowed)) + note)
        if self.thinking_budget is not None:
            if self.provider != "gemini":
                raise ValueError("thinking_budget is supported only for the gemini provider")
            limits = _GEMINI_THINKING_BUDGETS.get(self.model.removeprefix("models/"))
            if limits is None:
                raise ValueError("thinking_budget requires gemini-2.5-flash, gemini-2.5-flash-lite, "
                                 "or gemini-2.5-pro; Gemini 3 cannot disable thinking")
            minimum, maximum, can_disable = limits
            budget = self.thinking_budget
            if (type(budget) is not int or not (
                    budget == -1 or (can_disable and budget == 0) or minimum <= budget <= maximum)):
                allowed = f"-1 (dynamic), {minimum}..{maximum}"
                if can_disable:
                    allowed += ", or 0 (disabled)"
                raise ValueError(f"thinking_budget for {self.model} must be {allowed}")
        root = self.base_url if self.base_url is not None else _DEFAULTS[self.provider][0]
        if not root:
            raise ValueError(f"{self.provider} requires an explicit base_url API root")
        _validate_url(root)

    def public_dict(self) -> dict[str, Any]:
        """Serializable configuration; contains a key variable name, never its value."""
        self.validate()
        result = asdict(self)
        # Preserve existing manifests/fingerprints when this new knob is unused,
        # including the Grok run that may be active while Gemini is configured.
        if self.thinking_budget is None:
            result.pop("thinking_budget")
        if self.thinking_level is None:
            result.pop("thinking_level")
        if self.thinking_mode is None:
            result.pop("thinking_mode")
        result["api_key_env"] = self.api_key_env or _DEFAULTS[self.provider][1]
        result["base_url"] = self.base_url or _DEFAULTS[self.provider][0]
        return result


def _validate_url(url: str) -> None:
    if not isinstance(url, str) or any(c.isspace() for c in url):
        raise ValueError("base_url must be an HTTPS API root (or localhost HTTP)")
    try:
        parsed = parse.urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError("base_url is not a valid API root") from None
    if not host or parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url requires a host and must not include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not include a query or fragment; use api_key_env")
    loopback = host.lower() == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        raise ValueError("base_url must use HTTPS, except for localhost HTTP")
    if port == 0:
        raise ValueError("base_url port must be positive")


def build_request(config: ProviderConfig, prompt: str, image_bytes: bytes,
                  mime_type: str) -> dict[str, Any]:
    """Build one prompt + image request without reading credentials or doing I/O."""
    config.validate()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("image_bytes must contain the image bytes")
    if mime_type not in _IMAGE_TYPES:
        raise ValueError("unsupported image MIME type")
    if config.provider == "deepinfra" and mime_type == "image/gif":
        raise ValueError("deepinfra supports JPEG, PNG, and WebP images, not GIF")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    if config.provider == "openai":
        payload = {
            "model": config.model,
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url,
                 "detail": config.image_detail},
            ]}],
            "max_output_tokens": config.max_output_tokens,
            "store": False,
        }
        if config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": config.reasoning_effort}
    elif config.provider == "anthropic":
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type,
                                               "data": encoded}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": config.max_output_tokens,
        }
        if config.reasoning_effort == "none":
            payload["thinking"] = {"type": "disabled"}
    elif config.provider == "gemini":
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": prompt}, {"inlineData": {"mimeType": mime_type, "data": encoded}},
            ]}],
            "generationConfig": {"maxOutputTokens": config.max_output_tokens},
        }
        if config.thinking_budget is not None:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": config.thinking_budget}
        elif config.thinking_level is not None:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": config.thinking_level}
    else:
        image_url = {"url": data_url}
        if config.image_detail is not None:
            image_url["detail"] = config.image_detail
        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url},
            ]}],
            "max_tokens": config.max_output_tokens,
        }
        if config.provider == "zai" and config.thinking_mode is not None:
            payload["thinking"] = {"type": config.thinking_mode}
        elif config.provider in {"zai", "deepseek"} and config.reasoning_effort == "none":
            payload["thinking"] = {"type": "disabled"}
        elif config.provider == "openai-compatible" and config.reasoning_effort is not None:
            payload["reasoning_effort"] = config.reasoning_effort
        # Qwen3-VL Instruct is a separate, non-thinking checkpoint. Its none
        # setting is validated above and recorded locally; do not send the
        # hybrid-model enable_thinking flag or OpenAI's reasoning_effort to it.
        # DeepInfra Scout likewise records standard Instruct model selection;
        # neither a thinking toggle nor image_url.detail is supported here.
    if config.temperature is not None:
        target = payload["generationConfig"] if config.provider == "gemini" else payload
        target["temperature"] = config.temperature
    return payload


def check_api_key(config: ProviderConfig) -> None:
    """Check the selected variable; never print or return its value."""
    _api_key(config)


def _api_key(config: ProviderConfig) -> str:
    config.validate()
    name = config.api_key_env or _DEFAULTS[config.provider][1]
    value = os.environ.get(name, "")
    if not value.strip():
        raise ProviderError(f"Missing API key: set the {name} environment variable", uncertain=False)
    if any(c.isspace() for c in value):
        raise ProviderError(f"Invalid API key in {name}: whitespace is not allowed", uncertain=False)
    return value


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward authentication to a redirect target.
        return None


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str],
               timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    opener = request.build_opener(_NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        raw = response.read()
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProviderError("Provider returned invalid JSON", transient=True) from None
    if not isinstance(result, dict):
        raise ProviderError("Provider returned a non-object JSON response", transient=True)
    return result


def _retry_after_seconds(value: str | None) -> float | None:
    """Keep only a valid Retry-After delay; never expose raw response headers."""
    if not isinstance(value, str):
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                return None
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) and seconds >= 0 else None


def _json_error_detail(exc: error.HTTPError, key: str) -> str:
    """Read only a provider's JSON error message; redact credentials before clipping.

    Full error bodies, headers, URLs, and nested details are never persisted. An
    unreadable or unexpected body must not hide the original HTTP failure.
    """
    try:
        raw = exc.read(16385)
        if len(raw) > 16384:
            return ""
        envelope = json.loads(raw)
        detail = envelope.get("error") if isinstance(envelope, dict) else None
        # Compatible endpoints can return either an error object (OpenAI
        # shape) or a plain error string (xAI shape).
        message = detail.get("message") if isinstance(detail, dict) else detail
        if not isinstance(message, str) or not message.strip():
            return ""
        # Decode JSON before redaction so JSON-escaped keys are also removed.
        for secret in (key, parse.quote(key, safe=""), parse.quote_plus(key)):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        message = re.sub(r"(?:AIza|sk-|xai-)[A-Za-z0-9_-]+", "[REDACTED]", message)
        # Quoting prevents terminal control characters/newlines from becoming
        # executable terminal output; limit diagnostic text without leaking a
        # key prefix at the truncation boundary.
        clipped = message[:800] + ("..." if len(message) > 800 else "")
        return ": " + json.dumps(clipped, ensure_ascii=True)
    except (ValueError, TypeError, OSError, http.client.HTTPException):
        return ""


def call_model(config: ProviderConfig, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one API request (potentially billable); retries default to zero."""
    config.validate()
    key = _api_key(config)
    root = (config.base_url or _DEFAULTS[config.provider][0]).rstrip("/")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.provider == "anthropic":
        url = root + "/messages"
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    elif config.provider == "gemini":
        model = config.model.removeprefix("models/")
        url = root + "/models/" + parse.quote(model, safe="") + ":generateContent"
        headers["x-goog-api-key"] = key
    else:
        url = root + ("/responses" if config.provider == "openai" else "/chat/completions")
        headers["Authorization"] = "Bearer " + key
    uncertain_attempt = False
    for attempt in range(config.retries + 1):
        try:
            response = _post_json(url, payload, headers, config.timeout)
            if response.get("error"):
                raise ProviderError(f"{config.provider} returned an API error")
            try:
                result = _extract_response(config.provider, response)
                if config.provider == "zai" and config.thinking_mode == "enabled":
                    # Keep provider-returned reasoning as audit evidence only.
                    # Scoring always uses the separate final answer in text.
                    reasoning = response["choices"][0]["message"].get("reasoning_content")
                    if isinstance(reasoning, str):
                        result["reasoning_content"] = reasoning
                return result
            except (TypeError, AttributeError, KeyError, IndexError, ValueError):
                raise ProviderError(f"{config.provider} returned an unexpected response shape",
                                    transient=True) from None
        except error.HTTPError as exc:
            status = exc.code
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None
            try:
                detail = (_json_error_detail(exc, key)
                          if config.provider in {"gemini", "anthropic", "qwen", "zai", "deepseek",
                                                 "openai-compatible", "deepinfra"} else "")
            finally:
                exc.close()
            # Preserve uncertainty from earlier submissions even if a later
            # retry is definitively rejected (for example timeout, then 429).
            uncertain_attempt |= status not in {400, 401, 403, 404, 429}
            if status in _TRANSIENT_STATUS and attempt < config.retries:
                time.sleep(max(min(2 ** attempt, 8), retry_after or 0))
                continue
            raise ProviderError(f"{config.provider} request failed with HTTP {status}{detail}",
                                uncertain=uncertain_attempt,
                                transient=status in _TRANSIENT_STATUS, status_code=status,
                                retry_after_seconds=retry_after) from None
        except (error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException):
            uncertain_attempt = True
            if attempt < config.retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise ProviderError(f"{config.provider} request failed (network/timeout error)",
                                transient=True) from None
        except ProviderError as exc:
            uncertain_attempt |= exc.uncertain
            if exc.transient and attempt < config.retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            exc.uncertain = uncertain_attempt
            raise
    raise AssertionError("unreachable")


def _tokens(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _usage(provider: str, response: dict[str, Any]) -> dict[str, int]:
    raw = response.get("usageMetadata" if provider == "gemini" else "usage") or {}
    if not isinstance(raw, dict):
        return {}
    if provider == "gemini":
        incoming = _tokens(raw.get("promptTokenCount"))
        outgoing = _tokens(raw.get("candidatesTokenCount"))
        thinking = _tokens(raw.get("thoughtsTokenCount"))
        if thinking is not None:
            outgoing = (outgoing or 0) + thinking
        total = _tokens(raw.get("totalTokenCount"))
    elif provider in _CHAT_PROVIDERS:
        incoming = _tokens(raw.get("prompt_tokens"))
        outgoing = _tokens(raw.get("completion_tokens"))
        thinking = _tokens((raw.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        total = _tokens(raw.get("total_tokens"))
    else:
        incoming = _tokens(raw.get("input_tokens"))
        outgoing = _tokens(raw.get("output_tokens"))
        thinking_field = "thinking_tokens" if provider == "anthropic" else "reasoning_tokens"
        thinking = _tokens((raw.get("output_tokens_details") or {}).get(thinking_field))
        total = _tokens(raw.get("total_tokens"))
        if provider == "anthropic" and incoming is not None:
            incoming += (_tokens(raw.get("cache_creation_input_tokens")) or 0)
            incoming += (_tokens(raw.get("cache_read_input_tokens")) or 0)
    if total is None and incoming is not None and outgoing is not None:
        total = incoming + outgoing
    values = {"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": total,
              "reasoning_tokens": thinking,
              "cost_in_usd_ticks": _tokens(raw.get("cost_in_usd_ticks"))}
    if provider == "deepseek":
        # These are a breakdown of input_tokens, not additional token charges.
        # Preserve absent counts as absent, including missing reasoning usage.
        for name in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            values[name] = _tokens(raw.get(name))
    return {name: value for name, value in values.items() if value is not None}


def _extract_response(provider: str, response: dict[str, Any]) -> dict[str, Any]:
    refusal = False
    response_id = response.get("id")
    if provider == "openai":
        messages = [item for item in response["output"] if item.get("type") == "message"]
        refusal = any(part.get("type") == "refusal" for item in messages
                      for part in item.get("content", []))
        final = [item for item in messages if item.get("phase") == "final_answer"]
        selected = final or [item for item in messages if item.get("phase") != "commentary"]
        text = "\n".join(part["text"] for item in selected for part in item.get("content", [])
                         if part.get("type") == "output_text")
        reason = (response.get("incomplete_details") or {}).get("reason") or response.get("status")
        complete = response.get("status") == "completed" and all(
            item.get("status", "completed") == "completed" for item in selected
        ) and not response.get("incomplete_details")
    elif provider == "anthropic":
        text = "\n".join(part["text"] for part in response["content"] if part.get("type") == "text")
        reason = response.get("stop_reason")
        refusal = ((response.get("stop_details") or {}).get("type") == "refusal"
                   or reason == "refusal"
                   or any(part.get("type") == "refusal" for part in response["content"]))
        complete = reason in {"end_turn", "stop_sequence"}
    elif provider == "gemini":
        response_id = response.get("responseId")
        candidates = response.get("candidates") or []
        if candidates:
            candidate = candidates[0]
            text = "\n".join(part["text"] for part in candidate.get("content", {}).get("parts", [])
                             if "text" in part and not part.get("thought", False))
            reason = candidate.get("finishReason")
            complete = reason == "STOP"
        else:
            text = ""
            reason = (response.get("promptFeedback") or {}).get("blockReason") or "no_candidates"
            complete = False
    else:
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(part["text"] for part in content if part.get("type") == "text")
        else:
            text = content or ""
        refusal = bool(message.get("refusal"))
        reason = choice.get("finish_reason")
        complete = reason == "stop"
    if not isinstance(text, str):
        raise ValueError("non-string text")
    if refusal:
        reason = "refusal"
    elif not text.strip() and complete:
        reason = "empty_output"
    return {
        "text": text,
        "usage": _usage(provider, response),
        "response_id": response_id if isinstance(response_id, str) else None,
        "finish_reason": reason if isinstance(reason, str) else None,
        "complete": bool(complete and text.strip() and not refusal),
    }
