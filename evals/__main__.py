"""Run with python -m evals; no network access unless --execute is supplied."""

import argparse
import json
import sys

from .run import evaluate
from .providers import PROVIDERS, ProviderError
from .prompts import DEFAULT_PROMPT_POLICY, PROMPT_POLICIES
from .scoring import ANSWER_FORMATS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="provider/model ID; providers: " + ", ".join(PROVIDERS))
    parser.add_argument("--data", default="symbolic", help="symbolic, natural-language, identification, identification-symbolic, or a model-input JSONL path relative to repo root")
    parser.add_argument("--answers", help="Answer JSONL/JSON path (default: the format's released answer file)")
    parser.add_argument("--repo-root", help="Repository root (default: inferred from evals location, not cwd)")
    parser.add_argument("--output", help="New output directory, relative to cwd; default: evals/runs/<unique-run>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Send paid model requests; without this flag the run is entirely offline")
    mode.add_argument("--dry-run", action="store_true", help="Explicit offline preparation (also the default)")
    parser.add_argument("--limit", type=int, help="Positive maximum row count")
    parser.add_argument("--offset", type=int, default=0, help="Skip rows after track filtering and optional shuffling")
    parser.add_argument("--track", choices=("synthetic", "photo"))
    parser.add_argument("--seed", type=int, help="Deterministically shuffle before offset/limit; omission preserves file order")
    parser.add_argument("--resume", action="store_true", help="Resume at --output; settled IDs are skipped, definite rejections retried")
    parser.add_argument("--retry-pending", action="store_true", help="With --resume, retry saved errors, including uncertain requests that may already have been billed; otherwise --continue-on-error defers saved errors")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Move to the next ID after transient API errors. On resume, defer saved errors unless --retry-pending is supplied. Exit 3 means all IDs attempted with errors pending")
    parser.add_argument("--max-consecutive-errors", type=int, default=3,
                        help="With --continue-on-error, stop after this many consecutive request failures (default: 3). Authentication/configuration failures always stop")
    parser.add_argument("--request-interval", type=float, default=0,
                        help="Minimum seconds between question request starts; local pacing, may change on resume (default: 0)")
    parser.add_argument("--rate-limit-cooldown", type=float, default=60,
                        help="With --continue-on-error, wait at least this many seconds after HTTP 429, or longer if Retry-After asks; local policy, may change on resume (default: 60)")
    parser.add_argument("--api-key-env", help="Name of environment variable containing API key; never put the key itself here")
    parser.add_argument("--base-url", help="Provider API root; required for openai-compatible and qwen (copy its region/workspace endpoint from Model Studio)")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=None, help="Omitted by default for model compatibility")
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=0, help="Explicit transient HTTP retries; default 0")
    parser.add_argument("--max-image-edge", type=int, default=2048, help="Longest image edge after preparation; default 2048, also capped at 4 MiB")
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh"),
                        help="Reasoning effort for openai/openai-compatible; anthropic, zai (glm-4.5v, glm-4.6v-flash, glm-4.6v-flashx), and deepseek (deepseek-flash) accept only none and disable thinking; qwen and deepinfra accept none for Qwen3-VL Instruct and Llama 4 Scout Instruct respectively, by model selection. Omitted by default")
    parser.add_argument("--image-detail", choices=("auto", "low", "high", "original"),
                        help="Image detail for openai/openai-compatible/deepseek models; defaults to high ('original' is openai/deepseek only)")
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking-mode", choices=("enabled", "disabled"),
                          help="Native on/off thinking switch for Z.ai GLM-4.5V/4.6V/4.6V-Flash/4.6V-FlashX; cannot combine with --reasoning-effort. No low/medium/high levels")
    thinking.add_argument("--thinking-budget", type=int, default=None,
                          help="Gemini 2.5 native thinking budget: 0 disables thinking on Flash/Flash-Lite; -1 is dynamic; positive values enable thinking. Omitted by default; unsupported for Gemini 3")
    thinking.add_argument("--thinking-level", choices=("minimal", "low", "medium", "high"),
                          help="Native thinking level for Gemini 3 or hosted Gemma 4. Gemma 4: minimal disables thinking, high enables it. Gemini 3: minimal is not guaranteed thinking-off. Omitted by default")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="plain",
                        help="Scoring only: plain (original rule) or plain-or-bold (also accepts **Yes**, __No__, **AF**). Must match on resume")
    parser.add_argument("--prompt-policy", choices=PROMPT_POLICIES, default=DEFAULT_PROMPT_POLICY,
                        help="Default: append the one-word instruction to Boolean prompts only. answer-only-v1 also appends a label-only instruction to identification prompts. Use original for unmodified prompts. Must match on resume")
    args = parser.parse_args(argv)
    try:
        report = evaluate(**vars(args))
    except (ValueError, OSError, ImportError, ProviderError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Saved predictions can be resumed with identical settings.", file=sys.stderr)
        return 130
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if (args.continue_on_error and report["mode"] == "execute" and report.get("all_attempted")
            and not report.get("stopped_on_error") and not report["complete"]):
        print(f"All IDs attempted; {report['pending_rows']} API errors remain pending. "
              "Exit code 3. Retry later with --resume --retry-pending.", file=sys.stderr)
        return 3
    return 1 if report.get("stopped_on_error") or (report["mode"] == "execute" and not report["complete"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
