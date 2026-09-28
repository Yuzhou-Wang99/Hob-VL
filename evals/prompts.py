"""Versioned request prompts, prepared after validating the unchanged dataset."""

from dataclasses import replace

from .data import Dataset


BOOLEAN_OUTPUT_INSTRUCTION = (
    "Respond with exactly one word: Yes or No. "
    "Do not include explanations, Markdown, punctuation, or any other text."
)
IDENTIFICATION_OUTPUT_INSTRUCTION = (
    "Respond with exactly one candidate label. "
    "Do not include explanations, Markdown, punctuation, or any other text."
)
DEFAULT_PROMPT_POLICY = "boolean-one-word-v1"
ANSWER_ONLY_PROMPT_POLICY = "answer-only-v1"
PROMPT_POLICIES = (DEFAULT_PROMPT_POLICY, ANSWER_ONLY_PROMPT_POLICY, "original")


def prompt_preparation(task: str, policy: str) -> dict | None:
    if policy not in PROMPT_POLICIES:
        raise ValueError(f"Unsupported prompt policy: {policy}")
    if policy == "original":
        return None
    if task == "yes-no":
        # Both policies send identical Boolean text; retain its existing identity.
        return {"policy": DEFAULT_PROMPT_POLICY, "suffix": "\n\n" + BOOLEAN_OUTPUT_INSTRUCTION}
    if task == "identification" and policy == ANSWER_ONLY_PROMPT_POLICY:
        return {"policy": policy, "suffix": "\n\n" + IDENTIFICATION_OUTPUT_INSTRUCTION}
    return None


def prepare_prompts(dataset: Dataset, policy: str = DEFAULT_PROMPT_POLICY) -> Dataset:
    """Derive request text from freshly loaded inputs; never edit source files."""
    preparation = prompt_preparation(dataset.task, policy)
    if preparation is None:
        return dataset
    return replace(dataset, examples=[
        replace(example, prompt=example.prompt + preparation["suffix"])
        for example in dataset.examples
    ])


def recorded_prompt_policy(identity: dict) -> str:
    """Legacy manifests have no preparation field and used the original text."""
    if "prompt_preparation" not in identity:
        return "original"
    preparation = identity["prompt_preparation"]
    if not isinstance(preparation, dict) or preparation.get("policy") not in PROMPT_POLICIES:
        raise ValueError("Unsupported or modified prompt_preparation in manifest")
    expected = prompt_preparation(identity["data"]["task"], preparation["policy"])
    if expected is None or preparation != expected:
        raise ValueError("Unsupported or modified prompt_preparation in manifest")
    return expected["policy"]
