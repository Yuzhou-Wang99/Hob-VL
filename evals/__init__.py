"""Offline-first, API-backed evaluation of the Hob-VL dataset."""

__version__ = "1.0.0"


def evaluate(**kwargs):
    """Run an evaluation; dry-run by default. See :func:`evals.run.evaluate`."""
    from .run import evaluate as run_evaluation

    return run_evaluation(**kwargs)
