"""``keepout validate``: run the NASA CARA validation cases."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Annotated, TypeAlias

import typer
from rich.table import Table

from keepout.cli.options import CacheDir, JsonFlag, Workers
from keepout.cli.output import JsonValue, fail, show
from keepout.sources.cara import CARA_COMMIT
from keepout.validation.cases import load_all, load_monte_carlo_cases
from keepout.validation.evaluation import (
    MethodSummary,
    Outcome,
    evaluate,
    summarise,
)

__all__ = ["validate_command"]

MonteCarloFlag: TypeAlias = Annotated[
    bool,
    typer.Option(
        "--monte-carlo",
        help="Run the Monte Carlo cases instead (minutes to hours).",
    ),
]
"""``--monte-carlo``: the Monte Carlo cases instead of the others."""


def validate_command(
    *,
    monte_carlo: MonteCarloFlag = False,
    workers: Workers = None,
    cache_dir: CacheDir = None,
    as_json: JsonFlag = False,
) -> None:
    """Compare keepout's Pc with NASA CARA's on every validation case.

    Downloads CARA's files at the pinned commit on first use, runs every
    case (2-D, 3-D and maximum Pc; with --monte-carlo the Monte Carlo
    cases) and prints, per suite and method, the cases passed and the worst
    relative error. Exits with status 1 if a case fails. The repository's
    tools/validation_report.py publishes the same comparison as
    docs/validation.md.
    """
    cases = (load_monte_carlo_cases if monte_carlo else load_all)(cache_dir)
    outcomes = evaluate(cases, workers=workers or os.cpu_count() or 1)
    summary = summarise(outcomes)
    failed = [o for o in outcomes if not o.passed]
    payload: dict[str, JsonValue] = {
        "cara_commit": CARA_COMMIT,
        "monte_carlo": monte_carlo,
        "comparisons": len(outcomes),
        "failed": len(failed),
        "summary": [_summary_json(s) for s in summary],
        "outcomes": [_outcome_json(o) for o in outcomes],
    }
    lines = [
        f"FAIL {_name(o)}: {o.result.value:.6e} vs CARA {o.case.reference:.6e}"
        for o in failed
    ]
    show(
        payload,
        _table(summary, monte_carlo=monte_carlo),
        *lines,
        f"{len(outcomes) - len(failed)} of {len(outcomes)} comparisons passed "
        f"(CARA commit {CARA_COMMIT[:12]}).",
        as_json=as_json,
    )
    if failed:
        fail(f"{len(failed)} of {len(outcomes)} comparisons failed")


def _name(outcome: Outcome) -> str:
    return f"{outcome.case.suite} / {outcome.case.case_id} / {outcome.result.method}"


def _table(summary: Sequence[MethodSummary], *, monte_carlo: bool) -> Table:
    title = "Monte Carlo validation" if monte_carlo else "Validation"
    table = Table(title=f"{title} against NASA CARA")
    for column in ("Suite", "Method", "Cases passed"):
        table.add_column(column)
    table.add_column("Worst relative error", justify="right")
    for s in summary:
        worst = s.worst_relative_error
        table.add_row(
            s.suite,
            s.method,
            f"{s.passed} / {s.total}",
            "-" if worst is None else f"{worst:.1e}",
        )
    return table


def _summary_json(summary: MethodSummary) -> dict[str, JsonValue]:
    return {
        "suite": summary.suite,
        "method": summary.method,
        "passed": summary.passed,
        "total": summary.total,
        "worst_relative_error": summary.worst_relative_error,
    }


def _outcome_json(outcome: Outcome) -> dict[str, JsonValue]:
    case, verdict = outcome.case, outcome.verdict
    return {
        "suite": case.suite,
        "case_id": case.case_id,
        "method": outcome.result.method,
        "reference": case.reference,
        "value": outcome.result.value,
        "error": verdict.error,
        "relative": verdict.relative,
        "passed": verdict.passed,
        "ci": None if outcome.ci is None else list(outcome.ci),
    }
