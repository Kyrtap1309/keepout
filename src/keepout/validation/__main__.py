"""``python -m keepout.validation [--check]``: regenerate the CARA report."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from keepout.validation.cases import load_all
from keepout.validation.report import (
    evaluate,
    render_page,
    render_summary,
    replace_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Write (or with ``--check`` verify) the validation page and README summary.

    Returns
    -------
    int
        0 on success; 1 if a case fails or, with ``--check``, a file is stale.
    """
    parser = argparse.ArgumentParser(prog="python -m keepout.validation")
    parser.add_argument("--check", action="store_true", help="fail if out of date")
    parser.add_argument("--page", type=Path, default=Path("docs/validation.md"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    args = parser.parse_args(argv)

    outcomes = evaluate(load_all())
    wanted = {
        args.page: render_page(outcomes),
        args.readme: replace_summary(
            args.readme.read_text(encoding="utf-8"), render_summary(outcomes)
        ),
    }
    status = 0
    for path, text in wanted.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if args.check:
            print(f"{path} is out of date; run python -m keepout.validation")
            status = 1
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    failed = [o for o in outcomes if not o.passed]
    for outcome in failed:
        print(f"FAIL {outcome.case.suite} {outcome.case.case_id} {outcome.method}")
    return 1 if failed else status


if __name__ == "__main__":
    sys.exit(main())
