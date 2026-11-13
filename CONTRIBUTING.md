# Contributing to keepout

keepout is developed with [uv](https://docs.astral.sh/uv/). Clone the
repository and install every dependency, with the development tools:

```bash
git clone https://github.com/Kyrtap1309/keepout
cd keepout
uv sync
uv run pre-commit install     # every CI lint check, run on commit
```

## Tests

```bash
uv run pytest -n auto         # offline test suite, without the slow tests
uv run pytest -n auto -m slow # the slow tests CI also runs (tens of seconds each)
uv run pytest -m validation   # reference checks; downloads CARA and Kelvins data
uv run pytest -n0 --inline-snapshot=fix   # update the CLI snapshots (not under -n)
```

## Checks

```bash
uv run ruff check && uv run ruff format --check
uv run mypy && uv run basedpyright
uv run lint-imports && uv run codespell && uv run deptry src
uv run mkdocs build --strict  # the documentation, with its links checked
uv run mkdocs serve           # the documentation, served locally
```

## Reports

The reports in `docs/` and their summaries in `README.md` are generated.
Regenerate them with:

```bash
uv run python tools/validation_report.py   # validation report
uv run python tools/validation_report.py --monte-carlo   # Monte Carlo (slow)
uv run python tools/risk_report.py   # Kelvins risk report (downloads 221 MB once)
uv run python tools/screening_report.py   # SOCRATES screening report (offline)
uv run python tools/screening_benchmark.py   # screening benchmark (offline, minutes)
uv run python tools/screening_benchmark.py --source spacetrack   # on the full catalog
uv run python tools/synthetic_operator_report.py   # synthetic operator report (offline)
uv run python tools/maneuver_report.py   # manoeuvre report (downloads one CARA CDM)
uv run python tools/spacetrack_snapshot.py verify   # frozen Space-Track catalog (offline)
```

## Test data

Third-party data committed for the tests is not under keepout's MIT
licence; each data directory has a `NOTICE` or `README.md` saying where
it comes from. The sample CDMs in `tests/cdm/data/` are keepout's own.
