# Contributing to keepout

keepout is developed with [uv](https://docs.astral.sh/uv/): every command here runs through it.
Install it first ([other ways](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # or: pipx install uv
```

Then clone the repository and install every dependency, with the
development tools:

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
# A fresh stack under its own project name: `down -v` deletes that project's
# volume only, never the default project's `data` volume with your own store.
# If your own stack holds port 8501, add KEEPOUT_PORT=8601 (and that URL below).
docker compose -p keepout-e2e down -v && docker compose -p keepout-e2e up --wait
uv run pytest -m e2e --base-url http://127.0.0.1:8501   # needs uv run playwright install chromium
docker compose -p keepout-e2e down -v   # when done
# (CI's e2e job also restarts the services and checks the upload is kept)
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

The reports in `docs/` and their summaries, in `README.md` (validation) and
`docs/reports.md` (the others), are generated. Regenerate them with:

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
uv run python tools/dashboard_seed.py   # the dashboard's seed and the demo's update CDM (offline)
uv run python tools/iridium_cosmos_report.py   # Iridium-Cosmos case study (offline, minutes)
uv run python tools/iridium_cosmos_data.py verify   # its 2009 excerpt (offline)
uv run python tools/dashboard_gif.py   # README GIF; on a fresh stack, -p keepout-docs (see its --help)
```

The case study's excerpt is rebuilt only from Space-Track's 2009 bulk TLE
file, downloaded by hand (see `tools/iridium_cosmos_data.py`):
`uv run python tools/iridium_cosmos_data.py build tle2009.txt.zip`.

## Compatibility

A change to what [Stability](docs/stability.md) covers follows its
deprecation policy. Every change goes into the
[changelog](docs/changelog.md)'s **Unreleased** section.

## Releasing

1. Move the changelog's **Unreleased** entries under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading with the day's date, add its link at
   the bottom, and set `version` in `pyproject.toml` (then `uv lock`).
2. Merge to `main`, then tag and push: `git tag -a vX.Y.Z -m "keepout X.Y.Z"`
   and `git push origin vX.Y.Z`. The release workflow checks the tag
   against the version, builds the sdist and wheel and creates the GitHub
   Release with the changelog section as its notes.

## Test data

Third-party data committed for the tests is not under keepout's MIT
licence; each data directory has a `NOTICE` or `README.md` saying where
it comes from. The sample CDMs in `tests/cdm/data/` are keepout's own.
