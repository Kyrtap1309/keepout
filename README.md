# keepout

Spacecraft conjunction assessment in Python: CCSDS CDM ingest, probability of
collision (Pc), risk trends and collision-avoidance manoeuvre planning.

> **Status:** pre-alpha (milestone M0 — project skeleton).

Pc algorithms are implemented from the published literature and validated
against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
keepout is not affiliated with NASA; CARA files are downloaded on demand and
never redistributed.

## Development

```bash
uv sync
uv run pytest                 # offline test suite
uv run pytest -m validation   # downloads CARA reference data
uv run ruff check && uv run ruff format --check
uv run mypy
```

## Licence

MIT
