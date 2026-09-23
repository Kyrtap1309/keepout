# keepout

Spacecraft conjunction assessment in Python: CCSDS CDM ingest, probability of
collision (Pc), risk trends and collision-avoidance manoeuvre planning.

!!! warning "Pre-alpha"
    Milestone M0 — project skeleton. APIs will change.

## Validation philosophy

Every Pc algorithm in keepout is implemented from the published literature
and validated against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
CARA files are downloaded on demand from a pinned commit and verified by
SHA-256; they are never redistributed with keepout.

## Development

```bash
uv sync
uv run pytest                 # offline test suite
uv run pytest -m validation   # downloads CARA reference data
uv run mkdocs serve           # this documentation
```
