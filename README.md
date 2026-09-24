# keepout

Spacecraft conjunction assessment in Python: CCSDS CDM ingest, probability of
collision (Pc), risk trends and collision-avoidance manoeuvre planning.

> **Status:** pre-alpha (milestone M1 — CDM + covariance).

Pc algorithms are implemented from the published literature and validated
against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
keepout is not affiliated with NASA; CARA files are downloaded on demand and
never redistributed.

## Reading a CDM

```python
import keepout.cdm
from keepout.encounter import encounter_plane

cdm = keepout.cdm.load("event.cdm")  # KVN or XML, CDM 1.0 or 2.0
primary, secondary, diagnostics = cdm.object_states()  # GCRF, SI units
plane = encounter_plane(primary, secondary)
print(plane.miss_distance, plane.covariance, diagnostics)
```

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
