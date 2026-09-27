# keepout

Spacecraft conjunction assessment in Python: CCSDS CDM ingest, probability of
collision (Pc), risk trends and collision-avoidance manoeuvre planning.

> **Status:** pre-alpha (milestone M2 — 2-D probability of collision).

Pc algorithms are implemented from the published literature and validated
against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
keepout is not affiliated with NASA; CARA files are downloaded on demand and
never redistributed.

## Validation

2-D Pc against every CARA reference case
([full report](docs/validation.md)):

<!-- validation-summary:start -->
| Method | Cases passed | Worst relative error |
|---|---|---|
| foster | 66 / 66 | 8.9e-08 |
| elrod | 66 / 66 | 8.9e-08 |
<!-- validation-summary:end -->

## Reading a CDM

```python
import keepout.cdm
from keepout.encounter import encounter_plane

cdm = keepout.cdm.load("event.cdm")  # KVN or XML, CDM 1.0 or 2.0
primary, secondary, diagnostics = cdm.object_states()  # GCRF, SI units
plane = encounter_plane(primary, secondary)
print(plane.miss_distance, plane.covariance, diagnostics)
```

## Probability of collision

```python
from keepout.conjunction import Conjunction
from keepout.pc import compute_pc

conjunction = Conjunction.from_cdm(cdm, hard_body_radius=20.0)
result = compute_pc(conjunction)  # Foster; or compute_pc(conjunction, "elrod")
print(result.value, result.diagnostics)
```

## Development

```bash
uv sync
uv run pytest                 # offline test suite
uv run pytest -m validation   # downloads CARA reference data
uv run python -m keepout.validation   # regenerate the validation report
uv run ruff check && uv run ruff format --check
uv run mypy
```

## Licence

MIT
