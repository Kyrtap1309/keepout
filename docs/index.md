# keepout

Spacecraft conjunction assessment in Python: CCSDS CDM ingest, probability of
collision (Pc), risk trends and collision-avoidance manoeuvre planning.

!!! warning "Pre-alpha"
    Milestone M2 — 2-D probability of collision. APIs will change.

## Reading a CDM

```python
import keepout.cdm
from keepout.encounter import encounter_plane

cdm = keepout.cdm.load("event.cdm")  # KVN or XML, CDM 1.0 or 2.0
primary, secondary = cdm.object_states()  # GCRF, SI units
plane = encounter_plane(primary, secondary)
print(plane.miss_distance, plane.covariance, primary.covariance.repairs)
```

## Probability of collision

```python
import keepout.cdm
from keepout.conjunction import Conjunction
from keepout.pc import compute_pc

conjunction = Conjunction.from_cdm(keepout.cdm.load("event.cdm"), hard_body_radius=20.0)
result = compute_pc(conjunction)  # Foster; or compute_pc(conjunction, "elrod")
print(result.value, result.diagnostics)  # repairs, 2-D warnings, method notes
print(compute_pc(conjunction, "elrod", order=64).value)  # method options
```

Without `hard_body_radius`, the sum of both objects' `HBR` (CDM 2.0) is used;
keepout never falls back to a default radius.

## Validation philosophy

Every Pc algorithm in keepout is implemented from the published literature
and validated against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
CARA files are downloaded on demand from a pinned commit and verified by
SHA-256; they are never redistributed with keepout. See the
[validation report](validation.md).

## Development

```bash
uv sync
uv run pytest                 # offline test suite
uv run pytest -m validation   # downloads CARA reference data
uv run python tools/validation_report.py   # regenerate the validation report
uv run mkdocs serve           # this documentation
```
