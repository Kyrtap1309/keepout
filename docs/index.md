# keepout

Spacecraft conjunction assessment in Python: CCSDS CDM ingest, probability of
collision (Pc), risk trends and collision-avoidance manoeuvre planning.

!!! warning "Pre-alpha"
    Milestone M4 — risk levels, trends and decisions over a conjunction
    event. APIs will change.

## Reading a CDM

```python
import keepout.cdm
from keepout import Conjunction

cdm = keepout.cdm.load("event.cdm")  # KVN or XML, CDM 1.0 or 2.0
primary, secondary = cdm.object_states()  # GCRF, SI units
plane = Conjunction(primary, secondary, hard_body_radius=20.0).plane
print(plane.miss_distance, plane.covariance, primary.covariance.repairs)
```

## Probability of collision

```python
import keepout.cdm
from keepout import Conjunction, compute_pc

conjunction = Conjunction.from_cdm(keepout.cdm.load("event.cdm"), hard_body_radius=20.0)
result = compute_pc(conjunction)  # picks Foster, Hall 3D-Nc or a maximum Pc
print(result.method, result.value, result.diagnostics)
print(compute_pc(conjunction, "hall3d").details["nc"])  # a method by name
mc = compute_pc(conjunction, "montecarlo", samples=100_000, seed=1)
print(mc.value, mc.details["ci_low"], mc.details["ci_high"])
```

Without `hard_body_radius`, the sum of both objects' `HBR` (CDM 2.0) is used;
keepout never falls back to a default radius.

## Risk of an event

```python
from pathlib import Path

import keepout.cdm
from keepout import ConjunctionEvent, assess

cdms = [keepout.cdm.load(path) for path in Path("event").glob("*.cdm")]
assessment = assess(ConjunctionEvent.from_cdms(cdms).observations())
print(assessment.decision.action.name, assessment.decision.reasons)
```

See [Risk assessment](risk.md) for the rules and the
[Kelvins report](risk-kelvins.md) for keepout's decisions on 15 321 real
events of ESA missions.

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
uv run pre-commit install     # ruff, mypy and the lock checked on commit
uv run pytest                 # offline test suite
uv run pytest -m validation   # downloads CARA and Kelvins data
uv run python tools/validation_report.py   # regenerate the validation report
uv run python tools/validation_report.py --monte-carlo   # Monte Carlo (slow)
uv run python tools/risk_report.py   # Kelvins risk report (downloads 221 MB once)
uv run mkdocs serve           # this documentation
```
