# keepout

Spacecraft conjunction assessment in Python: screening against the public
catalog, CCSDS CDM ingest and generation, probability of collision (Pc),
risk trends, a synthetic operator on public data, a command line and
collision-avoidance manoeuvre planning.

!!! warning "Pre-alpha"
    Milestone M6 — synthetic operator and command line. APIs will change.

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
from keepout import compute_pc

conjunction = keepout.cdm.load("event.cdm").conjunction(hard_body_radius=20.0)
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

## Screening

```python
from collections import Counter

from astropy.time import Time

from keepout import screen
from keepout.sources.celestrak import fetch_catalog

catalog = fetch_catalog()  # CelesTrak's public catalog, cached for two hours
iss = [omm for omm in catalog.omms if omm.catalog_number == 25544]
result = screen(iss, catalog.omms, Time.now())  # the next 7 days, a 5 km sphere
for approach in result.approaches:
    print(
        approach.tca.isot, approach.secondary_number, f"{approach.miss_distance:.0f} m"
    )
print(Counter(skipped.reason.value for skipped in result.skipped))  # sets left out
```

See [Screening](screening.md) for the method, the screening volumes and the
limits of GP data, and [Screening against SOCRATES](screening-socrates.md)
for keepout's results next to CelesTrak's.

## Synthetic operator and command line

```bash
pip install "keepout[cli]"
keepout screen --fleet examples/synthetic-operator/fleet.toml \
    --catalog tests/screening/data/celestrak-public.csv.gz \
    --start 2026-10-06 --days 7 --cdm-dir cdms
keepout assess cdms/*.kvn   # exit status 3 if an event needs a manoeuvre
```

In a clone of the repository, this screens three real satellites against
the public catalog frozen on 2026-10-06, issues a CCSDS CDM for each of the
12 close approaches of the week and assesses each. The CDMs' covariances are
keepout's model of GP element-set errors, an assumption, not orbit
determination. See [Synthetic operator](synthetic.md) for what a run does,
[Assumptions](assumptions.md) for the model and its limits,
[Command line](cli.md) for every command, [Space-Track data](spacetrack.md)
for the full catalog, and the [report](synthetic-operator.md) for the
week's approaches.

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
uv run pytest -m validation   # slow checks; downloads CARA and Kelvins data
uv run python tools/validation_report.py   # regenerate the validation report
uv run python tools/validation_report.py --monte-carlo   # Monte Carlo (slow)
uv run python tools/risk_report.py   # Kelvins risk report (downloads 221 MB once)
uv run python tools/screening_report.py   # SOCRATES screening report (offline)
uv run python tools/screening_benchmark.py   # screening benchmark (offline, minutes)
uv run python tools/screening_benchmark.py --source spacetrack   # on the full catalog
uv run python tools/synthetic_operator_report.py   # synthetic operator report (offline)
uv run python tools/spacetrack_snapshot.py verify   # frozen Space-Track catalog (offline)
uv run mkdocs serve           # this documentation
```
