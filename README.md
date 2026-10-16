# keepout

Spacecraft conjunction assessment in Python: screening against the public
catalog, CCSDS CDM ingest, probability of collision (Pc), risk trends and
collision-avoidance manoeuvre planning.

> **Status:** pre-alpha (milestone M5 — screening).

Pc algorithms are implemented from the published literature and validated
against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
keepout is not affiliated with NASA; CARA files are downloaded on demand and
never redistributed.

## Validation

Every Pc method against every applicable CARA reference case
([full report](docs/validation.md)):

<!-- validation-summary:start -->
| Suite | Method | Cases passed | Worst relative error |
|---|---|---|---|
| Alfano (2009) | foster | 11 / 11 | 8.9e-08 |
| Alfano (2009) | elrod | 11 / 11 | 8.9e-08 |
| Alfano (2009), 3-D | hall3d | 12 / 12 | 5.5e-04 |
| CARA Pc method test set | foster | 53 / 53 | 2.7e-08 |
| CARA Pc method test set | elrod | 53 / 53 | 2.7e-08 |
| CARA Pc method test set, 3-D | hall3d | 53 / 53 | 2.1e-03 |
| CARA unit tests | foster | 3 / 3 | 3.7e-08 |
| CARA unit tests | elrod | 3 / 3 | 3.7e-08 |
| CARA unit tests | hall3d | 2 / 2 | 7.1e-04 |
| CARA unit tests, maximum Pc | frisbee | 2 / 2 | - |
| CARA unit tests, maximum Pc | dilution | 4 / 4 | 1.0e-03 |
<!-- validation-summary:end -->

Monte Carlo against CARA's two-body Monte Carlo
([full report](docs/validation-monte-carlo.md), checked nightly):

<!-- validation-monte-carlo-summary:start -->
| Suite | Method | Cases passed | Worst relative error |
|---|---|---|---|
| Alfano (2009), Monte Carlo | montecarlo | 12 / 12 | 7.3e-02 |
| CARA Pc method test set, Monte Carlo | montecarlo | 21 / 21 | 1.6e-01 |
<!-- validation-monte-carlo-summary:end -->

<!-- screening-socrates-summary:start -->
Close approaches from `screen` against CelesTrak's SOCRATES, on the 1275 SOCRATES approaches computed from the same element sets ([report](docs/screening-socrates.md)):

| Approaches | Compared | Within tolerance | Worst TCA | Worst range |
|---|---|---|---|---|
| fast (>= 1 km/s) | 266 | 261 | 0.493 ms | 0.49565 m |
| slow (< 1 km/s) | 1008 | 1008 | 324.653 ms | 0.49998 m |

5 approaches outside tolerance involve 2 objects whose element sets evidently differ from SOCRATES's. keepout also reports 5 minima of these pairs that SOCRATES does not list.
<!-- screening-socrates-summary:end -->

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
from keepout import compute_pc

conjunction = cdm.conjunction(hard_body_radius=20.0)
result = compute_pc(conjunction)  # picks Foster, Hall 3D-Nc or a maximum Pc
print(result.method, result.value, result.diagnostics)
print(compute_pc(conjunction, "hall3d").details["nc"])  # a method by name
mc = compute_pc(conjunction, "montecarlo", samples=100_000, seed=1)
print(mc.value, mc.details["ci_low"], mc.details["ci_high"])
```

## Risk of an event

```python
from pathlib import Path

import keepout.cdm
from keepout import ConjunctionEvent, assess

cdms = [keepout.cdm.load(path) for path in Path("event").glob("*.cdm")]
event = ConjunctionEvent.from_cdms(cdms)  # one encounter, oldest CDM first
assessment = assess(event.observations(hard_body_radius=20.0))
print(assessment.decision.action.name, assessment.decision.reasons)
print(assessment.trend.direction, assessment.level)
```

<!-- risk-kelvins-summary:start -->
keepout's decision for each of the 15321 events of ESA's Kelvins data, from ESA's Pc of every CDM ([report](docs/risk-kelvins.md)):

| Decision | Events | Share |
|---|---|---|
| ignore | 13823 | 90.2% |
| monitor | 1468 | 9.6% |
| monitor, prepare manoeuvre | 6 | <0.1% |
| manoeuvre | 24 | 0.2% |

![Kelvins event 734: manoeuvre](docs/img/risk-kelvins/event-734.svg)
<!-- risk-kelvins-summary:end -->

Kelvins data: ESA, *Collision Avoidance Challenge dataset*, Zenodo, 2021,
[doi:10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683); © ESA,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

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

Each approach holds both objects' GCRF states at TCA, the miss distance, the
miss vector in the primary's RTN frame and the relative velocity. Screening
GP element sets finds geometry, not risk: they have no covariance and are
kilometres off within days. [Screening](docs/screening.md) explains the
method, the screening volumes (5 km as SOCRATES by default, NASA's 10 km for
GP data, the USSPACECOM Spaceflight Safety Handbook's by orbit regime), the
limits of GP data and the gap between the public and the full catalog; the
comparison with SOCRATES is under [Validation](#validation).

GP data: USSPACECOM/18 SDS via Space-Track.org, retrieved from CelesTrak
(celestrak.org); the frozen copy in `tests/screening/data/` was retrieved
on 2026-10-06 and is not covered by keepout's MIT licence (see its `NOTICE`).

### Performance

The target: 20 primaries against the public catalog for a week in under 60 s
with 4 worker processes. CI also times each change against its merge base and
fails on a slowdown of more than 20 %.

<!-- screening-benchmark-summary:start -->
Screening 20 LEO primaries against the 19914 element sets of the frozen public catalog (CelesTrak, retrieved 2026-10-06) for 7 days from 2026-10-06 00:00 UTC, with 4 worker processes (`tools/screening_benchmark.py`):

| Volume | Screened | Left out | Approaches | Wall time | CPU time |
|---|---|---|---|---|---|
| 5 km sphere (default) | 19886 | 28 stale | 715 | 8.5 s | 40.3 s |
| 10 km sphere (NASA, GP data) | 19886 | 28 stale | 2551 | 9.5 s | 44.3 s |
| SFS Handbook Table 3 | 19886 | 28 stale | 345 | 14.8 s | 65.2 s |

Times: median of 3 runs on 2026-10-07, Intel(R) Core(TM) i5-14600K (20 logical CPUs), Python 3.13.7; `--check` checks the counts, not the times.
<!-- screening-benchmark-summary:end -->

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
uv run python tools/synthetic_operator_report.py   # synthetic operator report (offline)
uv run python tools/spacetrack_snapshot.py verify   # frozen Space-Track catalog (offline)
uv run ruff check && uv run ruff format --check
uv run mypy
```

## Licence

MIT. The third-party data committed for tests is not covered: the CelesTrak
data in `tests/screening/data/`, the Space-Track data in
`tests/screening/spacetrack-data/` (see their `NOTICE` files) and the Kelvins
excerpt in `tests/sources/data/` (see its `README.md`).
