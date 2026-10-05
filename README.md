# keepout

Spacecraft conjunction assessment in Python: CCSDS CDM ingest, probability of
collision (Pc), risk trends and collision-avoidance manoeuvre planning.

> **Status:** pre-alpha (milestone M3 — 3-D, Monte Carlo and maximum
> probability of collision).

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
from keepout import Conjunction, compute_pc

conjunction = Conjunction.from_cdm(cdm, hard_body_radius=20.0)
result = compute_pc(conjunction)  # picks Foster, Hall 3D-Nc or a maximum Pc
print(result.method, result.value, result.diagnostics)
print(compute_pc(conjunction, "hall3d").details["nc"])  # a method by name
mc = compute_pc(conjunction, "montecarlo", samples=100_000, seed=1)
print(mc.value, mc.details["ci_low"], mc.details["ci_high"])
```

## Development

```bash
uv sync
uv run pytest                 # offline test suite
uv run pytest -m validation   # downloads CARA reference data
uv run python tools/validation_report.py   # regenerate the validation report
uv run python tools/validation_report.py --monte-carlo   # Monte Carlo (slow)
uv run ruff check && uv run ruff format --check
uv run mypy
```

## Licence

MIT
