# keepout

Spacecraft conjunction assessment in Python: screening against the public
catalog, CCSDS CDM ingest and generation, probability of collision (Pc),
risk trends, a synthetic operator on public data, a command line and
collision-avoidance manoeuvre planning.

> **Status:** pre-alpha; the API may change without notice.

Pc algorithms are implemented from the published literature and validated
against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
keepout is not affiliated with NASA. CARA files are downloaded on demand and
never redistributed.

![The dashboard: events list, manoeuvre planner, a CDM uploaded and the event's Pc trend rising to red](docs/img/dashboard/dashboard.gif)

The dashboard on its demo data, from a clone: `docker compose up`, then
http://127.0.0.1:8501 ([Deployment](docs/deployment.md)).

## Install

keepout needs Python 3.11 or later. It is not on PyPI yet:

```bash
pip install "keepout[cli,store] @ git+https://github.com/Kyrtap1309/keepout"
```

The extras (`cli`, `store`, `dashboard`, `validation`) are in
[Install](docs/index.md#install).

## Example

The examples read a sample CDM, a fleet file and a frozen catalog committed
to the repository. From a clone:

```bash
git clone https://github.com/Kyrtap1309/keepout && cd keepout
uv sync --extra cli --extra store
```

Assess the sample CDM from Python (`uv run python`):

```python
import keepout.cdm
from keepout import compute_pc

# Any CDM 1.0 or 2.0, KVN or XML, works.
cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")
conjunction = cdm.conjunction(combined_hard_body_radius=20.0)  # [m]
result = compute_pc(conjunction)  # picks Foster, Hall 3D-Nc or a maximum Pc
print(result.method, f"{result.value:.2e}")  # hall3d 4.50e-02
```

The same from the command line: screen three real satellites (a
[fleet file](docs/synthetic.md#the-fleet-file)) against the public catalog
frozen in the repository, write a CDM per close approach, then assess them.

```bash
uv run keepout screen --fleet examples/synthetic-operator/fleet.toml \
    --catalog tests/screening/data/celestrak-public.csv.gz \
    --start 2026-10-06 --days 7 --cdm-dir cdms
uv run keepout assess cdms/*.kvn   # exit status 3 if an event needs a manoeuvre
```

GP data has no covariance, so these CDMs carry keepout's
[synthetic covariance model](docs/assumptions.md): their Pc shows what the
model implies, not the satellites' real risk.

New to conjunction assessment? [Concepts](docs/concepts.md) explains
the workflow and the terms (Pc, TCA, HBR, CDM, ...).

## Documentation

The documentation is in [`docs/`](docs/); `uv run mkdocs serve` renders
it as a website, with the API reference generated from the docstrings.

- [Getting started](docs/index.md): install, where to start and a tour of
  every feature; [Concepts](docs/concepts.md) and the glossary.
- [Working with CDMs](docs/cdm.md): reading, converting, writing and
  grouping CDMs.
- [Screening](docs/screening.md) and the
  [synthetic operator](docs/synthetic.md), with its
  [synthetic covariance model](docs/assumptions.md).
- [2-D probability of collision](docs/pc-2d.md) and
  [3-D, Monte Carlo and maximum Pc](docs/pc-3d.md).
- [Risk assessment](docs/risk.md) and
  [manoeuvre planning](docs/maneuver.md).
- [Command line](docs/cli.md), [the conjunction store](docs/store.md),
  [the dashboard](docs/dashboard.md) and
  [Space-Track data](docs/spacetrack.md).
- [FAQ](docs/faq.md) and [changelog](docs/changelog.md).

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

## Reports

Each report is generated from the data it names and rerun by CI; their
summaries are in the [reports overview](docs/reports.md).

- [Monte Carlo validation](docs/validation-monte-carlo.md): keepout's Monte
  Carlo against CARA's, checked nightly.
- [Screening against SOCRATES](docs/screening-socrates.md): `screen`
  against CelesTrak's close approaches from the same element sets.
- [Risk on ESA's Kelvins data](docs/risk-kelvins.md): keepout's decision
  for each real event of ESA's Collision Avoidance Challenge.
- [Screening performance](docs/reports.md#screening-performance): 20
  primaries against the public and the full catalog for a week.
- [Synthetic operator](docs/synthetic-operator.md): a small real fleet
  over a week of public data.
- [Manoeuvre planning on a CARA event](docs/maneuver-report.md): the
  burns that clear a red event.

## Licence

MIT. The third-party data committed for tests is not covered:

- the CelesTrak data in `tests/screening/data/` (GP data: USSPACECOM/18 SDS
  via Space-Track.org, retrieved from CelesTrak on 2026-10-06; see its
  `NOTICE`);
- the Space-Track data in `tests/screening/spacetrack-data/` (retrieved from
  Space-Track.org on 2026-10-15; see its `NOTICE`);
- the Kelvins excerpt in `tests/sources/data/` (see its `README.md`).

See [`LICENSE`](LICENSE) for the licence text and
[Data and licences](docs/index.md#data-and-licences) for the data's
sources. Development commands are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
To cite keepout, see [Citing keepout](docs/index.md#citing-keepout).
