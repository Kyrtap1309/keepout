# keepout

Spacecraft conjunction assessment in Python: screening against the public
catalog, CCSDS CDM ingest and generation, probability of collision (Pc),
risk trends, a synthetic operator on public data, a command line and
collision-avoidance manoeuvre planning.

!!! note "Stable"
    Since 1.0.0 the Python API, the command line, its JSON output and the
    store's schema follow semantic versioning ([Stability](stability.md)).

## Quickstart

1. Install [uv](https://docs.astral.sh/uv/) if you do not have it
    (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `pipx install uv`;
    [other ways](https://docs.astral.sh/uv/getting-started/installation/)).

2. Clone the repository and install keepout with its command line, since
    the examples read files committed to it:

    ```bash
    git clone https://github.com/Kyrtap1309/keepout && cd keepout
    uv sync --extra cli --extra store
    ```

3. Assess a CDM: its Pc, the risk of its event and the action to take.
    Here a sample CDM with a combined hard-body radius of 20 m:

    ```bash
    uv run keepout assess tests/cdm/data/tracss_v2.kvn --hbr 20
    ```

    It ends with exit status 3: this sample event needs a manoeuvre.

4. Screen a fleet against the public catalog and assess each close
    approach: see [Synthetic operator and command
    line](#synthetic-operator-and-command-line). From Python, start with
    [Probability of collision](#probability-of-collision).

## Install

keepout needs Python 3.11 or later. It is not on PyPI (another project
holds the name), so install it from GitHub:

```bash
pip install "keepout[cli,store] @ git+https://github.com/Kyrtap1309/keepout"
```

The extras add optional parts; without them you get the library only:

| Extra | Adds | Needed for |
|---|---|---|
| `cli` | Typer and Rich | the `keepout` command |
| `store` | SQLAlchemy and Alembic | the [conjunction store](store.md): `--store`, `keepout store`, `keepout jobs`, `keepout worker` |
| `dashboard` | Streamlit and Plotly, and the `store` extra | the [dashboard](dashboard.md): `keepout dashboard` |
| `validation` | openpyxl and xlrd | `keepout validate` and the [validation report](validation.md): reading CARA's spreadsheets |

Give several extras as `keepout[cli,store]`. The examples on these pages
read files committed to the repository: a sample CDM, the example fleet
and the frozen catalogs. To run them, work inside a clone as in the
[Quickstart](#quickstart), with the extras you need (`uv sync
--all-extras` for every one).

A clone needs [uv](https://docs.astral.sh/uv/): `uv sync` installs keepout and the locked versions of
its dependencies into the clone's `.venv`, and `uv run` runs a command
there, so every `uv run ...` on these pages works from the clone's root.
`pip install` above needs no uv; the dashboard's `docker compose up`
([Deployment](deployment.md)) needs Docker only.

## Where to start

- **New to conjunction assessment?** [Concepts](concepts.md) explains the
  workflow and every term (Pc, TCA, HBR, CDM, ...) in one page.
- **Have CDMs?** [Working with CDMs](cdm.md) →
  [2-D probability of collision](pc-2d.md) → [Risk assessment](risk.md).
- **Only public data?** [Screening](screening.md) →
  [Synthetic operator](synthetic.md).
- **Prefer a terminal?** [Command line](cli.md) walks through a week of
  a synthetic operator.
- **Prefer a browser?** [The dashboard](dashboard.md) shows a store's
  events, their trends and manoeuvre plans.
- **Stuck?** [FAQ and troubleshooting](faq.md).

## Reading a CDM

`tests/cdm/data/tracss_v2.kvn` is a sample CDM 2.0 in the repository;
use your own file in its place.

```python
import keepout.cdm
from keepout import Conjunction

cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")  # KVN or XML, CDM 1.0 or 2.0
primary, secondary = cdm.object_states()  # GCRF, SI units
plane = Conjunction(primary, secondary, combined_hard_body_radius=20.0).plane
print(f"{plane.miss_distance:.1f} m")  # on the encounter plane
print(plane.covariance.round(1).tolist())  # combined 2x2 covariance [m²]
print(primary.covariance.repairs)  # what was fixed in the CDM's covariance
```

```text
100.0 m
[[19462.0, 0.8], [0.8, 500.0]]
()
```

[Frames, units and covariances](frames-and-units.md) explains the frames keepout accepts,
the units and how a CDM's covariance is checked and repaired.

[Working with CDMs](cdm.md) covers the CDM versions keepout reads,
converting and writing CDMs, and grouping many CDMs into events.

## Probability of collision

```python
import keepout.cdm
from keepout import compute_pc

cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")
conjunction = cdm.conjunction(combined_hard_body_radius=20.0)  # [m]
result = compute_pc(conjunction)  # picks Foster, Hall 3D-Nc or a maximum Pc
print(result.method, f"{result.value:.4e}")
print(result.diagnostics)  # why the result may need a closer look
print(f"{compute_pc(conjunction, 'foster').value:.4e}")  # a method by name
mc = compute_pc(conjunction, "montecarlo", samples=100_000, seed=1)
low, high = mc.interval  # its 99% confidence interval
print(f"{mc.value:.4e} ({low:.4e} to {high:.4e})")
```

Here `auto` chose Hall's 3-D method, because the CDM's states are not
exactly at the closest approach:

```text
hall3d 4.5044e-02
('states are 300 m from closest approach along the relative velocity; 2-D Pc uses the straight-line closest approach',)
4.5028e-02
4.4950e-02 (4.3279e-02 to 4.6664e-02)
```

Without `combined_hard_body_radius`, keepout uses the sum of both objects'
`HBR` (CDM 2.0). It never falls back to a default radius.
[2-D probability of collision](pc-2d.md) and
[3-D, Monte Carlo and maximum Pc](pc-3d.md) describe the methods.

## Risk of an event

An event is all the CDMs of one encounter. Here it has a single CDM;
with your own, pass every CDM of the event. [Risk assessment](risk.md)
starts with a sample event of three CDMs.

```python
import keepout.cdm
from keepout import ConjunctionEvent, assess

cdms = [keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")]
event = ConjunctionEvent.from_cdms(cdms)  # one encounter, in any order
assessment = assess(event.observations(combined_hard_body_radius=20.0))
print(assessment.decision.action.name)
for reason in assessment.decision.reasons:
    print("-", reason)
```

```text
MANEUVER
- latest Pc 4.50e-02 is red, 0.84 d before TCA
- trend undetermined
- within the decision point (2.00 d before TCA)
- diagnostic: states are 300 m from closest approach along the relative velocity; 2-D Pc uses the straight-line closest approach
```

See [Risk assessment](risk.md) for the rules. The
[Kelvins report](risk-kelvins.md) shows keepout's decisions on 15 321 real
events of ESA missions.

## Screening

```python
from collections import Counter

from astropy.time import Time

import keepout.omm
from keepout import screen

# CelesTrak's public catalog as it was on 2026-10-06, frozen in the repository.
# For the live one: catalog = keepout.sources.celestrak.fetch_catalog().omms
catalog = keepout.omm.load("tests/screening/data/celestrak-public.csv.gz")
iss = [omm for omm in catalog if omm.catalog_number == 25544]
start = Time("2026-10-06T00:00:00", scale="utc")  # or Time.now()
result = screen(iss, catalog, start)  # the next 7 days, a 5 km sphere
for approach in result.approaches:
    print(
        approach.tca.isot, approach.secondary_number, f"{approach.range_at_tca:.0f} m"
    )
print(Counter(skipped.reason.value for skipped in result.skipped))  # sets left out
```

```text
2026-10-06T05:12:33.772 58180 4633 m
2026-10-06T12:10:37.724 58180 4902 m
2026-10-07T16:10:03.846 46275 4104 m
2026-10-10T14:08:54.848 49773 2751 m
Counter({'stale': 28})
```

Each approach holds both objects' GCRF states at TCA, the range at TCA, the
miss vector in the primary's [RTN frame](frames-and-units.md#rtn-frame)
(radial, transverse, normal) and the relative velocity. Screening GP
element sets finds geometry, not risk. The element sets have no
covariance and are kilometres off within days.

[Screening](screening.md) explains the method, the three built-in screening
volumes and the limits of GP data.
[Screening against SOCRATES](screening-socrates.md) puts keepout's results
next to CelesTrak's.

## Synthetic operator and command line

From a clone of the repository (see the [Quickstart](#quickstart)):

```bash
uv run keepout screen --fleet examples/synthetic-operator/fleet.toml \
    --catalog tests/screening/data/celestrak-public.csv.gz \
    --start 2026-10-06 --days 7 --cdm-dir cdms
uv run keepout assess cdms/*.kvn   # exit status 3 if an event needs a manoeuvre
```

`keepout screen` acts as a conjunction assessment service for a fleet of
real satellites, listed in a [fleet file](synthetic.md#the-fleet-file).
Here it screens the ISS, Metop-B and Intelsat 39 against
the public catalog frozen on 2026-10-06. It writes a CDM for each close
approach and assesses each event (12 approaches that week, 2 to monitor).

!!! warning "Assumed covariances"
    GP data has no covariance, so these CDMs carry keepout's
    [synthetic covariance model](assumptions.md) of GP element-set errors,
    not orbit determination. Their Pc values show what the model implies,
    not the satellites' real risk.

Other commands download data and check keepout against CARA:

- `keepout fetch catalog` downloads the live catalog, from CelesTrak (no
  account) or from Space-Track.org (the full catalog, with an account).
- `keepout fetch cdm` lists Space-Track's public CDM summaries.
- `keepout validate` checks keepout's Pc against CARA's test cases.

The `store` extra (`keepout[store]`) keeps CDMs, events and their Pc in a
local SQLite file. `screen` and `assess` use it with `--store`, as do
`keepout store add` and `keepout maneuver --event ID`. Each event has a
stable id of 12 hexadecimal characters. `keepout worker` computes queued
manoeuvre plans in the background. See [The conjunction store](store.md).
`keepout dashboard` shows a store in the browser, by default on this
machine only (the `dashboard` extra; `--host` and its warning are in
[The dashboard](dashboard.md)).

See [Synthetic operator](synthetic.md) for what a run does,
[Command line](cli.md) for every command,
[Space-Track data](spacetrack.md) for the full catalog and the
[report](synthetic-operator.md) for the week's approaches.

## Manoeuvre planning

```python
import keepout.cdm
from keepout import ConjunctionEvent
from keepout.maneuver import plan_maneuver

event = ConjunctionEvent.from_cdms([keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")])
plan = plan_maneuver(event, combined_hard_body_radius=20.0)  # target Pc 1e-5
for option in (plan.retrograde, plan.prograde):  # None if no burn meets it
    if option is not None:
        hours = option.burn_lead / 3600
        print(f"{option.delta_v:+.4f} m/s {hours:.1f} h before TCA, Pc {option.pc:.1e}")
```

```text
-0.0063 m/s 17.1 h before TCA, Pc 8.5e-06
+0.0040 m/s 18.9 h before TCA, Pc 9.6e-06
```

keepout plans an impulsive along-track burn of the primary from the latest
CDM of an event. It computes the Pc for a grid of burns (Δv against burn
time) and picks the smallest prograde and retrograde burn below the target
Pc. It can also screen each burn against a catalog for the approaches it
brings. On the command line this is `keepout maneuver`.

[Manoeuvre planning](maneuver.md) describes the method and its assumptions
and limits. The [report](maneuver-report.md) shows the Δv × burn-time
heatmap of a CARA test event.

## Validation

Every Pc algorithm in keepout is implemented from the published literature
and validated against the public test cases of the
[NASA CARA Analysis Tools](https://github.com/nasa/CARA_Analysis_Tools).
keepout is not affiliated with NASA. CARA files are downloaded on demand
from a pinned commit and verified by SHA-256. They are never redistributed
with keepout.

- The [validation report](validation.md) compares every Pc method with
  every applicable CARA case.
- The [Monte Carlo validation report](validation-monte-carlo.md), checked
  nightly, compares keepout's Monte Carlo with CARA's.
- [Screening against SOCRATES](screening-socrates.md) compares `screen`
  with CelesTrak's SOCRATES.

## Citing keepout

keepout has no paper yet. Cite the repository, with the version
(`keepout.__version__`) or the commit you used:

> Niczke, P. *keepout: spacecraft conjunction assessment in Python*,
> version 1.0.0, 2026. <https://github.com/Kyrtap1309/keepout>

Cite the data as well, as below.

## Data and licences

keepout is under the MIT licence. The third-party data committed for its
tests is not covered by it:

- **GP data.** USSPACECOM/18 SDS via Space-Track.org. The frozen public
  catalog in `tests/screening/data/` was retrieved from CelesTrak
  (celestrak.org) on 2026-10-06. The frozen full catalog in
  `tests/screening/spacetrack-data/` was retrieved from Space-Track.org on
  2026-10-15. The excerpt of Space-Track.org's 2009 bulk TLE file in
  `tests/case_study/data/iridium-cosmos/` is the case study's. See their
  `NOTICE` files.
- **Kelvins data.** ESA, *Collision Avoidance Challenge dataset*, Zenodo,
  2021, [doi:10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683);
  © ESA, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
  excerpt in `tests/sources/data/` has its own `README.md`.
- **CARA test CDMs.** NASA CARA Analysis Tools (NASA Open Source
  Agreement), downloaded on demand. Only numbers derived from them and the
  manoeuvre report's figure are committed.

Development commands (tests, reports, linting) are in
[`CONTRIBUTING.md`](https://github.com/Kyrtap1309/keepout/blob/main/CONTRIBUTING.md).
