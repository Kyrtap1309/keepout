# Synthetic operator

A satellite operator receives conjunction data messages (CDMs) from a
conjunction assessment service, computes the risk of each event and
decides whether to manoeuvre. keepout can play the service itself on
public data: it screens a fleet of real satellites against the GP catalog,
issues a CCSDS CDM for every close approach and assesses each, so the whole
workflow runs end to end without access to an operational service. This is
the *synthetic operator*.

!!! warning "The covariances are assumed"
    GP element sets carry no covariance, and a Pc needs one. The synthetic
    operator gives every object the covariance of keepout's model of GP
    element-set errors: a documented assumption from the literature, not
    orbit determination. Its Pc values show what the model implies, not the
    risk of the real satellites. [Assumptions](assumptions.md) explains the
    model, its sources and its limits.

```python
from pathlib import Path

from astropy.time import Time

import keepout.omm
from keepout.cdm import kvn
from keepout.synthetic import load_fleet
from keepout.synthetic import run


def main() -> None:
    fleet = load_fleet("examples/synthetic-operator/fleet.toml")
    catalog = keepout.omm.load("tests/screening/data/celestrak-public.csv.gz")
    start = Time("2026-10-06T00:00:00", scale="utc")  # live data: Time.now()
    result = run(fleet, catalog, start, workers=4)  # 7 days, 5 km sphere
    for event in result.events:
        approach, pc = event.approach, event.observation.pc
        print(
            approach.tca.isot[:16],
            approach.primary_number,
            approach.secondary_number,
            f"{approach.miss_distance:.0f} m",
            "no Pc" if pc is None else f"Pc {pc:.1e}",
            event.action.name.lower(),
        )
    folder = Path("cdms")
    folder.mkdir(exist_ok=True)
    for cdm in result.cdms:
        kvn.dump(cdm, folder / f"{cdm.message_id}.kvn")


if __name__ == "__main__":  # needed for workers > 1
    main()
```

Run from a clone of the repository (the example fleet and the public
catalog frozen on 2026-10-06 are committed), it prints the twelve close
approaches of the week, writes their CDMs to `cdms/` and takes about 7 s
with 4 workers:

```text
2026-10-06T05:12 25544 58180 4633 m Pc 1.4e-30 ignore
2026-10-06T09:25 38771 35434 4085 m Pc 1.5e-17 ignore
2026-10-06T12:10 25544 58180 4902 m Pc 4.3e-59 ignore
2026-10-06T23:45 38771 31882 3374 m Pc 4.3e-33 ignore
2026-10-07T08:06 38771 30041 4511 m Pc 8.5e-30 ignore
2026-10-07T16:10 25544 46275 4104 m Pc 7.5e-10 ignore
2026-10-07T22:10 38771 35884 3314 m Pc 1.6e-09 ignore
2026-10-08T12:03 44476 55841 3647 m Pc 7.4e-06 monitor
2026-10-09T09:20 38771 29734 1476 m Pc 3.9e-10 ignore
2026-10-10T14:08 25544 49773 2751 m Pc 1.1e-18 ignore
2026-10-11T10:56 38771 30131 2019 m Pc 2.7e-10 ignore
2026-10-12T11:28 38771 35196 212 m Pc 4.9e-06 monitor
```

The same run from the terminal is `keepout screen` (see
[Command line](cli.md#a-week-of-a-synthetic-operator)), and
[Synthetic operator on public data](synthetic-operator.md) is its generated
report: the fleet, the model's parameters, every approach with its Pc and
maximum Pc, and an example CDM.

## The fleet file

A fleet file (TOML) names the satellites to protect by catalog number, with
what GP data does not say about them: each satellite's hard-body radius
(the radius of a sphere enclosing it) and whether it can manoeuvre, and the
radius assumed for every other object. The example,
`examples/synthetic-operator/fleet.toml`, starts:

```toml
# Radius assumed for every catalog object [m]: debris is mostly smaller,
# some spacecraft are much larger.
catalog_hard_body_radius = 1.0

# The International Space Station: crewed, low Earth orbit at 51.6 deg.
[[satellites]]
catalog_number = 25544
name = "ISS (ZARYA)"
hard_body_radius = 50.0
maneuverable = true
```

and adds Metop-B (38771, a sun-synchronous orbit at about 820 km, 10 m) and
Intelsat 39 (44476, geostationary, 20 m). The three were chosen by
screening candidates against the frozen catalog: each has close approaches
in the week of 2026-10-06 (no GPS or other MEO satellite had one). The radii
are illustrative round numbers, not their operators' values, and keepout is
not their operator.

Every key but `name` is required and no other key is allowed: a typo is an
error, not a silent default, since a hard-body radius changes every Pc.
`keepout.synthetic.load_fleet(path)` reads a file and `read_fleet(text)` its
text; both raise `FleetError` naming the satellite and the key at fault.

## What a run does

`keepout.synthetic.run(fleet, catalog, start, duration, volume)`
does what a conjunction assessment service does for an operator:

1. **Element sets.** Each fleet satellite's newest element set is taken
   from the catalog; a satellite missing from it is a `FleetError`.
2. **Screening.** The fleet is screened against the catalog with
   [`screen`](screening.md) (by default 7 days and a 5 km sphere, as
   CelesTrak's SOCRATES; any `ScreeningVolume` or rule per satellite works).
3. **Conjunctions.** At each approach's TCA both objects get the
   covariance of the model for their own element sets (by default
   `DEFAULT_MODEL`; `model=` takes another) and their hard-body radii: a
   fleet satellite its own, any other object the fleet's
   `catalog_hard_body_radius`.
4. **CDMs.** Each conjunction becomes a CDM (below).
5. **Risk.** Each CDM is assessed as its recipient would assess it: the Pc
   is computed from the message (`"auto"` by default; `compute=` takes a
   `keepout.pc.computation`), and the event
   is assessed with keepout's [risk rules](risk.md) (`thresholds=` and
   `decision_config=` change them).

The result (`OperatorResult`) holds the screening, with the element sets it
left out, and one `OperatorEvent` per approach: the approach, its event
(one CDM), keepout's Pc of it and the assessment; `result.cdms` lists the
messages.

**One event per approach.** Two close approaches of the same pair are two
local minima of their range, however close in time, so each is its own
event with a single CDM. With a single Pc the trend is undetermined, and
the decision rests on the Pc's risk level and the time from the CDM's
creation to TCA: a red Pc within 2 days of TCA is `manoeuvre`, a red Pc
further ahead `prepare manoeuvre`. Two fleet satellites approaching each
other give two events, one CDM for each satellite, as a service sends each
operator its own.

## The CDMs

The messages are built by `keepout.cdm.make_cdm`, which turns any
`Conjunction` into a CDM and knows nothing of where its covariances come
from; they read back to the same conjunction, also after a KVN or XML
round trip. For the synthetic operator:

- **Header.** `ORIGINATOR = KEEPOUT`; `MESSAGE_FOR` the fleet satellite's
  name; `CREATION_DATE` the start of the window unless `created=` says
  otherwise (no clock is read, so a run gives the same bytes again);
  `MESSAGE_ID` both catalog numbers, the TCA and the creation time, e.g.
  `38771_35196_20261012T112828.710Z_20261006T000000.000Z`; two `COMMENT`
  lines saying the covariances are assumed.
- **Relative metadata.** TCA (to the microsecond), miss distance, the
  secondary's position and velocity relative to the fleet satellite in its
  RTN frame, the screening period and the screening volume
  (`SCREEN_VOLUME_X`, `_Y`, `_Z`, a comment saying they are semi-axes).
- **Objects.** `OBJECT1` is the fleet satellite (`OBJECT_TYPE = PAYLOAD`,
  `MANEUVERABLE = YES` or `NO` from the fleet file), `OBJECT2` the other
  object (`MANEUVERABLE = N/A`; GP data has no object type).
  `OBJECT_DESIGNATOR` is the catalog number, `CATALOG_NAME = SATCAT`,
  `EPHEMERIS_NAME = NONE`, `REF_FRAME = GCRF`; a name or international
  designator the element set lacks is `UNKNOWN`. Each object's `HBR` makes
  the message CDM version 2.0.
- **Covariance.** `COVARIANCE_METHOD = DEFAULT` (CCSDS's value for a
  covariance not from orbit determination), and six `COMMENT` lines naming
  the model, its parameters, the element set's epoch and age and the
  resulting sigmas. The 6×6 RTN covariance has the model's position sigmas
  on the diagonal and zeros elsewhere (the velocity block is unknown).

The [report](synthetic-operator.md#example-cdm) shows one in full.

## Running it again

The `MESSAGE_ID` carries the creation time, so a run on another day issues
new messages: the CDMs of one approach from several runs (the same pair,
TCAs within 10 minutes) are successive CDMs of one event for
`keepout assess` and `keepout.events.group_events`, which then follow the Pc's
trend. With new element sets the TCA and the Pc move as an operational
service's would; with unchanged element sets only the time to TCA changes.

## References

- CCSDS 508.0-B-1, *Conjunction Data Message*, Blue Book, 2013.
- NASA, *Spacecraft Conjunction Assessment and Collision Avoidance Best
  Practices Handbook*, NASA/SP-20205011318 Rev. 2.
- The model's sources are listed under [Assumptions](assumptions.md#references).
