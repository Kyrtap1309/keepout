# Risk assessment

A conjunction is rarely a single message: the screening service sends a CDM
every few hours for days before TCA, and each one gives a new Pc. keepout
turns that sequence into a recommendation: **ignore**, **monitor**,
**prepare maneuver** or **maneuver**, always with the reasons.

```python
from pathlib import Path

import keepout.cdm
from keepout import ConjunctionEvent, assess

# Three CDMs of one sample encounter; use the CDMs of your own event instead.
paths = Path("tests/cdm/data/event").glob("*.kvn")
event = ConjunctionEvent.from_cdms(keepout.cdm.load(path) for path in paths)
observations = event.observations(combined_hard_body_radius=20.0)  # [m]
for observation in observations:  # oldest first
    days = observation.time_to_tca / 86400
    print(f"{days:.2f} d before TCA: Pc {observation.pc:.2e}")
assessment = assess(observations)
print(assessment.decision.action.name)
for reason in assessment.decision.reasons:
    print("-", reason)
```

```text
4.01 d before TCA: Pc 6.41e-05
2.51 d before TCA: Pc 2.53e-03
0.84 d before TCA: Pc 4.50e-02
MANEUVER
- latest Pc 4.50e-02 is red, 0.84 d before TCA
- trend rising (+0.90 decades per day)
- within the decision point (2.00 d before TCA)
- diagnostic: states are 300 m from closest approach along the relative velocity; 2-D Pc uses the straight-line closest approach
```

The sample event's CDMs are made up from the repository's sample CDM: its
covariances shrink from one CDM to the next, so the Pc rises
(`tests/cdm/data/event/README.md` says how they were made). On the command
line, [`keepout assess`](cli.md#assess) does the same for CDM files, and
its `--json` output is described in [JSON output](cli-json.md#screen-and-assess):

```bash
keepout assess tests/cdm/data/event/*.kvn  # exit status 3: a manoeuvre
```

## Events and observations

A `ConjunctionEvent` holds the CDMs of one encounter: the same two objects
(`OBJECT1` and `OBJECT2` by `OBJECT_DESIGNATOR`) and TCAs within `tca_tolerance`
(600 s by default). It refuses CDMs that do not belong together: other
objects, a repeated `MESSAGE_ID`, TCAs further apart or different
`CONJUNCTION_ID` values. `group_events` sorts a pile of CDMs into events,
in TCA order: a CDM joins the event that admits it (`EventBounds.admits`)
or starts a new one, so CDMs of different `CONJUNCTION_ID` make different
events. A CDM without one that fits two events is an error.

`event.observations()` computes keepout's own Pc for every CDM with
`compute_pc`. The method is `"auto"` by default. `compute=` picks another
method, with its options:

```python
from keepout.pc import computation

event.observations(compute=computation("elrod"), combined_hard_body_radius=20.0)
event.observations(
    compute=computation("montecarlo", samples=100_000, seed=1),
    combined_hard_body_radius=20.0,
)
```

The originator's `COLLISION_PROBABILITY` is kept next to keepout's Pc in
`reported_pc`.

Each result becomes a `PcObservation`: the time from the CDM's creation to
TCA, the Pc, what kind of value it is, and the diagnostics. A CDM whose Pc
cannot be computed (an unusable covariance, no convergence) gives an
observation with the error instead of stopping the others. A missing
hard-body radius is the one exception. It is reported before anything is
computed, since every CDM would fail the same way.

An observation's `source` names the method that gave the value, e.g.
`keepout (foster)` when `"auto"` chose Foster. A failed one names the
method asked for, e.g. `keepout (auto)`.

Observations can also come from elsewhere. ESA's Kelvins data
(`keepout.sources.kelvins`) gives ESA's own Pc per CDM, which keepout
assesses as reported.

| Kind | Meaning | Source |
|---|---|---|
| `estimate` | an estimate of the Pc | Foster, Elrod, Hall 3D-Nc, Monte Carlo |
| `upper_bound` | a maximum Pc, used when a covariance is unknown | Frisbee, dilution, Alfano (2005) |
| `below_floor` | the source reports only that the Pc is at most this | ESA's floor of 1e-30 |

## Risk levels

| Level | Pc | Default |
|---|---|---|
| green | below `green_below` | Pc < 1e-7 |
| yellow | between the two | 1e-7 <= Pc < 1e-4 |
| red | from `red_from` | Pc >= 1e-4 |

The default thresholds follow common operational practice (NASA's CARA
mitigates at 1e-4). They are a starting point, not a standard for every
mission. `Thresholds(green_below=..., red_from=...)` sets your own.

By default the comparison is **conservative**: a Monte Carlo Pc is compared
by the upper end of its confidence interval, so 0 hits in too few samples is
not green. `Thresholds(basis="estimate")` compares the estimate itself.

## Trend

`analyse_trend` orders the observations from the oldest to the latest. When
observations share a time, a failed one is ordered after one with a Pc.
Among those with a Pc, the highest (or the highest interval upper end)
comes last. CDMs issued after TCA and CDMs without a Pc are left out, with
a note. The **direction** is the slope of a least-squares line through
log10 Pc against time over the last three observations
(`TrendConfig(window=...)`):

- **rising**: more than 0.5 decades of Pc per day (`steady_within`);
- **falling**: less than -0.5 decades per day;
- **steady**: in between;
- **undetermined**: fewer than two observations, or all at one time.

Pc values below 1e-30 (including 0) are fitted at 1e-30. These settings are a
heuristic, not a standard. The [Kelvins report](risk-kelvins.md) shows how
they behave on 15 321 real events.

## Decision

`decide` looks at the latest observation before TCA. The actions are
ordered from the mildest to the most drastic (`Action.IGNORE <
Action.MONITOR < Action.PREPARE_MANEUVER < Action.MANEUVER`):

| Latest observation | Action |
|---|---|
| red, at most `decision_point` before TCA (2 days by default) | **maneuver** |
| red, earlier | **prepare maneuver** (monitor and prepare a manoeuvre) |
| yellow | **monitor** |
| green, rising fast enough to reach yellow by TCA | **monitor** |
| green, an upper bound or with diagnostics | **monitor** |
| green otherwise | **ignore** |
| no Pc (the computation failed) | **monitor**, or the red rules (below) |
| none before TCA (the event is past) | **monitor** |

When the latest observation has no Pc but the latest observation with a Pc
was red, the red rules apply, judged at the failed observation's time. That
gives maneuver within the decision point and prepare maneuver before it.

A green Pc that rises is projected to TCA along the fitted slope. A rise
from 1e-18 to 1e-17 is reported but does not need attention. A doubtful
green is never ignored: an upper bound, a repaired covariance, a
questionable 2-D assumption or a failed computation.

The decision point leaves time to plan the manoeuvre and screen the new
orbit. Set it with `DecisionConfig(decision_point=...)`, in seconds.

## Assumptions and limits

- The rules are transparent heuristics for a small operator's triage, not an
  operational standard and not a substitute for an operational conjunction
  assessment service.
- The latest observation decides. If it has no Pc, the latest one with a Pc
  decides with it. Earlier observations shape the trend only.
- The trend extrapolates nothing except the rising-green check. Forecasting
  the Pc at TCA is not implemented.
- Each CDM is computed independently with `compute_pc`. In `"auto"` mode a
  slow encounter can take up to a minute per CDM, while
  `computation("foster")` takes milliseconds.
