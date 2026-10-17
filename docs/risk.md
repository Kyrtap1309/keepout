# Risk assessment

A conjunction is rarely a single message: the screening service sends a CDM
every few hours for days before TCA, and each one gives a new Pc. keepout
turns that sequence into a recommendation: **ignore**, **monitor**,
**prepare manoeuvre** or **manoeuvre**, always with the reasons.

```python
from pathlib import Path

import keepout.cdm
from keepout import ConjunctionEvent, assess

cdms = [keepout.cdm.load(path) for path in Path("event").glob("*.cdm")]
event = ConjunctionEvent.from_cdms(cdms)
assessment = assess(event.observations(hard_body_radius=20.0))
print(assessment.decision.action.name)  # e.g. MONITOR
for reason in assessment.decision.reasons:
    print("-", reason)
```

## Events and observations

A `ConjunctionEvent` holds the CDMs of one encounter: the same two objects
(`OBJECT1` and `OBJECT2` by designator) and TCAs within `tca_tolerance`
(600 s by default). It refuses CDMs that do not belong together: other
objects, a repeated `MESSAGE_ID`, TCAs further apart or different
`CONJUNCTION_ID` values. `group_events` sorts a pile of CDMs into events.

`event.observations()` computes keepout's own Pc for every CDM with
`compute_pc` (`"auto"` by default; `compute=computation("elrod")` for
another method, `computation("montecarlo", samples=100_000, seed=1)` for a
method with options) and keeps the originator's `COLLISION_PROBABILITY`
next to it in `reported_pc`. Each result becomes a `PcObservation`: the time
from the CDM's creation to TCA, the Pc, what kind of value it is, and the
diagnostics. A CDM whose Pc cannot be computed (an unusable covariance, no
convergence) gives an observation with the error instead of stopping the
others. An observation's `source` names the method that gave the value,
e.g. `keepout (foster)` when `"auto"` chose Foster; a failed one names the
method asked for, e.g. `keepout (auto)`. A missing hard-body radius is the one exception: it is reported
before anything is computed, since every CDM would fail the same way.

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
mitigates at 1e-4); they are a starting point, not a standard for every
mission: `Thresholds(green_below=..., red_from=...)` sets your own.

By default the comparison is **conservative**: a Monte Carlo Pc is compared
by the upper end of its confidence interval, so 0 hits in too few trials is
not green. `Thresholds(basis="estimate")` compares the estimate itself.

## Trend

`analyse_trend` orders the observations from the oldest to the latest. At
one time, a failed observation counts as later than one with a Pc, and the
higher Pc (or interval upper end) as the latest. CDMs issued after TCA and
CDMs without a Pc are left out, with a note. The **direction** is the slope
of a least-squares line through log10 Pc against time over the last three
observations (`TrendConfig(window=...)`):

- **rising**: more than 0.5 decades of Pc per day (`steady_within`);
- **falling**: less than -0.5 decades per day;
- **steady**: in between;
- **undetermined**: fewer than two observations, or all at one time.

Pc values below 1e-30 (including 0) are fitted at 1e-30. These settings are a
heuristic, not a standard; the [Kelvins report](risk-kelvins.md) shows how
they behave on 15 321 real events.

## Decision

`decide` looks at the latest observation before TCA; the actions are
ordered from the mildest to the most drastic (`Action.IGNORE <
Action.MONITOR < Action.PREPARE_MANOEUVRE < Action.MANOEUVRE`):

| Latest observation | Action |
|---|---|
| red, at most `decision_point` before TCA (2 days by default) | **manoeuvre** |
| red, earlier | **prepare manoeuvre** (monitor and prepare a manoeuvre) |
| yellow | **monitor** |
| green, rising fast enough to reach yellow by TCA | **monitor** |
| green, an upper bound or with diagnostics | **monitor** |
| green otherwise | **ignore** |
| no Pc (the computation failed) | **monitor**; if the latest observation with a Pc was red, the red rules judged at this observation's time (manoeuvre within the decision point, otherwise prepare manoeuvre) |
| none before TCA (the event is past) | **monitor** |

A green Pc that rises is projected to TCA along the fitted slope; a rise from
1e-18 to 1e-17 is reported but does not need attention. A doubtful green (an
upper bound, a repaired covariance, a questionable 2-D assumption, a failed
computation) is never ignored.

The decision point leaves time to plan the manoeuvre and screen the new
orbit; set it with `DecisionConfig(decision_point=...)` (seconds).

## Assumptions and limits

- The rules are transparent heuristics for a small operator's triage, not an
  operational standard and not a substitute for an operational conjunction
  assessment service.
- The latest observation decides (if it has no Pc, together with the latest
  one with a Pc); earlier ones shape the trend only.
- The trend extrapolates nothing except the rising-green check; forecasting
  the Pc at TCA is out of scope until v1.x.
- Each CDM is computed independently with `compute_pc`: in `"auto"` mode a
  slow encounter can take up to a minute per CDM; `computation("foster")`
  takes milliseconds.
