# Manoeuvre planning

When an event stays red, its operator plans a collision-avoidance
manoeuvre. That is a burn of the primary, some hours before TCA, that takes
the Pc below a target without bringing the primary close to anything else.
keepout plans such a burn from the CDMs of one event. It computes the Pc of
a grid of burns, picks the smallest burn each way that meets the target,
and can screen each burn against a catalog for the approaches it brings.

The burns are impulsive, ideal and along the velocity, and the CDM's
covariances are used unchanged. [Assumptions and limits](#assumptions-and-limits)
lists everything the planner leaves out, with measured numbers. The
[report](maneuver-report.md) shows the planner on a CARA test event and the
re-screen on the frozen public catalog.

## Terms

These terms are used on the rest of this page:

- A **cell** is one burn of the grid: a burn time and a signed Δv.
- A burn's **lead** (`burn_lead`) is the time from the burn to TCA, in
  seconds.
- The **baseline** is the event without a burn: the Pc of the latest CDM,
  by the event's own method.
- An **option** is the smallest burn in one direction that meets the
  target, confirmed by the event's own method (see
  [The options](#the-options)).
- An **event return** is the event's own pair meeting again, away from its
  TCA, at or above the target. Only a [re-screen](#re-screen) finds one.

## Example

From a clone of the repository, this plans the burns of
`tests/cdm/data/tracss_v2.kvn`, a test CDM 2.0 with made-up states and
covariances (`CALCULATED`), a manoeuvrable primary and a red Pc. It gives
`HBR` for one object only, so the combined hard-body radius is given:

```python
import keepout.cdm
from keepout import ConjunctionEvent
from keepout.maneuver import plan_maneuver

cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")  # or several CDMs
event = ConjunctionEvent.from_cdms([cdm])
plan = plan_maneuver(event, combined_hard_body_radius=20.0)  # target 1e-5
print(f"{plan.baseline.pc:.2e}", plan.baseline.method)  # the Pc without a burn
for option in (plan.retrograde, plan.prograde):  # None if no burn meets it
    if option is not None:
        hours = option.burn_lead / 3600
        print(f"{option.delta_v:+.4f} m/s {hours:.1f} h {option.pc:.2e}")
print(plan.pc.shape)  # Foster's Pc per (burn time, Δv) cell
for note in plan.notes:
    print("Note:", note)
```

```text
4.50e-02 hall3d
-0.0063 m/s 17.1 h 8.46e-06
+0.0040 m/s 18.9 h 9.63e-06
(72, 62)
Note: states are 300 m from closest approach along the relative velocity; 2-D Pc uses the straight-line closest approach
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS, and with a perigee at 400 km drag changes this grid: on one orbit each (CD·A/m = 0.02, 25 h) it moved log10 Pc by up to 0.18 at 400 km and 1.29 at 300 km, measurements and no bound
```

A retrograde burn of 6.3 mm/s 17.1 h before TCA, or a prograde one of
4.0 mm/s 18.9 h before, takes the Pc from 4.5e-2 below the target, as
confirmed by Hall 3D-Nc, the event's own method. The script took 6.1 to
6.3 s over 3 runs on one thread, imports included (Intel Core i5-14600K;
see [Time](#time)).

!!! tip "Run the library on one thread"
    From Python, set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`
    before starting the interpreter (e.g.
    `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python plan.py`). The
    `keepout` command does this itself, unless the variables are set (see
    [Threads](cli.md#threads)).

`evaluate_burn(plan, burn_lead=..., delta_v=...)` gives one burn of the
plan's event, on the grid or off it, with the event's own Pc method. For a
burn of your own, `prepare_maneuver(event)` gives the event's baseline
without the grid, in a fraction of the time, and `evaluate_burn` takes it
in place of a plan. The command line runs the same as `keepout maneuver`
([Command line](cli.md#maneuver)). It also plans an event kept in a
[conjunction store](store.md) by its id, `keepout maneuver --event ID`,
and re-screens its burns against the catalog the event was screened with
([Stored events](cli.md#stored-events)).

## The grid

The planner computes the Pc of a grid of cells (see [Terms](#terms)):

| | Default |
|---|---|
| Burn times | from 2 days before TCA, or the latest CDM's creation if that is later, to 6 hours before TCA, 8 per orbital period of the primary |
| Δv | 1 mm/s to 1 m/s, 31 values per direction, geometrically spaced; prograde (positive) and retrograde (negative) |
| Pc | Foster's 2-D Pc in every cell, with the CDM's covariances unchanged |
| Target | Pc below 1e-5 (`TARGET_PC`) |

A burn before the CDM's creation is left out, since the operator did not
know of the event then. The last burn time is 6 hours before TCA because an
along-track burn acts through the drift it builds up. That drift has little
time to grow so close to TCA. `BurnGrid` changes any of these, and
`--target-pc` or `target_pc` the target.

The default target, 1e-5, is ten times below the red
threshold `red_from` (1e-4). The margin leaves the event clearly out of the
red rather than just under it. The target is compared on the basis of the
plan's `thresholds`, which `evaluate_burn` and `rescreen` use too. On the
default, conservative basis a Monte Carlo Pc counts by the upper end of its
confidence interval.

Every cell uses Foster's Pc, whatever the event's own method. Foster takes
about 0.1 ms per cell and Hall 3D-Nc 0.1 to 0.2 s, so Hall on the 12 896
cells of a full grid would take over 15 minutes. Instead the event's own
method (the one `auto` chose for the CDM, or the one given) confirms the
options. A cell that Foster puts below the target but the event's method
does not is therefore never an option. The rest of the grid, and the
heatmap, are Foster's.

## The burn as a difference

A CDM gives both objects' states and covariances at TCA, from the
originator's orbit determination. keepout has no better orbit than that.
So the planner does not integrate a burned orbit and compare it with the
CDM. It computes what the burn changes at TCA and adds that change to the
CDM's state. The secondary is not moved, since the burn does not move it.

For small burns the change grows as the along-track drift does, about
3·Δv·Δt. Take the Metop-B CDM of the report: 25.1 h from the CDM's creation
to TCA, gravity to J6 without drag. A burn at the CDM's creation moves the
primary at TCA by 2.73 km for 0.01 m/s, 27.3 km for 0.1 m/s and 273.0 km
for 1 m/s. The estimate 3·Δv·Δt gives 2.72, 27.2 and 271.5 km.

??? note "Implementation: the integration"
    From the primary's state at TCA, a Cowell integration goes back to the
    earliest burn time. It uses DOP853, gravity to J6, and drag if the CDM
    gives `CD_AREA_OVER_MASS`. The integration then goes forward to TCA
    with every cell of the grid in one batch: the orbit without a burn and,
    for each burn time, the orbits with each Δv added along the velocity
    there. The change at TCA is a cell minus the orbit without a burn,
    carried in the same batch. The solver's steps are shared, so its error
    is nearly the same with and without the burn and cancels in the
    difference. On Metop-B's CDM the integration back and forth ended
    1.5 cm from where it started (one measurement). The secondary is not
    integrated.

## The new closest approach

After a burn the objects no longer meet at the CDM's TCA. Each cell finds
its own closest approach by Newton's method on the range of a second-order
expansion of both relative positions about TCA. The expansion uses
position, velocity and the acceleration of the force model, and is trusted
within a twentieth of the primary's orbital period.

The expansion may have no minimum there, or a curvature that is not
positive. That happens in a slow, nearly co-orbital encounter, in which a
small change of speed can move the minimum by hours. Such a cell gets no
Pc, is never an option and is counted in a note. A miss distance the
expansion cannot vouch for would read as a safe burn.

The planner looks for the new TCA only within that twentieth of an orbit.
In a slow encounter a burn can bring the same two objects back hours later,
outside it, at a Pc no cell of the grid shows. Only a
[re-screen](#re-screen) shows such an event return. Below 100 m/s of
relative speed the plan notes this.

Both objects are then moved to the new TCA along the same expansion, with
the CDM's covariances, and the cell's Pc is computed there.

## The options

The option of a direction is its smallest burn below the target,
confirmed by the method that gave the baseline Pc. Usually that is the
cell of smallest |Δv| below the target on Foster's Pc. Among cells of
equal |Δv|, the latest burn wins, as the one decided with the most
information.

The event's own method computes the Pc of each candidate in turn, in that
order, and the first it also puts below the target is the option. So:

- a cell Foster puts below the target but the event's method does not is
  skipped, and the next candidate is tried;
- at most 8 candidates per direction are tried;
- a direction where none is confirmed has no option, and the notes say
  how many candidates the method refused or failed on.

`ManeuverPlan` keeps the whole grid (`burn_leads`, `delta_v`, `pc`,
`range_at_tca`), the baseline, both options and the notes. The notes cover:

- assumed covariances;
- drag left off (a warning with measured numbers when the perigee is below
  450 km);
- a slow encounter;
- the cells without a Pc.

## Re-screen

A burn that clears one event moves the primary for days afterwards, into
approaches it did not have. `rescreen(plan, option, catalog)` screens the
primary against a GP catalog twice for 7 days from the burn, with the same
[screening](screening.md) as everywhere else. One run uses the primary as
SGP4 gives it, the other adds the burn. The burn is a
`ManeuveredPropagator`: SGP4's states plus the Cowell difference of the
burn, rotated to TEME at every node.

The burned orbit is accurate to metres on the orbits measured, far below
a screening volume of kilometres.

??? note "Implementation: accuracy of the burned orbit"
    Against a Cowell integration of the whole orbit after a 1 m/s burn,
    the difference over 7 days reached at most 2.24 m on the 38 fixture
    orbits with e > 0.3. On less eccentric orbits it was 0.09 m for 68377
    (a circular LEO) and 0.82 m for 89251 (e = 0.28, perigee near 500 km).
    The tests hold the eccentric orbits to 3 m and 68377 to 1 m. These are
    measurements on those orbits, not a bound, and other orbits are not
    covered by them.

??? note "Implementation: radius bounds of the burned primary"
    The apogee/perigee filter needs bounds on the primary's radius after
    the burn. The difference lies mostly along the track (1800 km after a
    week of 1 m/s on a LEO orbit) and moves the radius by far less, so the
    bounds are taken from the radius itself: sampled every 10 s, and
    widened between samples by a bound on its curvature, they are within
    metres of its range. A 1 m/s re-screen of NOAA-20 against the public
    catalog then takes 8 s, not 32 s, and a slow test checks that it finds
    the same approaches as bounds widened by the whole difference. Where
    SGP4 fails for the primary inside the window, those wide bounds are
    used.

The result compares the two screenings:

- **New approaches**: approaches with the burn that the screening without it
  does not have within 10 minutes of their TCA.
- **Closer approaches**: approaches it does have, matched to the nearest in
  time, whose range at TCA fell by any amount. The event's own approach
  (its secondary within 10 minutes of the event's TCA) is not among them.
- **Event returns**: the new and closer approaches of the event's own
  secondary more than 10 minutes from the event's TCA whose Pc is at or
  above the plan's target. The burn clears the event at its TCA but brings
  the pair back elsewhere above the bar it was planned to clear.
- **The event's range** after the burn, if its secondary was screened.
- **Coverage**: whether the event's secondary was screened at all, and how
  many catalog objects screening left out. The secondary may be missing
  from the catalog given. `secondary_screened` is then false, and the
  re-screen says nothing of the event.

Each new or closer approach gets a Pc and a risk level. The Pc is the
synthetic operator's: `auto` unless `compute` says otherwise, compared on
the conservative basis of the thresholds. `conflict` is true when any of
these approaches is red. `clean` is true when there is no conflict and no
event return.

!!! warning "Assumed covariances"
    GP element sets have no covariance. The Pc of an approach the re-screen
    finds rests on the [synthetic covariance model](assumptions.md) of both
    element sets, with all its limits. It is labelled assumed wherever it
    is shown.

The primary must be in the catalog and screened in both runs, and SGP4
must carry it through the whole window. Otherwise `rescreen` raises
`ManeuverError`, so "nothing new" never stands for "nothing looked at".

The re-screen checks the options. It does not search for another one when
an option brings an approach. Weighing a burn that clears one event against
an approach it brings is left to the analyst. The analyst can evaluate any
burn with `evaluate_burn` or `keepout maneuver --burn-before --dv` and
re-screen it.

## The outcome

The plan `keepout maneuver` shows is a `ManeuverOutcome`. `plan_event`
does what the command does. It plans the event with a target, Pc method,
covariance policy, hard-body radius and thresholds; `plan_arguments`
reads them from an operator's [settings](reference/settings.md). If a
catalog is given, it re-screens each option against it.

A `ManeuverOutcome` holds everything the command shows:

- the baseline and the notes;
- the burns (`PlannedBurn`: an option, and its re-screen or `None`);
- the Pc grid (`ManeuverGrid`);
- what the burns were re-screened against.

`outcome.verdict` applies `judge` to its burns and its baseline: when the
baseline already meets the target, the verdict is that no burn is needed
(`no_burn_needed`), whatever the burns, and none is recommended. With
`burn=(lead, dv)`,
`plan_event` evaluates a burn of one's own and plans no grid, as
`--burn-before --dv` does.

With `event` from the [first example](#example), and the combined
hard-body radius in the settings since the CDM gives `HBR` for one object
only:

```python
from keepout.maneuver import plan_event
from keepout.settings import Settings, plan_arguments

settings = Settings(combined_hard_body_radius=20.0)  # [m]
outcome = plan_event(event, **plan_arguments(settings))  # no catalog: no re-screen
print(outcome.verdict.recommendable, [burn.direction.value for burn in outcome.burns])
```

```text
True ['prograde', 'retrograde']
```

The [conjunction store](store.md#background-jobs) caches an outcome as
a result row. It leaves out the event, which the store holds already. A
background job computes the outcome with `plan_event`.

## Time

All times were measured on an Intel Core i5-14600K, 3 runs each, on
2026-11-03:

| Run | One thread | Default threads |
|---|---|---|
| Default grid, LEO event 48 h before TCA (12 896 cells) | 29.9 to 31.2 s | 33.4 to 46.1 s |
| The report's Metop-B event, 25.1 h (5 642 cells, the report's own runs) | 9.7 to 10.1 s | 12.3 to 16.2 s |

Most of the time is the batched integration. With default threading the
run uses about 20 times more CPU time and is still slower; run the library
on one thread (see the tip under [Example](#example)).

A confirmation by Hall 3D-Nc adds 0.1 to 0.2 s per candidate. A
confirmation by Monte Carlo adds up to about a minute for a slow encounter.
A re-screen of the report's GEO burn against the public catalog took 4.2 s
with 4 workers in each of 3 runs.

## Assumptions and limits

The planner answers "what does this burn do to the event", from the CDM's
states and covariances. These assumptions apply to any CDM. The numbers
below are measurements on sample cases, not bounds.

- **The burn.** Impulsive, ideal (no execution error, no attitude or
  thrust profile) and along the primary's velocity only. There are no
  radial or normal burns, and no burn of the secondary. An along-track
  burn acts through the drift it builds up, so it loses effect close to
  TCA. That is why the latest burn time is 6 hours before TCA, not
  because later burns are impossible.
- **The CDM's state is the truth.** The planner adds the change a burn
  makes at TCA to the CDM's state. It has no better orbit to start from.
- **Covariances unchanged.** Every cell uses the CDM's covariances of
  both objects. This treats the burn as adding no uncertainty, and the
  orbit determination as not changing with time.
- **The target.** 1e-5 by default, ten times below the red threshold (see
  [The grid](#the-grid)). `--target-pc` or `target_pc` sets another.
- **Forces.** Zonal gravity to J6 (EGM2008) about a pole fixed for the
  integration. There are no tesseral terms, no third bodies (Moon, Sun)
  and no solar radiation pressure. These matter most at GEO, where
  lunisolar gravity and SRP dominate the perturbations. A GEO plan
  therefore rests on a simpler orbit than a LEO one.
- **Drag only with CD·A/m.** Drag is on only when the CDM gives the
  primary's `CD_AREA_OVER_MASS`. Otherwise the plan says it is off. What
  that costs depends on the altitude. It was measured on one circular
  orbit per altitude, with CD·A/m = 0.02 and burns up to 25 h before TCA
  on the default grid. Switching drag on moved log10 Pc by at most 0.003
  at 630 km, 0.03 at 500 km, 0.18 at 400 km and 1.29 at 300 km. Below a
  450 km perigee the note without drag is a warning with these numbers.
- **The atmosphere.** An exponential density table by altitude band
  (Vallado, table 8-4), without solar activity or the day-night bulge.
  Its density is a rough guess for any given day. An eccentric orbit
  that swings across the table's band edges integrates to metres less
  accurately than the solver's tolerances suggest, against drag effects of
  hundreds of kilometres or more; most of this error cancels in the planner's
  differences.

    ??? note "Implementation: band edges and accuracy"
        The table's bands do not join exactly: up to 0.14 % at 25 km,
        0.01 % or less above. Three eccentric orbits were integrated for
        25 h and compared with tolerances 10 times tighter (rtol 1e-12):

        | Orbit | Difference |
        |---|---|
        | 240 km, e = 0.005, CD·A/m = 0.05 | 1.5 m |
        | 300 km, e = 0.01, CD·A/m = 0.02 | 4.1 m |
        | 400 km, e = 0.02, CD·A/m = 0.1 | 9.5 m |
        | 300 km, circular, CD·A/m = 0.02 | 0.03 m |

        Drag itself moved these orbits by 170 to 2 650 km. The planner
        works on differences of orbits integrated together, in which most
        of this error cancels.
- **Foster on the grid.** Every cell uses Foster's 2-D Pc, for speed, and
  the event's own method confirms the options (see
  [The grid](#the-grid)).
- **Slow encounters.** The planner looks for each cell's new closest
  approach only within a twentieth of an orbit of TCA (see
  [The new closest approach](#the-new-closest-approach)). A burn can bring
  the pair back hours later, outside that window. Only a re-screen shows
  such an event return. Inside the window, the new TCA of a cell was
  measured up to 28 m off at 10 m/s among the cells sampled. The
  covariances there are hundreds of metres.
- **Re-screen accuracy.** The burned orbit of a re-screen is SGP4 plus a
  Cowell difference, accurate to metres on the orbits measured (see
  [Re-screen](#re-screen)).
- **Re-screen Pc assumed.** The Pc of a new or closer approach is assumed
  (see [Re-screen](#re-screen)). The re-screen also
  sees only the objects in the catalog given, and the public catalog
  lacks most debris (see [Screening](screening.md#the-public-catalog)).

## References

- D. A. Vallado, *Fundamentals of Astrodynamics and Applications*,
  Microcosm Press: Cowell's method and the exponential atmosphere
  (table 8-4).
- N. K. Pavlis, S. A. Holmes, S. C. Kenyon, J. K. Factor, "The development
  and evaluation of the Earth Gravitational Model 2008 (EGM2008)",
  *J. Geophys. Res.* 117, B04406, 2012: the zonal coefficients J2 to J6.
- J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta formulae",
  *J. Comput. Appl. Math.* 6(1), 1980; E. Hairer, S. P. Nørsett, G. Wanner,
  *Solving Ordinary Differential Equations I*, 2nd ed., Springer, 1993
  (DOP853, as `scipy.integrate.solve_ivp` implements it).
- NASA, *Spacecraft Conjunction Assessment and Collision Avoidance Best
  Practices Handbook*, NASA/SP-20205011318 Rev. 2: collision avoidance
  planning and the re-screen of a planned manoeuvre.
