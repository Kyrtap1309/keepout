# Manoeuvre planning

When an event stays red, its operator plans a collision-avoidance
manoeuvre: a burn of the primary, some hours before TCA, that takes the Pc
below a target without bringing the primary close to anything else. keepout
plans such a burn from the CDMs of one event. It computes the Pc of a grid
of burns, picks the smallest burn each way that meets the target, and can
screen each burn against a catalog for the approaches it brings.

The burns are impulsive, ideal and along the velocity, and the CDM's
covariances are used unchanged; [Assumptions](assumptions.md#manoeuvre-planning)
lists everything the planner leaves out, with measured numbers. The
[report](maneuver-report.md) shows the planner on a CARA test event and the
re-screen on the frozen public catalog.

```python
import keepout.cdm
from keepout import ConjunctionEvent
from keepout.maneuver import plan_maneuver

cdm = keepout.cdm.load("event.cdm")  # the event's latest CDM, or several
event = ConjunctionEvent.from_cdms([cdm])
plan = plan_maneuver(event, combined_hard_body_radius=20.0)  # target 1e-5
print(plan.baseline.pc, plan.baseline.method)  # the Pc without a burn
for option in (plan.retrograde, plan.prograde):  # None if no burn meets it
    if option is not None:
        print(f"{option.delta_v:+.4f} m/s {option.burn_lead / 3600:.1f} h", option.pc)
print(plan.pc.shape, plan.notes)  # Foster's Pc per (burn time, Δv) cell
```

`evaluate_burn(plan, burn_lead=..., delta_v=...)` gives one burn of the
plan's event, on the grid or off it, with the event's own Pc method. The
command line runs the same as `keepout maneuver`
([Command line](cli.md#maneuver)).

## The burn as a difference

A CDM gives both objects' states and covariances at TCA, from the
originator's orbit determination; keepout has no better orbit than that.
So the planner does not propagate the CDM's state to TCA and compare: it
computes what a burn *changes* there and adds that change to the CDM's
state.

From the primary's state at TCA, a Cowell integration (DOP853, gravity to
J6, drag if the CDM gives `CD_AREA_OVER_MASS`) goes back to the earliest
burn time, then forward to TCA with every cell of the grid in one batch:
the orbit without a burn and, for each burn time, the orbits with each Δv
added along the velocity there. The change at TCA is a cell minus the
orbit without a burn, carried in the same batch. The solver's steps are
shared, so its error is nearly the same with and without the burn and
cancels in the difference; on Metop-B's CDM the integration back and forth
ended 1.5 cm from where it started (one measurement). The secondary is not integrated: the burn does not
move it.

For small burns the change grows as the along-track drift does, about
3·Δv·Δt. On the Metop-B CDM of the report (25.1 h from the CDM's creation
to TCA, gravity to J6 without drag), the burn at the CDM's creation moves
the primary at TCA by 2.73 km for 0.01 m/s, 27.3 km for 0.1 m/s and
273.0 km for 1 m/s, against 2.72, 27.2 and 271.5 km from 3·Δv·Δt.

## The new closest approach

After a burn the objects no longer meet at the CDM's TCA. Each cell finds
its own closest approach by Newton's method on the range of a second-order
expansion of both relative positions about TCA (position, velocity and
the acceleration of the force model). The expansion is trusted within a
twentieth of the primary's orbital period. Where it has no minimum there,
or its curvature is not positive (a slow, nearly co-orbital encounter, in
which a small change of speed can move the minimum by hours), the cell
gets no Pc, is never an option and is counted in a note: a miss distance
the expansion cannot vouch for would read as a safe burn.

The planner looks for the new TCA only within that twentieth of an orbit.
In a slow encounter a burn can bring the same two objects back hours later,
outside it, at a Pc no cell of the grid shows: only a
[re-screen](#re-screen) shows it. Below 100 m/s of relative speed the plan
notes this.

Both objects are then moved to the new TCA along the same expansion, with
the CDM's covariances, and the cell's Pc is computed there.

## The grid

| | Default |
|---|---|
| Burn times | from 2 days before TCA, or the latest CDM's creation if that is later, to 6 hours before TCA, 8 per orbital period of the primary |
| Δv | 1 mm/s to 1 m/s, 31 values per direction, geometrically spaced; prograde (positive) and retrograde (negative) |
| Pc | Foster's 2-D Pc in every cell, with the CDM's covariances unchanged |
| Target | Pc below 1e-5, keepout's default: one order of magnitude below `red_from` (1e-4), as a margin. Compared on the thresholds' basis (the conservative basis takes the upper end of a Monte Carlo interval) |

A burn before the CDM's creation is left out: the operator did not know of
the event then. The last burn time is 6 hours before TCA because an
along-track burn acts through the drift it builds up, which has little time
to grow so close to TCA. `BurnGrid` changes any of these, and `--target-pc`
or `target_pc` the target.

Every cell uses Foster's Pc, whatever the event's own method: Foster takes
about 0.1 ms per cell, Hall 3D-Nc 0.1 to 0.2 s, so Hall on the 12 896 cells
of a full grid would take over 15 minutes. The event's own method (the one
`auto` chose for the CDM, or the one given) confirms the options instead.

## The options

In each direction, the candidates are the cells below the target on
Foster's Pc, from the smallest |Δv| and, among equals, from the latest burn
(the one decided with the most information). The event's own method
computes the Pc of each candidate in turn, at most 8 per direction, and the
first it also puts below the target is the option of that direction. An
option is therefore always confirmed by the method that gave the baseline
Pc; a direction without one has no option, and the notes say how many
candidates the method refused or failed on.

`ManeuverPlan` keeps the whole grid (`burn_leads`, `delta_v`, `pc`,
`range_at_tca`), the baseline (the CDM's own Pc with no burn), both
options and the notes: assumed covariances, drag left off (a warning with
measured numbers when the perigee is below 450 km), a slow encounter, and
the cells without a Pc.

## Re-screen

A burn that clears one event moves the primary for days afterwards, into
approaches it did not have. `rescreen(plan, option, catalog)` screens the
primary against a GP catalog twice for 7 days from the burn, with the same
[screening](screening.md) as everywhere else: once as SGP4 gives it, and
once with the burn. The burn is a `ManoeuvredPropagator`: SGP4's states
plus the Cowell difference of the burn, rotated to TEME at every node.
Against a Cowell integration of the whole orbit after the burn, at 1 m/s
over 7 days, that difference reached 2.24 m at worst on the 38 fixture
orbits with an eccentricity above 0.3, and below 0.3 0.09 m on 68377 (a
circular LEO) and 0.82 m on 89251 (e = 0.28, perigee near 500 km); the
tests hold the eccentric orbits to 3 m and 68377 to 1 m. These are
measurements on those orbits, not a bound.
The screening volume is kilometres.

The result compares the two screenings:

- **New approaches**: approaches with the burn that the screening without it
  does not have within 10 minutes of their TCA.
- **Closer approaches**: approaches it does have, matched to the nearest in
  time, whose range at TCA fell by any amount. The event's own approach
  (its secondary within 10 minutes of the event's TCA) is not among them.
- **Event returns**: the new and closer approaches of the event's own
  secondary more than 10 minutes from the event's TCA whose Pc is at or
  above the plan's target: the burn clears the event at its TCA but brings
  the pair back elsewhere above the bar it was planned to clear.
- **The event's range** after the burn, if its secondary was screened.
- **Coverage**: whether the event's secondary was screened at all (an
  analyst object is in no public catalog; `secondary_screened` is then
  false and the re-screen says nothing of the event) and how many catalog
  objects screening left out.

Each new and closer approach gets Foster's Pc on the
[synthetic covariance model](assumptions.md) of both element sets, an
assumed covariance, and a risk level; `conflict` is true when one is red,
and `clean` when there is no conflict and no event return.
The primary must be in the catalog and screened in both runs, and SGP4
must carry it through the whole window: otherwise `rescreen` raises
`ManeuverError`, so "nothing new" never stands for "nothing looked at".

The re-screen checks the options; it does not search for another one when
an option brings an approach. Weighing a burn that clears one event against
an approach it brings is left to the analyst, who can evaluate any burn
with `evaluate_burn` or `keepout maneuver --burn-before --dv` and re-screen
it.

## Time

Measured on an Intel Core i5-14600K, 3 runs each (2026-11-03): the default
grid on a LEO event 48 h before TCA (12 896 cells) took 29.9 to 31.2 s with
the numerical libraries on one thread and 33.4 to 46.1 s on their default
threads; most of it is the batched integration. On the report's Metop-B
event (25.1 h, 5 642 cells) it took 9.7 to 10.1 s on one thread and 12.3 to
16.2 s on the default threads (the report's own runs), at about twenty times the one-thread CPU
time. Setting
`OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS=1`) is therefore faster as
well as cheaper. A confirmation by Hall 3D-Nc adds 0.1 to 0.2 s per
candidate, and by Monte Carlo up to about a minute for a slow encounter. A
re-screen of the report's GEO burn against the public catalog took 4.2 s
with 4 workers in each of 3 runs.

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
