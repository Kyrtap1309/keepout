# Assumptions

keepout computes from the data it is given and says when the data is not
enough. In one place it must assume more: the [synthetic operator](synthetic.md)
gives GP element sets, which carry no covariance, a covariance from a model
of their errors, so that a Pc can be computed at all. This page states that
model, where its numbers come from, what it is for and what it cannot do,
and the synthetic operator's other assumptions. Nothing in those sections
applies to CDMs from an operational service, whose covariances come from
orbit determination. The last section states what the
[manoeuvre planner](maneuver.md) assumes, which applies to any CDM.

!!! warning "Not a calibrated product"
    The synthetic covariance is a documented assumption built from published
    error statistics, not a fit to the errors of the objects it is applied
    to. Pc values computed with it show what the model implies; they are not
    the risk of the real satellites, and no decision about a real satellite
    should rest on them.

## The covariance model

`keepout.synthetic.SyntheticCovarianceModel` gives an element set, at a time
`t` (the TCA of an approach), a diagonal **position** covariance in the
object's RTN frame (radial, transverse or along-track, normal or
cross-track):

```text
sigma_R = f sigma_R0
sigma_T = f (sigma_T0 + k |t - epoch|)
sigma_N = f sigma_N0
```

The radial and normal sigmas are constant and the along-track sigma grows
linearly with the element set's age, as GP errors do: along-track errors
dominate and grow, radial and cross-track errors grow much more weakly
(Flohrer et al. 2009; Geul et al. 2017). The age is `|t - epoch|`, so an
element set newer than `t` is covered too. The velocity block is zero.

| Regime | sigma_R0 | sigma_T0 | sigma_N0 | k |
|---|---|---|---|---|
| LEO | 0.102 km | 0.471 km | 0.126 km | 1.5 km/day |
| MEO | 0.073 km | 0.131 km | 0.054 km | 1.5 km/day |
| GTO | 1.960 km | 3.897 km | 1.808 km | 1.5 km/day |
| HEO | 0.824 km | 1.367 km | 1.059 km | 1.5 km/day |
| GEO | 0.359 km | 0.432 km | 0.086 km | 1.5 km/day |

- **Epoch sigmas** `sigma_R0`, `sigma_T0`, `sigma_N0`: Table 1 of Flohrer et
  al. (2009), the 1-sigma errors of the US catalog's element sets of
  2008-01-01 (11 286 objects) near their epoch, each against a numerical
  orbit fitted to its own SGP4 states over 24 hours.
- **Growth** `k`: the "typical growth of 1,500 m/day" that Levit and
  Marshall (2011) measured with SGP4 against precise orbits of laser-ranged
  satellites (three in LEO, Etalon-2 in MEO); their Table 2 spans 100 to
  3000 m/day. NASA's CA handbook (Rev. 2, Appendix E, after Hejduk et al.)
  reports 600 m to 3 km in the first 72 hours in LEO. No growth was
  measured for GTO, HEO or GEO; keepout applies the same rate there.
- **Safety factor** `f = 2`, on every sigma and so on the rate: in Flohrer et
  al.'s own validation against radar-tracked objects the true errors were
  about twice their estimates, and their GEO estimates "might be too
  optimistic". With it the along-track growth is 3 km/day, the top of
  Levit and Marshall's range.

With these numbers a LEO element set four days old has an along-track
sigma of about 13 km, and in the
[report's week](synthetic-operator.md#covariance-model) the along-track
sigmas at the TCAs range from 1.0 to 20.4 km. Every parameter can be
replaced:

```python
from astropy.time import Time

import keepout.omm
from keepout.synthetic import OrbitRegime, RegimeParameters, SyntheticCovarianceModel

catalog = keepout.omm.load("tests/screening/data/celestrak-public.csv.gz")
iss = next(omm for omm in catalog if omm.catalog_number == 25544)
when = Time("2026-10-10T00:00:00", scale="utc")
print("\n".join(SyntheticCovarianceModel().comments(iss, when)))
default = SyntheticCovarianceModel()
leo = RegimeParameters(50.0, 200.0, 50.0, transverse_rate=0.01)
mine = SyntheticCovarianceModel(
    regimes={**default.regimes, OrbitRegime.LEO: leo}, safety_factor=1.0
)
print(f"{mine.sigmas(iss, when).transverse:.0f} m")
```

```text
Synthetic covariance: keepout GP element-set age model, an assumption
sigma_T = f (sigma_T0 + k |t - epoch|); sigma_R, sigma_N constant
Regime LEO: sigma_R0 0.102 km, sigma_T0 0.471 km, sigma_N0 0.126 km
k 1.500 km/day, safety factor f 2
Element set epoch 2026-10-06T00:20:47.860512 UTC, age +3.986 days
Sigmas: R 0.204 km, T 12.899 km, N 0.252 km
```

```text
3644 m
```

The six comment lines are those each object of a synthetic CDM carries
after `COVARIANCE_METHOD = DEFAULT`, so the message itself says how its
covariance was made. `covariance(omm, time)` gives the `Covariance`
(RTN, in m², at `time`).

## Orbit regimes

The regimes and their bounds are Flohrer et al.'s, so that each regime's
sigmas apply to the kind of orbit they were measured on. keepout computes
perigee and apogee altitudes from SGP4's own mean elements (as for the
[screening volumes](screening.md#screening-volumes)):

| Regime | Perigee altitude | Apogee altitude |
|---|---|---|
| LEO | | below 2 000 km |
| GEO | above 33 786 km | below 37 786 km |
| MEO | above 2 000 km | below 33 786 km |
| GTO | below 2 000 km | above 33 786 km |
| HEO | any other orbit | |

The bounds are about altitude, not purpose, so some orbits land in a
regime their name does not suggest: Molniya orbits are GTO; inclined
geosynchronous orbits (IGSO) and inclined GEO are GEO; QZS-4 is HEO;
an eccentric LEO orbit with an apogee above 2 000 km (e.g. a fragment at
845 × 2 496 km) is HEO, with sigmas of about 1 km instead of LEO's 0.1 to
0.5 km. Of the 19 914 element sets of the frozen public catalog, 19 032 are
LEO, 182 MEO, 27 GTO, 87 HEO and 586 GEO.

## What the model is for

The model is for the **2-D Pc** (Foster, Elrod), which uses the position
covariance only. The velocity block is zero because the velocity
uncertainty is not modelled, and a zero velocity block tells a 3-D method
that the velocities are exact:

- `compute_pc(..., "auto")` keeps the 2-D Pc, with a note, when a 2-D
  assumption is questionable (a long or slow encounter) and a covariance is
  known only in position; with velocity blocks it would choose Hall 3D-Nc
  (see [Choosing a method](pc-3d.md#choosing-a-method)).
- Hall 3D-Nc and Monte Carlo, when used anyway (by name, or by `auto` at
  zero relative velocity, where no 2-D Pc exists), add a diagnostic that
  they take the velocity of such an object as exact.
- A CDM records the model as `COVARIANCE_METHOD = DEFAULT`; reading it
  back, `Cdm.assumed_covariance`, `ConjunctionEvent.assumed_covariance` and
  each `PcObservation.assumed_covariance` say that its Pc rests on it.
- Measured on approaches of the screening fixture (its 222 objects
  screened against each other for a week) with the model's covariances:
  Hall 3D-Nc and Monte Carlo accept the zero velocity block. On 8 sampled
  fast approaches (1 to 14 km/s) Hall agrees with Foster within 0.02 to
  7 %. On 8 sampled slow ones (0.2 to 1.1 m/s, mostly satellites flying in
  formation) Hall gives 1.3 to 2 times Foster's Pc on five, and on three
  its peak-overlap iteration does not converge (Monte Carlo fails on the
  same scan). Given a plausible velocity uncertainty instead, Hall
  converges there but gives 13 times Foster's Pc: a 3-D Pc of a slow
  encounter depends on the velocity uncertainty, which the model does not
  know.

`hall3d` and `montecarlo` can still be chosen explicitly; their result then
rests on exact velocities as well as on the assumed positions.

## What the Pc values mean

A Pc computed from an assumed covariance depends on the covariance's size
in two opposite ways, and the [report](synthetic-operator.md#approaches)
shows both for each approach with its maximum Pc over a scale factor of the
combined covariance (Alfano 2005):

- **Dilution region.** When the covariance is large against the miss
  distance, a smaller covariance (a better-known orbit) would *raise* the
  Pc. In the report's week, this holds for 2 of the 12 approaches, the two
  yellow ones (scale factors 0.72 and 0.10 at the maximum).
- **Far from the hard body.** For the other 10 the maximum needs a
  covariance 10 to 129 times larger: the miss lies far from the hard body
  against the covariance's narrow radial and normal sigmas (0.17 to
  0.72 km), so their tiny Pc values would rise steeply, up to 2.8e-5, if the
  covariance were larger.

Either way the decisions hold only as far as the model does.

## Limits

A zero-mean Gaussian of this shape cannot hold what is known about GP
errors:

- **Biases.** GP errors have clear biases, and the smallest error is not
  at the epoch (Kelso 2007; Geul et al. 2017).
- **Asymmetry.** Errors grow differently forward and backward in time
  from the epoch; the model uses `|t - epoch|` for both.
- **Object by object.** "Error characteristics for satellites in similar
  orbits can be considerably different" (Kelso 2007): one set of numbers
  per regime cannot know a particular object, its drag, its manoeuvres or
  its tracking.
- **Growth.** Linear growth at one rate for every regime; for drag-dominated
  low orbits the growth is expected to be quadratic (Geul et al. 2017), and
  space weather changes it.
- **Correlations.** The real errors are strongly correlated (along-track
  position with radial velocity); the model is diagonal and has no velocity
  block.

It is also not checked against any truth: Space-Track's public CDM
summaries (`cdm_public`, see [Space-Track data](spacetrack.md#public-cdms))
carry no covariance, list only emergency-reportable events and may not be
redistributed, so they cannot calibrate it.

## The synthetic operator's other assumptions

- **Hard-body radii.** The example fleet's radii (ISS 50 m, Metop-B 10 m,
  Intelsat 39 20 m) are illustrative round numbers, not their operators'
  values, and every catalog object is given 1 m; a fleet file must state
  every radius.
- **Issued at the start of the window.** Each CDM's creation date is the
  start of the window, so every approach is assessed as of that moment: an
  approach six days ahead counts as six days before TCA.
- **One event per approach, one CDM per event.** No Pc trend, so the
  decision rests on the risk level and the time to TCA.
- **GP data.** The states at TCA come from SGP4 and GP element sets, with
  the [limits of GP data](screening.md#limits-of-gp-data); the screening
  volume decides which approaches exist at all.

## Manoeuvre planning

The [manoeuvre planner](maneuver.md) answers "what does this burn do to
the event", from the CDM's states and covariances, under these
assumptions. The numbers below are measurements on sample cases, not
bounds.

- **The burn.** Impulsive, ideal (no execution error, no attitude or
  thrust profile) and along the primary's velocity only: no radial or
  normal burns, and no burn of the secondary. An along-track burn acts
  through the drift it builds up, so it loses effect close to TCA; the
  latest burn time is 6 hours before TCA for that reason, not because
  later burns are impossible.
- **The CDM's state is the truth.** The planner adds the change a burn
  makes at TCA to the CDM's state; it has no better orbit to start from.
- **Covariances unchanged.** Every cell uses the CDM's covariances of
  both objects, as if the burn added no uncertainty and the orbit
  determination did not change with time.
- **The target.** 1e-5 by default (`TARGET_PC`): keepout's default, one
  order of magnitude below `red_from` (1e-4), chosen as a margin, so that
  a burn leaves the event clearly out of the red rather than just under
  it. `--target-pc` or `target_pc` sets another.
- **Forces.** Zonal gravity to J6 (EGM2008) about a pole fixed for the
  integration. No tesseral terms, no third bodies (Moon, Sun) and no
  solar radiation pressure: these matter most at GEO, where lunisolar
  gravity and SRP dominate the perturbations, so a GEO plan rests on a
  simpler orbit than a LEO one.
- **Drag only with CD·A/m.** Drag is on only when the CDM gives the
  primary's `CD_AREA_OVER_MASS`; otherwise the plan says it is off. What
  that costs depends on the altitude. On one circular orbit each (CD·A/m
  = 0.02, burns up to 25 h before TCA, the default grid), switching drag
  on moved log10 Pc by at most 0.003 at 630 km, 0.03 at 500 km, 0.18 at
  400 km and 1.29 at 300 km; below a 450 km perigee the note without
  drag is a warning with these numbers.
- **The atmosphere.** An exponential density table by altitude band
  (Vallado, table 8-4), without solar activity or the day-night bulge;
  its density is a rough guess for any given day. The table's bands do
  not join exactly (up to 0.14 % at 25 km, 0.01 % or less above), and an
  orbit whose altitude swings across band edges integrates less accurately
  than the solver's tolerances suggest. Against tolerances 10 times
  tighter (rtol 1e-12), after 25 h the three eccentric orbits measured
  differed by 1.5 m (240 km, e = 0.005, CD·A/m = 0.05), 4.1 m (300 km,
  e = 0.01, CD·A/m = 0.02) and 9.5 m (400 km, e = 0.02, CD·A/m = 0.1),
  and a circular orbit at 300 km (CD·A/m = 0.02) by 0.03 m; drag itself
  moved these orbits by 170 to 2 650 km. The planner works on
  differences of orbits integrated together, in which most of this error
  cancels.
- **Foster on the grid.** Every cell uses Foster's 2-D Pc, whatever the
  event's own method, for speed (0.1 ms per cell against 0.1 to 0.2 s for
  Hall 3D-Nc). The options are confirmed with the event's own method, so
  a cell Foster puts below the target that the method does not is never
  an option; the rest of the grid, and the heatmap, are Foster's.
- **Slow encounters.** Each cell's new closest approach comes from a
  second-order expansion about TCA, and the planner looks for it only
  within a twentieth of an orbit of TCA. A cell where the expansion has
  no true minimum there gets no Pc and is never an option; the plan
  counts these cells. In a slow encounter a burn can bring the pair back
  hours later, outside that window, at a Pc no cell shows: only a
  re-screen shows it, as an event return. Below 100 m/s relative speed
  the plan notes it. Inside the window the new TCA of a cell was measured
  up to 28 m off at 10 m/s among the cells sampled, against covariances
  of hundreds of metres.
- **Re-screen accuracy.** The burned orbit of a re-screen is SGP4 plus a
  Cowell difference. Against a Cowell integration of the whole orbit
  after the burn, at 1 m/s over 7 days, the difference reached 2.24 m at
  worst on the 38 fixture orbits with an eccentricity above 0.3, and
  below 0.3 0.09 m on 68377 (a circular LEO) and 0.82 m on 89251
  (e = 0.28, perigee near 500 km); other orbits are not covered by these
  numbers. Screening's volume is kilometres.
- **Re-screen Pc assumed.** The catalog's element sets have no covariance,
  so the Pc of a new or closer approach, by `auto` as for the synthetic
  operator, rests on the synthetic covariance model above, with all its limits, and is labelled assumed wherever it is
  shown. The re-screen also sees only the objects in the catalog given:
  the public catalog lacks most debris (see
  [Screening](screening.md)).

## References

- T. Flohrer, H. Krag, H. Klinkrad, B. Bastida Virgili, C. Früh, "Improving
  ESA's collision risk estimates by an assessment of the TLE orbit errors
  of the US SSN catalogue", *Proc. 5th European Conference on Space
  Debris*, ESA SP-672, 2009.
- C. Levit, W. Marshall, "Improved orbit predictions using two-line
  elements", *Advances in Space Research* 47, 2011 (arXiv:1002.2277).
- NASA, *Spacecraft Conjunction Assessment and Collision Avoidance Best
  Practices Handbook*, NASA/SP-20205011318 Rev. 2, Appendix E.
- T. S. Kelso, "Validation of SGP4 and IS-GPS-200D Against GPS Precision
  Ephemerides", AAS 07-127, 2007.
- J. Geul, E. Mooij, R. Noomen, "TLE uncertainty estimation using robust
  weighted differencing", *Advances in Space Research* 59(10), 2522-2535,
  2017.
- S. Alfano, "Relating Position Uncertainty to Maximum Conjunction
  Probability", *Journal of the Astronautical Sciences* 53(2), 193-205,
  2005.
- D. A. Vallado, *Fundamentals of Astrodynamics and Applications*,
  Microcosm Press, table 8-4 (exponential atmosphere).
