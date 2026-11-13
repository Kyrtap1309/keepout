# Synthetic covariance model

keepout computes from the data it is given and says when the data is not
enough. In one place it must assume more. GP element sets carry no
covariance, so the [synthetic operator](synthetic.md) gives each one a
covariance from a model of GP errors. Without it no Pc could be computed.

This page states that model, where its numbers come from, what it is for
and what it cannot do. It also lists the synthetic operator's other
assumptions. None of this applies to CDMs from an operational service,
whose covariances come from orbit determination. The manoeuvre planner
states its own assumptions under
[Manoeuvre planning](maneuver.md#assumptions-and-limits).

!!! warning "Not a calibrated product"
    The synthetic covariance is a documented assumption built from published
    error statistics, not a fit to the errors of the objects it is applied
    to. Pc values computed with it show what the model implies. They are not
    the risk of the real satellites, and no decision about a real satellite
    should rest on them.

## The covariance model

[`SyntheticCovarianceModel`](reference/synthetic.md#keepout.synthetic.SyntheticCovarianceModel)
gives an element set a
diagonal **position** covariance at a time `t`, such as the TCA of an
approach. The covariance is in the object's RTN frame: radial, transverse
(along-track) and normal (cross-track).

```text
sigma_R = f sigma_R0
sigma_T = f (sigma_T0 + k |t - epoch|)
sigma_N = f sigma_N0
```

The radial and normal sigmas are constant. The along-track sigma grows
linearly with the element set's age, as GP errors do. Along-track errors
dominate and grow, while radial and cross-track errors grow much more
weakly (Flohrer et al. 2009; Geul et al. 2017). The age is `|t - epoch|`,
so an element set newer than `t` is covered too. The velocity block is
zero.

| Regime | sigma_R0 | sigma_T0 | sigma_N0 | k |
|---|---|---|---|---|
| LEO | 0.102 km | 0.471 km | 0.126 km | 1.5 km/day |
| MEO | 0.073 km | 0.131 km | 0.054 km | 1.5 km/day |
| GTO | 1.960 km | 3.897 km | 1.808 km | 1.5 km/day |
| HEO | 0.824 km | 1.367 km | 1.059 km | 1.5 km/day |
| GEO | 0.359 km | 0.432 km | 0.086 km | 1.5 km/day |

- **Epoch sigmas** `sigma_R0`, `sigma_T0`, `sigma_N0` come from Table 1 of
  Flohrer et al. (2009). They are the 1-sigma errors near the epoch of the
  US catalog's element sets of 2008-01-01 (11 286 objects). Each element
  set was compared with a numerical orbit fitted to its own SGP4 states over
  24 hours.
- **Growth** `k` is the "typical growth of 1,500 m/day" that Levit and
  Marshall (2011) measured. They compared SGP4 with precise orbits of
  laser-ranged satellites (three in LEO, Etalon-2 in MEO). Their Table 2
  spans 100 to 3000 m/day. NASA's CA handbook (Rev. 2, Appendix E, after
  Hejduk et al.) reports 600 m to 3 km in the first 72 hours in LEO. No
  growth was measured for GTO, HEO or GEO, so keepout applies the same rate
  there.
- **Safety factor** `f = 2` applies to every sigma, and so to the rate. In
  Flohrer et al.'s own validation against radar-tracked objects, the true
  errors were about twice their estimates. They also note that their GEO
  estimates "might be too optimistic". With the factor, the along-track
  growth is 3 km/day, the top of Levit and Marshall's range.

With these numbers a LEO element set four days old has an along-track
sigma of about 13 km. The
[synthetic operator report](synthetic-operator.md#covariance-model) gives
the along-track sigmas at the TCAs of its week. Every parameter can be
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

Each object of a synthetic CDM carries these six comment lines after
`COVARIANCE_METHOD = DEFAULT`, so the message itself says how its
covariance was made.
[`covariance(omm, time)`](reference/synthetic.md#keepout.synthetic.SyntheticCovarianceModel.covariance)
gives the `Covariance` at
`time`, in RTN and in m².

## Orbit regimes

The regimes and their bounds are Flohrer et al.'s, so that each regime's
sigmas apply to the kind of orbit they were measured on. keepout computes
perigee and apogee altitudes from SGP4's own mean elements, as for the
[screening volumes](screening.md#screening-volumes). The rows are tried in
order, and the first that fits gives the regime:

| Regime | Perigee altitude | Apogee altitude |
|---|---|---|
| LEO | any | below 2 000 km |
| GEO | above 33 786 km | below 37 786 km |
| MEO | above 2 000 km | below 33 786 km |
| GTO | below 2 000 km | above 33 786 km |
| HEO | any | any |

HEO takes every orbit the rows above it do not, including an altitude
exactly on a bound.

The bounds are about altitude, not purpose, so some orbits land in a
regime their name does not suggest:

- Molniya orbits are GTO.
- Inclined geosynchronous orbits (IGSO) and inclined GEO are GEO.
- QZS-4 is HEO.
- An eccentric LEO orbit with an apogee above 2 000 km (e.g. a fragment at
  845 × 2 496 km) is HEO. Its sigmas are then about 1 km instead of LEO's
  0.1 to 0.5 km.

The repository keeps a copy of CelesTrak's public GP catalog, downloaded
on 2026-10-06 and never updated since
(`tests/screening/data/celestrak-public.csv.gz`), so that the tests and
reports repeat exactly. Of its 19 914 element sets, 19 032 are LEO, 182
MEO, 27 GTO, 87 HEO and 586 GEO.

## What the model is for

The model is for the **2-D Pc** (Foster, Elrod), which uses the position
covariance only. The velocity block is zero because the velocity
uncertainty is not modelled. A zero velocity block tells a 3-D method that
the velocities are exact, so keepout guards against it:

- [`compute_pc(..., "auto")`](reference/pc.md#keepout.pc.compute_pc) keeps the 2-D Pc, with a note, when a 2-D
  assumption is questionable (a long or slow encounter) and a covariance is
  known only in position. With velocity blocks it would choose Hall 3D-Nc
  (see [Choosing a method](pc-3d.md#choosing-a-method)).
- Hall 3D-Nc and Monte Carlo can still be used, by name or by `auto` at
  zero relative velocity, where no 2-D Pc exists. They then add a
  diagnostic that they take the velocity of such an object as exact.
- A CDM records the model as `COVARIANCE_METHOD = DEFAULT`. Read back,
  [`Cdm.assumed_covariance`](reference/cdm.md#keepout.cdm.Cdm.assumed_covariance),
  [`ConjunctionEvent.assumed_covariance`](reference/events.md#keepout.events.ConjunctionEvent.assumed_covariance)
  and each
  `PcObservation.assumed_covariance` say that its Pc rests on it.

A 3-D Pc of a slow encounter depends on the velocity uncertainty, which the
model does not know. A result of `hall3d` or `montecarlo` chosen explicitly
therefore rests on exact velocities as well as on the assumed positions.

??? details "How much the zero velocity block matters"
    The effect was measured on approaches between the 222 objects of the
    screening fixture (`tests/screening/data/celestrak-fixture.csv`, a
    sample of the copied catalog from LEO to HEO that the screening tests
    use). They were screened against each other for a week with the model's
    covariances. Hall 3D-Nc and Monte Carlo accept the zero velocity block.

    - On 8 sampled fast approaches (1 to 14 km/s), Hall agrees with Foster
      within 0.02 to 7 %.
    - On 8 sampled slow ones (0.2 to 1.1 m/s, mostly satellites flying in
      formation), Hall gives 1.3 to 2 times Foster's Pc on five. On the
      other three its peak-overlap iteration does not converge, and Monte
      Carlo fails on the same scan.
    - Given a plausible velocity uncertainty instead, Hall converges on
      those three but gives 13 times Foster's Pc.

## What the Pc values mean

A Pc computed from an assumed covariance depends on the covariance's size
in two opposite ways. The maximum Pc over a scale factor of the combined
covariance (Alfano 2005) shows which way applies:

- **Dilution region.** When the covariance is large against the miss
  distance, a smaller covariance (a better-known orbit) would *raise* the
  Pc. The maximum then needs a scale factor below 1.
- **Far from the hard body.** When the miss lies far from the hard body
  against the covariance's narrow directions (its radial and normal
  sigmas), the Pc is tiny, and it would rise steeply if the covariance were
  larger. The maximum then needs a scale factor above 1.

The [synthetic operator report](synthetic-operator.md#approaches) gives
both for each of its approaches: the maximum Pc and its scale factor.
Either way the decisions hold only as far as the model does.

## Limits

A zero-mean Gaussian of this shape cannot hold what is known about GP
errors:

- **Biases.** GP errors have clear biases, and the smallest error is not
  at the epoch (Kelso 2007; Geul et al. 2017).
- **Asymmetry.** Errors grow differently forward and backward in time
  from the epoch. The model uses `|t - epoch|` for both.
- **Object by object.** "Error characteristics for satellites in similar
  orbits can be considerably different" (Kelso 2007). One set of numbers
  per regime cannot know a particular object, its drag, its manoeuvres or
  its tracking.
- **Growth.** The model grows linearly, at one rate for every regime. For
  drag-dominated low orbits the growth is expected to be quadratic (Geul et
  al. 2017), and space weather changes it.
- **Correlations.** The real errors are strongly correlated (along-track
  position with radial velocity). The model is diagonal and has no
  velocity block.

The model is also not checked against any truth. Space-Track's public CDM
summaries (`cdm_public`, see [Space-Track data](spacetrack.md#public-cdms))
cannot calibrate it. They carry no covariance, list only
emergency-reportable events and may not be redistributed.

## The synthetic operator's other assumptions

- **Hard-body radii.** The example fleet's radii (ISS 50 m, Metop-B 10 m,
  Intelsat 39 20 m) are illustrative round numbers, not their operators'
  values. Every catalog object is given 1 m. A fleet file must state every
  radius.
- **Issued at the start of the window.** Each CDM's creation date is the
  start of the window, so every approach is assessed as of that moment. An
  approach six days ahead counts as six days before TCA.
- **One event per approach, one CDM per event.** There is no Pc trend, so
  the decision rests on the risk level and the time to TCA.
- **GP data.** The states at TCA come from SGP4 and GP element sets, with
  the [limits of GP data](screening.md#limits-of-gp-data). The screening
  volume decides which approaches exist at all.

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
