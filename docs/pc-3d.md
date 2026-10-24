# 3-D, Monte Carlo and maximum probability of collision

The [2-D methods](pc-2d.md) assume straight-line motion and constant
covariances during a short encounter. keepout adds three kinds of method for
when that does not hold, or when a covariance is missing:

| Method | Function | For |
|---|---|---|
| Hall 3D-Nc | `hall3d` | slow, long or repeated encounters (curved motion, growing uncertainty) |
| Monte Carlo | `montecarlo` | an independent, assumption-light estimate with a confidence interval |
| Frisbee | `frisbee` | one covariance unknown: the maximum over it |
| Dilution | `dilution` | both covariances known but perhaps too large: the maximum over their scale |
| Alfano | `alfano` | no covariance at all: the maximum over every covariance |

## Two-body motion in equinoctial elements

Both 3-D methods move the objects by two-body motion around TCA and describe
each object's uncertainty as a Gaussian in **equinoctial elements**
`(n, af, ag, chi, psi, lambda_M)` rather than in Cartesian space (Hall 2021,
sec. IV). Under two-body motion only the mean longitude changes,
`lambda_M(t) = lambda_M(TCA) + n t`, and a sample displaced along track stays
on its curved orbit. A Cartesian Gaussian would move it along the tangent
instead: with the 238 km along-track sigma of one CARA test conjunction, a
sample 180 km along the tangent sits 2.3 km above the orbit, and a Cartesian
Monte Carlo finds no collisions at all where CARA's finds 1.4e-4.

The elements are non-singular for circular and equatorial orbits; a
retrograde factor `fr = -1` handles `i = 180 deg` (Broucke & Cefola 1972).
`to_equinoctial` raises `GeometryError` for an unbound orbit, a rectilinear
state (zero angular momentum) and an equatorial orbit whose retrograde factor
puts it on the singularity. Kepler's equation is solved by Newton's method
from Danby's starting value, which converges for every eccentricity below 1;
if it does not converge (e.g. NaN elements), `GeometryError` is raised.
Jacobians are taken by the complex step, exact to machine precision
(`complex_step` raises `TypeError` if the function returns a real
array, which would give a zero Jacobian).
Both methods need elliptic orbits and 6x6 covariances; a covariance with a
zero velocity block is allowed.

## Hall 3D-Nc

Hall (2021) writes the expected number of collisions `Nc` as the time
integral of the rate at which the secondary's centre enters the hard-body
sphere of radius `R` around the primary:

$$
\dot N_c(t) = R^2 \oint \mathcal N(R\hat r;\ \breve r, \tilde A)\,
\nu(\hat r)\, d\hat r,
\qquad
\nu = \frac{\sigma}{\sqrt{2\pi}}\left[e^{-\tilde\nu^2}
      - \sqrt\pi\,\tilde\nu\,\operatorname{erfc}\tilde\nu\right],
$$

with `r` and `A` the mean and covariance of the relative position and `sigma`,
`nu~` from the relative velocity conditioned on the position (eqs. 51e, 52).
The Gaussians come from a first-order expansion of the two-body motion,
centred not on the mean orbits but on the states at the **peak overlap** of
the two position densities, found by iteration at each time (sec. VII.B;
convergence when the squared Mahalanobis step of the peak is below 1e-6).
Where the gap between the mean positions, less the radius, exceeds
`sqrt(1500)` (about 39) of the largest standard deviations of `A`, the rate
underflows and is taken as 0 without iterating.
Because this Gaussian is linear in the elements about the peak-overlap
states, not in Cartesian position, its curved map brings the far tail
slightly closer: with the miss `k` standard deviations out, `hall3d` exceeds
the 2-D Pc of even a fast, straight-line encounter by a relative
`4e-6 k^3` or so in LEO (7e-5 at 3 sigma, 0.4% at 10 sigma), scaling as
`miss^3 / r` with `r` the orbit radius. This is an intended consequence of
the curvilinear Gaussian, not a numerical error; the mean-centred expansion
does not show it.
Where the iteration fails, keepout falls back to the mean-centred expansion
only in the far tail: a failure inside an encounter segment, or within one
scan step of one, raises `PcConvergenceError` whatever the mean-centred rate
there, and elsewhere so does a failure where that rate exceeds 1e-10 of the
peak.

**Sphere integral.** When the smallest standard deviation of `A` is much
smaller than `R` (the Alfano cases have 0.14 m against a 10 m radius), the
density on the sphere is a thin band or small blobs that a fixed rule —
including CARA's 5810-point Lebedev rule — under-resolves. keepout puts the
pole on the narrowest axis: Gauss-Legendre panels in `theta` cluster around
the band `R cos(theta) = r . e1`, and a periodic trapezoid rule in `phi` has a
spacing set by the second standard deviation.

**Time integral.** A cheap screen on the mean trajectories first marks the
stretches of the window where the rate can be non-zero: cell by cell, a lower
bound on the gap between the mean positions is compared with the largest
position variance nearby, and cells where the rate is certainly 0 are never
evaluated. Each remaining stretch is scanned on a detection grid with a
spacing of 6.07 time standard deviations of the rate (`DETECTION_SIGMAS`),
fine enough that no encounter whose peak exceeds 1e-8 of the largest falls
between two nodes; the grid is then refined to an eighth of a straight-line
crossing time, but only around the nodes where the rate is significant.
Stretches where the rate exceeds 1e-16 of its peak become encounter segments,
split at their maxima and integrated by nested Clenshaw-Curtis rules (each
doubling reuses the previous nodes) whose order doubles until two agree to
`rtol` (1e-6 by default; it must be finite and positive), or until their
difference, already below 1e-4, stops halving: the rate itself is only
resolved to about 1e-5 by its sphere quadrature. Without an explicit
`window`, the window covers TCA and the straight-line closest approach of the
states, widened by the straight-line bound, which doubles until the rate at the edges is negligible; the window is
never wider than the shorter orbital period, so the next revolution's
encounter is not counted twice. `hall3d` raises `PcConvergenceError`
naming the window only when detecting its encounters would need more than
20 001 grid points; pass a shorter window then.

**Diagnostics.** `hall3d` reports a rate negligible over the whole window
(`Nc = 0`), `Nc > 1` (Pc set to 1), several encounters (below), a rate still
above 1e-16 of its peak at the edge of an automatic window, and a time
integral whose worst piece agrees to worse than both `rtol` and 1e-4 of `Nc`
(`QUIET_DISAGREEMENT`; smaller misses cannot affect a result at the validation
accuracy).

**Several encounters, long encounters.** `Nc` counts every entry into the
hard-body sphere, so it is an upper bound on Pc, the probability of at least
one entry. For a single short, straight-line encounter each sample enters at
most once and `Nc = Pc`. With several segments in the window, `Nc` is an
upper bound and the largest segment's `Nc` a lower bound on Pc; keepout
returns `min(Nc, 1)` and says so. In a long or zero-relative-velocity
encounter a sample can leave and re-enter, so even one segment's `Nc`
exceeds Pc: for two objects 10 m apart on the same circular orbit (5 m
sigmas, 5 m radius, 120 s), `Nc` = 1.50e-4 matches the expected number of
entries of a brute-force simulation (1.65e-4) while the probability of a
first entry is about 8e-5. `hall3d` adds a diagnostic when the relative
velocity is zero or the encounter lasts longer than the applicability limit
(1% of the primary's period, as for the 2-D warning): "long encounter: Nc
counts repeated entries into the hard-body sphere, so it is an upper bound on
Pc".

## Monte Carlo

Each sample draws both TCA states from their equinoctial Gaussians and moves
them by two-body motion. A sample is a collision when the relative distance
**first drops below** `R` inside the window; samples already overlapping at the
window start are not collisions in the window (CARA's convention: for Alfano
case 9 they are 8.4% of the samples, matching Alfano's "instantaneous
probability" 0.0842, and CARA's 0.280 excludes them).

Time is searched only inside the encounter segments of the 3D-Nc rate scan:
outside them the collision rate is below 1e-16 of its peak. On the scan
grid, a lower bound on each sample's distance between two points selects the
samples whose closest approach is refined by Newton's method on `r.v = 0`
(slope `d(r.v)/dt = v.v + r.a`, with `a` the two-body relative
acceleration). Because it
uses that scan, `montecarlo` raises `PcConvergenceError` when the scan does
(the rate cannot be computed, or its encounters cannot be detected within
the point budget).

Samples run in batches of 50 000 until the Clopper-Pearson interval
(99% by default) has a half-width below `rtol = 0.1` of the estimate, or the
budget of `max_samples = 1e7` is spent; `samples=` fixes the count and
`seed=` makes a run reproducible. The interval is the result's `interval`
(`low`, `high`), its level `details["confidence"]`.

## Maximum Pc

All three are **upper bounds**, not estimates: their results have
`kind == "upper_bound"` (the other methods' `"estimate"`), and their
diagnostics say so.

- **Frisbee** (one covariance unknown). The unknown covariance is taken as a
  variance `s >= 0` added along the miss direction `u = m/|m|` to the known
  projected covariance `C_k`, and the value is the maximum of the Foster Pc
  over `s`. For a small hard body the density at the miss vector `m` under
  `C_k + s u u^T` is largest at `s = |m|^2 (K^2 - 1) / K^2`,
  `K^2 = m^T C_k^{-1} m` (Frisbee's analytic optimum). It is the Pc optimum
  only for a small hard body, so it seeds a bounded search over `log s`
  (`SEARCH_DECADES` = 4 decades either side). The seed is chosen in two
  steps, each with the margin `SEARCH_GAIN_RTOL` = 1e-8 relative: `s = 0`
  is kept unless the analytic `s` beats its Pc by more than that margin, and
  the point kept is then replaced by the search's maximum only if that beats
  it by the same margin. A small-body result therefore reports Frisbee's `s`
  exactly; the Pc at the analytic `s` is kept in
  `details["pc_analytic"]`. Random positive semi-definite additions to `C_k`
  find nothing larger.
- **Dilution** (both known). The maximum of Pc over a scale factor `k` of the
  combined covariance, by the same bounded search seeded at the small-body
  optimum `k = K^2/2`; the nominal `k = 1` is reported unless a scale beats
  its Pc by more than 1e-8 relative. The conjunction is *diluted* when the
  maximum needs a smaller covariance (`k < 1`) and exceeds the nominal Pc by
  more than 0.1% (`result.diluted`, `None` for the other methods). With the
  miss inside the hard body the supremum is 1, as the covariance shrinks to
  zero.
- **Alfano** (none known). Over every 2-D covariance, the maximum for miss
  distance `d` is reached by a line-like covariance along the miss vector
  with `sigma^2 = 2 d R / ln((d+R)/(d-R))` (Alfano 2005); for `d >> R` it is
  `(R/d) sqrt(2/(pi e))`. Inside the hard body (`d < R`) the supremum is 1,
  and on its edge (`d = R`) exactly 1/2, both reached only as the covariance
  vanishes.

A CDM whose 6x6 covariance is all zero gives an object with an unknown
covariance (operators send zeros when they have none).

## Choosing a method

`compute_pc(conjunction)` (`method="auto"`):

1. zero relative velocity: `hall3d` (there is no encounter plane, so a
   `GeometryError` names the cause if a covariance is unknown or the 3-D
   methods cannot use an object's orbit, e.g. one that is not elliptic);
2. both covariances unknown: `alfano`; one unknown: `frisbee`;
3. a questionable 2-D assumption (encounter longer than 1% of an orbit, or
   states off TCA), with elliptic orbits and both covariances known in
   velocity too: `hall3d`, with the 2-D Pc kept in `details["pc_2d"]`;
4. otherwise `foster`, with a note when a 2-D assumption is questionable
   but a covariance is known only in position (a zero velocity block, as
   keepout's [synthetic covariances](assumptions.md#what-the-model-is-for)):
   the 3-D methods would take that object's velocity as exact.

However chosen (by `auto` at zero relative velocity, or by name), `hall3d`
and `montecarlo` add a diagnostic when a covariance is known only in
position: they take that object's velocity as exact.

If `hall3d` raises `PcConvergenceError` after its encounter scan (the
peak-overlap iteration failed where the rate matters, the rate cannot be
computed at a quadrature node, or `Nc` is not finite), `auto` falls back to
`montecarlo` with at most `AUTO_MC_SAMPLES` (1 000 000) samples and the fixed
seed `AUTO_MC_SEED` (0), so the same conjunction always gives the same
result; the diagnostic says so. The fallback reuses Hall's encounter scan
rather than repeating it: the samples are searched in the segments Hall
found. A failure of the scan itself (its encounters need more than 20 001
detection points, the rate is significant where the screen ruled it
negligible, or it cannot be computed on the scan grid) is not covered: Monte
Carlo has no scan to reuse, and would need the same one, so the
`PcConvergenceError` propagates from `compute_pc` even in `auto` mode.

The applicability warnings (e.g. "states are ... m from closest approach
along the relative velocity; 2-D Pc uses the straight-line closest
approach") are attached, in `auto` and with an explicit method alike,
whenever the encounter plane is defined (non-zero relative velocity) and
they fire, also for the maximum-Pc methods with a covariance unknown: the
distance from closest approach along the relative velocity does not depend
on the covariances, and a duration
computed from the known covariance alone is a lower bound on the true one,
so a duration warning is still valid (it then says "at least about").
`details["encounter_duration_s"]` is attached only when both covariances are
known.

The choice costs time: the 2-D and maximum-Pc methods take milliseconds,
`hall3d` about 1-60 s and the Monte Carlo fallback up to about a minute. For a
batch of conjunctions, look at `result.method` or choose a method explicitly:
`computation("foster")`, or `computation("montecarlo", samples=100_000,
seed=1)` for a method with options, is a `PcComputation` to call on each
conjunction (its options are checked once, when it is made).
`keepout.pc.METHODS` says what each method is (`kind`: `"2d"`, `"3d"`,
`"montecarlo"` or `"maxpc"`) and which options it takes.

## Slow encounters: the methods compared

The twelve Alfano (2009) test cases are slow or long encounters built to break
the 2-D model. From the [validation report](validation.md), with CARA's
two-body Monte Carlo (1e8 trials) for comparison; keepout's own Monte Carlo
estimates and their 99% intervals are in the
[Monte Carlo report](validation-monte-carlo.md):

| Case | 2-D Pc (Foster) | 3-D Pc (Hall) | Monte Carlo, CARA |
|---|---|---|---|
| 1 | 0.1467 | 0.2168 | 0.2169 |
| 2 | 0.006222 | 0.01556 | 0.01555 |
| 3 | 0.1004 | 0.1004 | 0.1003 |
| 4 | 0.04932 | 0.07364 | 0.07364 |
| 5 | 0.04449 | 0.04449 | 0.04447 |
| 6 | 0.004335 | 0.004335 | 0.004324 |
| 7 | 1.581e-4 | 1.618e-4 | 1.601e-4 |
| 8 | 0.03695 | 0.03525 | 0.03524 |
| 9 | 0.2902 | 0.2798 | 0.2798 |
| 10 | 0.2902 | 0.3641 | 0.3640 |
| 11 | 0.002672 | 0.002441 | 0.002436 |
| 12 | undefined | 0.002442 | 0.002423 |

The 2-D Pc is off by up to a factor 2.5 (case 2) and cannot express case 12
(zero relative velocity) or the window of cases 9 and 10 (same states,
different windows); Hall's 3D-Nc and the Monte Carlo estimates agree.

## Validation

The [validation report](validation.md) compares Hall with CARA's 3D-Nc on the
twelve Alfano (2009) cases, CARA's 53-CDM test set and two unit-test cases,
and the maximum-Pc methods with CARA's Frisbee and dilution unit tests. The
[Monte Carlo report](validation-monte-carlo.md) (checked nightly) checks
that CARA's two-body Monte Carlo value lies within keepout's interval.

**Tolerances.** CARA's own `Pc3D_Hall_UnitTest.m` reproduces its Alfano 3-D
values to 1e-3, which is the criterion here; keepout agrees to 5.5e-4 or
better (7.1e-4 on the two unit-test cases). On the 53-CDM set CARA's `Nc3D`
sits systematically above its own `Nc2D` where the encounter is nearly
straight-line and the two should agree. On the 18 cases where keepout's
3D-Nc and 2-D Pc agree within 1e-3, CARA's `Nc3D / Nc2D - 1` averages
+8.2e-4 (all 18 lie between +7.6e-4 and +8.7e-4; against its `Pc2D`, 15 of
the 18 lie between +6.7e-4 and +9.1e-4), while keepout's
Hall / Foster - 1 is spread from -7.7e-4 to +7.0e-4 around a mean of
-3e-5 (median magnitude 2.3e-4; within 1e-5 in 3 cases). keepout agrees
with CARA's `Nc3D` to 2.1e-3 (criterion 3e-3). Frisbee agrees with
Frisbee's published example to 5e-7 absolute (CARA's criterion) and dilution
with CARA's unit tests to 1.0e-3 relative (criterion 5e-3).

## References

- Hall, D. T. (2021). Expected collision rates for tracked satellites.
  *Journal of Spacecraft and Rockets* 58(3), 715-728.
  doi:10.2514/1.A34919.
- Broucke, R. A. and Cefola, P. J. (1972). On the equinoctial orbit elements.
  *Celestial Mechanics* 5, 303-310.
- Vallado, D. A. and Alfano, S. (2015). Updated analytical partials for
  covariance transformations and optimization. AAS/AIAA Astrodynamics
  Specialist Conference, AAS 15-537 (equinoctial elements and their
  conversions).
- Danby, J. M. A. (1987). The solution of Kepler's equation, III.
  *Celestial Mechanics* 40, 303-312.
- Alfano, S. (2005). Relating position uncertainty to maximum conjunction
  probability. *Journal of the Astronautical Sciences* 53(2), 193-205.
- Frisbee, J. H. (2015). An upper bound on high speed satellite collision
  probability when only one object has position uncertainty information.
  AAS/AIAA Astrodynamics Specialist Conference.
- Hejduk, M. D. et al. (2019). Satellite conjunction assessment risk analysis for
  "dilution region" events: issues and operational approaches. Space Traffic
  Management Conference, paper 28.
- Alfano, S. (2009). Satellite conjunction Monte Carlo analysis. AAS 09-233.
- Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial
  limits illustrated in the case of the binomial. *Biometrika* 26, 404-413.
