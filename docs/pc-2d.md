# 2-D probability of collision

The probability of collision (Pc) is the chance that two objects come
closer than the sum of their sizes, given the uncertainty of their orbits.
It is what an operator decides on: a miss distance of 200 m means little
without knowing whether the positions are uncertain by metres or by
kilometres. [Concepts](concepts.md) puts it in context.

For most conjunctions in low Earth orbit the objects pass each other within
seconds, at kilometres per second. The 2-D methods on this page are
then exact enough and take milliseconds. `compute_pc` uses them unless the
encounter is slow or a covariance is missing
([3-D, Monte Carlo and maximum Pc](pc-3d.md) covers those cases).

```python
import keepout.cdm
from keepout import compute_pc

cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")
conjunction = cdm.conjunction(combined_hard_body_radius=20.0)  # [m]
result = compute_pc(conjunction, "foster")
print(f"{result.value:.2e}", result.method, result.kind)
print(result.diagnostics)
auto = compute_pc(conjunction)  # this CDM's states are off TCA: auto picks hall3d
print(auto.method, f"{auto.value:.2e}")
```

```text
4.50e-02 foster estimate
('states are 300 m from closest approach along the relative velocity; 2-D Pc uses the straight-line closest approach',)
hall3d 4.50e-02
```

A `PcResult` holds:

- `value`: the Pc, in [0, 1];
- `method`: the method that gave it, which `"auto"` reports too;
- `kind`: `"estimate"`, or `"upper_bound"` for a maximum Pc;
- `diagnostics`: warnings about the value, such as a repaired covariance or
  a questionable 2-D assumption; empty when there is nothing to report;
- `details`: method-specific numbers, e.g. Foster's `error_estimate`.

This page's CDM gives states 300 m from the straight-line closest
approach, so Foster warns and `"auto"` uses Hall's 3-D method instead
([Choosing a method](pc-3d.md#choosing-a-method)).

A conjunction need not come from a CDM: `Conjunction(primary, secondary,
combined_hard_body_radius)` takes two `ObjectState`s of your own, in GCRF
([Working with CDMs](cdm.md#building-a-cdm) builds them from states and
RTN covariances). keepout never assumes a hard-body radius.
`cdm.conjunction()` without `combined_hard_body_radius` adds both objects'
`HBR` (CDM 2.0), and raises `HardBodyRadiusError` when an object has none.

## Model

For a short-duration encounter the two objects move on straight lines near
the time of closest approach (TCA) and their position errors are Gaussian and
constant. The collision probability is then the mass of a 2-D Gaussian inside
a circle on the **encounter plane**, the plane perpendicular to the relative
velocity (Foster & Estes 1992; Chan 2008):

- the Gaussian has mean equal to the miss vector and covariance equal to the
  sum of both objects' position covariances, projected onto the plane;
- the circle is centred on the primary and has the **combined hard-body
  radius** (HBR): the radius of a sphere that the secondary's centre must enter
  for a collision.

keepout projects the relative position onto the encounter plane. If the
states are not exactly at TCA, this gives the straight-line closest
approach. If the states are more than 1% of the relative distance from that
point, `compute_pc` adds a diagnostic.

## Reduction to one dimension

In the principal axes of the plane covariance (wide axis `u`, narrow axis `v`)
the density separates, and the 2-D integral over the circle becomes

$$
P_c = \int_{-R}^{R} \frac{1}{\sqrt{2\pi}\,\sigma_u}
      e^{-\frac{(u-\mu_u)^2}{2\sigma_u^2}}\;
      \tfrac12\left[\operatorname{erf}\frac{h(u)-\mu_v}{\sqrt2\,\sigma_v}
      - \operatorname{erf}\frac{-h(u)-\mu_v}{\sqrt2\,\sigma_v}\right] du,
\qquad h(u) = \sqrt{R^2-u^2}.
$$

The error-function difference is evaluated with `erfc` on the far side of the
mean, so small probabilities keep their relative precision.

## Methods

| Method | How | Default accuracy |
|---|---|---|
| `foster` | Adaptive Gauss-Kronrod quadrature (QUADPACK) of the 1-D integral with `u = R sin(theta)`, within 12 sigma of the mean | relative tolerance 1e-10 |
| `elrod` | Gauss-Chebyshev quadrature (second kind) of the same integral, at increasing orders | relative 1e-10 between orders |

**Foster** integrates over the band within 12 sigma of the mean. Outside it
the density is below `exp(-72)` of its peak. A circle entirely outside the
band gives exactly 0, where Elrod may still return e.g. 3e-58. Breakpoints
are placed where a thin narrow axis makes the chord mass a step.

**Elrod** starts at the order where the nodes resolve the Gaussian, at least
63. It steps along the nested sequence `n -> 2n + 1` (orders `2^j - 1`, up
to 65535), each step evaluating only the new nodes. It stops when two
orders agree to 1e-10. `order=` fixes the order (CARA uses 64).

Both methods agree with the NASA CARA reference values to better than 1e-7
relative (see the [validation report](validation.md)).

## When 2-D Pc is questionable

The straight-line model breaks down for slow encounters. keepout estimates
the encounter duration as the time the secondary needs to cross five
standard deviations of the combined covariance along the relative velocity,
plus the HBR, on each side. It warns when that exceeds 1% of the primary's
orbital period. A 3-D method may then give a different Pc (Alfano 2009,
cases 1, 2, 4 and 8).

`compute_pc` in `auto` mode then moves to a 3-D method where it can (see
[Choosing a method](pc-3d.md#choosing-a-method)).

## Degenerate covariances

A standard deviation on the plane below `1e-4 * HBR` is raised to that value,
with a diagnostic. A line-like covariance therefore still gives a finite Pc. A plane
covariance with no positive variance raises `CovarianceError`.

## References

- Foster, J. L. and Estes, H. S. (1992). *A parametric analysis of orbital
  debris collision probability and maneuver rate for space vehicles.*
  NASA JSC-25898.
- Chan, F. K. (2008). *Spacecraft Collision Probability.* The Aerospace Press.
- Alfano, S. (2009). *Satellite conjunction Monte Carlo analysis.* AAS 09-233.
- Elrod, C. (2019). Doctoral dissertation, Department of Statistical Science,
  Baylor University (as cited by NASA CARA for its Gauss-Chebyshev 2-D Pc).
- NASA, *Spacecraft Conjunction Assessment and Collision Avoidance Best
  Practices Handbook*, NASA/SP-20205011318 Rev. 2: the 2-D Pc in operational
  conjunction assessment, and the 1e-4 manoeuvre threshold that keepout's
  red level follows.
