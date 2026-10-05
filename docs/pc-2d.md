# 2-D probability of collision

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
states are not exactly at TCA, this is the straight-line closest approach;
`compute_pc` reports states more than 1% of the relative distance away from it.

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
| `foster` | Adaptive Gauss-Kronrod quadrature (QUADPACK) of the 1-D integral with `u = R sin(theta)`, over the band within 12 sigma of the mean (outside it the density is below `exp(-72)` of its peak; a circle entirely outside gives exactly 0, where Elrod may still return e.g. 3e-58), with breakpoints where a thin narrow axis makes the chord mass a step | relative tolerance 1e-10 |
| `elrod` | Gauss-Chebyshev quadrature (second kind) of the same integral; the order starts where the nodes resolve the Gaussian (at least 63) and steps along the nested sequence `n -> 2n + 1` (orders `2^j - 1`, each step evaluating only the new nodes, up to 65535) until two orders agree to 1e-10; `order=` fixes it (CARA uses 64) | relative 1e-10 between orders |

`compute_pc(conjunction, "auto")` uses Foster when the 2-D assumptions hold
(see [3-D, Monte Carlo and maximum Pc](pc-3d.md) for the other cases). Both
methods agree with the
NASA CARA reference values to better than 1e-7 relative; see the
[validation report](validation.md).

## When 2-D Pc is questionable

The straight-line model breaks down for slow encounters. keepout estimates
the encounter duration as the time the secondary needs to cross five standard
deviations of the combined covariance along the relative velocity (plus the
HBR) on each side, and warns
when it exceeds 1% of the primary's orbital period. A 3-D method may then give
a different Pc (Alfano 2009, cases 1, 2, 4 and 8); `compute_pc` in `auto` mode
then uses [Hall's 3D-Nc](pc-3d.md) if both covariances are known and both
orbits are elliptic, and otherwise stays on Foster (or, with a covariance
unknown, uses a maximum Pc).

## Degenerate covariances

A standard deviation on the plane below `1e-4 * HBR` is raised to that value
with a diagnostic, so a line-like covariance still gives a finite Pc. A plane
covariance with no positive variance raises `CovarianceError`.

## References

- Foster, J. L. and Estes, H. S. (1992). *A parametric analysis of orbital
  debris collision probability and maneuver rate for space vehicles.*
  NASA JSC-25898.
- Chan, F. K. (2008). *Spacecraft Collision Probability.* The Aerospace Press.
- Alfano, S. (2009). *Satellite conjunction Monte Carlo analysis.* AAS 09-233.
- Elrod, C. (2019). Doctoral dissertation, Department of Statistical Science,
  Baylor University (as cited by NASA CARA for its Gauss-Chebyshev 2-D Pc).
