# Frames and units

## Units

Every keepout type uses SI units: metres, metres per second, seconds and
kilograms. CDM files give the state vector in km and km/s and OD spans in
days; the readers convert on input and the writers convert back on output.

## Reference frames

| CDM `REF_FRAME` | Treatment in keepout |
|-----------------|----------------------|
| `GCRF` | used as is |
| `EME2000` | rotated to GCRF by the IAU 2006 frame bias (about 23 mas, ≈0.8 m at 7000 km), so a CDM whose objects use different frames stays consistent. |
| `ITRF` | converted to GCRF with astropy (ITRS → GCRS), including the Earth-rotation term ω × r of the velocity |
| `TEME` | rotated to GCRF with ERFA (`keepout.frames.teme_to_gcrf`); see below |

TEME (true equator, mean equinox of date) is the frame of SGP4 output, and
so of every OMM and TLE. Following Vallado et al. (2006), keepout turns it
to the pseudo-Earth-fixed frame by GMST (IAU 1982). From there it goes
through ITRF to GCRF by the IAU 2006/2000A model. As one rotation,
r_GCRF = Cᵀ R₃(GMST82 − ERA) r_TEME, where C is the GCRS → CIRS matrix and
ERA the Earth rotation angle.

In this rotation polar motion cancels. UT1 − UTC changes the result by
less than 0.3 mm at GEO, so keepout uses UTC for UT1 and needs no
Earth-orientation data, for any epoch. The velocity includes the slow turn
of the frame (precession and nutation, up to 5·10⁻⁴ m/s at GEO). The result
reproduces the paper's Appendix C example to 3 mm and 0.5 mm/s. It agrees
with astropy's TEME → GCRS, which reads IERS data, to 10⁻¹⁰ of the radius.

- `teme_to_gcrf` converts arrays of states at one or many epochs in one
  call.
- `to_gcrf_all` converts all TEME `StateVector`s of a list together.
- `frame_to_gcrf` converts arrays of states in any one frame, as a
  propagator gives them. TEME goes through `teme_to_gcrf`.

Any other frame is rejected with `UnsupportedFrame`, a `GeometryError` that
is also a `ValueError`. A covariance in a frame outside `COVARIANCE_FRAMES`
is rejected the same way.

### Earth-orientation (IERS) data

The ITRF conversion needs Earth-orientation parameters. keepout never changes
astropy's configuration: by default astropy downloads IERS-A predictions when
it needs them and otherwise falls back to its bundled tables with a warning.
For Pc this does not matter. UT1 and polar-motion errors rotate both objects
together, and the ω × r term does not depend on them. Absolute ITRF
positions do depend on them. The TEME conversion needs none (see above).

## RTN frame

RTN is built from an object's inertial (GCRF) state:
R = r/|r|, N = (r × v)/|r × v|, T = N × R.

CDM covariances are given in each object's RTN frame. For ITRF messages keepout
builds RTN from the state **after** conversion to GCRF. This assumes that
the originator's RTN is inertial. The 6×6 covariance is rotated with the
same matrix for position and velocity, and the rotation rate of the RTN
frame is neglected. That changes only the velocity terms. The 2-D Pc uses
the position block only and is unaffected. The 3-D and Monte Carlo Pc use
the velocity block, and are validated against CARA with this convention.

## Encounter plane

For relative position r = r₂ − r₁ and relative velocity v = v₂ − v₁ at TCA
(Foster & Estes 1992; Chan 2008):

- y = v/|v| is normal to the encounter plane;
- x is the unit component of r perpendicular to v;
- z = x × y.

The plane is spanned by (x, z); the miss vector is (|r⊥|, 0) and the combined
position covariance C₁ + C₂ is projected onto it. When r is parallel to v, x is
any unit vector perpendicular to v. Pc does not depend on that choice.

## Covariance validation and repair

A `Covariance` must be symmetric and positive semi-definite. Both are judged
on the correlation matrix `D⁻¹ C D⁻¹`, `D = diag(sqrt|Cᵢᵢ|)`. Every block is
thus checked at its own scale, such as a mm/s velocity variance next to a
km position variance. `Cᵢⱼ` and `Cⱼᵢ` may differ by 1e-9 of
`sqrt|Cᵢᵢ Cⱼⱼ|`, and no correlation-matrix eigenvalue may be below −1e-10.

Operational CDMs often violate this, so `CdmObject.covariance()` and
`Cdm.object_states()` default to `policy="repair"`:

- The matrix is symmetrised.
- A negative eigenvalue is clipped to zero where it lies. If the 3×3
  position block is itself invalid, its eigenvalues are clipped. Otherwise
  only the velocity covariance conditional on position (`V − B P⁺ Bᵀ`) is,
  so a defect in the velocity terms never changes the position block or
  the 2-D Pc.

Each change is recorded in `Covariance.repairs`. `Conjunction.diagnostics`
lists both objects' repairs (e.g. `primary (12345): clipped 1 negative
position-block eigenvalue(s) ...`), and every Pc result repeats them.
`policy="raise"` raises `CovarianceError` instead.

## References

- CCSDS 508.0-B-1, *Conjunction Data Message*, Blue Book, 2013 (Cor. 2, 2021).
- J. L. Foster and H. S. Estes, *A Parametric Analysis of Orbital Debris
  Collision Probability and Maneuver Rate for Space Vehicles*, NASA JSC-25898,
  1992.
- F. K. Chan, *Spacecraft Collision Probability*, The Aerospace Press, 2008.
- D. A. Vallado, P. Crawford, R. Hujsak and T. S. Kelso, *Revisiting
  Spacetrack Report #3*, AIAA 2006-6753, 2006.
