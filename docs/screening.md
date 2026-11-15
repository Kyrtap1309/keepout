# Screening

Screening finds the close approaches of your satellites (the *primaries*)
with a catalog of other objects over a window of days. An approach is any
time a catalog object comes inside a *screening volume* around a primary.
keepout reports each with its time of closest approach (TCA), range at TCA
and relative velocity. It screens the general perturbations (GP) element
sets that USSPACECOM publishes, with SGP4, as CelesTrak's
[SOCRATES](https://celestrak.org/SOCRATES/) does.

From a clone of the repository, this screens the ISS for 7 days against
CelesTrak's public catalog as it was on 2026-10-06, frozen in
`tests/screening/data/`:

```python
from collections import Counter

from astropy.time import Time

import keepout.omm
from keepout import screen


def main() -> None:
    catalog = keepout.omm.load("tests/screening/data/celestrak-public.csv.gz")
    iss = [omm for omm in catalog if omm.catalog_number == 25544]
    start = Time("2026-10-06T00:00:00", scale="utc")
    result = screen(iss, catalog, start, workers=4)  # 7 days, 5 km sphere
    for approach in result.approaches:
        radial, transverse, normal = approach.miss_rtn  # in the primary's RTN [m]
        print(
            approach.tca.isot,
            approach.secondary_number,
            f"{approach.range_at_tca:.0f} m",
            f"(R {radial:.0f}, T {transverse:.0f}, N {normal:.0f})",
            f"{approach.relative_speed:.0f} m/s",
        )
    print(Counter(skipped.reason.value for skipped in result.skipped))


if __name__ == "__main__":  # needed for workers > 1
    main()
```

```text
2026-10-06T05:12:33.772 58180 4633 m (R 3186, T -2873, N 1748) 7914 m/s
2026-10-06T12:10:37.724 58180 4902 m (R -4629, T -1376, N -842) 7937 m/s
2026-10-07T16:10:03.846 46275 4104 m (R -1521, T 1998, N 3246) 13035 m/s
2026-10-10T14:08:54.848 49773 2751 m (R -2365, T -957, N 1030) 11225 m/s
Counter({'stale': 28})
```

`miss_rtn` is the miss vector in the primary's
[RTN frame](frames-and-units.md#rtn-frame): radial (away from the Earth),
transverse (along the track) and normal (perpendicular to the orbit
plane), in metres. The `Counter` counts the element sets left out, by
reason (see [Skipped element sets](#skipped-element-sets)).

On live data, download the catalog and start the window now. A window
must end within 30 days of an element set's epoch, or the set is left out
as stale (see [Element age](#element-age)):

```python
from keepout.sources.celestrak import fetch_catalog

catalog = fetch_catalog().omms  # CelesTrak's public catalog, one set per object
start = Time.now()
```

!!! warning "Screening finds geometry, not risk"
    GP element sets carry no covariance, and their errors reach kilometres
    within days. A close approach found on GP data is a candidate for a
    closer look, not a collision risk. See
    [Limits of GP data](#limits-of-gp-data).

## Element sets

`keepout.omm` reads element sets into `Omm` objects, each one object's SGP4
mean elements in SI units. It reads every format they come in, recognised
by content: CCSDS OMM in KVN or XML (CCSDS 502.0-B-3), CelesTrak's CSV and
JSON, and legacy two- and three-line element sets (TLE).

```python
import keepout.omm

catalog = keepout.omm.load("catalog.csv")  # also gzip, KVN, XML, JSON, TLE
```

Catalog numbers above 99 999 have been assigned since July 2026; they have no
TLE form, so use OMM data for them. SGP4 accepts catalog numbers up to
339 999.

## CelesTrak's public catalog

`keepout.sources.celestrak.fetch_catalog()` downloads, as CSV, the GP
groups that together form CelesTrak's public catalog (`PUBLIC_GROUPS`).
They are the active satellites, the last 30 days' launches, analyst objects
and the debris of the three largest fragmentations. It keeps one element
set per object, the newest when an object is in several groups.
`Catalog.duplicates` lists the others.

`fetch_group` fetches one group. `fetch_socrates` fetches the latest
SOCRATES report, whose `approaches()` are its rows in SI units
(`keepout.sources.socrates.SocratesApproach`).

The client follows CelesTrak's
[usage policy](https://celestrak.org/usage-policy.php), which blocks clients
that ignore it:

- A download is cached under `$KEEPOUT_CACHE_DIR`, else the user cache
  directory (e.g. `~/.cache/keepout` on Linux). It is reused for at least
  two hours (`TTL`, CelesTrak's update interval) without asking CelesTrak
  again.
- CelesTrak answers a repeated download of unchanged data with HTTP 403. The
  cached copy is then current and is returned. Without one,
  `SourceUnavailable` says why.
- Any other status (`HttpStatusError`) or network error raises at once.
  Nothing is retried.
- Requests name keepout, its version and its home page in the `User-Agent`.

Only what is in the public catalog can be screened this way; see
[The public catalog](#the-public-catalog).

## Screening volumes

A `ScreeningVolume` is an ellipsoid centred on the primary with radial,
transverse (in-track) and normal (cross-track) semi-axes along the primary's
RTN axes. A sphere is three equal semi-axes. `screen` takes one volume for
every primary, or a rule choosing each primary's own. A rule
(`VolumeRule`) is a function of the primary's `Omm`. It refuses a primary
by raising a `KeepoutError`, and the primary is then skipped as
`NO_VOLUME`. Any other exception is taken for a bug in the rule and
propagates.

| Volume | Size | Source |
|---|---|---|
| `DEFAULT_VOLUME` | 5 km sphere | the threshold of CelesTrak's SOCRATES reports |
| `ScreeningVolume.nasa_gp_data()` | 10 km sphere | NASA CA Best Practices Handbook, Rev. 2, Appendix E |
| `ScreeningVolume.sfs_handbook` (a rule) | by orbit regime, below | USSPACECOM Spaceflight Safety Handbook for Satellite Operators, v1.7, Table 3 |
| `ScreeningVolume.sphere(radius)`, `ScreeningVolume(radial, transverse, normal)` | your own | |

```python
from keepout import ScreeningVolume
from keepout.screening import DEFAULT_VOLUME

print(DEFAULT_VOLUME.label, DEFAULT_VOLUME.radial)  # semi-axes in metres
print(ScreeningVolume.nasa_gp_data().label)
flat = ScreeningVolume(400.0, 25_000.0, 25_000.0, label="flat")  # R, T, N [m]
```

```text
5 km sphere (as in CelesTrak's SOCRATES) 5000.0
10 km sphere (NASA CA Handbook Rev. 2, App. E, GP data)
```

Pass a volume or a rule as `screen`'s `volume` argument, e.g.
`screen(primaries, catalog, start, volume=ScreeningVolume.sfs_handbook)`.

**5 km**, the default, makes the results comparable with SOCRATES. It is
not a safety margin for GP data.

**10 km.** NASA's analysis of screenings on GP data against screenings on
precision orbits finds that a miss-distance threshold of about 10 km is
needed to catch 95 % of the high-Pc events, because of the errors of GP
data (NASA Spacecraft Conjunction Assessment and Collision Avoidance Best
Practices Handbook, NASA/SP-20205011318 Rev. 2, 2026, Appendix E).

**Spaceflight Safety Handbook, Table 3** ("HAC screening volumes"), radial ×
in-track × cross-track:

| Regime (`SfsRegime`) | Orbit | Volume |
|---|---|---|
| LEO 1 | perigee altitude up to 500 km | 0.4 × 44 × 51 km |
| LEO 2 | perigee above 500 km, up to 750 km | 0.4 × 25 × 25 km |
| LEO 3 | perigee above 750 km, up to 1200 km | 0.4 × 12 × 12 km |
| LEO 4 | perigee above 1200 km, up to 2000 km | 0.4 × 2 × 2 km |
| Deep Space | period between 1300 and 1800 min, inclination below 35° | 10 × 10 × 10 km |
| outside Table 3 | anything else (MEO, HEO, inclined geosynchronous) | 10 × 10 × 10 km |

Every row of the table also needs an eccentricity below 0.25. keepout makes
three choices the table leaves open:

- **Semi-axes.** The table does not say whether its sizes are semi-axes or
  full extents. keepout reads them as semi-axes, the larger of the two
  readings, so no approach the handbook would report is missed.
- **Fixed ellipsoids.** The LEO rows are covariance screenings, for which
  the 18th Space Defense Squadron builds a volume from both objects'
  covariances. GP data has none, so keepout uses the rows' sizes as fixed
  ellipsoids.
- **Outside the table.** An orbit in no row gets the deep-space volume,
  which the table labels "Deep Space (GEO/HEO/MEO)". Its regime is
  `SfsRegime.OUTSIDE`, so the classification stays honest.

The regime comes from SGP4's own mean elements (its perigee and un-Kozai'd
period, with the WGS 72 constants). A two-body reading of the element set's
mean motion would put LEO perigees 3 km lower to 6 km higher, enough to move
some orbits across a regime boundary.

## From an approach to a Pc

A `CloseApproach` holds the catalog numbers and both GCRF states at TCA.
The range at TCA (`range_at_tca`), the miss vector in the primary's RTN
frame (`miss_rtn`), the relative velocity and the speed follow from them.

`to_conjunction` turns an approach into a `Conjunction` for `compute_pc`.
It takes a hard-body radius, either the combined one or each object's
(`primary_radius` and `secondary_radius`). It also takes covariances at
TCA from elsewhere, such as your own orbit determination or a CDM. Without
covariances only the maximum-Pc methods apply, which give an upper bound,
not an estimate. The [synthetic operator](synthetic.md) supplies assumed
covariances from a model of GP errors instead, and turns each approach into
a CDM.

```python
from astropy.time import Time

import keepout.omm
from keepout import ScreeningVolume, compute_pc, screen

catalog = keepout.omm.load("catalog.csv")  # OMM KVN/XML, CelesTrak CSV/JSON, TLE
primaries = [omm for omm in catalog if omm.catalog_number in {63216, 58509}]
start = Time("2026-10-06T00:00:00", scale="utc")
result = screen(primaries, catalog, start, volume=ScreeningVolume.sfs_handbook)
for number, volume in result.volumes.items():
    print(number, volume.label)
closest = min(result.approaches, key=lambda approach: approach.range_at_tca)
print(closest.tca.isot, closest.secondary_number, f"{closest.range_at_tca:.0f} m")
conjunction = closest.to_conjunction(combined_hard_body_radius=10.0)  # no covariance
pc = compute_pc(conjunction)
print(pc.method, f"{pc.value:.1e}")
```

With the 222 element sets of `tests/screening/data/celestrak-fixture.csv`
as `catalog.csv`:

```text
58509 SFS Handbook v1.7 Table 3, LEO 1
63216 SFS Handbook v1.7 Table 3, LEO 1
2026-10-06T08:34:04.437 57062 671 m
alfano 7.2e-03
```

## How `screen` works

`screen` returns a `ScreeningResult`. It holds the `CloseApproach`es in
chronological order, the element sets it left out (`skipped`), the window
and the volume of each primary. Its arguments, with their defaults (times
in seconds):

```python
from keepout.screening import DEFAULT_VOLUME

result = screen(
    primaries,  # element sets of your satellites
    catalog,  # element sets to screen them against
    start,  # astropy Time; no default
    duration=7 * 86400.0,  # the window is (start, start + duration]
    volume=DEFAULT_VOLUME,  # a ScreeningVolume or a rule
    workers=1,  # processes; None for one per CPU, at most 4 (see Workers)
    max_age=30 * 86400.0,  # see Element age
    max_lead=3 * 86400.0,
)
```

Two more keyword arguments are rarely needed: `max_states` bounds the
memory of a process, and `adjust_primaries` replaces the primaries'
propagator (the [re-screen](maneuver.md#re-screen) of a manoeuvre uses
it).

1. **Element sets.** Of several element sets of one object, the newest is
   screened; the others are skipped. An element set is also skipped when it
   is too old or too new for the window (below), when SGP4 cannot be
   initialised with it, or, for a primary, when the volume rule refuses it.
2. **Pairs.** Each primary is paired with every catalog object except
   itself (a primary may be in the catalog). The apogee/perigee filter
   drops the pairs whose distances from the Earth's centre never come
   within the primary's volume of each other, before anything is
   propagated.
3. **Search.** The remaining pairs are propagated with SGP4 on a time grid,
   in TEME and in time chunks of bounded memory. Sieves rule out the grid
   intervals in which a pair cannot come inside the volume. In the other
   intervals every local minimum of the range is found, with its TCA.
4. **Approaches.** Each minimum inside the primary's volume becomes a
   `CloseApproach`: both objects' states at TCA, rotated from TEME to GCRF.

### Skipped element sets

Nothing is dropped silently. `result.skipped` lists every element set that
was not screened, as a `SkippedObject`. Each holds the element set, its
`ScreeningRole` (`PRIMARY` or `CATALOG`), a `SkipReason` and a
human-readable detail.

| `SkipReason` | When |
|---|---|
| `DUPLICATE` | another element set of the same catalog number has a newer epoch |
| `STALE` | the window ends more than `max_age` (30 days) after the element epoch |
| `FUTURE_EPOCH` | the window starts more than `max_lead` (3 days) before the element epoch |
| `UNSUPPORTED` | SGP4 cannot be initialised with the elements (e.g. a catalog number above 339 999) |
| `NO_VOLUME` | the volume rule gives no volume for this primary |
| `PROPAGATION_FAILED` | SGP4 reports an error inside the window (e.g. the orbit has decayed) |

An object with an SGP4 error inside the window is left out with every
approach it has, since its states before the error are not trusted either. An
object the filter rules out of every pair is never propagated, so an error
it would have is not seen.

### Element age

An element set is screened only near its epoch:

- `max_age` of 30 days from the epoch to the end of the window;
- `max_lead` of 3 days from the start of the window to a later epoch.
  CelesTrak serves epochs up to about a day and a half after the day of the
  download.

Inside these limits the search finds every approach (next section),
except for an object re-entering the atmosphere. GP data loses accuracy
much sooner than 30 days, so a smaller `max_age` may suit your
application better. In the first example, this leaves out the sets more
than 7 days old at the end of the window:

```python
result = screen(iss, catalog, start, max_age=7 * 86400.0, workers=4)
```

??? note "Implementation: why 30 days and 3 days"
    The search's guarantee assumes that every acceleration beyond the
    point mass and J2 stays below 1 m/s². On the public
    catalog of October 2026, SGP4 stays below 0.2 m/s² from 3 days before
    to 38 days after each element epoch. Further away some objects leave
    the bound, from 39 days after the epoch and by orders of magnitude.
    Hence the defaults.

    One kind of object breaks the assumption inside these limits: an
    object re-entering the atmosphere. With its perigee 113 km up, SGP4's
    drag on object 270365 exceeds 1 m/s² from 8 days after its epoch, days
    before SGP4 fails for it. The guarantee does not cover such objects.

### The search and its guarantee

Both objects are sampled on a time grid of 120 s. Sieves rule out
the grid intervals in which the pair cannot come inside the volume, and
every minimum of the range in the other intervals is found, with its TCA
to 1 µs. The sieves hold for any grid step, so the step changes the time
screening takes, not what it finds. In a co-orbital or formation-flying
pair, a minimum and a maximum of the range less than 10 s apart are not
told apart.

keepout's tests compare the search with a brute-force one (every pair
sampled densely) on a small catalog, and check with property tests that no
filter or sieve drops an approach the brute-force search finds.

??? note "Implementation: the sieves and the minimum search"
    - **Speed sieve.** Each object's speed is bounded from its radius bounds
      over the window (the perigee speed of the most eccentric orbit that fits,
      plus 5 %), so the relative speed is below a bound *V*. A pair whose
      ranges at the two ends of an interval of length *h* add up to more than
      2*R* + *V h* (*R* the volume's bounding radius) cannot come within *R*
      in between.
    - **Chord sieve.** With *A* a bound on the relative acceleration (the
      gravity gradient, 2μ/*r*³ at the lowest radius *r* on the segment
      joining the objects, 5 % added for the zonal terms, times the largest
      range over the interval, or at most the sum of both gravities; plus
      twice the 1 m/s² bound on the other accelerations: drag, the Sun and
      Moon, SGP4's own terms), the relative position stays within *A h*²/8
      of the straight line between its two end points; if that line does not
      come within *R* + *A h*²/8 of the primary, neither does the pair. The
      largest range is the smaller of the speed sieve's bound,
      (|ρ*a*| + |ρ*b*| + *V h*)/2, and the chord's own,
      max(|ρ*a*|, |ρ*b*|) + *A h*²/8.

    In an interval that passes, a minimum of the range is a root of
    ρ·dρ/dt (ρ the relative position), found with Brent's method to 1 µs.
    dρ/dt comes from positions by a central difference, not from SGP4's
    velocities. SGP4's velocity is not exactly the derivative of its position
    (by up to 12 m/s), which would move the TCA of a slow pair by seconds. The
    states reported at TCA are SGP4's own, velocities included. Within two
    difference steps of the window's ends, where a central difference would
    take positions outside the window, a one-sided fourth-order difference
    over five positions inside it is used instead: an SGP4 error just
    outside the window (an object decaying a second after it) is not the
    object's.

    ρ·dρ/dt is increasing over an interval when the relative velocity, which
    stays within *A h*/2 of the chord velocity *c* = (ρ*b* − ρ*a*)/*h*,
    satisfies (|*c*| − *A h*/2)² > ρ*max* *A* (ρ*max* the largest range). The
    interval then holds a minimum exactly when ρ·dρ/dt goes from negative
    at its start to non-negative at its end. Over a week of the public
    catalog (20 LEO primaries, a 5 km sphere, the default step), every
    interval in which the pair moves faster than 0.5 km/s is decided this
    way. The range of a co-orbital or formation-flying pair can change too
    slowly for that. Such an interval is resampled every 10 s, and each
    sign change between the samples is a minimum. A minimum and a maximum
    of the range less than 10 s apart are not told apart there (on the
    frozen catalog the closest are 75 s apart).

    An SGP4 error while an interval is refined loses that interval's minima
    only; the pair's other intervals are refined as if it had none. Each
    interval, and each failure found while refining it, is decided from
    its own pair's states only, so the result does not depend on how the
    grid is cut into chunks (`max_states`) or the catalog into
    [workers](#workers).

### What counts as an approach

- **Every local minimum** of the range inside the volume is its own
  `CloseApproach`; a pair in formation can have two within minutes.
- **The window is `(start, start + duration]`**: an approach is reported
  when its TCA lies in it. A range still falling at the end of the window,
  or already rising at its start, has its minimum outside. Consecutive
  windows therefore find the minima of their union, each once.
- **Inside the volume at TCA.** For an ellipsoid, a pass that crosses the
  volume away from its closest point does not count: with a flat volume
  (400 m radial, tens of kilometres in-track), a pair can enter it without
  its point of closest approach being inside.
- **Docked spacecraft.** CelesTrak gives docked spacecraft the element set
  of their station. A pair at exactly the same position is taken as one
  object, with no approach.

### Workers

`workers=4` deals the catalog into 4 parts, screened in separate processes,
and merges the results. Each approach depends on its own pair's states only,
so the result is the same, bit for bit, for any number of workers.

The processes are started with `spawn` on every platform, which imports the
main module again. A script must therefore call `screen` with `workers`
above 1 under `if __name__ == "__main__":`, as in the example above. A
script read from standard input (`python -`) cannot use workers.

## Limits of GP data

GP element sets are mean elements that the 18th Space Defense Squadron fits
to its tracking and publishes for every cataloged object. They are what
anyone can screen with. They are not precise orbits:

- **No covariance.** GP data says nothing about its own uncertainty, so a
  probability of collision cannot be computed from it. An approach found on
  GP data gives a miss distance, not a risk.
- **Kilometre errors.** SGP4's analytic theory and the fit leave errors of
  kilometres in low orbits within days of the element epoch, growing with
  the time since it. This is why NASA finds a 10 km threshold necessary on
  GP data to catch 95 % of the high-Pc events (Appendix E of its handbook).
  The 5 km default only matches SOCRATES.
- **Each object its own epoch.** Element sets are hours to days old, each
  from its own fit. SOCRATES and keepout give the same approach only from
  the same element sets, and two element sets of the same epoch can still
  differ (see the [SOCRATES report](screening-socrates.md)).

Use GP screening to find what to look at. Decisions need precise orbits
with covariance, such as the conjunction data messages (CDMs) of an
operational conjunction assessment service, which keepout reads with
`keepout.cdm` and assesses as in [Risk assessment](risk.md).

## The public catalog

CelesTrak publishes only part of the catalog: about 20 000 objects in its
groups (19 914 on 2026-10-06). SOCRATES screens the whole catalog of
Space-Track.org, and 24 194 objects appear in its report of that day. Most
rocket bodies and debris are in no public group. 40 211 of the 186 229
approaches of SOCRATES's report of 2026-10-06 involve an object outside the
public catalog, and screening against `fetch_catalog()` misses such
approaches.

The full catalog, about 32 000 objects (31 930 on 2026-10-15), comes from
Space-Track.org with a free account.
`keepout.sources.spacetrack.fetch_catalog()` gives it as the same `Catalog`
(see [Space-Track data](spacetrack.md)).

## Validation and performance

- **[Screening against SOCRATES](screening-socrates.md).** It covers the
  1 275 approaches of SOCRATES's report of 2026-10-06 whose objects are in
  the public catalog with the element sets SOCRATES used. Screened with
  keepout's defaults, all agree within SOCRATES's print precision (1 ms,
  0.5 m, 0.5 m/s; 0.5 s for the TCA of slow pairs). The exceptions are 5
  approaches of two objects whose element sets evidently differ from
  SOCRATES's, and one left out as stale. `tools/screening_report.py`
  generates the report, and CI runs it with `--check`.
- **Benchmark.** 20 LEO primaries screened for 7 days with 4 workers on a
  desktop CPU take about 9 s against the frozen public catalog (19 914
  objects). Against the frozen full catalog (31 930 objects, Space-Track,
  2026-10-15) they take 13 s and find 866 approaches in the 5 km sphere.
  The target is under 60 s. `tools/screening_benchmark.py` writes the full
  tables to the [README](https://github.com/Kyrtap1309/keepout#performance)
  and the numbers to [`screening-benchmark.json`](screening-benchmark.json)
  and [`screening-benchmark-spacetrack.json`](screening-benchmark-spacetrack.json).
- **Regressions.** CI times every change against its merge base on the same
  runner (a reduced catalog, one process, five runs each, alternately). It
  fails when the median CPU time rises by more than 20 %.

## Data

The frozen catalog, test fixture and SOCRATES excerpt are in
`tests/screening/data/` (see the `NOTICE` there). Data: USSPACECOM/18 SDS
via Space-Track.org, retrieved from CelesTrak (celestrak.org) on
2026-10-06. This is basic space situational awareness data, which the
Space-Track.org user agreement allows to be redistributed with this
citation. It is not covered by keepout's MIT licence. Cite data you
download from CelesTrak in the same way when you publish results computed
from it.

The frozen full catalog is in `tests/screening/spacetrack-data/` (see the
`NOTICE` there). Data: USSPACECOM/18 SDS via Space-Track.org, retrieved on
2026-10-15. It is also basic SSA data, redistributed with this citation and
not covered by keepout's MIT licence.
