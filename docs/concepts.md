# Concepts

This page is for readers new to conjunction assessment. It explains what
keepout computes and why, and defines the terms the other pages use.
Readers who know the field can go straight to the
[Glossary](#glossary) or [Getting started](index.md).

## Conjunction assessment in one page

About 30 000 tracked objects orbit the Earth: working satellites, dead
ones, rocket bodies and debris. A satellite operator must know when one of
them comes close to their satellite, and whether to move the satellite out
of the way. That is **conjunction assessment** (CA).

1. **Screening.** Someone propagates the orbits of the operator's
   satellites (the *primaries*) and of every catalogued object (the
   *secondaries*) for the coming days. Every time a secondary comes within
   a *screening volume* around a primary is a **close approach**, with a
   time of closest approach (**TCA**) and a miss distance.
2. **CDMs.** For each close approach a conjunction assessment service,
   such as the US Space Force's 18th Space Defense Squadron or the Office
   of Space Commerce's TraCSS, sends the operator a **Conjunction Data
   Message** (CDM). A CDM gives both objects' states at TCA and the
   uncertainty of each, a **covariance** from orbit determination.
3. **Probability of collision.** From the states, the covariances and the
   size of the objects (the **hard-body radius**), the operator computes
   the probability that the objects collide, the **Pc**. A miss distance
   alone says little: 200 m with an uncertainty of 10 km is less worrying
   than 500 m with an uncertainty of 100 m.
4. **Risk assessment.** A new CDM arrives every few hours until TCA, each
   with its own Pc. The operator watches the latest Pc and its trend and
   decides: ignore, monitor, prepare a manoeuvre, or manoeuvre. A common
   threshold for acting is a Pc of 1 in 10 000 (1e-4).
5. **Manoeuvre planning.** To act, the operator plans a small burn some
   hours before TCA that takes the Pc below a target. The new orbit is
   screened again so that the burn does not bring a new close approach.

keepout does each step:

| Step | keepout | Guide |
|---|---|---|
| Screening | `screen`, `keepout screen` | [Screening](screening.md) |
| CDMs | `keepout.cdm`, and the [synthetic operator](synthetic.md) on public data | [Getting started](index.md#reading-a-cdm) |
| Pc | `compute_pc` | [2-D Pc](pc-2d.md), [3-D, Monte Carlo and maximum Pc](pc-3d.md) |
| Risk | `assess`, `keepout assess` | [Risk assessment](risk.md) |
| Manoeuvre | `plan_maneuver`, `keepout maneuver` | [Manoeuvre planning](maneuver.md) |
| History | the conjunction store, `keepout store` | [The conjunction store](store.md) |

### Public data and its limits

Operators receive CDMs for their own satellites only. Anyone can download
the **GP element sets** of the whole catalog (from CelesTrak or
Space-Track), but they carry no covariance, and their errors reach
kilometres within days. A Pc cannot be computed from them alone. keepout
therefore screens public data for geometry, and its synthetic operator
adds an *assumed* covariance to make CDMs from it. Its Pc values show what
that assumption implies, not the real risk
([Synthetic covariance model](assumptions.md)).

!!! warning "keepout is not an operational service"
    keepout is a library, not a service. Its risk rules are transparent
    heuristics, not an operational standard. No decision about a real
    satellite should rest on keepout alone.

## Glossary

**18 SDS**
:   The US Space Force's 18th Space Defense Squadron. It maintains the
    catalog of tracked objects, publishes GP element sets and issues CDMs.

**Action**
:   keepout's recommendation for an event: *ignore*, *monitor*, *prepare
    maneuver* or *maneuver* ([Decision](risk.md#decision)).

**Alfano, Frisbee, dilution**
:   The three maximum-Pc methods keepout implements, for no covariance, one
    covariance, and two covariances that may be too large
    ([Maximum Pc](pc-3d.md#maximum-pc)).

**Analyst object**
:   A tracked object that 18 SDS has not yet identified. CelesTrak's
    public catalog includes its `analyst` group, but a CDM's secondary may
    be an analyst object missing from the catalog given to a re-screen
    ([Re-screen](maneuver.md#re-screen)).

**Basis (conservative, estimate)**
:   How a Pc is compared with the risk thresholds and the manoeuvre target.
    On the default *conservative* basis a Monte Carlo Pc counts by the upper
    end of its confidence interval; on the *estimate* basis by the estimate
    itself ([Risk levels](risk.md#risk-levels)).

**Burn, Δv, prograde, retrograde**
:   A burn changes a satellite's velocity by Δv (m/s). Prograde is along
    the velocity (positive Δv), retrograde against it (negative). A burn's
    *lead* is the time from the burn to TCA ([Manoeuvre planning](maneuver.md#the-grid)).

**CARA**
:   NASA's Conjunction Assessment Risk Analysis team. Its open-source
    *CARA Analysis Tools* provide the test cases keepout is validated
    against ([Validation report](validation.md)).

**CDM**
:   Conjunction Data Message, the CCSDS 508.0 standard message describing
    one close approach of two objects: both states at TCA, their
    covariances and metadata. keepout reads and writes CDM 1.0 and the
    TraCSS CDM 2.0, in KVN or XML.

**CelesTrak**
:   A free website that republishes GP element sets in groups, without an
    account. Its groups form the *public catalog*, about 20 000 objects
    ([Screening](screening.md#celestraks-public-catalog)).

**Close approach**
:   One local minimum of the distance between two objects inside the
    screening volume, found by screening. keepout's `CloseApproach`.

**Conjunction**
:   Two objects near their closest approach, with their states,
    covariances and hard-body radius: what a Pc is computed from.
    keepout's `Conjunction`.

**Covariance**
:   The 6×6 matrix describing the uncertainty of an object's position and
    velocity, from orbit determination. keepout checks it and, by default,
    repairs small defects ([Covariance validation and repair](frames-and-units.md#covariance-validation-and-repair)).

**Decision point**
:   The time before TCA (2 days by default) after which a red event's
    action is *maneuver* rather than *prepare maneuver*. It leaves time to
    plan the manoeuvre and screen the new orbit ([Decision](risk.md#decision)).

**Diagnostic**
:   A warning attached to a Pc: a repaired covariance, a questionable 2-D
    assumption, a quadrature problem. A green Pc with a diagnostic is
    monitored, never ignored ([Decision](risk.md#decision)).

**Encounter plane**
:   The plane through the primary perpendicular to the relative velocity
    at TCA. The 2-D Pc is computed on it ([Frames and units](frames-and-units.md#encounter-plane)).

**Event**
:   All the CDMs of one encounter of the same two objects, issued over the
    days before TCA. keepout's `ConjunctionEvent`; in the store, each has a
    12-character id ([Event ids](store.md#event-ids)).

**GCRF, EME2000, ITRF, TEME**
:   Reference frames. GCRF is keepout's inertial frame for every state.
    EME2000 is nearly the same; ITRF rotates with the Earth; TEME is the
    frame of SGP4's output ([Frames, units and covariances](frames-and-units.md)).

**GP element set, OMM, TLE**
:   *General perturbations* element sets: mean orbital elements that
    18 SDS fits for every catalogued object, propagated with SGP4. They are
    published as CCSDS Orbit Mean-elements Messages (OMM) or in the older
    two-line element (TLE) format. They have no covariance.

**Grid cell**
:   One burn of the manoeuvre planner's grid: a burn time and a signed Δv,
    with the Pc that burn gives ([Manoeuvre planning](maneuver.md#terms)).

**HBR (hard-body radius)**
:   The radius of a sphere enclosing an object. A collision is the
    secondary's centre entering a sphere of the *combined* HBR (both radii
    added) around the primary. keepout never assumes a default HBR.

**Kelvins**
:   ESA's *Collision Avoidance Challenge* data set: about 15 000 real
    events of ESA missions, with ESA's Pc for every CDM
    ([Kelvins report](risk-kelvins.md)).

**Manoeuvre (maneuver)**
:   A collision-avoidance burn
    ([which spelling](faq.md#getting-started)).

**Maximum Pc**
:   The largest Pc any covariance in a family could give: an upper bound,
    used when a covariance is unknown. keepout marks such results
    `kind == "upper_bound"`.

**Miss distance**
:   The distance between the two objects at TCA.

**Nc**
:   The expected number of collisions, which Hall's 3-D method computes.
    For a single short encounter it equals the Pc; otherwise it is an upper
    bound on it ([Hall 3D-Nc](pc-3d.md#hall-3d-nc)).

**Pc (probability of collision)**
:   The probability that the two objects come closer than the combined HBR
    during the encounter. keepout computes it with 2-D, 3-D, Monte Carlo or
    maximum-Pc methods; `compute_pc` picks one.

**Primary, secondary**
:   The primary is the satellite being protected (the CDM's `OBJECT1`), the
    secondary the other object (`OBJECT2`).

**Re-screen**
:   Screening the primary again after a planned burn, for 7 days from the
    burn, and comparing with the run without it. It finds the approaches
    the burn brings, which the Pc grid cannot see
    ([Re-screen](maneuver.md#re-screen)).

**Risk level**
:   *Green*, *yellow* or *red*, from the Pc and two thresholds (by default
    1e-7 and 1e-4) ([Risk levels](risk.md#risk-levels)).

**RTN**
:   An object's local frame: **R**adial (away from the Earth),
    **T**ransverse (along the track) and **N**ormal (perpendicular to the
    orbit plane). CDM covariances are given in it.

**Screening volume**
:   The region around a primary inside which an approach is reported: a
    sphere or an ellipsoid along RTN ([Screening volumes](screening.md#screening-volumes)).

**SGP4**
:   The analytic propagator that GP element sets are made for.

**SOCRATES**
:   CelesTrak's public daily report of close approaches, which keepout's
    screening is compared with ([Screening against SOCRATES](screening-socrates.md)).

**Space-Track.org**
:   18 SDS's website for GP element sets (the full catalog, about 32 000
    objects) and public CDM summaries, with a free account
    ([Space-Track data](spacetrack.md)).

**Stale element set**
:   An element set whose epoch is more than `max_age` (30 days by default)
    before the end of the screening window. Screening leaves it out, since
    its errors are too large by then ([Element age](screening.md#element-age)).

**Store, job, worker**
:   The conjunction store is a SQLite file keeping events, their CDMs and
    Pc. A job is a manoeuvre plan queued in it; a worker computes the jobs
    ([The conjunction store](store.md)).

**Synthetic operator, assumed covariance**
:   keepout acting as a conjunction assessment service on public data: it
    screens a fleet, writes a CDM per approach with an *assumed* covariance
    from a model of GP errors, and assesses it ([Synthetic operator](synthetic.md)).

**TCA (time of closest approach)**
:   The time at which the two objects are closest.

**TraCSS**
:   The US Office of Space Commerce's Traffic Coordination System for
    Space, which issues CDMs in version 2.0.

**Trend**
:   Whether an event's Pc is *rising*, *falling* or *steady* over its
    latest CDMs ([Trend](risk.md#trend)).
