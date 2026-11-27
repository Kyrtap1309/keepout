# Changelog

Every change to keepout, by version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and keepout follows
[semantic versioning](https://semver.org/) from 1.0.0: what that covers,
and how a name is deprecated before it goes, is in [Stability](stability.md).

`keepout.__version__` gives the version installed.

## [Unreleased]

## [1.0.0] - YYYY-MM-DD

The first release. Everything is new in it; the development versions before
it were never released, so their changes are not listed. A store made by a
development version is migrated with `keepout store upgrade`
([Upgrading](store.md#upgrading)).

### Added

- **Conjunction data messages.** CDM reading and writing (CCSDS 508.0-B-1
  and TraCSS CDM 2.0, KVN and XML), covariance checks and repairs, frames
  and the encounter plane ([Working with CDMs](cdm.md)).
- **Probability of collision.** 2-D Pc (Foster, Elrod), 3-D Pc (Hall
  3D-Nc), Monte Carlo with a confidence interval, and maximum Pc (Frisbee,
  dilution, Alfano), each validated against NASA CARA's reference cases
  ([2-D](pc-2d.md), [3-D, Monte Carlo and maximum Pc](pc-3d.md),
  [validation](validation.md)).
- **Risk assessment** of conjunction events: risk levels, trend and the
  recommended action at a decision point ([Risk assessment](risk.md)),
  shown on ESA's Kelvins data ([report](risk-kelvins.md)).
- **Screening.** Element sets (OMM, CelesTrak CSV and JSON, TLE), batched
  SGP4 and screening against a catalog, compared with CelesTrak's SOCRATES
  ([Screening](screening.md)); clients for CelesTrak and Space-Track.org
  within their published limits ([Space-Track data](spacetrack.md)).
- **Synthetic operator.** CDMs for a fleet from public GP data with an
  assumed covariance model ([Synthetic operator](synthetic.md),
  [Assumptions](assumptions.md)).
- **Manoeuvre planning.** Along-track burns over a grid of size and time,
  with a Cowell propagator, and the re-screen of a planned burn against the
  catalog ([Manoeuvre planning](maneuver.md)).
- **Command line.** `keepout fetch`, `screen`, `assess`, `maneuver`,
  `validate`, `store`, `jobs`, `worker` and `dashboard`, with `--json`
  output ([Command line](cli.md), [JSON output](cli-json.md)).
- **Conjunction store** with stable event ids, background jobs and a worker
  ([The conjunction store](store.md)).
- **Dashboard.** Events, Pc trends, the manoeuvre planner and CDM uploads
  on a local store ([The dashboard](dashboard.md)); `docker compose up`
  runs it with a worker on a demo store ([Deployment](deployment.md)).
- **Case study.** The Iridium 33 – Cosmos 2251 collision of 2009 on the
  element sets public then: screening, synthetic CDMs, Pc, decision, rank,
  manoeuvre and re-screen ([case study](case-study-iridium-cosmos.md)), and
  its dashboard (`docker compose --profile iridium up`).

[Unreleased]: https://github.com/Kyrtap1309/keepout/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Kyrtap1309/keepout/releases/tag/v1.0.0
