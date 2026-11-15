# Changelog

keepout is pre-alpha (version `0.1.0.dev0`) and has not been released.
Until the first release, any change may break the API, the command line,
the JSON output or the store's schema. A store made by an older keepout is
migrated with `keepout store upgrade` ([Upgrading](store.md#upgrading)).
From the first release on, this page will list every change by version.

`keepout.__version__` gives the version installed.

## Unreleased

The features in the order they were added:

- **2026-09:** CDM reading and writing (CCSDS 508.0-B-1 and TraCSS CDM 2.0,
  KVN and XML); covariance checks and repairs; frames and the encounter
  plane. 2-D Pc (Foster, Elrod), then 3-D (Hall 3D-Nc), Monte Carlo and
  maximum Pc (Frisbee, dilution, Alfano), validated against NASA CARA.
- **2026-10:** risk assessment of conjunction events (levels, trend,
  action). Element sets (OMM, CelesTrak CSV and JSON, TLE), SGP4
  propagation and screening against CelesTrak's public catalog, compared
  with SOCRATES. The synthetic operator, the Space-Track client and the
  `keepout` command.
- **2026-11:** collision-avoidance manoeuvre planning and its re-screen
  (`keepout maneuver`). The conjunction store with stable event ids
  (`keepout store`), then background jobs and the worker (`keepout jobs`,
  `keepout worker`). The dashboard (`keepout dashboard`, the `dashboard` extra):
  the events list, each event's Pc trend, encounter plane and geometry, and
  the manoeuvre planner, with CDM uploads and `refresh` jobs that compute
  the Pc an event lacks. `keepout.figures`, the plot data the reports and
  the dashboard share. The store's schema revision `0003` (a job's kind,
  the events list's indexes): `keepout store upgrade` migrates an older
  store.
