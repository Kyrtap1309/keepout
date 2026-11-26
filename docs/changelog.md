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
  `keepout worker`). The [dashboard](dashboard.md) (`keepout dashboard`,
  the `dashboard` extra), with CDM uploads and refresh jobs. `keepout.figures`
  and a colour-blind-safe palette shared by the reports and the dashboard
  (the report SVGs changed colours). Store schema revision `0003`:
  `keepout store upgrade` migrates an older store. `docker compose up`
  runs the dashboard and a worker on a demo store
  ([Deployment](deployment.md)); `keepout dashboard --host` sets the
  address it listens on, warning off loopback.

Then a review of the whole code base changed:

### Fixed

- `dilution`: a conjunction is diluted only when the maximum beats the
  nominal Pc. `frisbee` gives Alfano's line maximum when the known object's
  covariance is zero. Monte Carlo counts samples whose orbit is no ellipse
  as misses (`details["non_elliptic"]`, with a diagnostic) instead of
  carrying NaN.
- Screening takes each object's newest element set usable in the window: an
  older set is no longer lost behind a newer one outside it.
- `keepout maneuver` says "No burn needed" when the Pc without a burn
  already meets the target, and the event's own approach is never reported
  as a new one by the re-screen.
- CDM XML with any `USER_DEFINED` parameter name is read; epoch fractions
  that round up to the next second are accepted; SOCRATES text with blank
  lines is read.
- A refresh queued while another one runs is no longer lost; a stopped
  refresh keeps the Pc it computed. Two processes opening a new store at
  once no longer fail with "database is locked".
- `keepout dashboard`: Ctrl+C or SIGTERM stops Streamlit too (killed after
  10 s or a second Ctrl+C); a bad `$KEEPOUT_AS_OF` is named as such.
- Dashboard plots: the green threshold line is green; the planner names
  the demo's catalog by its path in the repository.

### Changed

- `elrod` refuses a non-integer quadrature order. UTC epochs before 1972
  are refused ([Frames and units](frames-and-units.md)).
- Command line: options that do not go together are usage errors; progress
  goes to standard error on a terminal; `assess` and `screen` tables have
  an Event column and are printed whole when piped; every `--json` document
  starts with `"schema": 1` ([JSON output](cli-json.md)); a scoped IPv6
  `--host` is refused.
- `keepout worker --heartbeat-file`, and a healthcheck on the worker in
  `compose.yaml`.
- The dashboard's tables are sortable, the risk reads as "red ·
  manoeuvre", red events get a primary planner button, past events are
  shown apart and the planner's grid cells and notes are collapsed
  ([The dashboard](dashboard.md)).
- `PcObservation.method` is a `Method`; `trend_figure` takes only the
  assessment.

### Removed

- `keepout.omm.tle.checksum`, `keepout.figures.RISK_LABELS` and
  `ACTION_LABELS`, `keepout.validation.cases.MonteCarloSizing`.
