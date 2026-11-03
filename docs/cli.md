# Command line

The `keepout` command runs the library's main workflows from a terminal:
download data, screen a fleet and issue CDMs as a
[synthetic operator](synthetic.md), assess CDM files, plan a
collision-avoidance burn, and check keepout's Pc against NASA CARA's test
cases. It is a thin layer over the library
functions the other pages describe, so it computes exactly what they do.

```bash
pip install "keepout[cli]"
keepout --help
```

The `cli` extra installs [Typer](https://typer.tiangolo.com/) (and Rich, for
the tables); the library itself never imports either. Without the extra,
`keepout` prints the line to install it and exits with status 1.
`python -m keepout.cli` is the same command.

| Command | What it does |
|---|---|
| `keepout fetch catalog [--source celestrak\|spacetrack]` | Download the GP element sets of every object in orbit, for `screen` |
| `keepout fetch cdm` | Download Space-Track's public CDM summaries of approaches to come |
| `keepout fetch cara` | Download NASA CARA's validation files (pinned commit, SHA-256) |
| `keepout fetch kelvins` | Download ESA's Kelvins collision avoidance data set (221 MB) |
| `keepout screen --fleet FILE [--days 7] [--cdm-dir DIR]` | Screen a fleet, issue a CDM per approach, assess each |
| `keepout assess CDM...` | Compute each CDM's Pc and each event's risk and action |
| `keepout maneuver CDM... [--catalog FILE]` | Find the smallest burn each way that takes an event below a target Pc, and re-screen it |
| `keepout validate` | Compare keepout's Pc with CARA's on every validation case |

`keepout COMMAND --help` lists every option of a command.

## A week of a synthetic operator

From a clone of the repository, this screens the
[example fleet](synthetic.md#the-fleet-file) (the ISS, Metop-B and
Intelsat 39) against the public catalog frozen on 2026-10-06, offline, and
writes a CDM for each close approach:

```bash
keepout screen --fleet examples/synthetic-operator/fleet.toml \
    --catalog tests/screening/data/celestrak-public.csv.gz \
    --start 2026-10-06 --days 7 --cdm-dir cdms
```

```text
Screened 3 fleet satellites against 19886 objects (tests/screening/data/celestrak-public.csv.gz) from 2026-10-06T00:00:00 to 2026-10-13T00:00:00 UTC: 12 close approaches.
Close approaches
TCA (UTC)         Objects        Miss (m)  CDMs        Pc  Level   Action
──────────────────────────────────────────────────────────────────────────
2026-10-06 05:12  25544 / 58180      4633     1  1.44e-30  green   ignore
2026-10-06 09:25  38771 / 35434      4085     1  1.46e-17  green   ignore
2026-10-06 12:10  25544 / 58180      4902     1  4.33e-59  green   ignore
2026-10-06 23:45  38771 / 31882      3374     1  4.28e-33  green   ignore
2026-10-07 08:06  38771 / 30041      4511     1  8.48e-30  green   ignore
2026-10-07 16:10  25544 / 46275      4104     1  7.48e-10  green   ignore
2026-10-07 22:10  38771 / 35884      3314     1  1.55e-09  green   ignore
2026-10-08 12:03  44476 / 55841      3647     1  7.38e-06  yellow  monitor
2026-10-09 09:20  38771 / 29734      1476     1  3.90e-10  green   ignore
2026-10-10 14:08  25544 / 49773      2751     1  1.09e-18  green   ignore
2026-10-11 10:56  38771 / 30131      2019     1  2.67e-10  green   ignore
2026-10-12 11:28  38771 / 35196       212     1  4.86e-06  yellow  monitor
28 element sets left out (stale, future epochs, SGP4 failures; --json lists them).
44476/55841/2026-10-08T12:03:02.771792: monitor (latest Pc 7.38e-06 is yellow, 2.50 d before TCA; trend undetermined; diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 2.51e+03 s, 2.9% of the primary's orbital period (limit 1%); a 3-D Pc may differ; diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used)
38771/35196/2026-10-12T11:28:28.709977: monitor (latest Pc 4.86e-06 is yellow, 6.48 d before TCA; trend undetermined)
Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
12 CDMs written to cdms
```

Each row is one close approach (the fleet satellite first), with keepout's
Pc computed from the CDM issued for it, the Pc's risk level and the
recommended action; the events that need more than `ignore` are explained
below the table. A fleet satellite that could not be screened (its element
set too old for the window, too far in the future, or failing in SGP4) has
no row and is not protected: a line above the table names it, e.g.
`Fleet satellite 44476 (INTELSAT 39 (IS-39)) was NOT screened: stale, ...`,
and the first line says `Screened 2 of 3 fleet satellites`. The covariances behind these Pc values are assumed: read
[Assumptions](assumptions.md) before reading anything into them.

The CDMs are ordinary CCSDS messages that any CDM reader takes, and
`assess` reads them back like CDMs from any other originator:

```bash
keepout assess cdms/*.kvn             # the same 12 events: exit status 0
keepout assess cdms/*.kvn --hbr 1000  # a 1 km hard body: exit status 3
```

With a combined hard-body radius of 1 km instead of each CDM's `HBR`, the
second prints `12 events from 12 CDMs: 2 manoeuvre, 4 prepare manoeuvre,
1 monitor, 5 ignore.` and ends with status 3, so a script can act on it.
The next line, `12 of 12 events rest on assumed covariances
(COVARIANCE_METHOD = DEFAULT), not orbit determination.`, is the reminder
that these decisions rest on keepout's covariance model.

On live data, download the catalog and screen from now on:

```bash
keepout fetch catalog                  # CelesTrak's public catalog, no account
keepout screen --fleet fleet.toml --days 7 --cdm-dir cdms
```

## Fetching data

Every download goes into keepout's cache: `--cache-dir DIR`, else
`$KEEPOUT_CACHE_DIR`, else the user cache directory (`~/.cache/keepout` on
Linux, `~/Library/Caches/keepout` on macOS). Each source's client keeps to
that source's rules, so running a command again soon reuses the cached copy
instead of downloading.

- **`fetch catalog`** downloads the GP catalog `screen` uses. From
  CelesTrak (the default, `--source celestrak`): its public groups, about
  20 000 objects, no account; a copy is reused for 2 hours (see
  [Screening](screening.md#celestraks-public-catalog)). From Space-Track
  (`--source spacetrack`): the full catalog, about 32 000 objects, with an
  account; a copy is reused for an hour, and Space-Track is asked at most
  once an hour (see [Space-Track data](spacetrack.md)).
- **`fetch cdm`** lists Space-Track's public CDM summaries (`cdm_public`)
  of the approaches to come: both objects, TCA, minimum range and the Pc
  Space-Track reports. They carry no states or covariances, so `assess`
  cannot read them, and they may not be redistributed (see
  [Space-Track data](spacetrack.md#public-cdms)).
- **`fetch cara`** and **`fetch kelvins`** download the reference data of
  the [validation report](validation.md) and the
  [Kelvins risk report](risk-kelvins.md); `validate` fetches CARA's files
  itself when they are missing.

## `screen`

`keepout screen` is the [synthetic operator](synthetic.md): it screens the
satellites of a fleet file against a catalog, turns each close approach
into a CDM whose covariances come from keepout's model of GP element-set
errors, computes the Pc from each CDM and assesses each event.

- **Window.** `--start` (ISO 8601 UTC, default now) and `--days` (default
  7). The start is also each CDM's `CREATION_DATE`: the risk is assessed as
  of the start of the window.
- **Catalog.** By default the cached copy of `keepout fetch catalog`
  (`--source celestrak` or `spacetrack`) while it is current, else a new
  download within the source's limits; `--catalog FILE` reads element sets
  from a file instead (OMM KVN or XML, CelesTrak CSV or JSON, TLE, gzipped
  or not).
- **Volume and model.** A 5 km sphere, as CelesTrak's SOCRATES, and the
  covariance model's published parameters; the library's
  `keepout.synthetic.run` takes others.
- **CDMs.** `--cdm-dir DIR` writes each CDM as `DIR/<MESSAGE_ID>.kvn`. The
  `MESSAGE_ID` is made of both catalog numbers, the TCA and the creation
  time (e.g.
  `38771_35196_20261012T112828.710Z_20261006T000000.000Z`), so the same
  run writes the same files again, and a run on another day adds new files
  beside them rather than overwriting them: `assess` then takes the CDMs of
  one approach from several runs as successive CDMs of one event.
- **Workers.** `--workers N` processes (default: one per CPU, at most 4;
  each runs its numerical libraries on one thread).

`screen` exits with status 0 whatever the decisions: it reports, `assess`
decides.

## `assess`

`keepout assess` reads CDM files (KVN or XML, CDM 1.0 or 2.0, gzipped or
not; the shell expands `*.kvn`), groups them into events (the same two
objects, TCAs within 10 minutes), computes keepout's Pc of every CDM and
assesses each event as in [Risk assessment](risk.md): the risk level of the
latest Pc, the trend over the event's CDMs and the action, decided as of
the latest CDM's creation date.

- `--method` picks the Pc method (`auto` by default: Foster, Hall 3D-Nc or
  a maximum Pc, as [`compute_pc`](pc-3d.md) chooses).
- `--hbr METRES` sets the combined hard-body radius; by default it is the
  sum of both objects' `HBR` (CDM 2.0), and a CDM without them is an error:
  keepout never falls back to a default radius.

A CDM object whose `COVARIANCE_METHOD` is `DEFAULT` has an assumed
covariance, not one from orbit determination; keepout's synthetic operator
writes all its CDMs so. `assess` counts the events with such a CDM in a
line below the counts, e.g. `1 of 2 events rests on assumed covariances
(COVARIANCE_METHOD = DEFAULT), not orbit determination.`, so that a Pc from
a model is not taken for one from tracking.

The exit status is 3 when at least one event's action is `manoeuvre`,
whether or not its covariances are assumed.

## `maneuver`

`keepout maneuver` plans an along-track burn of an event's primary
(`OBJECT1`), as in [Manoeuvre planning](maneuver.md). It reads CDM files
as `assess` does; they must make one event (several are an error that
lists them), and the burns are planned from its latest CDM. After the
week of the synthetic operator above, this plans the yellow Intelsat 39
event, with the target green, and re-screens both burns against the
frozen catalog:

```bash
keepout maneuver cdms/44476_55841_*.kvn --target-pc 1e-7 \
    --catalog tests/screening/data/celestrak-public.csv.gz
```

```text
Event 44476/55841/2026-10-08T12:03:02.771792: 44476 / 55841, TCA 2026-10-08 12:03 UTC, target Pc 1.00e-07.
Burns of the primary
Option      Before TCA (h)  Δv (m/s)        Pc  Range (m)  Method  Target
─────────────────────────────────────────────────────────────────────────
no burn                  -         -  7.38e-06       3647  foster  missed
prograde              12.1   +0.0794  6.54e-08       7771  foster  met
retrograde            12.1   -0.0398  4.64e-08       9157  foster  met
Note: 2-D Pc assumptions questionable: the encounter lasts about 2.51e+03 s, 2.9% of the primary's orbital period (limit 1%); a 3-D Pc may differ
Note: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used
Note: every Pc rests on an assumed covariance (COVARIANCE_METHOD = DEFAULT)
Note: drag is off: the CDM gives the primary no positive CD_AREA_OVER_MASS
Note: relative speed 1.99 m/s is below 100 m/s: the planner looks for the new TCA only within 1/20 of an orbit of TCA, so after a burn the pair can meet again hours away, at a Pc no cell shows; only a re-screen shows it (inside that window the new TCA was measured up to 28 m off at 10 m/s)
Note: 240 of 930 cells have no Pc and are never options: 240 without a closest approach to trust, 0 where Foster's Pc failed
Note: prograde: 2-D Pc assumptions questionable: the encounter lasts about 8.4e+03 s, 9.7% of the primary's orbital period (limit 1%); a 3-D Pc may differ
Note: retrograde: 2-D Pc assumptions questionable: the encounter lasts about 7.17e+03 s, 8.3% of the primary's orbital period (limit 1%); a 3-D Pc may differ
prograde re-screened against 19879 objects (tests/screening/data/celestrak-public.csv.gz) from 2026-10-07T23:57:11 to 2026-10-14T23:57:11 UTC; 35 catalog objects left out, of which it says nothing.
  Event secondary 55841 on the element sets (SGP4, not the CDM's orbit): no approach within 10 min of the CDM's TCA after the burn.
  No new or closer approach at red among the objects screened.
  The burn brings this pair back at 2026-10-09 00:07 UTC at Pc 9.87e-06 (assumed), above the target.
  1 new approach:
    55841 at 2026-10-09 00:07 UTC: range 4113 m, assumed Pc 9.87e-06, yellow
  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
retrograde re-screened against 19879 objects (tests/screening/data/celestrak-public.csv.gz) from 2026-10-07T23:57:11 to 2026-10-14T23:57:11 UTC; 35 catalog objects left out, of which it says nothing.
  Event secondary 55841 on the element sets (SGP4, not the CDM's orbit): no approach within 10 min of the CDM's TCA after the burn.
  No new or closer approach at red among the objects screened.
```

It took 8.7 to 10.6 s over 3 runs (Intel Core i5-14600K). The table gives the event without a burn and the
smallest burn each way that meets the target: its time before TCA, its
signed Δv (negative is retrograde), the Pc and range at the new closest
approach and the method that confirmed it (the event's own). The notes
are the plan's; here the slow GEO encounter leaves a quarter of the grid
without a trusted closest approach. Each re-screen then says what became
of the event's secondary on the element sets (SGP4, not the CDM's orbit),
whether any new or closer approach is at red, when the burn brings the
event's own pair back away from its TCA at or above the target, and lists
the approaches, the highest Pc first (at most 5 of each kind; `--json`
lists all). Here the prograde burn clears the event at its TCA but brings
the same object back twelve hours later at 9.87e-06, above the target: the
planner looks for the new TCA only within a twentieth of an orbit, so only
the re-screen sees it, and that option is not clean. The retrograde burn
is clean, so the command exits with status 0 and shows both. These Pc
values rest on assumed covariances, as the output says.

- **Target.** `--target-pc P` (default 1e-5), compared on the
  conservative basis as in `assess`.
- **One burn.** `--burn-before HOURS` and `--dv M/S` (together) report
  that burn, with the event's method, instead of the smallest burn each
  way; it may be off the grid. The whole grid is still computed first,
  so this takes as long as a plan.
- **Radius.** `--hbr METRES` as for `assess`.
- **Re-screen.** `--catalog FILE` or `--source celestrak|spacetrack` (as
  for `screen`, with `--cache-dir` and `--workers`) screens each burn
  against the catalog for 7 days from the burn. The primary must be in the
  catalog; the secondary need not be (an analyst object is in no public
  catalog), and then the output says in capitals that the event was NOT
  screened. Without a catalog there is no re-screen, and the output says
  that nothing is known of the approaches a burn brings.

Planning takes tens of seconds on the default grid, plus a run of the
event's own method per candidate burn (up to a minute each for a slow
encounter on Monte Carlo). A primary with `MANEUVERABLE = NO`, a latest
CDM created later than 6 h before TCA, an unknown covariance, or a primary
the re-screen cannot find or carry is an error (status 1).

The exit status is 4 when no burn on the grid meets the target, when the
burn given with `--burn-before` and `--dv` misses it, or when every burn
brings an approach at red or the event's own pair back above the target
(away from its TCA, by the re-screen); the last line then says which. One
burn that does neither is enough for status 0, and the others are shown
with what they bring.

## `validate`

`keepout validate` runs every CARA validation case (2-D, 3-D and maximum Pc;
with `--monte-carlo` the Monte Carlo cases, which take minutes to hours) and
prints, per suite and method, the cases passed and the worst relative
error: the table at the top of the [validation report](validation.md). It
exits with status 1 if a case fails. It downloads CARA's files at the pinned
commit on first use. The published report itself is generated in the
repository by `tools/validation_report.py`.

## Exit status

| Status | Meaning |
|---|---|
| 0 | Done. |
| 1 | An error in the data, a download or a file: one line `Error: ...` on standard error, no traceback (also for a refused Space-Track query, with the time to wait). |
| 2 | A usage error: an unknown option, a bad value, a missing file. |
| 3 | `assess`: at least one event needs a manoeuvre. |
| 4 | `maneuver`: no burn on the grid meets the target, the burn given with `--burn-before` and `--dv` misses it, or every burn brings an approach at red or the event's pair back above the target. |

## JSON output

`--json` makes any command print one JSON document on standard output
instead of tables (strict JSON: no `NaN`), for scripts and other tools;
errors still go to standard error with the same exit status. `screen` and
`assess` describe each event in the same shape: `event_id`, `tca`,
`object1` and `object2` (`object_id` and `name`), `miss_distance_m`, `cdms`
(their `MESSAGE_ID`s), `pc`, `level` (`green`, `yellow`, `red` or `null`),
`trend`, `action` (`ignore`, `monitor`, `prepare_manoeuvre` or
`manoeuvre`), `reasons`, `notes` and `observations` (each CDM's
`message_id`, `time_to_tca_s`, `pc`, `method`, `error`, `reported_pc`,
`diagnostics` and `assumed_covariance`) and `assumed_covariance` (true when a CDM of the event
has an object with `COVARIANCE_METHOD = DEFAULT`). Both commands also give
`assumed_covariance_events` at the top: the number of such events, every event
for `screen`. `screen` adds each approach's `relative_speed_m_s` and
`cdm_file`, and lists the element sets it `skipped` with the reason;
`assess` adds `cdm_count` (the CDMs read) and `manoeuvre` (true when it
exits with status 3).

`maneuver` gives `event`, `primary`, `secondary`, `target_pc`, `method`
(the event's own), `assumed_covariance`, `baseline` (`pc`,
`range_at_tca_m`, `method`, `meets_target`), `notes` and `options`: one
per burn, with `direction`, `burn_lead_s`, `delta_v_m_s`, the four keys
of `baseline` and `rescreen` (`null` without a catalog). A `rescreen` has
`conflict` (a new or closer approach is red), `secondary_screened`,
`event_range_m` (`null` when the secondary has no approach within 10
minutes of the CDM's TCA, or was not screened: read
`secondary_screened`), `catalog_objects` (the element sets screened),
`skipped_objects`, `window_start` and `window_end` (UTC, ISO 8601, as
every time in the JSON; the start is the burn), and `new_approaches`,
`closer_approaches` and `event_returns` (the new or closer approaches of
the event's own secondary more than 10 minutes from its TCA at or above
`target_pc`; any makes the burn count like a conflict for status 4), each
approach with `secondary`, `tca`, `range_at_tca_m`, `relative_speed_m_s`,
`pc`, `level` and `assumed_covariance` (always true). With no burn on the
grid, `options` is empty.
