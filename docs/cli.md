# Command line

The `keepout` command runs the library's main workflows from a terminal.
It downloads data, screens a fleet and issues CDMs as a
[synthetic operator](synthetic.md), and assesses CDM files. It also plans a
collision-avoidance burn and checks keepout's Pc against NASA CARA's test
cases. It is a thin layer over the library functions the other pages
describe, so it computes exactly what they do.

```bash
pip install "keepout[cli] @ git+https://github.com/Kyrtap1309/keepout"
keepout --help
```

The examples below read files committed to the repository. Run them from a
clone, after `uv sync --extra cli` (see [Install](index.md#install)), with
`uv run keepout ...` or in the activated environment.

The `cli` extra installs [Typer](https://typer.tiangolo.com/), and Rich for
the tables. The library itself never imports either. Without the extra,
`keepout` prints the line to install it and exits with status 1.
`python -m keepout.cli` is the same command.

Command, option and API names use the US spelling `maneuver`, as the CCSDS
keywords do. The prose of these pages uses the British "manoeuvre".

| Command | What it does |
|---|---|
| `keepout fetch catalog [--source celestrak\|spacetrack]` | Download the GP element sets of every object in orbit, for `screen` |
| `keepout fetch cdm` | Download Space-Track's public CDM summaries of approaches to come |
| `keepout fetch cara` | Download NASA CARA's validation files (pinned commit, SHA-256) |
| `keepout fetch kelvins` | Download ESA's Kelvins collision avoidance data set (221 MB) |
| `keepout screen --fleet FILE [--days 7] [--cdm-dir DIR]` | Screen a fleet, issue a CDM per approach, assess each |
| `keepout assess CDM...` | Compute each CDM's Pc and each event's risk and action |
| `keepout maneuver CDM...\|--event ID [--catalog FILE]` | Find the smallest burn each way that takes an event below a target Pc, and re-screen it |
| `keepout validate` | Compare keepout's Pc with CARA's on every validation case |
| `keepout store add CDM... --store FILE` | Keep CDMs in a [conjunction store](store.md), each in its event, and assess the events |
| `keepout store upgrade\|rebuild\|purge --store FILE` | Upgrade a store's schema, compute its missing Pc, delete its old events |
| `keepout jobs list\|cancel --store FILE` | List a store's [background jobs](store.md#background-jobs), cancel one |
| `keepout worker --store FILE [--until-idle] [--workers N]` | Compute the manoeuvre plans queued with `maneuver --event ID --queue` |

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
3befe6f1305d: monitor (latest Pc 7.38e-06 is yellow, 2.50 d before TCA; trend undetermined; diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 2.51e+03 s, 2.9% of the primary's orbital period (limit 1%); a 3-D Pc may differ; diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used)
b815c689dc80: monitor (latest Pc 4.86e-06 is yellow, 6.48 d before TCA; trend undetermined)
Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
12 CDMs written to cdms
```

Each row is one close approach, the fleet satellite first. It shows
keepout's Pc computed from the CDM issued for it, the Pc's risk level and
the recommended action. The events that need more than `ignore` are
explained below the table.

A fleet satellite may not be screened, because its element set is too old
for the window, too far in the future, or fails in SGP4. It then has no row
and is not protected. A line above the table names it, e.g.
`Fleet satellite 44476 (INTELSAT 39 (IS-39)) was NOT screened: stale, ...`,
and the first line says `Screened 2 of 3 fleet satellites`.

!!! warning "Assumed covariances"
    Every Pc on this page that comes from `screen`, or from its CDMs, rests
    on keepout's model of GP element-set errors, not on orbit
    determination. Read [Assumptions](assumptions.md) before reading
    anything into these values.

The CDMs are ordinary CCSDS messages that any CDM reader takes, and
`assess` reads them back like CDMs from any other originator:

```bash
keepout assess cdms/*.kvn             # the same 12 events: exit status 0
keepout assess cdms/*.kvn --hbr 1000  # a 1 km hard body: exit status 3
```

The second command sets a combined hard-body radius of 1 km instead of
each CDM's `HBR`. It prints `12 events from 12 CDMs: 2 maneuver,
4 prepare maneuver, 1 monitor, 5 ignore.` and ends with status 3, so a
script can act on it. The next line says that all 12 events rest on
assumed covariances (`COVARIANCE_METHOD = DEFAULT`), not orbit
determination.

On live data, download the catalog and screen from now on:

```bash
keepout fetch catalog                  # CelesTrak's public catalog, no account
keepout screen --fleet fleet.toml --days 7 --cdm-dir cdms
```

## Fetching data

Every download goes into keepout's cache. That is `--cache-dir DIR` if
given, else `$KEEPOUT_CACHE_DIR`, else the user cache directory
(`~/.cache/keepout` on Linux, `~/Library/Caches/keepout` on macOS). Each
source's client keeps to that source's rules, so running a command again
soon reuses the cached copy instead of downloading.

- **`fetch catalog`** downloads the GP catalog `screen` uses.
    - From CelesTrak (the default, `--source celestrak`) it gets the public
      groups, about 20 000 objects, with no account. A copy is reused for
      2 hours (see [Screening](screening.md#celestraks-public-catalog)).
    - From Space-Track (`--source spacetrack`) it gets the full catalog,
      about 32 000 objects, with an account. A copy is reused for an hour,
      and Space-Track is asked at most once an hour (see
      [Space-Track data](spacetrack.md)).
- **`fetch cdm`** lists Space-Track's public CDM summaries (`cdm_public`)
  of the approaches to come, with an account. Each gives both objects, the
  TCA, the minimum range and the Pc Space-Track reports. They carry no
  states or covariances, so `assess` cannot read them. A copy is reused
  for 8 hours, and Space-Track is asked at most once in 8 hours. They may
  not be redistributed (see [Space-Track data](spacetrack.md#public-cdms)).
- **`fetch cara`** and **`fetch kelvins`** download the reference data of
  the [validation report](validation.md) and the
  [Kelvins risk report](risk-kelvins.md). Each file is checked against its
  SHA-256, and a file already in the cache with the right digest is not
  downloaded again. `validate` fetches CARA's files itself when they are
  missing.

The options of the `fetch` commands:

| Option | `catalog` | `cdm` | `cara` | `kelvins` | Meaning |
|---|---|---|---|---|---|
| `--source celestrak\|spacetrack` | yes | | | | where the catalog comes from (default `celestrak`) |
| `--cache-dir DIR` | yes | yes | yes | yes | the cache root (default `$KEEPOUT_CACHE_DIR`, else the user cache directory) |
| `--json` | yes | yes | yes | yes | print the result as one JSON document |

Space-Track needs an account, read from `SPACETRACK_IDENTITY` and
`SPACETRACK_PASSWORD` (see [Space-Track data](spacetrack.md)).

## `screen`

`keepout screen` is the [synthetic operator](synthetic.md). It screens the
satellites of a fleet file against a catalog and turns each close approach
into a CDM. The CDM's covariances come from keepout's model of GP
element-set errors. It then computes the Pc from each CDM and assesses each
event.

- **Window.** `--start` (ISO 8601 UTC, default now) and `--days` (default
  7). The start is also each CDM's `CREATION_DATE`, so the risk is assessed
  as of the start of the window.
- **Catalog.** By default `screen` uses the cached copy of
  `keepout fetch catalog` (`--source celestrak` or `spacetrack`) while it is
  current, else a new download within the source's limits. `--catalog FILE`
  reads element sets from a file instead: OMM KVN or XML, CelesTrak CSV or
  JSON, or TLE, gzipped or not.
- **Volume and model.** A 5 km sphere, matching CelesTrak's SOCRATES, and
  the covariance model's published parameters. The library's
  `keepout.synthetic.run` takes others.
- **CDMs.** `--cdm-dir DIR` writes each CDM as `DIR/<MESSAGE_ID>.kvn`. The
  `MESSAGE_ID` is made of both catalog numbers, the TCA and the creation
  time, e.g. `38771_35196_20261012T112828.710Z_20261006T000000.000Z`. The
  same run therefore writes the same files again. A run on another day adds
  new files beside them rather than overwriting them, and `assess` takes
  the CDMs of one approach from several runs as successive CDMs of one
  event.
- **Workers.** `--workers N` processes (default: one per CPU, at most 4).
  Each runs its numerical libraries on one thread.
- **Settings.** `--settings FILE` gives the [operator settings](#settings)
  the Pc is computed and the risk assessed with.
- **Store.** `--store FILE` adds the CDMs to a
  [conjunction store](#store), with a copy of the catalog file screened
  against, so that `maneuver` can re-screen a stored event against the
  same catalog later. It also caches their Pc with the settings, as
  `keepout store add` computes them. The rows are then in the store's
  order, the reasons and `--json` name each approach by the id of its
  stored event, and a line under the table gives those ids row by row.
  Each row stays the approach as screened: its one CDM and its Pc. A
  screening on a later day adds its CDM to the same stored event, and
  `keepout store add` or `assess --store` show the event with all of them.

`screen` exits with status 0 whatever the decisions: it reports, and
`assess` decides.

## `assess`

`keepout assess` reads CDM files: KVN or XML, CDM 1.0 or 2.0, gzipped or
not (the shell expands `*.kvn`). It groups them into events, each the same
two objects with TCAs within 10 minutes and one `CONJUNCTION_ID`, if any. It computes keepout's Pc of every
CDM and assesses each event as in [Risk assessment](risk.md). The result is
the risk level of the latest Pc, the trend over the event's CDMs and the
action, decided as of the latest CDM's creation date.

- `--method` picks the Pc method. The default `auto` uses Foster, Hall
  3D-Nc or a maximum Pc, as [`compute_pc`](pc-3d.md#choosing-a-method)
  chooses.
- `--hbr METRES` sets the combined hard-body radius. By default it is the
  sum of both objects' `HBR` (CDM 2.0), and a CDM without them is an error.
  keepout never falls back to a default radius.
- `--settings FILE` gives the [operator settings](#settings): Pc method,
  covariance policy, hard-body radius and risk thresholds. `--method` and
  `--hbr` win over the file.
- `--store FILE` adds the CDMs to a [conjunction store](#store) first, as
  `keepout store add` does with the same settings. Each event is
  then the stored one: its id is the store's, and it has every CDM the
  store holds of it, from earlier runs too.

A CDM object whose `COVARIANCE_METHOD` is `DEFAULT` has an assumed
covariance, not one from orbit determination. keepout's synthetic operator
writes all its CDMs so. `assess` counts the events with such a CDM in a
line below the counts, e.g. `1 of 2 events rests on assumed covariances
(COVARIANCE_METHOD = DEFAULT), not orbit determination.` A Pc from a model
is then not taken for one from tracking.

The exit status is 3 when at least one event's action is `maneuver`,
whether or not its covariances are assumed.

## `maneuver`

`keepout maneuver` plans an along-track burn of an event's primary
(`OBJECT1`), as in [Manoeuvre planning](maneuver.md). It reads CDM files
as `assess` does. They must make one event, and several are an error that
lists them. `--event ID` plans a stored event instead (see
[Stored events](#stored-events) below). The burns are planned from the
event's latest CDM. After the week of the synthetic operator above, this
plans the yellow Intelsat 39 event with the target set to green. It then
re-screens both burns against the frozen catalog:

```bash
keepout maneuver cdms/44476_55841_*.kvn --target-pc 1e-7 \
    --catalog tests/screening/data/celestrak-public.csv.gz
```

```text
Event 3befe6f1305d: 44476 / 55841, TCA 2026-10-08 12:03 UTC, target Pc 1.00e-07.
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
prograde re-screened against 19879 objects (tests/screening/data/celestrak-public.csv.gz) from 2026-10-07T23:57:11 to 2026-10-14T23:57:11 UTC; 35 catalog objects left out (not checked).
  Event secondary 55841 on the element sets (SGP4, not the CDM's orbit): no approach within 10 min of the CDM's TCA after the burn.
  No new or closer approach at red among the objects screened.
  The burn brings this pair back at 2026-10-09 00:07 UTC at Pc 9.87e-06 (assumed), above the target.
  1 new approach:
    55841 at 2026-10-09 00:07 UTC: range 4113 m, assumed Pc 9.87e-06, yellow
  Covariances are assumed (keepout's model of GP element set errors), not orbit determination.
retrograde re-screened against 19879 objects (tests/screening/data/celestrak-public.csv.gz) from 2026-10-07T23:57:11 to 2026-10-14T23:57:11 UTC; 35 catalog objects left out (not checked).
  Event secondary 55841 on the element sets (SGP4, not the CDM's orbit): no approach within 10 min of the CDM's TCA after the burn.
  No new or closer approach at red among the objects screened.
```

It took 9.9 to 10.4 s over 3 runs (Intel Core i5-14600K).

The table gives the event without a burn and the smallest burn each way
that meets the target. For each burn it shows the time before TCA, the
signed Δv (negative is retrograde), the Pc and range at the new closest
approach, and the method that confirmed it (the event's own). The notes
are the plan's. Here the slow GEO encounter leaves a quarter of the grid
without a trusted closest approach.

Each re-screen then reports four things:

- what became of the event's secondary on the element sets (SGP4, not the
  CDM's orbit);
- whether any new or closer approach is at red;
- when the burn brings the event's own pair back, away from its TCA, at or
  above the target;
- the approaches, the highest Pc first (at most 5 of each kind; `--json`
  lists all).

Here the prograde burn clears the event at its TCA. But it brings the same
object back twelve hours later at 9.87e-06, above the target. The planner
looks for the new TCA only within a twentieth of an orbit, so only the
re-screen sees it, and that option is not clean. The retrograde burn is
clean, so the command exits with status 0 and shows both.

- **Target.** `--target-pc P` (default: the settings' `target_pc`, else
  1e-5), compared on the conservative basis as in `assess`.
- **One burn.** `--burn-before HOURS` and `--dv M/S`, given together,
  report that burn with the event's method instead of the smallest burn
  each way. The burn may be off the grid. The whole grid is still computed
  first, so this takes as long as a plan.
- **Radius and settings.** `--hbr METRES` and `--settings FILE` as for
  `assess`; the settings also give the target.
- **Re-screen.** `--catalog FILE` or `--source celestrak|spacetrack`
  screens each burn against the catalog for 7 days from the burn. Both
  work as for `screen`, with `--cache-dir` and `--workers`. The primary
  must be in the catalog. The secondary need not be, since an analyst
  object is in no public catalog, and the output then says in capitals
  that the event was NOT screened. Without a catalog there is no
  re-screen, and the output says that nothing is known of the approaches a
  burn brings.

### Stored events

`--event ID --store FILE` (or `$KEEPOUT_STORE`) plans the event of that id
in a [conjunction store](#store), from every CDM the store holds of it,
in place of CDM files: give one or the other. The JSON `"event"` is then
the store's id. The other options work as with files.

Usually the event was found by `screen --store`. It is then re-screened
against the catalog it was screened with: the store's copy of it, checked
against its SHA-256. The output names it as `stored copy of` the file or
source screened. The plan is cached in the store, on the event's latest
CDM, and the same command with the same settings shows it again at once,
until a newer CDM of the event arrives.

The exceptions:

- `--catalog` or `--source` re-screens against the catalog given instead,
  and plans without the cache. So does `--burn-before`.
- An event whose latest CDM came from `assess --store` or `store add` was
  screened by no one. It is not re-screened without a catalog.
- If the stored copy was deleted or changed since, the command stops with
  an error naming the file (status 1). It never re-screens against
  another catalog. A plan already cached is still shown, since it does
  not need the copy.
- An id the store does not hold is an error (status 1).
- A plan the store cannot take (e.g. locked for longer than 30 s) is
  shown all the same, with `Warning: the plan was not cached:` and the
  reason on standard error.

`--queue` leaves the plan to a [background job](store.md#background-jobs)
instead of computing it in the command. For instance, this queues the
plan of the Intelsat 39 event, in the store that
[`store add`](#store) makes further down:

```bash
keepout maneuver --event 3befe6f1305d --store keepout.db --target-pc 1e-7 --queue
```

```text
Job 1 queued for event 3befe6f1305d.
```

A worker computes the jobs queued, one at a time. With `--until-idle` it
stops once the queue is empty, and says how many it ran (its log goes to
standard error):

```bash
keepout worker --store keepout.db --until-idle
```

```text
Ran 1 job.
```

The same command without `--queue` then shows the plan from the store,
not re-screened, since `store add` stored no catalog:
`keepout maneuver --event 3befe6f1305d --store keepout.db --target-pc 1e-7`.
`keepout jobs list --store keepout.db` lists the jobs, the newest first,
with their status and the error of a failed one; `keepout jobs cancel ID
--store keepout.db` cancels one.

`--queue`:

- needs `--event`;
- takes `--target-pc`, `--hbr` and `--settings`;
- refuses `--catalog`, `--source`, `--burn-before`, `--dv` and `--workers`
  with a usage error (status 2). A job always plans on the default grid,
  against the stored catalog, and `keepout worker --workers` sets its
  processes;
- queues nothing for a plan already cached, and says so;
- prints the id of the same job when it is already queued or running.

Planning takes tens of seconds on the default grid. Each candidate burn
adds a run of the event's own method, up to a minute for a slow encounter
on Monte Carlo ([Time](maneuver.md#time) has the measurements). These are
errors (status 1):

- a primary with `MANEUVERABLE = NO`;
- a latest CDM created later than 6 h before TCA;
- an unknown covariance;
- a primary the re-screen cannot find or carry;
- with `--event`, an id the store does not hold, or a stored catalog
  copy deleted or changed.

The exit status is 4 when no burn can be recommended:

- no burn on the grid meets the target;
- the burn given with `--burn-before` and `--dv` misses it;
- every burn brings an approach at red, or brings the event's own pair
  back above the target away from its TCA (found by the re-screen).

The last line then says which. One burn that does neither is enough for
status 0, and the others are shown with what they bring.

## `store`

`keepout store` keeps CDMs, their events and their Pc in a
[conjunction store](store.md), a SQLite file. It needs the `store` extra
(`pip install "keepout[store]"`, or `uv sync --extra store`). Without it,
any command that uses a store prints the line to install it and exits with
status 1. Every `store` command takes the file as `--store FILE`, or from
`$KEEPOUT_STORE`, as `maneuver --event` does. `assess` and `screen` ignore
the variable: they write to a store only when given `--store`. After the
week of the synthetic operator above, this adds its CDMs to a new store:

```bash
keepout store add cdms/*.kvn --store keepout.db
```

```text
Conjunction events
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
12 events in the store for the 12 CDMs read: 2 monitor, 10 ignore.
12 of 12 events rest on assumed covariances (COVARIANCE_METHOD = DEFAULT), not orbit determination.
12 CDMs added to keepout.db (0 already stored), 12 new events.
Event ids, row by row: 026443d77699, ea4c6a976e3a, a351b16057ec, fd13fbdb9831, 6d45833fc2cc, 3f4c98d01163, 2aa19608e416, 3befe6f1305d, 527a3e2689d5, d604685adedc, 26151ac6b42f, b815c689dc80.
3befe6f1305d: monitor (latest Pc 7.38e-06 is yellow, 2.50 d before TCA; trend undetermined; diagnostic: 2-D Pc assumptions questionable: the encounter lasts about 2.51e+03 s, 2.9% of the primary's orbital period (limit 1%); a 3-D Pc may differ; diagnostic: the covariances of primary (44476) and secondary (55841) are known only in position (zero velocity block): the 3-D methods need the velocity uncertainty, so the 2-D Pc is used)
b815c689dc80: monitor (latest Pc 4.86e-06 is yellow, 6.48 d before TCA; trend undetermined)
```

The output is that of `assess`, with the store's event ids. The counts
line says the events are the store's, since a stored event may hold
CDMs from earlier runs as well as those read. The
[ids](store.md#event-ids) are those `screen` showed. Adding the same
files again stores nothing: the line under the counts then says
`0 CDMs added to keepout.db (12 already stored), 0 new events.` A CDM
from a later run joins its stored event, whose row then counts its CDMs
and whose trend runs over all of them. The exit status is that of
`assess`: 3 when an event needs a manoeuvre.

- **`store add CDM...`** adds CDM files, KVN or XML, as they are. Each
  joins its stored event or starts a new one, by the rule in
  [What it keeps](store.md#what-it-keeps). The Pc of every CDM of those
  events is computed, unless the store has it already. `--settings FILE`
  gives the [operator settings](#settings).
- **`store upgrade`** brings a store made by an older keepout to this
  keepout's schema, and prints the revisions, e.g.
  `Upgraded keepout.db: base -> 0001.` Any other command refuses such a
  store and says to run it ([Upgrading](store.md#upgrading)).
- **`store rebuild`** computes the Pc of every stored CDM that the store
  lacks for the settings (`--settings FILE`), and prints how many. Without
  `hbr_m` in the settings, it skips the CDMs without `HBR` and says how
  many: `1 CDM without HBR skipped: set hbr_m in the settings file.`
- **`store purge --before TIME`** deletes the events whose latest TCA is
  before `TIME` (ISO 8601 UTC), with their CDMs, their Pc and the catalog
  copies no other CDM needs. It asks `Delete N events with TCA before
  ...? [y/N]` first; `--yes` does not ask:

```bash
keepout store purge --before 2026-10-09 --yes --store keepout.db
```

```text
Deleted 8 events with TCA before 2026-10-09T00:00:00 UTC from keepout.db.
```

`screen --store` and `assess --store` add their CDMs to a store the same
way, and show the store's ids too. [The conjunction store](store.md)
explains what it keeps, and how to back it up.

## `validate`

`keepout validate` runs every CARA validation case: 2-D, 3-D and maximum
Pc. It prints the cases passed and the worst relative error per suite and
method, as in the table at the top of the
[validation report](validation.md). It exits with status 1 if a case fails.
It downloads CARA's files at the pinned commit on first use. The published
report itself is generated in the repository by
`tools/validation_report.py`.

- `--monte-carlo` runs the Monte Carlo cases instead of the others. They
  take minutes to hours.
- `--workers N` sets the processes (default: one per CPU, at most 4).
- `--cache-dir DIR` sets the cache root, as for `fetch`.
- `--json` prints the result as one JSON document.

## Threads

`keepout` runs the numerical libraries (OpenMP, OpenBLAS, MKL) on one
thread: it sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` to 1 unless they are already set. More threads made a
manoeuvre plan no faster, at 20 times the CPU time (see
[Time](maneuver.md#time)). Parallel work uses processes instead
(`--workers`). Set a variable yourself to override it.

## Settings

`assess`, `screen`, `maneuver`, `store add` and `store rebuild` take the
operator's settings from a TOML file, `--settings FILE`. Every key is
optional and takes the library's default when missing; an unknown key is
an error, so that a typo does not pass for a default:

```toml
[pc]
method = "foster"     # a Pc method, or "auto" (the default)
policy = "repair"     # what to do with an invalid covariance
hbr_m = 20.0          # combined hard-body radius [m]; else the CDMs' HBR

[risk]
green_below = 1e-7
red_from = 1e-4

[maneuver]
target_pc = 1e-5
```

An option given on the command line wins over the file: `--method`,
`--hbr` and `--target-pc`. A bad file ends the command with status 1,
naming the file and the key. From Python, `keepout.settings.load_settings`
reads such a file into [`Settings`](reference/settings.md).

## Exit status

| Status | Meaning |
|---|---|
| 0 | Done. |
| 1 | An error in the data, a download or a file. One line `Error: ...` goes to standard error, with no traceback. A refused Space-Track query also gives status 1, with the time to wait. |
| 2 | A usage error: an unknown option, a bad value, a missing file. |
| 3 | `assess`: at least one event needs a manoeuvre. |
| 4 | `maneuver`: no burn can be recommended (see [`maneuver`](#maneuver)). |

## JSON output

`--json` makes a command print one strict JSON document on standard
output instead of tables, for scripts and other tools. Errors still go to
standard error with the same exit status. [JSON output](cli-json.md)
gives the keys of each command; `keepout worker` has no `--json`.
