# The conjunction store

A conjunction is assessed again with every new CDM, over days. The
conjunction store keeps the CDMs of each event, the Pc keepout computed
from them and the catalog a screening used, in one SQLite file, so that
the next CDM adds to the event's history instead of starting it again.
The command line uses it with `--store FILE` (see
[`keepout store`](cli.md#store)); the library with `keepout.store`.

```bash
pip install "keepout[store] @ git+https://github.com/Kyrtap1309/keepout"
```

The `store` extra installs [SQLAlchemy](https://www.sqlalchemy.org/) and
[Alembic](https://alembic.sqlalchemy.org/). `import keepout` loads
neither, and neither does `keepout --help`. Without the extra, a command
given `--store` prints `--store needs the store extra: pip install
'keepout[store]'` and exits with status 1. The store needs SQLite 3.35 or
newer, the one Python's `sqlite3` module is built with
(`sqlite3.sqlite_version`); with an older one, opening a store is an
error naming its version.

## What it keeps

- **CDMs**, as their original text, KVN or XML, with the file they were
  read from (or `screen`). A CDM is stored once, by its `MESSAGE_ID`. The
  same message given again, in KVN or in XML, is reported as already
  stored. A different message under a stored `MESSAGE_ID` is an error,
  never a silent replacement.
- **Events.** Each CDM joins the stored event of the same `OBJECT1` and
  `OBJECT2` whose TCAs, with its own, spread over no more than 10 minutes
  (and whose `CONJUNCTION_ID`, if both give one, is the same), or starts a
  new event. This is the rule of `keepout assess` and
  [`group_events`](reference/events.md), applied one CDM at a time as
  they arrive (`group_events` takes them in TCA order). A CDM
  that fits two stored events is an error naming both, never a silent
  merge.
- **Pc observations**, one per CDM and per way of computing Pc: the
  method, the covariance policy and the hard-body radius. Settings that
  differ only in risk thresholds or target share them, because the risk
  level, trend and action are decided again on every read, from the
  thresholds of the moment. keepout's version is part of the way too, so
  a new release computes the Pc again.
- **Catalogs.** A CDM that `keepout screen --store` issued keeps a
  reference to the catalog it was screened against: each file's name,
  URL, retrieval time and SHA-256.
- **Manoeuvre plans**, one per event's latest CDM and per way of planning:
  keepout's version, the settings (target and thresholds among them) and
  the catalog re-screened against. `keepout maneuver --event ID` keeps
  the plan it computes, and a [background job](#background-jobs) the plan
  it computes; a newer CDM of the event is planned again.
- **Jobs**: the manoeuvre plans queued for a worker, and how far each got.

### Event ids

An event's id is the first 12 lowercase hexadecimal characters of the
SHA-256 of its first CDM's `MESSAGE_ID` (the oldest by `CREATION_DATE`),
such as `3befe6f1305d`: the same CDMs give the same id with a store or
without, on any machine. A stored event keeps its id however many CDMs
follow. So a CDM created before the event's first one, stored later, does
not change it, while `assess` on the files alone then names the event
after that older CDM. In the rare case that another stored event already
has the id, the new event takes the next free one. To see the ids again,
run `keepout store add` on the same files: it stores nothing new and
prints the ids row by row.

### The catalog copies

The catalog files themselves are copied beside the store file, in
`<store file>.catalogs/` (`keepout.db.catalogs/` for `keepout.db`), each
named by its SHA-256. A later download that replaces the cached catalog
does not change them, so a stored event is always re-screened against the
catalog it was found with (`keepout maneuver --event ID`, see
[Stored events](cli.md#stored-events)). The store refers to them by paths
relative to its own directory: the store file and its `.catalogs`
directory together are self-contained, and can be moved or copied
elsewhere as a pair. Every `screen --store` against a refreshed catalog
adds a copy of a few MB, and only `keepout store purge` removes them.

## Settings

`keepout store add` and `keepout store rebuild` compute Pc with the
[operator settings](cli.md#settings) of `--settings FILE`; `assess
--store` and `screen --store` cache it with the settings they compute
with.

The CDMs are stored before their Pc is computed. If the Pc fails, e.g.
`assess --store` on a CDM without `HBR`, the CDMs stay stored, and the
same command run again (with `--hbr`) computes their Pc. `keepout store
rebuild` without `hbr_m` skips the CDMs without `HBR`, computes the rest
and says how many it skipped.

## Upgrading

The store's schema is versioned (Alembic migrations shipped with
keepout). A keepout opens only a store at its own schema revision:

- a store made by an older keepout is refused with `... was made by an
  older keepout: run keepout store upgrade`. `keepout store upgrade
  --store FILE` migrates it in one transaction;
- a store made by a newer keepout is refused, and the file is left as it
  is: no downgrade.

Back the store up before upgrading it.

## Deleting old events

`keepout store purge --before TIME --store FILE` deletes the events whose
latest TCA is before `TIME`, with their CDMs, Pc, manoeuvre plans and
jobs, the catalog references no remaining CDM needs and their copies. A
worker computing a job of a purged event drops what it computed. It asks
first, with the number of events, unless given `--yes`. Nothing else
ever deletes from the store.

## Background jobs

Planning an event takes tens of seconds, up to minutes with a re-screen
or a slow Pc method. A job leaves it to a worker, and the plan is then read
from the store at once, by `keepout maneuver --event ID`.

```bash
keepout maneuver --event 3befe6f1305d --store keepout.db --queue
keepout worker --store keepout.db
```

[Stored events](cli.md#stored-events) runs both on the walk-through's
store.

- **Submitting.** `maneuver --event ID --queue` queues the plan of the
  event with the settings given (`--target-pc`, `--hbr`, `--settings`).
  A job is pinned to the event's latest CDM when queued, and the catalog
  that CDM was screened against (the store's copy): a CDM stored later
  changes nothing it computes. A job plans on the default burn grid and
  re-screens each burn against that catalog, if there is one. Queueing
  is idempotent: if the plan is cached already, nothing is queued; if the
  same plan (same CDM, same settings) is queued or running, that job is
  given back instead of a second one.
- **The worker.** `keepout worker --store FILE` takes the oldest queued
  job, computes it and takes the next, until stopped; with `--until-idle`
  it stops once the queue is empty. `--workers N` gives each re-screen N
  processes, as for `screen`. Each job is computed in a child process,
  which only reads the store; the worker alone writes the job's end and
  its plan. The worker logs each job on standard error, and prints how
  many it ran when it stops. Several workers can share a store: a job is
  taken by one of them only. In a container, run the worker under an init
  process (Docker's `init: true`, or `docker run --init`), so that the
  processes it kills are reaped.
- **Statuses.** A job is `queued`, then `running` while a worker holds
  it, and ends `succeeded`, `failed` (with its error) or `cancelled`.
  `keepout jobs list --store FILE` shows them, the newest first.
- **Leases and attempts.** A worker holds its job by a lease of 60 s,
  renewed every 20 s. A worker that dies (killed, its machine down)
  stops renewing it: once the lease lapses, the next worker to look at
  the queue puts the job back as `queued`, or `failed` if it was its
  second attempt (`the worker stopped twice without an answer`). A
  child process that exits without an answer (a crash, killed for lack
  of memory) costs an attempt the same way, at once.
- **Errors.** A job whose computation raises fails at once and is not
  tried again: the same input would fail the same way. A keepout error
  (an unknown covariance, a primary that cannot manoeuvre, a stored
  catalog copy deleted or changed) is stored as its message; any other
  error as `internal error`, with the traceback in the worker's log.
- **Time limit.** A job that computes for more than 10 minutes is
  stopped and fails (`stopped at the time limit of 10 min`). The child
  runs in a process group of its own and is killed with it, its
  re-screen's processes too. A re-screen whose process was killed (e.g.
  for lack of memory) hangs until the time limit: its processes ignore
  Ctrl+C and SIGTERM.
- **Stopping.** SIGTERM or Ctrl+C stops the worker: it kills the child,
  gives its job back to the queue without counting the attempt and
  exits with status 0. A worker killed at once (SIGKILL) cannot: its
  child notices within a second or so and goes, with its processes, and
  the job's lease lapses as for a dead worker.
- **Cancelling.** `keepout jobs cancel ID --store FILE` cancels a queued
  or running job; cancelling a job that has ended is an error (status
  1). The worker of a running job notices at its next lease renewal,
  within 20 s, kills the child and drops what it computed.
- **The cache.** A job that succeeds caches its plan on the CDM it was
  pinned to, as `keepout maneuver --event ID` caches the plan it
  computes in the command; whichever is first is kept. That command
  shows the cached plan while that CDM is the event's latest and the
  settings and keepout's version are the same. After a newer CDM, it
  plans again (or queue it again).

The jobs are a table of the store's schema since revision `0002`: a store
made before needs `keepout store upgrade`.

## Concurrent use and backups

The store uses SQLite's write-ahead log (WAL): readers do not wait for
the writer, and two writers queue, each waiting up to 30 s for the other.
A long Pc computation runs outside any transaction, so it does not hold
the others up; the Pc computed is written every 100 CDMs or every second,
so an interrupted `store rebuild` keeps nearly all it computed. While the store is in use, the file `keepout.db` comes
with `keepout.db-wal` and `keepout.db-shm` beside it, and the latest
writes may be in the `-wal` file only.

So do not copy the store with `cp` while anything uses it: the copy can
miss writes or be inconsistent. Use SQLite's online backup, which is
consistent at any time, and copy the catalogs directory with it:

```bash
sqlite3 keepout.db ".backup backup/keepout.db"
cp -r keepout.db.catalogs backup/
```

Copy the catalogs after the database: a copy of a newer catalog does no
harm, while a missing one makes its events fail to re-screen.

## From Python

```python
from pathlib import Path

from keepout.settings import Settings
from keepout.store import CdmInput, open_store

with open_store("keepout.db") as store:
    report = store.ingest([CdmInput.read(path) for path in Path("cdms").glob("*.kvn")])
    for item in store.assess(report.events, Settings()):
        print(item.event.event_id, item.assessment.decision.action.label)
```

`assess` computes and caches the Pc the store lacks for the settings
(`refresh`), then judges each event from all its stored CDMs. `cache`
keeps Pc computed elsewhere, as `screen --store` does with the Pc of its
screening; the [reference](reference/store.md) has the rest.

`JobQueue(store).submit(event_id, settings)` queues a plan and
`keepout.store.worker.run_worker(path)` runs a worker, as the commands do
(see the [reference](reference/store.md#keepout.store.worker)).
