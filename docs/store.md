# The conjunction store

A conjunction is assessed again with every new CDM, over days. The
conjunction store keeps each event's history in one SQLite file, so the
next CDM adds to it instead of starting again. It holds the CDMs, the Pc
keepout computed from them and the catalog a screening used. It also
holds manoeuvre plans, computed in the command or by a background worker.
The command line uses it with `--store FILE` (see
[`keepout store`](cli.md#store)). The library uses `keepout.store`.

```bash
pip install "keepout[cli,store] @ git+https://github.com/Kyrtap1309/keepout"
```

The `cli` extra adds the `keepout` command; `keepout[store]` alone is
enough to use the store from Python. The `store` extra installs [SQLAlchemy](https://www.sqlalchemy.org/) and
[Alembic](https://alembic.sqlalchemy.org/). `import keepout` loads
neither, and neither does `keepout --help`. Without the extra, a command
given `--store` prints `--store needs the store extra: pip install
'keepout[store]'` and exits with status 1.

The store needs SQLite 3.35 or newer, for `UPDATE … RETURNING`, which the
job queue uses. This is the SQLite Python's `sqlite3` module is built
with (`sqlite3.sqlite_version`). With an older one, opening a store is an
error naming its version.

## What it keeps

- **CDMs**, as their original text, KVN or XML, with the file they were
  read from (or `screen`). A CDM is stored once, by its `MESSAGE_ID`. The
  same message given again, in KVN or in XML, is reported as already
  stored. A different message under a stored `MESSAGE_ID` is an error,
  never a silent replacement.
- **Events.** Each CDM joins the stored event it belongs to, or starts a
  new one, by the rule of `keepout assess` and `group_events` (see
  [Events and observations](risk.md#events-and-observations)), applied one
  CDM at a time as they arrive. A CDM that fits two stored events is an
  error naming both, never a silent merge.
- **Pc observations**, one per CDM and per way of computing Pc: the
  method, the covariance policy and the hard-body radius. Settings that
  differ only in risk thresholds or target share them, because the risk
  level, trend and action are decided again on every read, from the
  thresholds of the moment. keepout's version is part of the way too, so
  a new release computes the Pc again.
- **Catalogs.** A CDM that `keepout screen --store` issued keeps a
  reference to the catalog it was screened against: each file's name,
  URL, retrieval time and SHA-256.
- **Manoeuvre plans.** A plan is cached on the CDM it was planned from.
  There is one per settings, catalog and keepout version.
  `keepout maneuver --event ID` and [background jobs](#background-jobs)
  both write them. After a newer CDM, the event is planned again.
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
rebuild` without `hbr_m` in the [settings file](cli.md#settings) (it has
no `--hbr`) skips the CDMs without `HBR`, computes the rest
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
or a slow Pc method. A job leaves that work to a worker. The plan is then
read from the store at once.

```bash
keepout maneuver --event 3befe6f1305d --store keepout.db --queue   # Job 1 queued
keepout worker --store keepout.db --until-idle   # or leave it running; Ctrl+C stops it
keepout jobs list --store keepout.db             # succeeded?
keepout maneuver --event 3befe6f1305d --store keepout.db   # the plan, at once
```

The cache matches on the CDM, the settings and keepout's version. So give
the last command the same options as the queued one.
[Stored events](cli.md#stored-events) runs these commands on the
walk-through's store.

### Submitting

`maneuver --event ID --queue` queues the plan of the event. It takes
`--target-pc`, `--hbr` and `--settings`. A job plans on the default burn
grid.

A job plans the event as it was when queued. It plans from the event's
latest CDM at that time. If that CDM came from a screening, each burn is
re-screened against the catalog it was screened with (the store's copy).
CDMs stored afterwards do not change the job.

Queueing twice does no harm. If the plan is cached already, nothing is
queued. If the same plan (same CDM, same settings) is queued or running,
the command names that job instead of queueing a second one.

### Running a worker

`keepout worker --store FILE` takes the oldest queued job, computes it,
then takes the next. It runs until stopped. With `--until-idle` it stops
once the queue is empty. `--workers N` gives each re-screen N processes,
as for `screen`.

The worker logs each job on standard error. When it stops, it prints how
many jobs it ran.

### Deployment

Several workers can share a store. Each job is taken by one worker only.

Each job runs in a child process of the worker. In a container, run the
worker under an init process (Docker's `init: true`, or `docker run
--init`). The init process reaps the child processes the worker kills.

### What happens to a job

`keepout jobs list --store FILE` shows the jobs, the newest first. A job
is `queued` until a worker takes it, then `running`. It ends in one of
three statuses:

- `succeeded`: its plan is cached, as if `keepout maneuver --event ID`
  had computed it. If both computed it, the first plan is kept.
- `failed`: `jobs list` shows the error. A job whose computation raises
  an error is not tried again, since it would fail the same way. A
  keepout error, such as an unknown covariance or a stored catalog copy
  that was deleted or changed, is shown as its message. Any other error
  is shown as `internal error`, with the traceback in the worker's log.
- `cancelled`: `keepout jobs cancel ID --store FILE` cancels a queued or
  running job. A running job's worker notices at its next check-in and
  drops what it computed. Cancelling a job that has ended is an error
  (status 1).

A job may compute for 10 minutes. Past this time limit, the worker stops
the job and fails it (`stopped at the time limit of 10 minutes`).

SIGTERM or Ctrl+C stops the worker. It puts its job back in the queue
and exits with status 0. The stop does not count against the job.

A worker that dies (SIGKILL, a crash, its machine down) has its job queued
again once it has been silent for 60 s. A job given back like this twice fails
(`the worker stopped twice without an answer`). A job whose re-screen
loses one of its processes fails at the time limit.

??? note "Implementation: heartbeats and leases"
    Two more defaults set the timing:

    - a worker checks in on its job (a heartbeat) every 20 s;
    - a job whose worker has not checked in for 60 s is given back. This
      hold is the job's lease.

    A dead worker stops checking in. Once its lease lapses, the next worker
    to look at the queue queues the job again. A child process killed while
    computing (e.g. for lack of memory) gives its job back the same way, at
    once.

    The [reference](reference/store.md#keepout.store.worker) explains
    leases, claims and attempts.

??? note "Implementation: a re-screen process killed"
    When one process of a re-screen is killed, Python sends SIGTERM to the
    re-screen's other processes and waits for them to end. They ignore
    SIGTERM, like the child process that runs the job, because a stop is
    meant for the worker alone. So the wait never ends, and the job fails
    at the time limit.

The jobs are a table of the store's schema since revision `0002`. A store
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

This adds the CDMs in `cdms/`, the folder the
[synthetic operator's run](cli.md#a-week-of-a-synthetic-operator) writes
(`keepout screen ... --cdm-dir cdms`), to a store and assesses their events:

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
