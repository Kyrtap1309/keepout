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
'keepout[store]'` and exits with status 1.

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
  [`group_events`](reference/events.md), applied one CDM at a time. A CDM
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

### Event ids

An event gets an id when its first CDM is stored: eight random
lowercase hexadecimal characters, such as `81a87dfa`. It keeps that id
however many CDMs follow, and the command line shows it in place of the
`<object>/<object>/<TCA>` id that `assess` gives without a store. To see
the ids again, run `keepout store add` on the same files: it stores
nothing new and prints the ids row by row.

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
latest TCA is before `TIME`, with their CDMs and Pc, the catalog
references no remaining CDM needs and their copies. It asks first, with
the number of events, unless given `--yes`. Nothing else ever deletes
from the store.

## Concurrent use and backups

The store uses SQLite's write-ahead log (WAL): readers do not wait for
the writer, and two writers queue, each waiting up to 30 s for the other.
A long Pc computation runs outside any transaction, so it does not hold
the others up. While the store is in use, the file `keepout.db` comes
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

from keepout.risk import assess
from keepout.store import CdmInput, open_store
from keepout.settings import Settings

settings = Settings()
with open_store("keepout.db") as store:
    report = store.ingest([CdmInput.read(path) for path in Path("cdms").glob("*.kvn")])
    store.refresh(settings, event_ids=report.events)
    for event_id in report.events:
        observations = store.observations(event_id, settings)
        decision = assess(observations, thresholds=settings.thresholds).decision
        print(event_id, decision.action.label)
```

`refresh` computes the Pc the store lacks for the settings and
`observations` reads it back; the [reference](reference/store.md) has the
rest.
