# Deployment with Docker

`docker compose up` runs the [dashboard](dashboard.md) and a
[worker](store.md#background-jobs) on a demo store, from a clone of the
repository, with Docker and Docker Compose v2 as the only requirements:

```bash
git clone https://github.com/Kyrtap1309/keepout && cd keepout
docker compose up --wait     # builds the image the first time (a few minutes)
```

Then open http://127.0.0.1:8501. `docker compose down` stops it and keeps
the data; `docker compose down -v` also deletes them, and the next `up`
starts from the demo again.

!!! warning "For this machine only"
    The dashboard has no authentication: anyone who reaches its port can
    read every event, store CDMs and queue or cancel jobs. `compose.yaml`
    publishes the port on the host's 127.0.0.1 only. Do not change that to
    `0.0.0.0` or put the port behind a public proxy. From another machine,
    use an SSH tunnel: `ssh -L 8501:127.0.0.1:8501 HOST`, then open
    http://127.0.0.1:8501.

## What runs

`compose.yaml` runs three services from one image, `keepout:local`, built
from `Dockerfile`:

- **`init`** gives the `data` volume the demo store if it has no store
  (`/data/keepout.db`), then exits. A volume that has a store, from a
  previous `up` or your own, is left as it is.
- **`dashboard`** is `keepout dashboard` on that store. Inside the container
  it listens on every address (`--host 0.0.0.0`, which prints a warning in
  its log); the host publishes it on 127.0.0.1:8501 only. It names
  itself `http://localhost:8501` in its log, so Streamlit does not look the
  machine's public IP up over the network. Its healthcheck
  asks Streamlit's `/_stcore/health`, and `up --wait` waits for it.
- **`worker`** is `keepout worker` on the same store: it computes the Pc of
  uploaded CDMs and the manoeuvre plans queued from the dashboard.
  `docker compose stop` sends it SIGTERM; it puts its running job back in
  the queue and exits.

Both run under an init process (`init: true`), which reaps the child
processes Streamlit and the jobs run in. Both run as the user `keepout`
(uid 10001), never root, and restart unless stopped.

`KEEPOUT_PORT` picks another host port when 8501 is taken:
`KEEPOUT_PORT=8601 docker compose up --wait`, then http://127.0.0.1:8601.

## The demo

The image holds a store built from `examples/dashboard/` when it is built
(`tools/dashboard_seed.py --store`): the 12 events and 37 CDMs of the
[demo](dashboard.md#the-demo), their Pc, and the manoeuvre plan of event
`3befe6f1305d`, so the planner shows its heatmap at once.

`compose.yaml` fixes the time the data are seen at to 2026-10-07 18:00 UTC
(`--as-of`): 6 events are ahead. To see the event's Pc rise, upload
`examples/dashboard/demo/*.kvn` on the Events page (**Upload CDMs**). It is
a fifth CDM of event `3befe6f1305d`, issued at 18:00 with a smaller miss
and covariance. The worker computes its Pc within seconds, and the event
turns red with a rising trend. The [README's GIF](https://github.com/Kyrtap1309/keepout#readme)
shows exactly that.

The planner names the catalog its re-screen used as
`stored copy of /src/tests/screening/data/celestrak-public.csv.gz`: the
path it had when the image was built, kept by the store as a label.

## Your own data

- **Your own time.** Remove the two `--as-of` items from the `dashboard`
  command in `compose.yaml`: the dashboard then reads the clock.
- **CDM files.** Upload them on the Events page, or add them from the
  command line:

    ```bash
    docker compose cp ./cdms worker:/tmp/cdms
    docker compose exec worker sh -c \
        'keepout store add /tmp/cdms/*.kvn --store /data/keepout.db --settings /opt/keepout/settings.toml'
    ```

- **Your settings.** The services read `/opt/keepout/settings.toml`, a
  copy of `examples/dashboard/settings.toml`. Mount your own over it, in
  both services, in a `compose.override.yaml` next to `compose.yaml`, which
  Compose reads too:

    ```yaml
    services:
      dashboard:
        volumes:
          - ./settings.toml:/opt/keepout/settings.toml:ro
      worker:
        volumes:
          - ./settings.toml:/opt/keepout/settings.toml:ro
    ```

    A Pc cached with other settings shows as not computed; **Compute**
    queues it for the worker.

- **Without the demo.** Every demo event is before 2026-10-14; delete them
  with `docker compose exec worker keepout store purge --store
  /data/keepout.db --before 2026-10-14 --yes`.

## Backups

Stop the services, so that SQLite's write-ahead log is merged into the
file, then copy the whole volume:

```bash
docker compose stop
docker compose cp worker:/data/. ./keepout-backup/
docker compose start
```

`keepout-backup/` then holds `keepout.db` and its catalog copies
(`keepout.db.catalogs/`), which the re-screens need.

## Updating

After `git pull`, rebuild and restart: `docker compose up --build --wait`.
The volume keeps its store. If the new keepout changed the store's schema,
the dashboard and the worker stop at start, saying so in their logs
(`docker compose logs`), and restart over and over; `up --wait` reports the
dashboard unhealthy. Migrate the store in a container of its own, then
start again ([Upgrading](store.md#upgrading)):

```bash
docker compose run --rm --no-deps worker keepout store upgrade --store /data/keepout.db
docker compose up --wait
```

The base images (`python:3.13-slim` and `uv`) are pinned by digest in
`Dockerfile`, so a build gives the same image until they are changed. To
take their security updates, look up the current digests
(`docker buildx imagetools inspect python:3.13-slim`), replace them in
`Dockerfile` and rebuild.

## Checked in CI

CI's `e2e` job builds the image, runs `docker compose up --wait`, requests
the health check and the first page, runs a browser scenario through the
three pages with an upload (Playwright, `tests/e2e`), restarts the
services and checks that the upload is still stored.
