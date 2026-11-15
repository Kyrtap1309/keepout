# syntax=docker/dockerfile:1
# keepout's image: the dashboard and the worker (compose.yaml), with the
# demo's store. Base images are pinned by digest; refresh them by hand
# (docker buildx imagetools inspect python:3.13-slim) and rebuild.

FROM ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 AS uv

# The demo's store, built with the development environment: the seed tool
# imports the reports' helpers (matplotlib, typeguard). Only the store
# leaves this stage.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS seed
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked
COPY tools tools
COPY examples examples
COPY tests/screening/data tests/screening/data
# The store's catalog copies are read-only to their owner (root here): the
# init service, run as keepout, must read them.
RUN uv run --locked python tools/dashboard_seed.py --store /out/keepout.db \
    && chmod -R a+rX /out

# The runtime environment: keepout with the cli, store and dashboard extras,
# from the lock, without the development tools.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/keepout/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
    --extra cli --extra store --extra dashboard

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS runtime
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin keepout \
    && mkdir /data && chown keepout:keepout /data
COPY --from=deps /opt/keepout/venv /opt/keepout/venv
COPY --from=seed /out /opt/keepout/seed
COPY examples/dashboard/settings.toml /opt/keepout/settings.toml
COPY docker/init-store.sh /opt/keepout/init-store.sh
ENV PATH=/opt/keepout/venv/bin:$PATH PYTHONUNBUFFERED=1
USER keepout
WORKDIR /home/keepout
VOLUME /data
EXPOSE 8501
CMD ["keepout", "--help"]
