# syntax=docker/dockerfile:1
# keepout's image: the dashboard and the worker (compose.yaml), with the
# demo's store. Base images are pinned by digest; refresh them by hand
# (docker buildx imagetools inspect python:3.13-slim) and rebuild.

# The Python image of the build and of the runtime, one pin for both.
ARG PYTHON_IMAGE=python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

FROM ${PYTHON_IMAGE} AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./

# The demo's store, built with the seed dependency group: the seed tool
# imports the reports' helpers (matplotlib, typeguard). Only the store
# leaves this stage. The dependencies are installed before src is copied,
# so a change to src keeps their layer.
FROM base AS seed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group seed --no-install-project
COPY src src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group seed
COPY tools tools
COPY examples examples
COPY tests/screening/data tests/screening/data
# The store's catalog copies are read-only to their owner (root here): the
# init service, run as keepout, must read them.
RUN uv run --no-sync python tools/dashboard_seed.py --store /out/keepout.db \
    && chmod -R a+rX /out

# The runtime environment: keepout with the cli, store and dashboard extras,
# from the lock, without the development tools.
FROM base AS deps
ENV UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/opt/keepout/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project \
    --extra cli --extra store --extra dashboard
COPY src src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
    --extra cli --extra store --extra dashboard

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --gid 10001 keepout \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin keepout \
    && mkdir /data && chown keepout:keepout /data
COPY --from=deps /opt/keepout/venv /opt/keepout/venv
COPY --from=seed /out /opt/keepout/seed
COPY examples/dashboard/settings.toml /opt/keepout/settings.toml
COPY --chmod=0755 docker/init-store.sh /opt/keepout/init-store.sh
ENV PATH=/opt/keepout/venv/bin:$PATH PYTHONUNBUFFERED=1
# Numeric, so that a runtime can check it is not root without /etc/passwd.
USER 10001:10001
WORKDIR /home/keepout
VOLUME /data
EXPOSE 8501
CMD ["keepout", "--help"]
