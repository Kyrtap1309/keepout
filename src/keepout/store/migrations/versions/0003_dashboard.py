"""The dashboard: a job's kind, and the indexes of the events list.

A job of a store from before is a manoeuvre plan (``kind`` defaults to
``maneuver``); a kind is one of :class:`~keepout.store.models.JobKind`
(``ck_jobs_kind``). The events list filters and orders by the latest TCA
(``ix_events_tca_max``) and looks up the active jobs of its events
(``ix_jobs_event_id_status``, which replaces ``0002``'s
``ix_jobs_event_id``, its prefix).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade the schema."""
    # SQLite adds a CHECK by copying the table (batch mode), which keeps
    # its AUTOINCREMENT only if told.
    with op.batch_alter_table(
        "jobs", table_kwargs={"sqlite_autoincrement": True}
    ) as batch:
        batch.add_column(
            sa.Column("kind", sa.String(), server_default="maneuver", nullable=False)
        )
        batch.create_check_constraint(
            op.f("ck_jobs_kind"), "kind IN ('maneuver', 'refresh')"
        )
    op.create_index(op.f("ix_jobs_event_id_status"), "jobs", ["event_id", "status"])
    op.drop_index(op.f("ix_jobs_event_id"), table_name="jobs")
    op.create_index(op.f("ix_events_tca_max"), "events", ["tca_max"])


def downgrade() -> None:
    """Downgrade the schema: drop the indexes and the jobs' kind."""
    op.drop_index(op.f("ix_events_tca_max"), table_name="events")
    op.create_index(op.f("ix_jobs_event_id"), "jobs", ["event_id"])
    op.drop_index(op.f("ix_jobs_event_id_status"), table_name="jobs")
    with op.batch_alter_table(
        "jobs", table_kwargs={"sqlite_autoincrement": True}
    ) as batch:
        batch.drop_constraint(op.f("ck_jobs_kind"), type_="check")
        batch.drop_column("kind")
