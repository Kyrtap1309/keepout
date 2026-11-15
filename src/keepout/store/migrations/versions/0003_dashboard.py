"""The dashboard: a job's kind, and the indexes of the events list.

A job of a store from before is a manoeuvre plan (``kind`` defaults to
``maneuver``). The events list filters and orders by the latest TCA
(``ix_events_tca_max``) and looks up the active jobs of its events
(``ix_jobs_event_id_status``, which replaces ``0002``'s
``ix_jobs_event_id``, its prefix). Its other lookups use the indexes of
``0001``: the CDMs of an event by creation (``ix_cdms_event_id_creation_date``),
a result by CDM and key (the unique ``uq_results_message_id_key``), an event
by its objects (``ix_events_object1_object2_tca_min``).

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
    op.add_column(
        "jobs",
        sa.Column("kind", sa.String(), server_default="maneuver", nullable=False),
    )
    op.create_index(op.f("ix_jobs_event_id_status"), "jobs", ["event_id", "status"])
    op.drop_index(op.f("ix_jobs_event_id"), table_name="jobs")
    op.create_index(op.f("ix_events_tca_max"), "events", ["tca_max"])


def downgrade() -> None:
    """Downgrade the schema: drop the indexes and the jobs' kind."""
    op.drop_index(op.f("ix_events_tca_max"), table_name="events")
    op.create_index(op.f("ix_jobs_event_id"), "jobs", ["event_id"])
    op.drop_index(op.f("ix_jobs_event_id_status"), table_name="jobs")
    op.drop_column("jobs", "kind")
