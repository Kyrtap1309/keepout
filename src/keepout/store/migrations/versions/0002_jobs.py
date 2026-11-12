"""The jobs: manoeuvre plans queued on an event's CDM.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade the schema."""
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(12), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("claims", sa.Integer(), nullable=False),
        sa.Column("worker", sa.String(), nullable=True),
        sa.Column("lease_until", sa.String(), nullable=True),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("started_at", sa.String(), nullable=True),
        sa.Column("finished_at", sa.String(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
            name=op.f("fk_jobs_event_id_events"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["cdms.message_id"],
            name=op.f("fk_jobs_message_id_cdms"),
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_jobs_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sqlite_autoincrement=True,
    )
    op.create_index(op.f("ix_jobs_event_id"), "jobs", ["event_id"])
    op.create_index(op.f("ix_jobs_message_id"), "jobs", ["message_id"])
    op.create_index(op.f("ix_jobs_status_created_at"), "jobs", ["status", "created_at"])


def downgrade() -> None:
    """Downgrade the schema: drop the jobs."""
    op.drop_index(op.f("ix_jobs_status_created_at"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_message_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_event_id"), table_name="jobs")
    op.drop_table("jobs")
