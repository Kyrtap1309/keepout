"""The first schema: events, CDMs, catalogs and their files, results.

Revision ID: 0001
Revises:
Create Date: 2026-11-04
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade the schema."""
    op.create_table(
        "events",
        sa.Column("id", sa.String(12), nullable=False),
        sa.Column("object1", sa.String(), nullable=False),
        sa.Column("object2", sa.String(), nullable=False),
        sa.Column("conjunction_id", sa.String(), nullable=True),
        sa.Column("tca_min", sa.String(), nullable=False),
        sa.Column("tca_max", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_events")),
    )
    op.create_index(
        op.f("ix_events_object1_object2_tca_min"),
        "events",
        ["object1", "object2", "tca_min"],
    )
    op.create_table(
        "catalogs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalogs")),
    )
    op.create_table(
        "catalog_files",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("catalog_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("retrieved", sa.String(), nullable=True),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalogs.id"],
            name=op.f("fk_catalog_files_catalog_id_catalogs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog_files")),
    )
    op.create_index(
        op.f("ix_catalog_files_catalog_id"), "catalog_files", ["catalog_id"]
    )
    op.create_table(
        "cdms",
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(12), nullable=False),
        sa.Column("creation_date", sa.String(), nullable=False),
        sa.Column("tca", sa.String(), nullable=False),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("format", sa.String(3), nullable=False),
        sa.Column("origin", sa.String(), nullable=False),
        sa.Column("catalog_id", sa.Integer(), nullable=True),
        sa.Column("ingested_at", sa.String(), nullable=False),
        sa.CheckConstraint("format IN ('kvn', 'xml')", name=op.f("ck_cdms_format")),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
            name=op.f("fk_cdms_event_id_events"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalogs.id"],
            name=op.f("fk_cdms_catalog_id_catalogs"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_cdms")),
    )
    op.create_index(
        op.f("ix_cdms_event_id_creation_date"), "cdms", ["event_id", "creation_date"]
    )
    op.create_index(op.f("ix_cdms_catalog_id"), "cdms", ["catalog_id"])
    op.create_table(
        "results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["cdms.message_id"],
            name=op.f("fk_results_message_id_cdms"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_results")),
        sa.UniqueConstraint(
            "message_id", "key", name=op.f("uq_results_message_id_key")
        ),
    )


def downgrade() -> None:
    """Downgrade the schema: drop every table."""
    op.drop_table("results")
    op.drop_index(op.f("ix_cdms_catalog_id"), table_name="cdms")
    op.drop_index(op.f("ix_cdms_event_id_creation_date"), table_name="cdms")
    op.drop_table("cdms")
    op.drop_index(op.f("ix_catalog_files_catalog_id"), table_name="catalog_files")
    op.drop_table("catalog_files")
    op.drop_table("catalogs")
    op.drop_index(op.f("ix_events_object1_object2_tca_min"), table_name="events")
    op.drop_table("events")
