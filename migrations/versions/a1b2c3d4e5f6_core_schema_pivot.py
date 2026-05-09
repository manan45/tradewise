"""Core schema pivot: sessions, universe_membership, earnings_calendar, system_state.

Per IMPLEMENTATION_PLAN.md §6 first-PR scope and final_requirements §8.7.2.

- sessions: TimescaleDB hypertable, system memory for every scenario lifecycle.
  birth_state captures the structured snapshot at session open;
  birth_embedding is the BGE vector used for nearest-neighbour priors.
- universe_membership: point-in-time S&P 500 / Russell 2000 / commodity universe
  membership. Required for survivorship-safe backtests (BACKTESTING.md §3.1).
- earnings_calendar: per-symbol earnings dates + status, used by event blackout
  rules in the risk gateway.
- system_state: single-row table holding the global circuit breaker
  (trading_halted) plus generic key/value flags.

Revision ID: a1b2c3d4e5f6
Revises: 8104afe36029
Create Date: 2026-05-09
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "a1b2c3d4e5f6"
down_revision = "8104afe36029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("opened_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("closed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("scenario", sa.String(length=64), nullable=False),
        sa.Column("regime", sa.String(length=32), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False,
                  server_default="open"),
        sa.Column("mode", sa.String(length=16), nullable=False,
                  server_default="live"),  # live | paper | backtest
        sa.Column("birth_state", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False),
        sa.Column("birth_embedding", postgresql.ARRAY(sa.REAL),
                  nullable=True),  # placeholder; swapped to vector(768) below
        sa.Column("plan", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=True),
        sa.Column("outcome", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=True),
        sa.Column("pnl", sa.Numeric(18, 6), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", "opened_at"),
    )
    # Swap to native pgvector for nearest-neighbour priors.
    op.execute("ALTER TABLE sessions DROP COLUMN birth_embedding")
    op.execute("ALTER TABLE sessions ADD COLUMN birth_embedding vector(768)")

    op.execute(
        "SELECT create_hypertable('sessions', 'opened_at', "
        "if_not_exists => TRUE)"
    )
    op.create_index("ix_sessions_symbol_opened", "sessions",
                    ["symbol", "opened_at"])
    op.create_index("ix_sessions_scenario_opened", "sessions",
                    ["scenario", "opened_at"])
    op.create_index("ix_sessions_status", "sessions", ["status"])

    op.create_table(
        "universe_membership",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("universe", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("entered_at", sa.Date(), nullable=False),
        sa.Column("exited_at", sa.Date(), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.UniqueConstraint("universe", "symbol", "entered_at",
                            name="uq_universe_membership"),
    )
    op.create_index("ix_universe_membership_lookup", "universe_membership",
                    ["universe", "symbol", "entered_at"])

    op.create_table(
        "earnings_calendar",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("earnings_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("session", sa.String(length=8), nullable=True),  # bmo|amc|dmh
        sa.Column("status", sa.String(length=16), nullable=False,
                  server_default="projected"),  # projected|confirmed|reported
        sa.Column("eps_estimate", sa.Numeric(12, 4), nullable=True),
        sa.Column("eps_actual", sa.Numeric(12, 4), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.UniqueConstraint("symbol", "earnings_at",
                            name="uq_earnings_symbol_at"),
    )
    op.create_index("ix_earnings_symbol_at", "earnings_calendar",
                    ["symbol", "earnings_at"])

    op.create_table(
        "system_state",
        sa.Column("id", sa.SmallInteger(), primary_key=True),
        sa.Column("trading_halted", sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column("halt_reason", sa.Text(), nullable=True),
        sa.Column("halted_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("flags", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.CheckConstraint("id = 1", name="ck_system_state_singleton"),
    )
    op.execute("INSERT INTO system_state (id) VALUES (1)")


def downgrade() -> None:
    op.drop_table("system_state")
    op.drop_index("ix_earnings_symbol_at", table_name="earnings_calendar")
    op.drop_table("earnings_calendar")
    op.drop_index("ix_universe_membership_lookup",
                  table_name="universe_membership")
    op.drop_table("universe_membership")
    op.drop_index("ix_sessions_status", table_name="sessions")
    op.drop_index("ix_sessions_scenario_opened", table_name="sessions")
    op.drop_index("ix_sessions_symbol_opened", table_name="sessions")
    op.drop_table("sessions")
