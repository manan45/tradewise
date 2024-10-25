from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table('stocks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('current_price', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('symbol')
    )
    op.create_table('stock_prices',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_symbol', sa.String(), nullable=False),
        sa.Column('open', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('high', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('low', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('close', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('volume', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['stock_symbol'], ['stocks.symbol'], ),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('stock_prices')
    op.drop_table('stocks')
