-- Create enum types
CREATE TYPE trade_action AS ENUM ('buy', 'sell', 'hold');
CREATE TYPE timeframe AS ENUM ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w');

-- Trading Sessions table
CREATE TABLE IF NOT EXISTS trading_sessions (
    id VARCHAR(50) PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    symbol VARCHAR(20) NOT NULL,
    interval timeframe NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    avg_profit DECIMAL(10,2),
    max_drawdown DECIMAL(5,2),
    sharpe_ratio DECIMAL(5,2),
    psychological_state JSONB,
    technical_state JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES trading_sessions(id),
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    predicted_value DECIMAL(10,2) NOT NULL,
    actual_value DECIMAL(10,2),
    confidence DECIMAL(5,2),
    mae DECIMAL(10,4),
    mse DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trade Suggestions table
CREATE TABLE IF NOT EXISTS trade_suggestions (
    id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES trading_sessions(id),
    symbol VARCHAR(20) NOT NULL,
    action trade_action NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    stop_loss DECIMAL(10,2) NOT NULL,
    take_profit DECIMAL(10,2) NOT NULL,
    risk_reward DECIMAL(5,2) NOT NULL,
    timeframe timeframe NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    analysis JSONB,
    signals JSONB,
    psychological_state JSONB,
    technical_state JSONB,
    zone_state JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Session Logs table
CREATE TABLE IF NOT EXISTS session_logs (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES trading_sessions(id),
    timestamp TIMESTAMP NOT NULL,
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market Analysis table
CREATE TABLE IF NOT EXISTS market_analysis (
    id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    technical_analysis JSONB,
    psychological_analysis JSONB,
    zone_analysis JSONB,
    predictions JSONB,
    recommendations JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_sessions_symbol ON trading_sessions(symbol);
CREATE INDEX idx_sessions_timeframe ON trading_sessions(interval);
CREATE INDEX idx_predictions_session ON predictions(session_id);
CREATE INDEX idx_predictions_symbol ON predictions(symbol);
CREATE INDEX idx_suggestions_symbol ON trade_suggestions(symbol);
CREATE INDEX idx_suggestions_session ON trade_suggestions(session_id);
CREATE INDEX idx_logs_session ON session_logs(session_id);
CREATE INDEX idx_analysis_symbol ON market_analysis(symbol);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_trading_sessions_updated_at
    BEFORE UPDATE ON trading_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
