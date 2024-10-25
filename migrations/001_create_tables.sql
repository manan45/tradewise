-- Create stocks table
CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    current_price NUMERIC NOT NULL,
    predicted_price NUMERIC
);

-- Create stock_price_history table
CREATE TABLE IF NOT EXISTS stock_price_history (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER NOT NULL
);

-- Create trade_suggestions table
CREATE TABLE IF NOT EXISTS trade_suggestions (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    action VARCHAR(10) NOT NULL,
    price NUMERIC NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create aggregated stock data tables for different timeframes
CREATE TABLE IF NOT EXISTS stock_data_1min (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_data_5min (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_data_15min (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_data_1hour (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stock_data_1day (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume INTEGER NOT NULL
);

-- Create indexes for better query performance
CREATE INDEX idx_stock_price_history_stock_id ON stock_price_history(stock_id);
CREATE INDEX idx_stock_price_history_timestamp ON stock_price_history(timestamp);
CREATE INDEX idx_trade_suggestions_stock_id ON trade_suggestions(stock_id);
CREATE INDEX idx_stock_data_1min_stock_id_timestamp ON stock_data_1min(stock_id, timestamp);
CREATE INDEX idx_stock_data_5min_stock_id_timestamp ON stock_data_5min(stock_id, timestamp);
CREATE INDEX idx_stock_data_15min_stock_id_timestamp ON stock_data_15min(stock_id, timestamp);
CREATE INDEX idx_stock_data_1hour_stock_id_timestamp ON stock_data_1hour(stock_id, timestamp);
CREATE INDEX idx_stock_data_1day_stock_id_timestamp ON stock_data_1day(stock_id, timestamp);
