CREATE TABLE IF NOT EXISTS stock_data_1min (
    timestamp TIMESTAMP PRIMARY KEY,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
CREATE TABLE IF NOT EXISTS stock_data_2min (
    timestamp TIMESTAMP PRIMARY KEY,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
-- Repeat for other timeframes...
-- Repeat for other timeframes...
