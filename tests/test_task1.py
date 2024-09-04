import pandas as pd
import pandas_ta as ta

# Load your stock data
df = pd.read_csv('../data/NVDA_historical_data.csv')

# Ensure the data has the required columns
assert 'Date' in df.columns, "Date column is missing"
assert 'Close' in df.columns, "Close column is missing"
assert 'Open' in df.columns, "Open column is missing"
assert 'High' in df.columns, "High column is missing"
assert 'Low' in df.columns, "Low column is missing"
assert 'Volume' in df.columns, "Volume column is missing"

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date just in case
df = df.sort_values(by='Date')

# Test case: Calculate the Simple Moving Average (SMA)
df['SMA_20'] = ta.sma(df['Close'], length=20)
assert not df['SMA_20'].isnull().all(), "SMA_20 calculation failed"

# Test case: Calculate the Relative Strength Index (RSI)
df['RSI_14'] = ta.rsi(df['Close'], length=14)
assert not df['RSI_14'].isnull().all(), "RSI_14 calculation failed"

# Test case: Calculate the Bollinger Bands
df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.bbands(df['Close'], length=20)
assert not df['BB_upper'].isnull().all(), "BB_upper calculation failed"
assert not df['BB_middle'].isnull().all(), "BB_middle calculation failed"
assert not df['BB_lower'].isnull().all(), "BB_lower calculation failed"

# Print a success message if all tests pass
print("All tests passed successfully.")
