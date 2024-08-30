# scripts/load_and_analyze_data.py

import os
import pandas as pd

# Define the path to the data directory
data_dir = os.path.join(os.path.dirname(__file__), '../data')

# Load the financial news dataset
news_data_path = os.path.join(data_dir, 'raw_analyst_ratings.csv')
news_data = pd.read_csv(news_data_path)

# Load the stock price dataset
stock_data_path = os.path.join(data_dir, 'stock_prices.csv')
stock_data = pd.read_csv(stock_data_path)

# Display the first few rows of each dataset
print("Financial News Data:")
print(news_data.head(), "\n")

print("Stock Price Data:")
print(stock_data.head())
