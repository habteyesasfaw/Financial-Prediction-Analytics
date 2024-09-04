import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime

# Load financial news data
news_data = pd.read_csv('../data/raw_analyst_ratings.csv')

# Load stock data
stock_files = [
    'NVDA_historical_data.csv',
    'AAPL_historical_data.csv',
    'AMZN_historical_data.csv',
    'GOOG_historical_data.csv',
    'META_historical_data.csv',
    'MSFT_historical_data.csv',
    'TSLA_historical_data.csv'
]

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Normalize dates and add sentiment scores to news data
news_data['date'] = pd.to_datetime(news_data['date'], format='%Y-%m-%d %H:%M:%S').dt.date
news_data['sentiment'] = news_data['headline'].apply(analyze_sentiment)

# Process each stock file
for file in stock_files:
    stock_data = pd.read_csv(f'../data/{file}')
    
    # Normalize dates
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    
    # Calculate daily returns
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    
    # Align news with stock data
    merged_data = pd.merge(news_data, stock_data, left_on='date', right_on='Date')
    
    # Aggregate daily sentiment scores
    daily_sentiment = merged_data.groupby('Date')['sentiment'].mean().reset_index()
    
    # Merge daily sentiment with stock data
    final_data = pd.merge(stock_data, daily_sentiment, on='Date')
    
    # Calculate correlation between sentiment and daily returns
    correlation = final_data['sentiment'].corr(final_data['Daily_Return'])
    
    print(f'Correlation between sentiment and daily returns for {file}: {correlation:.4f}')

    # Save results to a new CSV file
    final_data.to_csv(f'../results/{file.replace(".csv", "")}_correlation_results.csv', index=False)
