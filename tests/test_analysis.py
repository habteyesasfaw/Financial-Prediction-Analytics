import unittest
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import pearsonr

class TestFinancialAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load sample data
        cls.news_df = pd.DataFrame({
            'headline': [
                'Stock hits record high after positive earnings report',
                'Company faces lawsuit over alleged fraud',
                'Market sees dip as new trade policies introduced'
            ],
            'date': pd.to_datetime([
                '2024-08-01', '2024-08-02', '2024-08-03'
            ]),
            'stock': ['AAPL', 'AAPL', 'AAPL']
        })
        
        cls.stock_df = pd.DataFrame({
            'date': pd.to_datetime([
                '2024-08-01', '2024-08-02', '2024-08-03'
            ]),
            'close': [150.0, 155.0, 148.0]
        })
        
        cls.sia = SentimentIntensityAnalyzer()

    def test_sentiment_analysis(self):
        # Perform sentiment analysis on headlines
        self.news_df['sentiment'] = self.news_df['headline'].apply(lambda x: self.sia.polarity_scores(x)['compound'])
        
        # Check if sentiment scores are computed
        self.assertTrue('sentiment' in self.news_df.columns)
        self.assertEqual(len(self.news_df['sentiment']), len(self.news_df))

    def test_daily_returns(self):
        # Calculate daily returns
        self.stock_df['daily_return'] = self.stock_df['close'].pct_change()
        
        # Check if daily returns are computed correctly
        self.assertTrue('daily_return' in self.stock_df.columns)
        self.assertEqual(len(self.stock_df['daily_return']), len(self.stock_df))

    def test_correlation_analysis(self):
        # Ensure daily returns are calculated
        if 'daily_return' not in self.stock_df.columns:
            self.stock_df['daily_return'] = self.stock_df['close'].pct_change()

        # Perform sentiment analysis if not done
        if 'sentiment' not in self.news_df.columns:
            self.news_df['sentiment'] = self.news_df['headline'].apply(lambda x: self.sia.polarity_scores(x)['compound'])

        # Merge dataframes on date
        merged_df = pd.merge(self.news_df, self.stock_df, on='date')

        # Calculate average daily sentiment if multiple headlines per day
        daily_sentiment = merged_df.groupby('date')['sentiment'].mean().reset_index()
        
        # Calculate daily returns
        daily_returns = merged_df.groupby('date')['daily_return'].mean().reset_index()
        
        # Merge sentiment and returns for correlation analysis
        correlation_df = pd.merge(daily_sentiment, daily_returns, on='date')

        # Drop rows with NaN values
        correlation_df = correlation_df.dropna()

        # Calculate Pearson correlation
        if not correlation_df.empty:
            correlation, _ = pearsonr(correlation_df['sentiment'], correlation_df['daily_return'])
            
            # Check if correlation is within expected range
            self.assertTrue(-1 <= correlation <= 1)
        else:
            self.fail("Correlation DataFrame is empty after dropping NaNs, can't compute Pearson correlation.")

if __name__ == '__main__':
    unittest.main()
