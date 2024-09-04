import unittest
import pandas as pd
from textblob import TextBlob

# Sample data for testing
test_news_data = {
    'headline': [
        'Apple reaches all-time high in stock price',
        'Amazon announces major expansion plans',
        'Tesla faces regulatory scrutiny over safety concerns'
    ],
    'url': [
        'http://example.com/apple',
        'http://example.com/amazon',
        'http://example.com/tesla'
    ],
    'publisher': ['Reuters', 'Bloomberg', 'Reuters'],
    'date': ['2023-08-01', '2023-08-02', '2023-08-03'],
    'stock': ['AAPL', 'AMZN', 'TSLA']
}

class TestTask1EDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Convert the test data into a DataFrame
        cls.news_df = pd.DataFrame(test_news_data)
        cls.news_df['date'] = pd.to_datetime(cls.news_df['date'])

    def test_headline_length_statistics(self):
        """Test the calculation of headline length statistics."""
        self.news_df['headline_length'] = self.news_df['headline'].apply(len)
        mean_length = self.news_df['headline_length'].mean()
        self.assertEqual(mean_length, 44.0, "Mean headline length should be 44.0")

    def test_article_count_per_publisher(self):
        """Test the count of articles per publisher."""
        publisher_counts = self.news_df['publisher'].value_counts()
        self.assertEqual(publisher_counts['Reuters'], 2, "Reuters should have 2 articles")
        self.assertEqual(publisher_counts['Bloomberg'], 1, "Bloomberg should have 1 article")

    def test_sentiment_analysis(self):
        """Test sentiment analysis on headlines."""
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity
        
        self.news_df['sentiment'] = self.news_df['headline'].apply(get_sentiment)
        sentiments = self.news_df['sentiment'].tolist()
        expected_sentiments = [0.16, 0.0625, 0.0]
        self.assertEqual(sentiments, expected_sentiments, "Sentiments should match expected values")

if __name__ == '__main__':
    unittest.main()
