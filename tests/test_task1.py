import os
import pandas as pd
import unittest

class TestEDA(unittest.TestCase):
    def setUp(self):
        # Correct file path
        file_path = '../data/raw_analyst_ratings.csv'
        # Verify the file exists before attempting to read
        if not os.path.isfile(file_path):
            raise Exception(f"File not found: {file_path}")
        self.df = pd.read_csv(file_path)

    def test_dataframe_loaded(self):
        # Check if the DataFrame is loaded
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 0)

    def test_column_existence(self):
        # Check if specific columns exist
        required_columns = ['headline', 'publisher', 'publication_date']
        for column in required_columns:
            self.assertIn(column, self.df.columns)

    def test_headline_length_statistics(self):
        # Calculate the length of each headline
        self.df['headline_length'] = self.df['headline'].apply(len)
        
        # Test that headline length statistics are calculated correctly
        self.assertTrue('headline_length' in self.df.columns)
        self.assertGreater(self.df['headline_length'].mean(), 0)

    def test_publisher_counts(self):
        # Count the number of articles per publisher
        publisher_counts = self.df['publisher'].value_counts()
        
        # Test that publisher counts are calculated correctly
        self.assertGreater(len(publisher_counts), 0)
        self.assertGreater(publisher_counts.max(), 0)

    def test_publication_date_analysis(self):
        # Convert publication date to datetime
        self.df['publication_date'] = pd.to_datetime(self.df['publication_date'])
        
        # Test that publication dates are converted correctly
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.df['publication_date']))
        
        # Check if publication trends can be analyzed
        publication_trends = self.df.groupby(self.df['publication_date'].dt.to_period('M')).size()
        self.assertGreater(len(publication_trends), 0)
        self.assertGreater(publication_trends.max(), 0)

if __name__ == '__main__':
    unittest.main()
