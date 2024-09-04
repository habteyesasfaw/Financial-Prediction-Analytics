import unittest
import pandas as pd
import pandas_ta as ta

class TestTask1(unittest.TestCase):

    def setUp(self):
        # Mock data to avoid file dependency
        data = {
            'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'Open': pd.Series(100 + pd.np.random.randn(100).cumsum()),
            'High': pd.Series(102 + pd.np.random.randn(100).cumsum()),
            'Low': pd.Series(98 + pd.np.random.randn(100).cumsum()),
            'Close': pd.Series(100 + pd.np.random.randn(100).cumsum()),
            'Volume': pd.Series(pd.np.random.randint(1000, 5000, size=100))
        }
        self.df = pd.DataFrame(data)
    
    def test_sma_calculation(self):
        self.df['SMA_20'] = ta.sma(self.df['Close'], length=20)
        self.assertFalse(self.df['SMA_20'].isnull().all(), "SMA_20 calculation failed")

    def test_rsi_calculation(self):
        self.df['RSI_14'] = ta.rsi(self.df['Close'], length=14)
        self.assertFalse(self.df['RSI_14'].isnull().all(), "RSI_14 calculation failed")

    def test_bollinger_bands_calculation(self):
        self.df['BB_upper'], self.df['BB_middle'], self.df['BB_lower'] = ta.bbands(self.df['Close'], length=20)
        self.assertFalse(self.df['BB_upper'].isnull().all(), "BB_upper calculation failed")
        self.assertFalse(self.df['BB_middle'].isnull().all(), "BB_middle calculation failed")
        self.assertFalse(self.df['BB_lower'].isnull().all(), "BB_lower calculation failed")

if __name__ == '__main__':
    unittest.main()
