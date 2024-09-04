import unittest
import pandas as pd
import numpy as np
import talib

class TestTask1FinancialAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stock_df = pd.DataFrame({
            'date': pd.to_datetime([
                '2024-08-01', '2024-08-02', '2024-08-03'
            ]),
            'open': [100.0, 105.0, 102.0],
            'high': [110.0, 108.0, 107.0],
            'low': [95.0, 100.0, 99.0],
            'close': [105.0, 102.0, 100.0],
            'volume': [1000, 1500, 1200]
        })

    def test_sma_calculation(self):
        # Calculate SMA
        self.stock_df['SMA'] = talib.SMA(self.stock_df['close'], timeperiod=2)
        
        # Check if SMA is calculated correctly
        self.assertTrue('SMA' in self.stock_df.columns)
        self.assertEqual(len(self.stock_df['SMA']), len(self.stock_df))

    def test_rsi_calculation(self):
        # Calculate RSI
        self.stock_df['RSI'] = talib.RSI(self.stock_df['close'], timeperiod=2)
        
        # Check if RSI is calculated correctly
        self.assertTrue('RSI' in self.stock_df.columns)
        self.assertEqual(len(self.stock_df['RSI']), len(self.stock_df))

    def test_macd_calculation(self):
        # Calculate MACD
        macd, signal, hist = talib.MACD(self.stock_df['close'])
        self.stock_df['MACD'] = macd
        self.stock_df['MACD_Signal'] = signal
        self.stock_df['MACD_Hist'] = hist
        
        # Check if MACD is calculated correctly
        self.assertTrue('MACD' in self.stock_df.columns)
        self.assertTrue('MACD_Signal' in self.stock_df.columns)
        self.assertTrue('MACD_Hist' in self.stock_df.columns)
        self.assertEqual(len(self.stock_df['MACD']), len(self.stock_df))

    def test_daily_returns(self):
        # Calculate daily returns
        self.stock_df['daily_return'] = self.stock_df['close'].pct_change()
        
        # Check if daily returns are calculated correctly
        self.assertTrue('daily_return' in self.stock_df.columns)
        self.assertEqual(len(self.stock_df['daily_return']), len(self.stock_df))
        
        # Check that all values except the first are not NaN
        self.assertFalse(self.stock_df['daily_return'].iloc[1:].isnull().any())

if __name__ == '__main__':
    unittest.main()
