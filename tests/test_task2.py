import unittest
import pandas as pd
import numpy as np
import pandas_ta as ta
from pynance import Data
from matplotlib import pyplot as plt

class TestFinancialAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load sample stock price data
        cls.stock_df = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=5, freq='D'),
            'Open': [150.0, 152.0, 153.0, 155.0, 157.0],
            'High': [155.0, 154.0, 156.0, 158.0, 160.0],
            'Low': [148.0, 150.0, 151.0, 154.0, 156.0],
            'Close': [153.0, 153.0, 155.0, 157.0, 159.0],
            'Volume': [1000, 1500, 1200, 1300, 1400]
        })
        cls.stock_df.set_index('Date', inplace=True)

    def test_moving_average(self):
        # Calculate moving averages using TA-Lib
        self.stock_df['SMA'] = talib.SMA(self.stock_df['Close'], timeperiod=2)
        
        # Verify the moving average calculation
        self.assertFalse(self.stock_df['SMA'].isnull().any(), "SMA contains NaN values")
        self.assertEqual(self.stock_df['SMA'].iloc[1], (self.stock_df['Close'].iloc[0] + self.stock_df['Close'].iloc[1]) / 2, "SMA calculation is incorrect")

    def test_rsi(self):
        # Calculate RSI using TA-Lib
        self.stock_df['RSI'] = talib.RSI(self.stock_df['Close'], timeperiod=2)
        
        # Verify the RSI calculation
        self.assertFalse(self.stock_df['RSI'].isnull().any(), "RSI contains NaN values")
        self.assertTrue(0 <= self.stock_df['RSI'].max() <= 100, "RSI values are out of range")

    def test_macd(self):
        # Calculate MACD using TA-Lib
        macd, macd_signal, macd_hist = talib.MACD(self.stock_df['Close'], fastperiod=2, slowperiod=3, signalperiod=1)
        
        # Add MACD results to DataFrame
        self.stock_df['MACD'] = macd
        self.stock_df['MACD_Signal'] = macd_signal
        self.stock_df['MACD_Hist'] = macd_hist
        
        # Verify the MACD calculation
        self.assertFalse(self.stock_df[['MACD', 'MACD_Signal', 'MACD_Hist']].isnull().any().any(), "MACD contains NaN values")
        self.assertEqual(len(macd), len(self.stock_df), "MACD length does not match the length of the stock data")

    def test_data_visualization(self):
        # Visualize data and indicators
        plt.figure(figsize=(14, 7))
        
        # Plot Closing Price
        plt.plot(self.stock_df.index, self.stock_df['Close'], label='Close Price')
        
        # Plot Moving Average
        plt.plot(self.stock_df.index, self.stock_df['SMA'], label='SMA')
        
        # Plot RSI (on secondary axis)
        plt.figure(figsize=(14, 3))
        plt.plot(self.stock_df.index, self.stock_df['RSI'], label='RSI', color='orange')
        plt.axhline(70, color='r', linestyle='--', label='Overbought')
        plt.axhline(30, color='g', linestyle='--', label='Oversold')
        
        # Plot MACD (on secondary axis)
        plt.figure(figsize=(14, 3))
        plt.plot(self.stock_df.index, self.stock_df['MACD'], label='MACD', color='b')
        plt.plot(self.stock_df.index, self.stock_df['MACD_Signal'], label='MACD Signal', color='r')
        plt.bar(self.stock_df.index, self.stock_df['MACD_Hist'], label='MACD Histogram', color='gray')
        
        # Show all plots
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
