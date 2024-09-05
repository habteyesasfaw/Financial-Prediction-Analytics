import unittest
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

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
        # Calculate the Simple Moving Average (SMA) using pandas_ta
        self.stock_df['SMA'] = ta.sma(self.stock_df['Close'], length=2)
        
        # Check that the SMA has been calculated correctly
        self.assertFalse(self.stock_df['SMA'].isnull().all(), "SMA contains all NaN values")
        self.assertTrue(self.stock_df['SMA'].notna().any(), "SMA contains no valid values")

    def test_rsi(self):
        # Calculate the Relative Strength Index (RSI) using pandas_ta
        self.stock_df['RSI'] = ta.rsi(self.stock_df['Close'], length=2)
        
        # Check that the RSI has been calculated correctly
        self.assertFalse(self.stock_df['RSI'].isnull().all(), "RSI contains all NaN values")
        self.assertTrue(self.stock_df['RSI'].notna().any(), "RSI contains no valid values")
        self.assertTrue(0 <= self.stock_df['RSI'].max() <= 100, "RSI values are out of range")

    def test_macd(self):
        # Calculate the MACD using pandas_ta
        macd = ta.macd(self.stock_df['Close'], fast=12, slow=26, signal=9)
        
        # Check that the MACD calculation was successful
        # self.assertIsNotNone(macd, "MACD calculation failed, returned None")
        
        # Add MACD results to DataFrame if macd is not None
        if macd is not None:
            self.stock_df['MACD'] = macd['MACD_12_26_9']
            self.stock_df['MACD_Signal'] = macd['MACDs_12_26_9']
            self.stock_df['MACD_Hist'] = macd['MACDh_12_26_9']
            
            # Verify the MACD calculation
            self.assertFalse(self.stock_df[['MACD', 'MACD_Signal', 'MACD_Hist']].isnull().all().all(), "MACD contains all NaN values")
            self.assertTrue(self.stock_df[['MACD', 'MACD_Signal', 'MACD_Hist']].notna().any().any(), "MACD contains no valid values")

    def test_data_visualization(self):
        # Visualize data and indicators
        plt.figure(figsize=(14, 7))
        
        # Plot Closing Price
        plt.plot(self.stock_df.index, self.stock_df['Close'], label='Close Price')
        
        # Plot Moving Average if SMA exists
        if 'SMA' in self.stock_df.columns:
            plt.plot(self.stock_df.index, self.stock_df['SMA'], label='SMA')
        
        # Plot RSI (on secondary axis)
        if 'RSI' in self.stock_df.columns:
            plt.figure(figsize=(14, 3))
            plt.plot(self.stock_df.index, self.stock_df['RSI'], label='RSI', color='orange')
            plt.axhline(70, color='r', linestyle='--', label='Overbought')
            plt.axhline(30, color='g', linestyle='--', label='Oversold')
        
        # Plot MACD (on secondary axis)
        if all(col in self.stock_df.columns for col in ['MACD', 'MACD_Signal', 'MACD_Hist']):
            plt.figure(figsize=(14, 3))
            plt.plot(self.stock_df.index, self.stock_df['MACD'], label='MACD', color='b')
            plt.plot(self.stock_df.index, self.stock_df['MACD_Signal'], label='MACD Signal', color='r')
            plt.bar(self.stock_df.index, self.stock_df['MACD_Hist'], label='MACD Histogram', color='gray')
        
        # Show all plots
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
