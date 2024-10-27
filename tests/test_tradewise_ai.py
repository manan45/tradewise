import unittest
import pandas as pd
import numpy as np
from app.core.ai.tradewise_ai import TradewiseAI

class TestTradewiseAI(unittest.TestCase):
    def setUp(self):
        self.ai = TradewiseAI()
        self.sample_data = pd.DataFrame({
            'close': [100, 102, 101, 103, 105, 104, 106, 105, 107, 108],
            'timestamp': pd.date_range(start='2023-01-01', periods=10)
        })

    def test_calculate_moving_average(self):
        # Test case 1: Simple 5-day moving average
        expected_ma_5 = [np.nan, np.nan, np.nan, np.nan, 102.2, 103.0, 103.8, 104.6, 105.4, 106.0]
        result_ma_5 = self.ai.calculate_moving_average(self.sample_data['close'], window=5)
        np.testing.assert_array_almost_equal(result_ma_5, expected_ma_5, decimal=1)

        # Test case 2: Simple 3-day moving average
        expected_ma_3 = [np.nan, np.nan, 101.0, 102.0, 103.0, 104.0, 105.0, 105.0, 106.0, 106.67]
        result_ma_3 = self.ai.calculate_moving_average(self.sample_data['close'], window=3)
        np.testing.assert_array_almost_equal(result_ma_3, expected_ma_3, decimal=2)

    def test_calculate_exponential_moving_average(self):
        # Test case 1: 5-day EMA
        expected_ema_5 = [100.0, 101.0, 101.0, 102.0, 103.5, 103.75, 104.875, 104.9375, 106.0, 107.0]
        result_ema_5 = self.ai.calculate_exponential_moving_average(self.sample_data['close'], window=5)
        np.testing.assert_array_almost_equal(result_ema_5, expected_ema_5, decimal=4)

        # Test case 2: 3-day EMA
        expected_ema_3 = [100.0, 101.0, 101.0, 102.0, 103.5, 103.75, 104.875, 104.9375, 106.0, 107.0]
        result_ema_3 = self.ai.calculate_exponential_moving_average(self.sample_data['close'], window=3)
        np.testing.assert_array_almost_equal(result_ema_3, expected_ema_3, decimal=4)

    def test_moving_average_edge_cases(self):
        # Test with empty data
        empty_data = pd.Series([])
        result = self.ai.calculate_moving_average(empty_data, window=5)
        self.assertTrue(result.empty)

        # Test with window larger than data length
        short_data = pd.Series([1, 2, 3])
        result = self.ai.calculate_moving_average(short_data, window=5)
        self.assertTrue(np.isnan(result).all())

    def test_exponential_moving_average_edge_cases(self):
        # Test with empty data
        empty_data = pd.Series([])
        result = self.ai.calculate_exponential_moving_average(empty_data, window=5)
        self.assertTrue(result.empty)

        # Test with window larger than data length
        short_data = pd.Series([1, 2, 3])
        result = self.ai.calculate_exponential_moving_average(short_data, window=5)
        np.testing.assert_array_almost_equal(result, [1.0, 1.5, 2.25], decimal=4)

    def test_moving_average_types(self):
        # Test with different data types
        int_data = pd.Series([1, 2, 3, 4, 5])
        float_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        int_result = self.ai.calculate_moving_average(int_data, window=3)
        float_result = self.ai.calculate_moving_average(float_data, window=3)

        np.testing.assert_array_almost_equal(int_result, float_result, decimal=4)

if __name__ == '__main__':
    unittest.main()
