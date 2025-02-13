import unittest
import numpy as np
import polars as pl
from rctgan.rdt2.transformers.numerical import FloatFormatter, GaussianNormalizer, ClusterBasedNormalizer

class TestFloatFormatter(unittest.TestCase):

    def setUp(self):
        self.data = pl.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])

    def test_fit(self):
        formatter = FloatFormatter(missing_value_replacement='mean', model_missing_values=True, learn_rounding_scheme=True, enforce_min_max_values=True)
        formatter._fit(self.data)
        self.assertIsNotNone(formatter.null_transformer)
        self.assertIsNotNone(formatter._rounding_digits)
        self.assertIsNotNone(formatter._min_value)
        self.assertIsNotNone(formatter._max_value)
        self.assertEqual(formatter._dtype, self.data.dtype)

    def test_transform(self):
        formatter = FloatFormatter(missing_value_replacement='mean', model_missing_values=True)
        formatter._fit(self.data)
        transformed_data = formatter._transform(self.data)
        self.assertEqual(transformed_data.shape, (len(self.data), 2))
        self.assertEqual(transformed_data.dtype, np.float64)

        formatter = FloatFormatter(missing_value_replacement=None, model_missing_values=False)
        formatter._fit(self.data)
        transformed_data = formatter._transform(self.data)
        self.assertEqual(transformed_data.shape, (len(self.data),))
        self.assertEqual(transformed_data.dtype, np.float64)

    def test_reverse_transform(self):
        formatter = FloatFormatter(missing_value_replacement='mean', model_missing_values=True, learn_rounding_scheme=True, enforce_min_max_values=True)
        formatter._fit(self.data)
        transformed_data = formatter._transform(self.data)
        reversed_data = formatter._reverse_transform(transformed_data)
        pl.testing.assert_series_equal(self.data.fill_null(self.data.mean()).cast(self.data.dtype), pl.Series(reversed_data).cast(self.data.dtype), check_dtype=False)

        formatter = FloatFormatter(missing_value_replacement=None, model_missing_values=False)
        formatter._fit(self.data)
        transformed_data = formatter._transform(self.data)
        reversed_data = formatter._reverse_transform(transformed_data)
        pl.testing.assert_series_equal(self.data.cast(self.data.dtype), pl.Series(reversed_data).cast(self.data.dtype), check_dtype=False)

class TestGaussianNormalizer(unittest.TestCase):

    def setUp(self):
        self.data = pl.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])

    def test_fit(self):
        normalizer = GaussianNormalizer()
        normalizer._fit(self.data)
        self.assertIsNotNone(normalizer._univariate)

    def test_transform(self):
        normalizer = GaussianNormalizer()
        normalizer._fit(self.data)
        transformed_data = normalizer._transform(self.data)
        self.assertEqual(transformed_data.shape, (len(self.data),))
        self.assertAlmostEqual(transformed_data[0], -transformed_data[-1], places=2)
        self.assertEqual(transformed_data.dtype, np.float64)

    def test_reverse_transform(self):
        normalizer = GaussianNormalizer()
        normalizer._fit(self.data)
        transformed_data = normalizer._transform(self.data)
        reversed_data = normalizer._reverse_transform(transformed_data)

        original_without_nan = self.data.drop_nulls()
        non_nan_indices = self.data.drop_nulls().to_numpy().nonzero()[0]
        reversed_without_nan = pl.Series(reversed_data).to_numpy()[non_nan_indices]
        np.testing.assert_allclose(original_without_nan.to_numpy(), reversed_without_nan, rtol=0.1)


class TestClusterBasedNormalizer(unittest.TestCase):

    def setUp(self):
        self.data = pl.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])

    def test_fit(self):
        normalizer = ClusterBasedNormalizer()
        normalizer._fit(self.data)
        self.assertIsNotNone(normalizer._bgm_transformer)
        self.assertIsNotNone(normalizer.valid_component_indicator)

    def test_transform(self):
        normalizer = ClusterBasedNormalizer()
        normalizer._fit(self.data)
        transformed_data = normalizer._transform(self.data)
        self.assertEqual(transformed_data.shape, (len(self.data), 2))
        self.assertEqual(transformed_data.dtype, np.float64)

    def test_reverse_transform(self):
        normalizer = ClusterBasedNormalizer()
        normalizer._fit(self.data)
        transformed_data = normalizer._transform(self.data)
        reversed_data = normalizer._reverse_transform(transformed_data)

        original_without_nan = self.data.drop_nulls()
        non_nan_indices = self.data.drop_nulls().to_numpy().nonzero()[0]
        reversed_without_nan = pl.Series(reversed_data).to_numpy()[non_nan_indices]
        np.testing.assert_allclose(original_without_nan.to_numpy(), reversed_without_nan, rtol=0.2)

if __name__ == '__main__':
    unittest.main()