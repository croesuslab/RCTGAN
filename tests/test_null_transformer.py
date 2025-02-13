import unittest
import numpy as np
import polars as pl
from rctgan.rdt2.transformers.null import NullTransformer

class TestNullTransformer(unittest.TestCase):

    def setUp(self):
        self.data = pl.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10], strict=False)

    def test_models_missing_values(self):
        transformer = NullTransformer(model_missing_values=True)
        self.assertTrue(transformer.models_missing_values())

        transformer = NullTransformer(model_missing_values=False)
        self.assertFalse(transformer.models_missing_values())

    def test_get_missing_value_replacement(self):
        transformer = NullTransformer(missing_value_replacement='mean')
        replacement = transformer._get_missing_value_replacement(self.data)
        self.assertAlmostEqual(replacement, self.data.drop_nans().mean(), places=4)

        transformer = NullTransformer(missing_value_replacement='mode')
        replacement = transformer._get_missing_value_replacement(self.data)
        self.assertEqual(replacement, 1)

        transformer = NullTransformer(missing_value_replacement=100)
        replacement = transformer._get_missing_value_replacement(self.data)
        self.assertEqual(replacement, 100)

    def test_fit(self):
        transformer = NullTransformer(model_missing_values=True)
        transformer.fit(self.data)
        self.assertTrue(transformer.nulls)
        self.assertTrue(transformer._model_missing_values)

        transformer = NullTransformer(model_missing_values=True)
        transformer.fit(self.data.fill_nan(0))
        self.assertFalse(transformer.nulls)
        self.assertFalse(transformer._model_missing_values)  # Should be False because there are no nulls

    def test_transform(self):
        transformer = NullTransformer(missing_value_replacement='mean', model_missing_values=True)
        transformer.fit(self.data)
        transformed_data = transformer.transform(self.data)
        self.assertEqual(transformed_data.shape, (len(self.data), 2))  # Check for the added 'is_null' column 
        np.testing.assert_array_equal(transformed_data[:, 0], self.data.fill_nan(self.data.drop_nans().mean()).to_numpy())

        transformer = NullTransformer(missing_value_replacement=None, model_missing_values=False)
        transformer.fit(self.data)
        transformed_data = transformer.transform(self.data)
        self.assertEqual(transformed_data.shape, (len(self.data),))
        self.assertTrue(np.any(np.isnan(transformed_data)))

    def test_reverse_transform(self):
        # Test with model_missing_values = True
        transformer = NullTransformer(missing_value_replacement='mean', model_missing_values=True)
        transformer.fit(self.data)
        transformed_data = transformer.transform(self.data)
        reversed_data = transformer.reverse_transform(transformed_data)
        np.testing.assert_array_equal(self.data.to_numpy(), reversed_data.to_numpy())

        # Test with model_missing_values = False
        transformer = NullTransformer(missing_value_replacement=None, model_missing_values=False)
        transformer.fit(self.data)
        transformed_data = transformer.transform(self.data)
        reversed_data = transformer.reverse_transform(transformed_data)

if __name__ == '__main__':
    unittest.main()