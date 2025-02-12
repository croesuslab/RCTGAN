import unittest
import numpy as np
import polars as pl
from rctgan.rdt2.transformers.categorical import FrequencyEncoder, OneHotEncoder, LabelEncoder

class TestFrequencyEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = FrequencyEncoder()
        self.data = pl.Series("category", ["a", "b", "a", "c", "b", "a"])

    def test_fit(self):
        self.encoder._fit(self.data)
        self.assertIsNotNone(self.encoder.intervals)

        self.assertIsNotNone(self.encoder.means)
        self.assertAlmostEqual(self.encoder.means["a"], 0.25, places=2)
        self.assertAlmostEqual(self.encoder.means["b"], 0.67, places=2)
        
        self.assertIsNotNone(self.encoder.starts)
        
    def test_get_value(self):
        self.encoder._fit(self.data)
        value_a = self.encoder._get_value("a")
        value_b = self.encoder._get_value("b")
        self.assertAlmostEqual(value_a, 0.25, places=2)
        self.assertAlmostEqual(value_b, 0.67, places=2)

    def test_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        self.assertEqual(transformed.shape, (6,))
        self.assertEqual(transformed[0], transformed[2])
        self.assertEqual(transformed[1], transformed[4])
        
    def test_transform_by_category(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        transformed_category = self.encoder._transform_by_category(self.data)
        self.assertEqual(transformed.shape, (6,))
        np.testing.assert_array_equal(transformed_category, transformed)

    def test_transform_by_row(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        transformed_row = self.encoder._transform_by_row(self.data)
        self.assertEqual(transformed.shape, (6,))
        np.testing.assert_array_equal(transformed_row, transformed)

    def test_reverse_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        reversed_data = self.encoder._reverse_transform(pl.Series(transformed))
        self.assertTrue((reversed_data == self.data).all())
    
    def test_category_reverse_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        category_reversed_data = self.encoder._reverse_transform_by_category(pl.Series(transformed))
        self.assertTrue((category_reversed_data == self.data).all())
        
    def test_matrix_reverse_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        matrix_reversed_data = self.encoder._reverse_transform_by_row(pl.Series(transformed))
        self.assertTrue((matrix_reversed_data == self.data).all())
        
    def test_row_reverse_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        row_reversed_data = self.encoder._reverse_transform_by_row(pl.Series(transformed))
        self.assertTrue((row_reversed_data == self.data).all())

class TestOneHotEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = OneHotEncoder()
        self.data = pl.Series("category", ["a", "b", "a", "c", "b", "a"])

    def test_fit(self):
        self.encoder._fit(self.data)
        self.assertIsNotNone(self.encoder.dummies)
        self.assertIsNotNone(self.encoder._uniques)
        self.assertEqual(self.encoder._uniques, ['a', 'b', 'c'])
        self.assertEqual(self.encoder.dummies, ['a', 'b', 'c'])  
        
    def test_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        self.assertEqual(transformed.shape, (6, len(self.encoder.dummies)))
        np.testing.assert_array_equal(transformed[0], transformed[2])
        np.testing.assert_array_equal(transformed[1], transformed[4])

    def test_reverse_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        reversed_data = self.encoder._reverse_transform(transformed)
        self.assertTrue((reversed_data == self.data).all())

class TestLabelEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = LabelEncoder()
        self.data = pl.Series("category", ["a", "b", "a", "c", "b", "a"])

    def test_fit(self):
        self.encoder._fit(self.data)
        self.assertIsNotNone(self.encoder.values_to_categories)
        self.assertIsNotNone(self.encoder.categories_to_values)
        self.assertEqual(self.encoder.categories_to_values, {'a': 0, 'b': 1, 'c': 2})
        self.assertEqual(self.encoder.values_to_categories, {0: 'a', 1: 'b', 2: 'c'})

    def test_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        self.assertEqual(transformed.shape, (6,))
        np.testing.assert_array_equal(transformed, [0, 1, 0, 2, 1, 0])

    def test_reverse_transform(self):
        self.encoder._fit(self.data)
        transformed = self.encoder._transform(self.data)
        reversed_data = self.encoder._reverse_transform(pl.Series(transformed))
        self.assertTrue((reversed_data == self.data).all())

if __name__ == '__main__':
    unittest.main()