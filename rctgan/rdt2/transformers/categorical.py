"""Transformers for categorical data."""

import warnings

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, Union
import psutil
from scipy.stats import norm

from rctgan.rdt2.transformers.base import BaseTransformer


class FrequencyEncoder(BaseTransformer):
    """Transformer for categorical data.

    This transformer computes a float representative for each one of the categories
    found in the fit data, and then replaces the instances of these categories with
    the corresponding representative.

    The representatives are decided by sorting the categorical values by their relative
    frequency, then dividing the ``[0, 1]`` interval by these relative frequencies, and
    finally assigning the middle point of each interval to the corresponding category.

    When the transformation is reverted, each value is assigned the category that
    corresponds to the interval it falls in.

    Null values are considered just another category.

    Args:
        add_noise (bool):
            Whether to generate gaussian noise around the class representative of each interval
            or just use the mean for all the replaced values. Defaults to ``False``.
    """

    INPUT_SDTYPE = 'categorical'
    OUTPUT_SDTYPES = pl.Float64
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    mapping = None
    intervals = None
    starts = None
    means = None
    dtype = None
    _get_category_from_index = None

    def __setstate__(self, state):
        """Replace any ``null`` key by the actual ``np.nan`` instance."""
        intervals = state.get('intervals')
        if intervals:
            for key in list(intervals):
                if pd.isna(key):
                    intervals[np.nan] = intervals.pop(key)

        self.__dict__ = state

    def __init__(self, add_noise=False):
        self.add_noise = add_noise

    def is_transform_deterministic(self):
        """Return whether the transform is deterministic.

        Returns:
            bool:
                Whether or not the transform is deterministic.
        """
        return not self.add_noise

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        return self.COMPOSITION_IS_IDENTITY and not self.add_noise

    @staticmethod
    def _get_intervals(data: pl.Series) -> Dict:
        """Compute intervals for each categorical value.
    
        Args:
            data (pl.Series):
                Data to analyze.
    
        Returns:
            dict:
                intervals for each categorical value (start, end).
        """
        data = data.fill_null(np.nan)
        frequencies = data.value_counts().sort('category')

        start = 0
        end = 0
        elements = len(data)

        intervals = {}
        means = {}
        starts = []
        for value, frequency in zip(frequencies['category'], frequencies['count']):
            prob = frequency / elements
            end = start + prob
            mean = start + prob / 2
            std = prob / 6
            if value is None:
                value = np.nan

            intervals[value] = (start, end, mean, std)
            means[value] = mean
            starts.append((value, start))
            start = end

        starts = pl.DataFrame({"category": [s[0] for s in starts], "start": [s[1] for s in starts]}, strict=False)

        return intervals, means, starts

    def _fit(self, data: pl.Series) -> None:
        """Fit the transformer to the data.
    
        Compute the intervals for each categorical value.
    
        Args:
            data (pl.Series):
                Data to fit the transformer to.
        """
        self.dtype = data.dtype
        self.intervals, self.means, self.starts = self._get_intervals(data)

    def _transform_by_category(self, data: pl.Series) -> np.ndarray:
        """Transform the data by iterating over the different categories."""
        result = np.empty(shape=(len(data), ), dtype=float)

        # loop over categories
        for category, values in self.intervals.items():
            mean, std = values[2:]
            if category is np.nan:
                mask = data.is_null()
            else:
                mask = (data.to_numpy() == category)

            if self.add_noise:
                result[mask] = norm.rvs(mean, std, size=mask.sum())
            else:
                result[mask] = mean

        return result

    def _get_value(self, category: str) -> float:
        """Get the value that represents this category."""
        if category is None or category is np.nan:
            category = np.nan

        mean, std = self.intervals[category][2:]

        if self.add_noise:
            return norm.rvs(mean, std)

        return mean

    def _transform_by_row(self, data: pl.Series) -> np.ndarray:
        mapping = {k: self._get_value(k) for k in self.intervals}
        
        return data.fill_null(np.nan).map_elements(lambda x: mapping.get(x, np.nan), self.OUTPUT_SDTYPES).to_numpy()

    def _transform(self, data: pl.Series) -> np.ndarray:
        """Transform categorical values to float representatives.

        Args:
            data (pl.Series): Data to transform.

        Returns:
            np.ndarray
        """

        fit_categories = pl.Series(list(self.intervals.keys()))  # Polars Series
        has_nan = fit_categories.is_null().any()  # Polars is_null
        unseen_mask = ~(data.is_in(fit_categories) | (data.is_null() & has_nan))  # Polars is_in and is_null

        if unseen_mask.any():
            unseen_categories = data.filter(unseen_mask).slice(0, 5).to_list() # Polars filtering and slicing
            warnings.warn(
                f'The data contains {unseen_mask.sum()} new categories that were not '
                f'seen in the original data (examples: {unseen_categories}). Assigning '
                'them random values. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

            # Polars assignment with mask
            data = data.with_columns(
                pl.when(unseen_mask)
              .then(pl.Series(np.random.choice(fit_categories.to_numpy(), size=unseen_mask.sum())))
              .otherwise(pl.col("category")) # Keep original value if not unseen
              .alias("category")
            )
            
        if len(self.means) < len(data):
            return self._transform_by_category(data)

        return self._transform_by_row(data)

    def _reverse_transform_by_matrix(self, data: pl.Series) -> pl.Series:
        """Reverse transform the data with matrix operations."""
        num_rows = len(data)
        num_categories = len(self.means)

        data_np = np.broadcast_to(data.to_numpy(), (num_categories, num_rows)).T
        means_np = np.tile(np.array([self.means[cat] for cat in self.means.keys()]), (num_rows, 1)) 
        diffs = np.abs(data_np - means_np)
        indexes = np.argmin(diffs, axis=1)

        self._get_category_from_index = list(self.means.keys())
        reversed_data = [self._get_category_from_index[idx] for idx in indexes]
        return pl.Series(reversed_data, dtype=self.dtype, strict=False)

    def _reverse_transform_by_category(self, data: pl.Series) -> pl.Series:
        """Reverse transform the data by iterating over all the categories."""

        result = np.full(len(data), np.nan, dtype=object)

        for category, values in self.intervals.items():
            start, end, _, _ = values
            mask = (start <= data.to_numpy()) & (data.to_numpy() < end)
            result[mask] = category
            

        return pl.Series(result, dtype=self.dtype)

    def _get_category_from_start(self, value):
        lower = self.starts.filter(pl.col("start") <= value)
        return lower[-1, "category"]

    def _reverse_transform_by_row(self, data: pl.Series) -> pl.Series:
        """Reverse transform the data by iterating over each row."""
        return data.map_elements(self._get_category_from_start, return_dtype=self.dtype)

    def _reverse_transform(self, data: pl.Series) -> pl.Series:
        """Convert float values back to the original categorical values.

        Args:
            data (pl.Series): Data to revert.

        Returns:
            pl.Series
        """
        data = data.clip(0, 1)

        num_rows = len(data)
        num_categories = len(self.means)
        
        needed_memory = num_rows * num_categories * 8 * 3
        available_memory = psutil.virtual_memory().available

        if available_memory > needed_memory:
            return self._reverse_transform_by_matrix(data)

        if num_rows > num_categories:
            return self._reverse_transform_by_category(data)

        # loop over rows
        return self._reverse_transform_by_row(data)


class OneHotEncoder(BaseTransformer):
    """OneHotEncoding for categorical data.

    This transformer replaces a single vector with N unique categories in it
    with N vectors which have 1s on the rows where the corresponding category
    is found and 0s on the rest.

    Null values are considered just another category.
    """

    INPUT_SDTYPE = 'categorical'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True

    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.

        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.

        Args:
            data (pl.Series or pl.DataFrame):
                Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')

            data = data[:, 0]

        return data

    def get_output_sdtypes(self):
        """Return the output sdtypes produced by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced sdtypes.
        """
        output_sdtypes = {f'value{i}': 'float' for i in range(len(self.dummies))}

        return self._add_prefix(output_sdtypes)

    def _fit(self, data: pl.Series) -> None:
        """Fit the transformer to the data.

        Get the polars `dummies` which will be used later on for OneHotEncoding.

        Args:
            data (pl.Series or pl.DataFrame):
                Data to fit the transformer to.
        """
        data = self._prepare_data(data)

        null = data.is_null()
        self._uniques = list(data.filter(~null).unique().sort())
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()

        if not data.dtype.is_numeric:
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

    def _transform_helper(self, data: pl.Series) -> np.ndarray:
        if self._dummy_encoded:
            coder = self._indexer
            codes = pl.Categorical(data, categories=self._uniques).to_physical().to_numpy()
        else:
            coder = self._uniques
            codes = data.to_numpy()

        rows = len(data)
        dummies = np.broadcast_to(coder, (rows, self._num_dummies))
        coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
        array = (coded == dummies).astype(int)

        if self._dummy_na:
            null = np.zeros((rows, 1), dtype=int)
            null[data.is_null().to_numpy()] = 1
            array = np.append(array, null, axis=1)

        return array

    def _transform(self, data):
        """Replace each category with the OneHot vectors.

        Args:
            data (pl.Series, list or list of lists):
                Data to transform.

        Returns:
            np.ndarray
        """
        data = self._prepare_data(data)
        unique_data = {np.nan if x is None else x for x in data.unique()}
        unseen_categories = unique_data - set(self.dummies)
        if unseen_categories:
            # Select only the first 5 unseen categories to avoid flooding the console.
            examples_unseen_categories = set(list(unseen_categories)[:5])
            warnings.warn(
                f'The data contains {len(unseen_categories)} new categories that were not '
                f'seen in the original data (examples: {examples_unseen_categories}). Creating '
                'a vector of all 0s. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

        return self._transform_helper(data)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pl.Series or np.ndarray):
                Data to revert.

        Returns:
            pl.Series
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)
        reversed_data = [self.dummies[idx] for idx in indices]

        return pl.Series(reversed_data)


class LabelEncoder(BaseTransformer):
    """LabelEncoding for categorical data.

    This transformer generates a unique integer representation for each category
    and simply replaces each category with its integer value.

    Null values are considered just another category.

    Attributes:
        values_to_categories (dict):
            Dictionary that maps each integer value for its category.
        categories_to_values (dict):
            Dictionary that maps each category with the corresponding
            integer value.
    """

    INPUT_SDTYPE = pl.String
    OUTPUT_SDTYPES = pl.Int64
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    values_to_categories = None
    categories_to_values = None

    def _fit(self, data: pl.Series) -> None:
        """Fit the transformer to the data.

        Generate a unique integer representation for each category and
        store them in the `categories_to_values` dict and its reverse
        `values_to_categories`.

        Args:
            data (pl.Series):
                Data to fit the transformer to.
        """
        unique_data = data.fill_null(np.nan).unique().sort().to_list()
        self.values_to_categories = dict(enumerate(unique_data))
        self.categories_to_values = {
            category: value
            for value, category in self.values_to_categories.items()
        }

    def _transform(self, data: pl.Series) -> pl.Series:
        """Replace each category with its corresponding integer value.

        If a category has not been seen before, a random value is assigned.

        Args:
            data (pl.Series):
                Data to transform.

        Returns:
            pl.Series
        """
        mapped: pl.Series = data.fill_null(np.nan).map_elements(lambda x: self.categories_to_values.get(x, np.nan), self.OUTPUT_SDTYPES)
        is_null = mapped.is_null()
        if is_null.any():
            # Select only the first 5 unseen categories to avoid flooding the console.
            unseen_categories = set(data.filter(is_null).head(5).to_list())
            warnings.warn(
                f'The data contains {is_null.sum()} new categories that were not '
                f'seen in the original data (examples: {unseen_categories}). Assigning '
                'them random values. If you want to model new categories, '
                'please fit the transformer again with the new data.'
            )

            mapped = mapped.map_elements(lambda x: np.random.randint(len(self.categories_to_values)) if np.isnan(x) else x, self.OUTPUT_SDTYPES)

        return mapped

    def _reverse_transform(self, data: pl.Series) -> pl.Series:
        """Convert float values back to the original categorical values.

        Args:
            data (pl.Series or np.ndarray):
                Data to revert.

        Returns:
            pl.Series
        """
        if isinstance(data, np.ndarray):
            data = pl.Series(data)

        data = data.clip(lower_bound=min(self.values_to_categories), upper_bound=max(self.values_to_categories))
        data = data.round(0).map_elements(lambda x: self.values_to_categories.get(x, np.nan), self.INPUT_SDTYPE)
        return data
