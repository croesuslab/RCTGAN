"""Dataset Generators to test the rdt2 Transformers."""

from collections import defaultdict

from rdt2.performance.datasets import boolean, categorical, datetime, numerical, pii
from rdt2.performance.datasets.base import BaseDatasetGenerator

__all__ = [
    'boolean',
    'categorical',
    'datetime',
    'numerical',
    'BaseDatasetGenerator',
    'pii',
]


def get_dataset_generators_by_type():
    """Build a ``dict`` mapping sdtypes to dataset generators.

    Returns:
        dict:
            Mapping of sdtype to a list of dataset generators that produce
            data of that sdtype.
    """
    dataset_generators = defaultdict(list)
    for dataset_generator in BaseDatasetGenerator.get_subclasses():
        dataset_generators[dataset_generator.SDTYPE].append(dataset_generator)

    return dataset_generators
