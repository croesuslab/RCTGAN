"""Metadata module."""

from rctgan.metadata import visualization
from rctgan.metadata.dataset import Metadata
from rctgan.metadata.errors import MetadataError, MetadataNotFittedError
from rctgan.metadata.table import Table

__all__ = (
    'Metadata',
    'MetadataError',
    'MetadataNotFittedError',
    'Table',
    'visualization'
)
