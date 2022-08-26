"""Models for tabular data."""

from sdv.tabular.copulagan import CopulaGAN
from sdv.tabular.copulas import GaussianCopula
from sdv.tabular.ctgan import CTGAN, PC_CTGAN, TVAE

__all__ = (
    'GaussianCopula',
    'CTGAN',
    'PC_CTGAN',
    'TVAE',
    'CopulaGAN',
)
