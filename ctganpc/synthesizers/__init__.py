"""Synthesizers module."""

from ctganpc.synthesizers.ctganpc import CTGANSynthesizer, PC_CTGANSynthesizer
from ctganpc.synthesizers.tvae import TVAESynthesizer

__all__ = (
    'CTGANSynthesizer',
    'PC_CTGANSynthesizer',
    'TVAESynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
