"""Synthesizers module."""

from rctgan.ctganpc.synthesizers.ctganpc import CTGANSynthesizer, PC_CTGANSynthesizer

__all__ = (
    'CTGANSynthesizer',
    'PC_CTGANSynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
