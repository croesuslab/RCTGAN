# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.5.1'

from rctgan.ctganpc.demo import load_demo
from rctgan.ctganpc.synthesizers.ctganpc import CTGANSynthesizer, PC_CTGANSynthesizer

__all__ = (
    'CTGANSynthesizer',
    'PC_CTGANSynthesizer',
    'load_demo'
)
