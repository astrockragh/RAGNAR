"""RAGNAR creates accurate linelists for spectrograph wavelength calibration in the optical and near-infrared"""

__author__ = """Christian Kragh Jespersen"""
__email__ = 'ckragh@princeton.edu'
__version__ = '0.0.1'

from .dev import get_spectra_and_peaks, define_lines_and_intensities
from .dev import make_dataframe, add_absorption_origin