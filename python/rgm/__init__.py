"""
High-performance Python port of the RGM (Random Geological Model) generators.

This package provides a NumPy-first implementation of the Fortran `rgm3_curved`
model used by the examples in this repository.
"""

from .rgm3_curved import RGM3Curved, RGM3CurvedConfig, RGM3CurvedResult

__all__ = ["RGM3Curved", "RGM3CurvedConfig", "RGM3CurvedResult"]

