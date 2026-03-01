"""
SPGS-Net Area Estimation Package
=================================
Real-world area calculation from pixel measurements.
"""

from .area_calculator import AreaCalculator, calculate_defect_area

__all__ = [
    "AreaCalculator",
    "calculate_defect_area",
]
