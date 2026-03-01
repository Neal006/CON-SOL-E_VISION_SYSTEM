"""
SPGS-Net Defect Instance Package
=================================
Post-processing for defect instance separation.
"""

from .instance_separator import InstanceSeparator, separate_defect_instances

__all__ = [
    "InstanceSeparator",
    "separate_defect_instances",
]
