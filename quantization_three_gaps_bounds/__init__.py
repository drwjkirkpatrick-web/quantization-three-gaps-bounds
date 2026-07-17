"""
Quantization Three Gaps Bounds — Theoretical bounds for the three
approximation gaps in learned image compression quantization.
"""

from quantization_three_gaps_bounds.core import (
    GapBound,
    TotalGapDecomposition,
    discrete_gap_bound,
    entropy_estimation_gap_bound,
    smoothness_gap_bound,
    total_gap_bound,
    gap_dominance_analysis,
    optimal_quant_step,
    verify_discrete_gap,
    verify_entropy_gap,
    verify_smoothness_gap,
)

__version__ = "0.1.0"
__author__ = "Walker Kirkpatrick"
__license__ = "MIT"