"""
Mapping utilities for incremental verification.

Re-exports:
    gale_shapley_weighted_complete – the weighted Gale-Shapley implementation
    layer_match                    – convenience wrapper that builds the
                                     neuron-to-neuron mapping for one layer
"""

from .gale_shapley import gale_shapley_weighted_complete, layer_match
from .utils import neuron_ranges          # ← add this line

__all__ = [
    "gale_shapley_weighted_complete",
    "layer_match",
    "neuron_ranges",
]

