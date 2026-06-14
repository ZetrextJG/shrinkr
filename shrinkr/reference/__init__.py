"""Shrinkage reference implementations in numpy."""

from ._deal import ref_deal, ref_deal_objective
from ._lw_analytical import ref_lw_analytical, ref_lw_analytical_unstable
from ._lw_linear import ref_lw_linear
from ._oas import ref_oas

__all__ = [
    "ref_deal",
    "ref_deal_objective",
    "ref_lw_analytical_unstable",
    "ref_lw_analytical",
    "ref_lw_linear",
    "ref_oas",
]
