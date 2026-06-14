"""Functional C implementation of shrinkages and more."""

from ._deal import deal, deal_objective
from ._losses import loss_fm, loss_fr, loss_mv, loss_prial, mv_opt_cov
from ._lw_analytical import lw_analytical
from ._lw_linear import lw_linear
from ._oas import oas

__all__ = [
    "lw_analytical",
    "lw_linear",
    "oas",
    "deal",
    "deal_objective",
    "loss_fr",
    "loss_fm",
    "loss_mv",
    "loss_prial",
    "mv_opt_cov",
]
