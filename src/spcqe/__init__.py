from spcqe.functions import make_basis_matrix, make_regularization_matrix, pinball_slopes
from spcqe.quantiles import SmoothPeriodicQuantiles
from spcqe.solvers import solve_cvx_full, solve_osd, solve_cvx_sequential

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"