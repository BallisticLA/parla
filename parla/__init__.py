import parla.comps as comps
import parla.drivers as drivers
import parla.utils as utils
import parla.tests as tests

__version__ = '0.1.5'

from parla.utils.sketching import gaussian_operator, srct_operator, \
    sjlt_operator, sparse_sign_operator, orthonormal_operator, sampling_operator
from parla.comps.sketchers.oblivious import SkOpGA, SkOpTC, \
    SkOpSJ, SkOpSS, SkOpON, SkOpIN
from parla.comps.sketchers.aware import RS1, RowSketcher
from parla.comps.qb import QB1, QB2, QB3, QBDecomposer
from parla.comps.rangefinders import RF1, RangeFinder
from parla.drivers.least_squares import SPO, SSO1, OverLstsqSolver

from parla.drivers.svd import SVD1, SVDecomposer

