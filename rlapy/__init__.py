import rlapy.comps as comps
import rlapy.drivers as drivers
import rlapy.utils as utils
import rlapy.tests as tests

__version__ = '0.1.1'

from rlapy.utils.sketching import gaussian_operator, srct_operator, \
    sjlt_operator, sparse_sign_operator, orthonormal_operator, sampling_operator
from rlapy.comps.sketchers.oblivious import SkOpGA, SkOpTC, \
    SkOpSJ, SkOpSS, SkOpON, SkOpIN
from rlapy.comps.sketchers.aware import RS1, RowSketcher
from rlapy.comps.qb import QB1, QB2, QB3, QBDecomposer
from rlapy.comps.rangefinders import RF1, RangeFinder
from rlapy.drivers.least_squares import SPO3, SPO1, SSO1, OverLstsqSolver
from rlapy.drivers.svd import SVD1, SVDecomposer

