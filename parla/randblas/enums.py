from enum import Enum


class Layout(Enum):
    ColMajor = 'C'
    RowMajor = 'R'


class Op(Enum):
    NoTrans = 'N'
    Trans = 'T'
    ConjTrans = 'C'


class Uplo(Enum):
    Upper = 'U'
    Lower = 'L'
    General = 'G'


class Diag(Enum):
    NonUnit = 'N'
    Unit = 'U'


class Side(Enum):
    Left = 'L'
    Right = 'R'


class DenseDist(Enum):
    Gaussian = 'G'
    Uniform = 'U'
    Rademacher = 'R'
    Haar = 'H'  # needs LAPACK-level routines (or custom implementation).

