import pytest

from dft_dummy.bravis import BravisLattice, make_lattice_bravis
from dft_dummy.crystal_utils import make_mesh
from dft_dummy.kpoints import reduce_kpts


@pytest.fixture
def kpts():
    return make_mesh(4, 4, 4)


# fmt: off
FCC_LABELS = [
    0, 1, 2, 1, 1, 3, 4, 5, 2, 4, 6, 4, 1, 5, 4, 3, 1, 3, 4, 5, 3, 1,
    5, 4, 4, 5, 4, 7, 5, 4, 7, 4, 2, 4, 6, 4, 4, 5, 4, 7, 6, 4, 2, 4,
    4, 7, 4, 5, 1, 5, 4, 3, 5, 4, 7, 4, 4, 7, 4, 5, 3, 4, 5, 1
]

BCC_LABELS = [
    0, 1, 2, 1, 1, 1, 3, 3, 2, 3, 2, 3, 1, 3, 3, 1, 1, 3, 3, 1, 1, 4,
    5, 4, 3, 3, 5, 5, 3, 6, 3, 4, 2, 3, 2, 3, 3, 5, 5, 3, 2, 5, 7, 5,
    3, 3, 5, 5, 1, 1, 3, 3, 3, 4, 3, 6, 3, 5, 5, 3, 1, 4, 5, 4
]

HCP_LABELS = [
    0,  1,  2,  1,  3,  4,  5,  4,  6,  7,  8,  7,  3,  4,  5,  4,  3,
    4,  5,  4,  9, 10, 11, 10,  9, 10, 11, 10,  3,  4,  5,  4,  6,  7,
    8,  7,  9, 10, 11, 10,  6,  7,  8,  7,  9, 10, 11, 10,  3,  4,  5,
    4,  3,  4,  5,  4,  9, 10, 11, 10,  9, 10, 11, 10
]
# fmt: on


@pytest.mark.parametrize(
    "kwargs,brav,npts_exp,labels_exp",
    [
        (dict(a=1), BravisLattice.fcc, 8, FCC_LABELS),
        (dict(a=1), BravisLattice.bcc, 8, BCC_LABELS),
        (dict(a=1, c=2), BravisLattice.hcp, 12, HCP_LABELS),
    ],
)
def test_reduce_kpts(kwargs, brav, npts_exp, labels_exp, kpts):
    vec, _ = make_lattice_bravis(brav, **kwargs)
    npts, labels = reduce_kpts(kpts, vec)
    assert npts == npts_exp
    assert labels.tolist() == labels_exp
