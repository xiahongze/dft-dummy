import numpy as np
import pytest

from dft_dummy import bravis

NINETY = np.pi / 2
FOURTY_FIVE = np.pi / 4
COS45 = np.cos(FOURTY_FIVE)
ACOS0_5 = np.arccos(0.5)
VEC_FCC = np.array(
    [
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
    ]
)


@pytest.fixture(
    params=[
        (
            bravis.BravisLattice.triclinic,
            dict(a=1, b=2, c=3, alpha=ACOS0_5, beta=ACOS0_5 + 0.2, gamma=ACOS0_5 - 0.2),
        ),  # triclinic
        (
            bravis.BravisLattice.monoclinic,
            dict(a=1, b=2, c=3, alpha=NINETY, beta=ACOS0_5, gamma=NINETY),
        ),  # monoclinic
        (
            bravis.BravisLattice.orthorhombic,
            dict(a=1, b=2, c=3, alpha=NINETY, beta=NINETY, gamma=NINETY),
        ),  # orthorhombic
        (
            bravis.BravisLattice.tetragonal,
            dict(a=1, b=1, c=3, alpha=NINETY, beta=NINETY, gamma=NINETY),
        ),  # tetragonal
        (bravis.BravisLattice.cubic, dict(a=1)),  # cubic
        (bravis.BravisLattice.fcc, dict(a=1)),  # fcc
        (bravis.BravisLattice.bcc, dict(a=1)),  # bcc
        (bravis.BravisLattice.hexagonal, dict(a=1, c=2)),  # hcp
    ]
)
def bravis_kwargs(request):
    return request.param
