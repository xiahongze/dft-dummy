import numpy as np
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


def test_general_cubic():
    vec, vol = bravis.make_lattice_general(1, 1, 1, NINETY, NINETY, NINETY)
    assert np.allclose(vec, np.eye(3))
    assert vol == 1


def test_general_fcc():
    vec, vol = bravis.make_lattice_general(
        COS45, COS45, COS45, ACOS0_5, ACOS0_5, ACOS0_5
    )
    a = np.linalg.norm(vec[0])
    b = np.linalg.norm(vec[1])
    c = np.linalg.norm(vec[2])
    assert np.allclose([a, b, c], [COS45, COS45, COS45])
    assert np.allclose(vec[0].dot(vec[1]), 0.25)
    assert np.allclose(vec[1].dot(vec[2]), 0.25)
    assert np.allclose(vec[0].dot(vec[2]), 0.25)
    assert np.allclose(vol, 0.25)
