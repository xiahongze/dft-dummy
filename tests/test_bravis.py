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


@pytest.fixture(
    params=[
        (
            bravis.BravisLattice.triclinic,
            dict(a=1, b=2, c=3, alpha=ACOS0_5, beta=ACOS0_5, gamma=ACOS0_5),
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
        (bravis.BravisLattice.bcc, dict(a=1)),  # hcp
        (bravis.BravisLattice.hexagonal, dict(a=1, c=2)),
    ]
)
def bravis_kwargs(request):
    return request.param


SPECIALS = {
    bravis.BravisLattice.cubic,
    bravis.BravisLattice.fcc,
    bravis.BravisLattice.bcc,
    bravis.BravisLattice.hexagonal,
}


def test_bravis(bravis_kwargs):
    brav, kwargs = bravis_kwargs
    vec, vol = bravis.make_lattice_bravis(brav, **kwargs)
    if brav not in SPECIALS:
        vec_exp, vol_exp = bravis.make_lattice_general(**kwargs)
        assert np.allclose(vec, vec_exp)
    else:
        vol_exp = bravis.calc_volume(vec)
    assert np.allclose(vol, vol_exp)


ALL_PARAMS = {"a", "b", "c", "alpha", "beta", "gamma"}

MUST_HAVE_PARAMS = {
    bravis.BravisLattice.triclinic: ALL_PARAMS,
    bravis.BravisLattice.monoclinic: {"a", "b", "c", "beta"},
    bravis.BravisLattice.orthorhombic: {"a", "b", "c"},
    bravis.BravisLattice.tetragonal: {"a", "c"},
    bravis.BravisLattice.cubic: {"a"},
    bravis.BravisLattice.fcc: {"a"},
    bravis.BravisLattice.bcc: {"a"},
    bravis.BravisLattice.hexagonal: {"a", "c"},
}


def test_minimal_bravis_input(bravis_kwargs):
    """minimal input params should work"""
    brav, kwargs = bravis_kwargs
    redundant_keys = ALL_PARAMS - MUST_HAVE_PARAMS[brav]
    for k in redundant_keys:
        if k in kwargs:
            del kwargs[k]
    _ = bravis.make_lattice_bravis(brav, **kwargs)


def test_missing_param_bravis_input(bravis_kwargs):
    brav, kwargs = bravis_kwargs
    for k in MUST_HAVE_PARAMS[brav]:
        kw = kwargs.copy()
        del kw[k]
        with pytest.raises((KeyError, TypeError)):
            _ = bravis.make_lattice_bravis(brav, **kw)
