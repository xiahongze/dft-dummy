import pytest

from dft_dummy import bravis, symmetry


@pytest.fixture(scope="session")
def symmetry_ops():
    return symmetry.possible_unitary_rotations()


@pytest.fixture(scope="session")
def symmetry_op_names():
    return symmetry.possible_unitary_rotation_names()


def test_op_lengths(symmetry_ops, symmetry_op_names):
    assert len(symmetry_op_names) == len(symmetry_ops) == 32


POSSIBLE_OPS = {
    bravis.BravisLattice.triclinic: [0],
    bravis.BravisLattice.monoclinic: [0, 2],
    bravis.BravisLattice.orthorhombic: [0, 1, 2, 3],
    bravis.BravisLattice.tetragonal: list(range(8)),
    bravis.BravisLattice.cubic: list(range(24)),
    bravis.BravisLattice.fcc: list(range(24)),
    bravis.BravisLattice.bcc: list(range(24)),
    bravis.BravisLattice.hcp: [0, 1, 2, 3, 24, 25, 26, 27, 28, 29, 30, 31],
}


def test_symmetry(bravis_kwargs, symmetry_ops):
    brav, kwargs = bravis_kwargs
    vec, _ = bravis.make_lattice_bravis(brav, **kwargs)

    overlap = symmetry.calc_overlap_matrix(vec)
    checks = [
        i
        for i, sym in enumerate(symmetry_ops)
        if symmetry.check_symmetry(sym, vec, overlap)
    ]
    assert POSSIBLE_OPS.get(brav) == checks
