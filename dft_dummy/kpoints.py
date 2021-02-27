from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from dft_dummy.crystal_utils import calc_reciprocal, project_points
from dft_dummy.symmetry import (
    calc_overlap_matrix,
    check_symmetry,
    possible_unitary_rotations,
)


def reduce_kpts(kpts: np.ndarray, vec: np.ndarray) -> Tuple[int, np.ndarray]:
    """reduce kpoints to the irreducible ones

    Args:
        kpts (np.ndarray): Nx3 kmesh in the 1st Brillouin Zone in crystal coordinates
        vec (np.ndarray): 3x3 lattice basis vectors

    Returns:
        Tuple[int, np.ndarray]: number of irreducible kpoints and
            label array that maps the `kpts` to each (irreducible) kind.
    """
    overlap = calc_overlap_matrix(vec)
    symmetry_ops = possible_unitary_rotations()
    valid_ops = [
        (i, sym)
        for i, sym in enumerate(symmetry_ops)
        if check_symmetry(sym, vec, overlap)
    ]

    rvec = calc_reciprocal(vec)
    nkpts = len(kpts)

    def update_graph_matrix(sym, kcart):
        krot = sym.dot(kcart)  # rotated in cartisian system
        krot_crys = vec.dot(krot)  # back to crystal system
        krot_crys_moved = krot_crys - np.floor(krot_crys)  # move to the 1st BZ
        # find the index of the moved point in the original mesh
        norms = np.linalg.norm(kpts - krot_crys_moved, axis=1)
        idx = np.arange(len(norms))[norms < 1e-6]
        graph_matrix[i, idx] = 1

    graph_matrix = csr_matrix((nkpts, nkpts), dtype=int)
    for i, kcrys in enumerate(kpts):
        kcart = project_points(rvec, kcrys).ravel()  # flatten as it is one point
        for _, sym in valid_ops:
            update_graph_matrix(sym, kcart)
            update_graph_matrix(-sym, kcart)  # inverse symmetry

    return connected_components(
        csgraph=graph_matrix, directed=False, return_labels=True
    )
