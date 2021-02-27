"""
[Crystallographic Restriction]\
(https://en.wikipedia.org/wiki/Crystallographic_restriction_theorem)
[QE Source Code]\
(https://github.com/QEF/q-e/blob/master/PW/src/symm_base.f90)

Due to the Crystallographic Restriction, there are only 32 kinds of symmetry
operations allowed in the 3D world, if not counting the inversion. If inversion
is included, then there are 64 kinds of operations. Symmetry operations are by
definition a kind of unitary rotation, meaning the magnitude won't change before
and after the operation. `Magnitude` is a lousy word I used here. For accurate
discussion, one can refer to the Wiki.
"""
# flake8: noqa
from typing import List

import numpy as np


def possible_unitary_rotations() -> np.ndarray:
    """32 kinds of symmetry operations"""
    cos3, sin3 = np.cos(np.pi / 3), np.sin(np.pi / 3)
    mcos3, msin3 = -cos3, -sin3
    # fmt: off
    symms = [
        [ 1.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  1.   ],
        [-1.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   ,  0.   ,  0.   ,  1.   ],
        [-1.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   , -1.   ],
        [ 1.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   ,  0.   ,  0.   , -1.   ],
        [ 0.   ,  1.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.   ],
        [ 0.   , -1.   ,  0.   , -1.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.   ],
        [ 0.   , -1.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ],
        [ 0.   ,  1.   ,  0.   , -1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ],
        [ 0.   ,  0.   ,  1.   ,  0.   , -1.   ,  0.   ,  1.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   , -1.   ,  0.   , -1.   ,  0.   , -1.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   , -1.   ,  0.   ,  1.   ,  0.   ,  1.   ,  0.   ,  0.   ],
        [ 0.   ,  0.   ,  1.   ,  0.   ,  1.   ,  0.   , -1.   ,  0.   ,  0.   ],
        [-1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  1.   ,  0.   ],
        [-1.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   , -1.   ,  0.   ],
        [ 1.   ,  0.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   ,  1.   ,  0.   ],
        [ 1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   , -1.   ,  0.   ],
        [ 0.   ,  0.   ,  1.   ,  1.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ],
        [ 0.   ,  0.   , -1.   , -1.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ],
        [ 0.   ,  0.   , -1.   ,  1.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   ],
        [ 0.   ,  0.   ,  1.   , -1.   ,  0.   ,  0.   ,  0.   , -1.   ,  0.   ],
        [ 0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  1.   ,  1.   ,  0.   ,  0.   ],
        [ 0.   , -1.   ,  0.   ,  0.   ,  0.   , -1.   ,  1.   ,  0.   ,  0.   ],
        [ 0.   , -1.   ,  0.   ,  0.   ,  0.   ,  1.   , -1.   ,  0.   ,  0.   ],
        [ 0.   ,  1.   ,  0.   ,  0.   ,  0.   , -1.   , -1.   ,  0.   ,  0.   ],
        [ cos3 ,  sin3 ,  0.   , msin3 ,  cos3 ,  0.   ,  0.   ,  0.   ,  1.   ],
        [ cos3 , msin3 ,  0.   ,  sin3 ,  cos3 ,  0.   ,  0.   ,  0.   ,  1.   ],
        [mcos3 ,  sin3 ,  0.   , msin3 , mcos3 ,  0.   ,  0.   ,  0.   ,  1.   ],
        [mcos3 , msin3 ,  0.   ,  sin3 , mcos3 ,  0.   ,  0.   ,  0.   ,  1.   ],
        [ cos3 , msin3 ,  0.   , msin3 , mcos3 ,  0.   ,  0.   ,  0.   , -1.   ],
        [ cos3 ,  sin3 ,  0.   ,  sin3 , mcos3 ,  0.   ,  0.   ,  0.   , -1.   ],
        [mcos3 , msin3 ,  0.   , msin3 ,  cos3 ,  0.   ,  0.   ,  0.   , -1.   ],
        [mcos3 ,  sin3 ,  0.   ,  sin3 ,  cos3 ,  0.   ,  0.   ,  0.   , -1.   ],
    ]
    # fmt: on
    return np.array(symms).reshape(-1, 3, 3)


def possible_unitary_rotation_names() -> List[str]:
    """32 kinds of symmetry operations"""
    return [
        "identity                                ",
        "180 deg rotation - cart. axis [0,0,1]   ",
        "180 deg rotation - cart. axis [0,1,0]   ",
        "180 deg rotation - cart. axis [1,0,0]   ",
        "180 deg rotation - cart. axis [1,1,0]   ",
        "180 deg rotation - cart. axis [1,-1,0]  ",
        " 90 deg rotation - cart. axis [0,0,-1]  ",
        " 90 deg rotation - cart. axis [0,0,1]   ",
        "180 deg rotation - cart. axis [1,0,1]   ",
        "180 deg rotation - cart. axis [-1,0,1]  ",
        " 90 deg rotation - cart. axis [0,1,0]   ",
        " 90 deg rotation - cart. axis [0,-1,0]  ",
        "180 deg rotation - cart. axis [0,1,1]   ",
        "180 deg rotation - cart. axis [0,1,-1]  ",
        " 90 deg rotation - cart. axis [-1,0,0]  ",
        " 90 deg rotation - cart. axis [1,0,0]   ",
        "120 deg rotation - cart. axis [-1,-1,-1]",
        "120 deg rotation - cart. axis [-1,1,1]  ",
        "120 deg rotation - cart. axis [1,1,-1]  ",
        "120 deg rotation - cart. axis [1,-1,1]  ",
        "120 deg rotation - cart. axis [1,1,1]   ",
        "120 deg rotation - cart. axis [-1,1,-1] ",
        "120 deg rotation - cart. axis [1,-1,-1] ",
        "120 deg rotation - cart. axis [-1,-1,1] ",
        " 60 deg rotation - cryst. axis [0,0,1]  ",
        " 60 deg rotation - cryst. axis [0,0,-1] ",
        "120 deg rotation - cryst. axis [0,0,1]  ",
        "120 deg rotation - cryst. axis [0,0,-1] ",
        "180 deg rotation - cryst. axis [1,-1,0] ",
        "180 deg rotation - cryst. axis [2,1,0]  ",
        "180 deg rotation - cryst. axis [0,1,0]  ",
        "180 deg rotation - cryst. axis [1,1,0]  ",
    ]


def calc_overlap_matrix(vec: np.ndarray) -> np.ndarray:
    """calculate the overlap matrix given basis vectors.
    .. math::
        \\mathbf{O} = \\frac{1}{\\mathbf{V} \\cdot \\mathbf{V^T}}

    Args:
        vec (np.ndarray): 3x3 basis vectors

    Returns:
        np.ndarray: 3x3 overlap matrix
    """
    return np.linalg.inv(vec.dot(vec.T))


def check_symmetry(
    sym: np.ndarray, vec: np.ndarray, overlap: np.ndarray = None, tol: float = 1e-6
) -> bool:
    """check a symmetry operation is possible with regards to a set of
    basis vectors and the corresponding overlapping matrix

    Args:
        sym (np.ndarray): 3x3 unitary matrix
        vec (np.ndarray): 3x3 basis vectors
        overlap (np.ndarray): 3x3 overlap matrix. Defaults to None.
        tol (float): numerical tolerance. Defaults to 1e-6.

    Returns:
        bool: true for a valid operation
    """
    if overlap is None:
        overlap = calc_overlap_matrix(vec)

    # rotate basis according to sym
    vec_rotated = vec.dot(sym)
    # project the rotated basis onto the crystal coordinate
    # aka, onto the basis vectors
    vec_projected_crys = vec_rotated.dot(vec.T)
    # compute the overlap with the overlap matrix, which is
    # the inverse of the square of the basis vectors
    overlapped = overlap.dot(vec_projected_crys.T)
    # if all the elements in `overlapped` are integers, it means
    # the rotated vectors aligns with the original crystal.
    # otherwise, it means it is not a valid sym op.
    return np.all(np.abs(np.round(overlapped) - overlapped) < tol)
