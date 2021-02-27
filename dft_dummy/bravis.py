"""Bravis Lattice"""
from enum import Enum
from typing import Dict, Tuple

import numpy as np


def calc_volume(vec: np.ndarray) -> float:
    """calculate the volume of the cell
    [Reference](https://mathinsight.org/scalar_triple_product)

    Args:
        vec (np.ndarray): 3x3 array

    Returns:
        float: cell volume
    """
    return np.abs(np.linalg.det(vec))


def make_lattice_general(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> Tuple[np.ndarray, float]:
    """making a bravis lattice and return its lattice vectors and volume

    Args:
        a (float): first lattice constant, no unit
        b (float): second lattice constant, no unit
        c (float): third lattice constant, no unit
        alpha (float): angle between b&c, 0-pi
        beta (float): angle between a&c, 0-pi
        gamma (float): angle between a&b, 0-pi

    Returns:
        Tuple[np.ndarray, float]: Lattice vectors (3x3) and volume
    """
    cartisian = np.eye(3)
    a1 = a * cartisian[0]
    a2 = b * np.cos(gamma) * cartisian[0] + b * np.sin(gamma) * cartisian[1]
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c * c - cx * cx - cy * cy)
    a3 = np.dot([cx, cy, cz], cartisian)
    volume = a * b * cz * np.sin(gamma)
    return np.vstack((a1, a2, a3)), volume


class BravisLattice(str, Enum):
    triclinic = "Triclinic"
    monoclinic = "Monoclinic"
    orthorhombic = "orthorhombic"
    tetragonal = "Tetragonal"
    trigonal = "Trigonal"  # Rhomohedral
    cubic = "Cubic"
    fcc = "Face Centre Cubic"
    bcc = "Body Centre Cubic"
    hcp = "Hexagonal"


def make_lattice_bravis(
    bravis: BravisLattice, **kwargs: Dict[str, float]
) -> Tuple[np.ndarray, float]:
    """make a lattice according to the type of Lattice

    Args:
        bravis (BravisLattice): type of Bravis lattice

    Keyword Args:
        a, b, c, alpha, beta, gamma

    Raises:
        ValueError: wrong bravis lattice
        TypeError: missing parameters
        KeyError: missing parameters

    Returns:
        Tuple[np.ndarray, float]: Lattice vectors (3x3) and volume
    """
    a = kwargs["a"]  # first lattice constant
    if bravis == BravisLattice.triclinic:
        pass
    elif bravis == BravisLattice.monoclinic:
        kwargs["alpha"] = kwargs["gamma"] = np.pi / 2
    elif bravis == BravisLattice.orthorhombic:
        kwargs["alpha"] = kwargs["gamma"] = kwargs["beta"] = np.pi / 2
    elif bravis == BravisLattice.tetragonal:
        kwargs["alpha"] = kwargs["gamma"] = kwargs["beta"] = np.pi / 2
        kwargs["b"] = a
    elif bravis == BravisLattice.cubic:
        return (a * np.eye(3), a * a * a)
    elif bravis == BravisLattice.fcc:
        return (
            a
            * np.array(
                [
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                ]
            ),
            0.25 * a * a * a,
        )
    elif bravis == BravisLattice.bcc:
        return (
            0.5
            * a
            * np.array(
                [
                    [1, 1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                ]
            ),
            0.5 * a * a * a,
        )
    elif bravis == BravisLattice.hcp:
        c = kwargs["c"] / a
        vec = a * np.array([[1, 0, 0], [-1 / 2, np.sqrt(3) / 2, 0], [0, 0, c]])
        return (vec, calc_volume(vec))
    else:
        raise ValueError(f"unsupported Bravis Lattice, {bravis}")

    return make_lattice_general(**kwargs)
