"""Bravis Lattice"""
from enum import Enum
from typing import Tuple

import numpy as np


def make_lattice(
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
    hexagonal = "Hexagonal"
    cubic = "Cubic"
    fcc = "Face Centre Cubic"
    bcc = "Body Centre Cubic"
