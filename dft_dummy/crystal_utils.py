"""crystal system utilities such as coordinate transformation, etc"""

import numpy as np


def project_points(vec: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """project list of points onto the given basis vectors

    Args:
        vec (np.ndarray): 3x3 basis vectors
        pts (np.ndarray): Nx3 points in the basis vector coordinates

    Returns:
        np.ndarray: Nx3 projected points
    """
    if pts.shape[-1] != 3:
        raise ValueError("invalid points")
    pts = pts.reshape(-1, 3)

    return vec.dot(pts.T).T


def calc_reciprocal(vec: np.ndarray) -> np.ndarray:
    """calculate the reciprocal basis vectors

    Args:
        vec (np.ndarray): basis vectors

    Returns:
        np.ndarray: the reciprocal basis vectors
    """
    return np.linalg.inv(vec)


def make_mesh(
    nx: int, ny: int, nz: int, dx: bool = False, dy: bool = False, dz: bool = False
) -> np.ndarray:
    """make a uniformly distributed mesh, like in a Monkhorst-Pack grid but instead of
    centering at the origin, this mesh starts off from the origin. Both of them spread
    evenly in the first Brillouin Zone and are essentially equivalent. The returned
    array is in the crytal coordinates, NOT necessarily the cartisian system.

    Args:
        nx (int): number of points along 1st crytal axis
        ny (int): number of points along 2nd crytal axis
        nz (int): number of points along 3rd crytal axis
        dx (bool): to offset half a step along 1st crytal axis
        dy (bool): to offset half a step along 2nd crytal axis
        dz (bool): to offset half a step along 3rd crytal axis

    Returns:
        np.ndarray: Nx3 mesh, where N == nx * ny * nz
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("invalid number of points")
    nstep_x = np.linspace(0, 1, num=nx, endpoint=False)
    nstep_y = np.linspace(0, 1, num=ny, endpoint=False)
    nstep_z = np.linspace(0, 1, num=nz, endpoint=False)
    x, y, z = np.meshgrid(nstep_x, nstep_y, nstep_z)
    pts = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    offset = np.array([1 / nx / 2 * dx, 1 / ny / 2 * dy, 1 / nz / 2 * dz])
    return pts + offset
