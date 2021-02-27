import numpy as np
import pytest

from dft_dummy import crystal_utils


def test_project():
    vec = np.random.randn(3, 3)
    pts = np.random.randn(10, 3)
    pts_proj = crystal_utils.project_points(vec, pts)

    pts_proj_exp = np.array([vec.dot(p) for p in pts])
    assert np.allclose(pts_proj, pts_proj_exp)


def test_project_throw():
    vec = np.random.randn(3, 3)
    pts = np.random.randn(3, 10)
    with pytest.raises(ValueError):
        crystal_utils.project_points(vec, pts)


def test_reciprocal():
    vec = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
        ]
    )
    rvec = crystal_utils.calc_reciprocal(vec)
    rvec_exp = np.array(
        [
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1],
        ]
    )
    assert np.allclose(rvec, rvec_exp)


@pytest.mark.parametrize(
    "args,res_exp",
    [
        ((1, 1, 1, 0, 0, 0), [[0, 0, 0]]),
        ((1, 1, 1, 1, 1, 1), [[0.5, 0.5, 0.5]]),
        (
            (2, 2, 2, 0, 0, 0),
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ],
        ),
        (
            (2, 2, 2, 1, 1, 1),
            [
                [0.25, 0.25, 0.25],
                [0.25, 0.25, 0.75],
                [0.75, 0.25, 0.25],
                [0.75, 0.25, 0.75],
                [0.25, 0.75, 0.25],
                [0.25, 0.75, 0.75],
                [0.75, 0.75, 0.25],
                [0.75, 0.75, 0.75],
            ],
        ),
        (
            (1, 1, 2, 0, 0, 1),
            [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75]],
        ),
    ],
)
def test_mesh(args, res_exp):
    res = crystal_utils.make_mesh(*args)
    assert np.allclose(res, res_exp)


@pytest.mark.parametrize("args", [(0, 1, 1), (-1, 1, 1), (1, -1, 1)])
def test_mesh_throw(args):
    with pytest.raises(ValueError):
        crystal_utils.make_mesh(*args)
