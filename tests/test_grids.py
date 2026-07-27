import numpy as np

from pmhd.data.grids import k_grid, eps_grid, z_grid, theta_grid, theta_gridfull, B0_grid


def test_k_grid_shape_and_ordering():
    karr = k_grid()
    assert len(karr) == 69
    assert np.all(np.diff(karr) < 0)
    assert karr.min() > 0


def test_eps_grid_shape_and_range():
    epsarr = eps_grid()
    assert len(epsarr) == 100
    assert epsarr.min() == -1
    assert epsarr.max() == -0.01


def test_z_grid_decreasing():
    zarr = z_grid()
    assert zarr[0] == 1900
    assert zarr[-1] > 600
    assert np.all(np.diff(zarr) < 0)


def test_theta_grid_shape_and_range():
    thetaarr = theta_grid()
    assert len(thetaarr) == 17
    assert thetaarr[0] == 0
    assert np.isclose(thetaarr[-1], np.pi / 2)
    assert np.all(np.diff(thetaarr) > 0)


def test_theta_gridfull_shape_and_range():
    thetaarr = theta_gridfull()
    assert len(thetaarr) == 33
    assert thetaarr[0] == 0
    assert np.isclose(thetaarr[-1], np.pi)


def test_B0_grid_shape_and_ordering():
    B0arr = B0_grid(Bmax=5e-9, Bmin=5e-12, nB=61)
    assert len(B0arr) == 61
    assert np.all(np.diff(B0arr) > 0)
    assert np.isclose(B0arr[0], 5e-12)
    assert np.isclose(B0arr[-1], 5e-9)
