"""Test the `met` module."""

from hugs.calc import get_wind_components, get_wind_dir, get_wind_speed

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest


def test_speed():
    """Test calculating wind speed."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.])

    assert_array_almost_equal(true_speed, speed, 4)


def test_scalar_speed():
    """Test wind speed with scalars."""
    s = get_wind_speed(-3., -4.)
    assert_almost_equal(s, 5., 3)


def test_dir():
    """Test calculating wind direction."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    direc = get_wind_dir(u, v)

    true_dir = np.array([270., 225., 180., 270.])

    assert_array_almost_equal(true_dir, direc, 4)


def test_wind_components_scalar():
    """Test wind_components with scalar inputs."""
    u, v = get_wind_components(4., 2.)

    assert_array_almost_equal(u, -0.1395, 4)
    assert_array_almost_equal(v, -3.9976, 4)


def test_wind_components():
    """Test wind_components with array inputs."""
    speed = np.array([4., 2., 5., 0.])
    wdir = np.array([0., 300., 40., 135.])

    idx = 2
    u, v = get_wind_components(speed, wdir)
    uu, vv = get_wind_components(speed[idx], wdir[idx])
    assert(len(speed) == len(u))
    assert(len(speed) == len(v))
    assert_array_almost_equal(u[idx], uu, 4)
    assert_array_almost_equal(v[idx], vv, 4)


def test_warning_direction():
    """Test that warning is raised when wind direction > 360."""
    with pytest.warns(UserWarning):
        get_wind_components(4., 554.)
