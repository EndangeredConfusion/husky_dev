"""Tests for the Discretizer's world-to-grid transform."""

import math

import pytest

from husky_rl.grid_geometry import GridGeometry


def test_world_to_grid_uses_cell_containment() -> None:
    geometry = GridGeometry(width=17, height=11, cell_size_m=0.5)

    assert geometry.world_to_grid(0.0, 0.0) == (0, 0)
    assert geometry.world_to_grid(0.99, 1.01) == (1, 2)


def test_world_to_grid_applies_origin_offset() -> None:
    geometry = GridGeometry(
        width=17,
        height=11,
        cell_size_m=0.5,
        origin_x_m=-1.0,
        origin_y_m=2.0,
    )

    assert geometry.world_to_grid(-0.75, 2.75) == (0, 1)


def test_world_to_grid_can_invert_y_axis() -> None:
    geometry = GridGeometry(
        width=17,
        height=11,
        cell_size_m=1.0,
        invert_y_axis=True,
    )

    assert geometry.world_to_grid(3.2, 0.2) == (3, 10)
    assert geometry.world_to_grid(3.2, 10.2) == (3, 0)


@pytest.mark.parametrize(
    'x_m,y_m',
    [(-0.01, 0.0), (0.0, -0.01), (17.0, 0.0), (0.0, 11.0)],
)
def test_world_to_grid_rejects_out_of_bounds(x_m: float, y_m: float) -> None:
    geometry = GridGeometry(width=17, height=11, cell_size_m=1.0)

    assert geometry.world_to_grid(x_m, y_m) is None


def test_world_to_grid_rejects_non_finite_positions() -> None:
    geometry = GridGeometry(width=17, height=11, cell_size_m=1.0)

    assert geometry.world_to_grid(math.nan, 0.0) is None
    assert geometry.world_to_grid(0.0, math.inf) is None


@pytest.mark.parametrize('cell_size_m', [0.0, -1.0, math.nan])
def test_grid_geometry_requires_positive_finite_cell_size(cell_size_m: float) -> None:
    with pytest.raises(ValueError, match='cell_size_m'):
        GridGeometry(width=17, height=11, cell_size_m=cell_size_m)
