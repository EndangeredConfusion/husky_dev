"""Tests for the '86 Field cell table loader (no ROS needed)."""

import ast
import copy
import math
import os

import pytest
import yaml

from husky_rl.cell_table import (
    CellTable, LocalFrame, fix_accepted, parse_cell_key,
    STATUS_FIX, STATUS_GBAS_FIX, STATUS_NO_FIX, STATUS_SBAS_FIX)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(PKG, 'config', 'field86_cells.yaml')


def _training_grid():
    """Read OPEN_CELLS / GRID_* from astar_node.py without importing ROS."""
    src = open(os.path.join(PKG, 'husky_rl', 'astar_node.py'), encoding='utf-8').read()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', '')
            if name in ('OPEN_CELLS', 'GRID_WIDTH', 'GRID_HEIGHT'):
                out[name] = ast.literal_eval(node.value)
    return out['OPEN_CELLS'], out['GRID_WIDTH'], out['GRID_HEIGHT']


OPEN_CELLS, GRID_WIDTH, GRID_HEIGHT = _training_grid()


@pytest.fixture(scope='module')
def raw():
    with open(TABLE, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='module')
def table():
    return CellTable.load(TABLE)


def test_table_matches_training_grid(table):
    table.check_against(OPEN_CELLS, GRID_WIDTH, GRID_HEIGHT)
    assert set(table.centers) == OPEN_CELLS
    assert (table.width, table.height) == (GRID_WIDTH, GRID_HEIGHT)


def test_goals_match_policy_order(table):
    assert table.goals == {0: (1, 1), 1: (15, 1), 2: (1, 9), 3: (15, 9)}


def test_latlon_to_local_matches_table(raw, table):
    assert table.mismatches == []
    for key, entry in raw['cells'].items():
        east, north = table.center(parse_cell_key(key))
        assert east == pytest.approx(entry['east_m'], abs=0.02)
        assert north == pytest.approx(entry['north_m'], abs=0.02)


def test_origin_is_star_1(table):
    assert table.center((1, 1)) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert table.anchors[(1, 1)] == 'Star_1'


def test_local_latlon_round_trip():
    frame = LocalFrame(42.730705, -73.680592)
    for east, north in [(0.0, 0.0), (123.4, -56.7), (-10.0, 80.0)]:
        lat, lon = frame.to_latlon(east, north)
        assert frame.to_local(lat, lon) == pytest.approx((east, north), abs=1e-6)


def test_frame_matches_generator_formula():
    lat0, lon0 = 42.730705, -73.680592
    kx = math.pi / 180 * 6378137.0 * math.cos(math.radians(lat0))
    ky = math.pi / 180 * 6378137.0
    frame = LocalFrame(lat0, lon0)
    lat, lon = 42.7300, -73.6790
    assert frame.to_local(lat, lon) == pytest.approx(((lon - lon0) * kx, (lat - lat0) * ky))


def test_every_centre_is_its_own_nearest_cell(table):
    for cell, (east, north) in table.centers.items():
        nearest, dist = table.nearest_cell(east, north)
        assert nearest == cell
        assert dist == pytest.approx(0.0, abs=1e-9)
        assert table.distance_to_walkway(east, north) == pytest.approx(0.0, abs=1e-9)


def test_point_near_a_centre_maps_to_that_cell(table):
    east, north = table.center((8, 1))
    nearest, dist = table.nearest_cell(east + 1.0, north)
    assert nearest == (8, 1)
    assert dist == pytest.approx(1.0)


def test_midpoint_of_longest_gap_is_on_walkway(table):
    (ax, ay), (bx, by) = table.center((1, 2)), table.center((1, 3))
    assert math.hypot(bx - ax, by - ay) > 11.0
    mid = ((ax + bx) / 2, (ay + by) / 2)
    assert table.distance_to_walkway(*mid) == pytest.approx(0.0, abs=1e-9)
    # ...even though it is ~6 m from both centres
    assert table.nearest_cell(*mid)[1] > 5.5


def test_point_beside_a_walk_is_off_walkway(table):
    (ax, ay), (bx, by) = table.center((8, 1)), table.center((9, 1))
    mx, my = (ax + bx) / 2, (ay + by) / 2
    length = math.hypot(bx - ax, by - ay)
    nx, ny = -(by - ay) / length, (bx - ax) / length  # unit normal
    assert table.distance_to_walkway(mx + 10 * nx, my + 10 * ny) == pytest.approx(10.0, abs=0.5)


def test_path_to_local_flattens_centres(table):
    path = [(4, 1), (3, 1), (2, 1), (1, 1)]
    flat = table.path_to_local(path)
    assert len(flat) == 8
    assert all(isinstance(v, float) for v in flat)
    assert flat[:2] == pytest.approx(list(table.center((4, 1))))
    assert flat[-2:] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_missing_cell_fails_check(raw):
    data = copy.deepcopy(raw)
    del data['cells']['8,1']
    with pytest.raises(ValueError, match='missing cells'):
        CellTable.from_dict(data).check_against(OPEN_CELLS)


def test_extra_cell_fails_check(raw):
    data = copy.deepcopy(raw)
    data['cells']['0,0'] = dict(data['cells']['1,1'])
    with pytest.raises(ValueError, match='not open'):
        CellTable.from_dict(data).check_against(OPEN_CELLS)


@pytest.mark.parametrize('key', ['8', '8,1,2', 'a,b'])
def test_bad_key_raises(raw, key):
    data = copy.deepcopy(raw)
    data['cells'][key] = dict(data['cells']['1,1'])
    with pytest.raises(ValueError):
        CellTable.from_dict(data)


def test_cell_outside_grid_raises(raw):
    data = copy.deepcopy(raw)
    data['cells']['17,0'] = dict(data['cells']['1,1'])
    with pytest.raises(ValueError, match='outside'):
        CellTable.from_dict(data)


def test_goal_not_in_table_raises(raw):
    data = copy.deepcopy(raw)
    del data['cells']['15,9']
    with pytest.raises(ValueError, match='goal'):
        CellTable.from_dict(data)


def test_edited_latlon_without_east_north_is_reported(raw):
    data = copy.deepcopy(raw)
    data['cells']['8,1']['lat'] += 1e-5  # ~1.1 m north
    table = CellTable.from_dict(data)
    assert [c for c, _ in table.mismatches] == [(8, 1)]


@pytest.mark.parametrize('status,lat,lon,rtk,ok', [
    (STATUS_NO_FIX, math.nan, math.nan, False, False),
    (STATUS_FIX, math.nan, -73.68, False, False),
    (STATUS_FIX, 42.73, -73.68, False, True),
    (STATUS_SBAS_FIX, 42.73, -73.68, True, False),
    (STATUS_GBAS_FIX, 42.73, -73.68, True, True),
])
def test_fix_accepted(status, lat, lon, rtk, ok):
    assert fix_accepted(status, lat, lon, require_rtk=rtk)[0] is ok


def test_fix_rejected_by_covariance():
    cov = [4.0, 0, 0, 0, 4.0, 0, 0, 0, 9.0]  # 2 m std
    assert fix_accepted(STATUS_FIX, 42.73, -73.68, covariance=cov, covariance_type=2,
                        max_horizontal_std_m=1.0)[0] is False
    assert fix_accepted(STATUS_FIX, 42.73, -73.68, covariance=cov, covariance_type=0,
                        max_horizontal_std_m=1.0)[0] is True
