"""Tests for the reached-cell / hysteresis logic used by the discretizer."""

import math
import os

import pytest

from husky_rl.cell_table import CellTable, CellTracker

TABLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'field86_cells.yaml')


@pytest.fixture(scope='module')
def table():
    return CellTable.load(TABLE)


def _walk(a, b, step=0.5):
    """Points from a to b (inclusive) every ``step`` metres."""
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(math.ceil(d / step)))
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            for i in range(n + 1)]


def _offset(p, dx, dy=0.0):
    return p[0] + dx, p[1] + dy


def test_centre_is_reached(table):
    tracker = CellTracker(table)
    upd = tracker.update(*table.center((8, 1)))
    assert upd.cell == (8, 1) and upd.changed and not upd.bootstrapped


def test_bootstrap_nearest_on(table):
    tracker = CellTracker(table, bootstrap_nearest=True)
    upd = tracker.update(*_offset(table.center((8, 1)), 3.0))  # > 1.5 m, on the walk
    assert upd.cell == upd.nearest and upd.changed and upd.bootstrapped


def test_bootstrap_nearest_off(table):
    tracker = CellTracker(table, bootstrap_nearest=False)
    upd = tracker.update(*_offset(table.center((8, 1)), 3.0))
    assert upd.cell is None and not upd.changed


def test_bootstrap_only_once(table):
    tracker = CellTracker(table, bootstrap_nearest=True)
    a, b = table.center((8, 1)), table.center((9, 1))
    tracker.update(*a)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    upd = tracker.update(*mid)
    assert upd.cell == (8, 1) and not upd.changed


def test_hysteresis_walking_between_centres(table):
    tracker = CellTracker(table, reached_radius_m=1.5)
    a, b = table.center((8, 1)), table.center((9, 1))
    seen = []
    for p in _walk(a, b):
        upd = tracker.update(*p)
        dist_b = math.hypot(p[0] - b[0], p[1] - b[1])
        expected = (9, 1) if dist_b <= 1.5 or seen and seen[-1] == (9, 1) else (8, 1)
        assert upd.cell == expected, (p, upd)
        seen.append(upd.cell)
    assert seen[0] == (8, 1) and seen[-1] == (9, 1)
    assert sum(1 for x, y in zip(seen, seen[1:]) if x != y) == 1  # exactly one change


def test_turning_back_without_reaching_keeps_cell(table):
    tracker = CellTracker(table)
    a, b = table.center((8, 1)), table.center((9, 1))
    tracker.update(*a)
    tracker.update(*b)
    # walk back to 2 m short of A: still B
    back = _walk(b, a)
    for p in back:
        if math.hypot(p[0] - a[0], p[1] - a[1]) < 2.0:
            break
        assert tracker.update(*p).cell == (9, 1)


def test_change_flag_only_on_new_cell(table):
    tracker = CellTracker(table)
    c = table.center((8, 1))
    assert tracker.update(*c).changed
    assert not tracker.update(*_offset(c, 0.5)).changed
    assert not tracker.update(*_offset(c, -0.5)).changed


def test_off_walkway_does_not_change_cell(table):
    tracker = CellTracker(table, off_walkway_m=6.0)
    a = table.center((8, 1))
    tracker.update(*a)
    upd = tracker.update(a[0], a[1] + 20.0)  # 20 m north of the top walk
    assert upd.off_walkway and upd.cell == (8, 1) and not upd.changed
    upd = tracker.update(*table.center((9, 1)))  # back on the walk
    assert not upd.off_walkway and upd.cell == (9, 1) and upd.changed


def test_off_walkway_at_start_does_not_bootstrap(table):
    tracker = CellTracker(table, bootstrap_nearest=True)
    a = table.center((8, 1))
    upd = tracker.update(a[0], a[1] + 20.0)
    assert upd.off_walkway and upd.cell is None


def test_middle_of_longest_gap_is_not_off_walkway(table):
    tracker = CellTracker(table, off_walkway_m=6.0, bootstrap_nearest=False)
    (ax, ay), (bx, by) = table.center((1, 2)), table.center((1, 3))
    upd = tracker.update((ax + bx) / 2, (ay + by) / 2)
    assert not upd.off_walkway


def test_skipped_cell_is_reported(table):
    tracker = CellTracker(table)
    tracker.update(*table.center((8, 1)))
    upd = tracker.update(*table.center((10, 1)))
    assert upd.cell == (10, 1) and upd.skipped_from == (8, 1)
    upd = tracker.update(*table.center((11, 1)))
    assert upd.skipped_from is None


def test_closest_non_adjacent_pair_no_flicker(table):
    """(12,5) and (13,6) are only ~5.4 m apart but not neighbours on the grid."""
    a, b = table.center((12, 5)), table.center((13, 6))
    assert math.hypot(b[0] - a[0], b[1] - a[1]) < 6.0
    tracker = CellTracker(table, reached_radius_m=1.5)
    cells = [tracker.update(*p).cell for p in _walk(a, b)]
    changes = sum(1 for x, y in zip(cells, cells[1:]) if x != y)
    assert cells[0] == (12, 5) and cells[-1] == (13, 6) and changes == 1


def test_non_finite_position_ignored(table):
    tracker = CellTracker(table)
    tracker.update(*table.center((8, 1)))
    upd = tracker.update(math.nan, 0.0)
    assert upd.cell == (8, 1) and not upd.changed


def test_every_cell_reachable_at_its_centre(table):
    tracker = CellTracker(table)
    for cell, centre in table.centers.items():
        assert tracker.update(*centre).cell == cell


@pytest.mark.parametrize('radius,walkway', [(0.0, 6.0), (1.5, -1.0)])
def test_bad_parameters_rejected(table, radius, walkway):
    with pytest.raises(ValueError):
        CellTracker(table, reached_radius_m=radius, off_walkway_m=walkway)
