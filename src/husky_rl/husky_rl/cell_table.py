"""GPS cell table for the '86 Field: load, convert, look up, track.

Pure Python (PyYAML only, no ROS) so it can be unit-tested offline.

The table (config/field86_cells.yaml) gives every open training cell a real
GPS position.  Positions are converted to a local tangent plane in metres
(x = east, y = north, origin = ``local_frame.lat0/lon0``) with the same
equirectangular formula as config/gen_cell_table_v3.py, so a fix taken at a
cell's lat/lon lands exactly on that cell's centre.

Cell centres are computed from each cell's ``lat``/``lon`` (the GPS position
is the source of truth).  The table's ``east_m``/``north_m`` are only checked
for consistency; mismatches are listed in ``CellTable.mismatches``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

Cell = Tuple[int, int]

R_EARTH_M = 6378137.0

# sensor_msgs/NavSatStatus values (copied so this module stays ROS-free)
STATUS_NO_FIX = -1
STATUS_FIX = 0
STATUS_SBAS_FIX = 1
STATUS_GBAS_FIX = 2  # RTK fixed or float with nmea_navsat_driver

# NavSatFix.position_covariance_type value meaning "unknown"
COVARIANCE_TYPE_UNKNOWN = 0

# table east/north may differ from lat/lon-derived values by rounding (~1 cm)
MISMATCH_TOLERANCE_M = 0.05


def default_table_path() -> str:
    """Return the installed field86_cells.yaml (needs ament_index_python)."""
    import os

    from ament_index_python.packages import get_package_share_directory
    return os.path.join(
        get_package_share_directory('husky_rl'), 'config', 'field86_cells.yaml')


def parse_cell_key(key: str) -> Cell:
    """Turn ``"x,y"`` into ``(x, y)``."""
    parts = str(key).split(',')
    if len(parts) != 2:
        raise ValueError(f'cell key {key!r} is not "x,y"')
    return int(parts[0]), int(parts[1])


@dataclass(frozen=True)
class LocalFrame:
    """Equirectangular local tangent plane around (lat0, lon0)."""

    lat0: float
    lon0: float

    @property
    def kx(self) -> float:
        """Metres per degree of longitude at lat0."""
        return math.pi / 180.0 * R_EARTH_M * math.cos(math.radians(self.lat0))

    @property
    def ky(self) -> float:
        """Metres per degree of latitude."""
        return math.pi / 180.0 * R_EARTH_M

    def to_local(self, lat: float, lon: float) -> Tuple[float, float]:
        """Return (east_m, north_m) of a GPS position."""
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def to_latlon(self, east_m: float, north_m: float) -> Tuple[float, float]:
        """Return (lat, lon) of a local position."""
        return self.lat0 + north_m / self.ky, self.lon0 + east_m / self.kx


def _point_segment_distance(px: float, py: float,
                            ax: float, ay: float,
                            bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def are_adjacent(a: Cell, b: Cell) -> bool:
    """Return True for 4-neighbours on the training grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


class CellTable:
    """Open cells of the training grid with their real positions."""

    def __init__(self, frame: LocalFrame, width: int, height: int,
                 centers: Dict[Cell, Tuple[float, float]],
                 goals: Optional[Dict[int, Cell]] = None,
                 anchors: Optional[Dict[Cell, str]] = None,
                 mismatches: Optional[List[Tuple[Cell, float]]] = None) -> None:
        if width <= 0 or height <= 0:
            raise ValueError('grid width and height must be positive')
        if not centers:
            raise ValueError('cell table has no cells')
        for cell in centers:
            if not (0 <= cell[0] < width and 0 <= cell[1] < height):
                raise ValueError(f'cell {cell} is outside the {width}x{height} grid')
        for idx, goal in (goals or {}).items():
            if goal not in centers:
                raise ValueError(f'goal {idx} {goal} is not an open cell in the table')
        self.frame = frame
        self.width = width
        self.height = height
        self.centers = dict(centers)
        self.goals = dict(goals or {})
        self.anchors = dict(anchors or {})
        self.mismatches = list(mismatches or [])
        cells = sorted(self.centers)
        self.edges: List[Tuple[Cell, Cell]] = [
            (a, b) for i, a in enumerate(cells) for b in cells[i + 1:]
            if are_adjacent(a, b)]
        linked = {c for e in self.edges for c in e}
        self._isolated = [c for c in cells if c not in linked]

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> 'CellTable':
        """Load a table YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> 'CellTable':
        """Build a table from the parsed YAML structure."""
        try:
            lf = data['local_frame']
            frame = LocalFrame(float(lf['lat0']), float(lf['lon0']))
            width = int(data['grid']['width'])
            height = int(data['grid']['height'])
            raw_cells = data['cells']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'cell table is missing a required field: {exc}') from exc

        centers: Dict[Cell, Tuple[float, float]] = {}
        anchors: Dict[Cell, str] = {}
        mismatches: List[Tuple[Cell, float]] = []
        for key, entry in raw_cells.items():
            cell = parse_cell_key(key)
            if cell in centers:
                raise ValueError(f'cell {cell} appears twice')
            try:
                lat, lon = float(entry['lat']), float(entry['lon'])
            except (KeyError, TypeError) as exc:
                raise ValueError(f'cell {key} needs lat and lon') from exc
            if not (math.isfinite(lat) and math.isfinite(lon)):
                raise ValueError(f'cell {key} has non-finite lat/lon')
            east, north = frame.to_local(lat, lon)
            centers[cell] = (east, north)
            if 'east_m' in entry and 'north_m' in entry:
                err = math.hypot(east - float(entry['east_m']),
                                 north - float(entry['north_m']))
                if err > MISMATCH_TOLERANCE_M:
                    mismatches.append((cell, err))
            if entry.get('anchor'):
                anchors[cell] = str(entry['anchor'])

        goals = {int(k): parse_cell_key(v) for k, v in (data.get('goals') or {}).items()}
        return cls(frame, width, height, centers, goals, anchors, mismatches)

    def check_against(self, open_cells: Iterable[Cell],
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> None:
        """Raise ValueError unless the table covers exactly ``open_cells``."""
        expected = set(open_cells)
        have = set(self.centers)
        problems = []
        if width is not None and width != self.width:
            problems.append(f'grid width {self.width} != {width}')
        if height is not None and height != self.height:
            problems.append(f'grid height {self.height} != {height}')
        if expected - have:
            problems.append(f'missing cells {sorted(expected - have)}')
        if have - expected:
            problems.append(f'cells not open in training {sorted(have - expected)}')
        if problems:
            raise ValueError('cell table does not match the training grid: '
                             + '; '.join(problems))

    # ── lookups ─────────────────────────────────────────────────────────────

    def center(self, cell: Cell) -> Tuple[float, float]:
        """Return (east_m, north_m) of a cell centre (KeyError if not open)."""
        return self.centers[tuple(cell)]

    def nearest_cell(self, east_m: float, north_m: float) -> Tuple[Cell, float]:
        """Return the open cell whose centre is closest, and the distance."""
        best_cell, best_d = None, math.inf
        for cell, (cx, cy) in self.centers.items():
            d = math.hypot(east_m - cx, north_m - cy)
            if d < best_d:
                best_cell, best_d = cell, d
        return best_cell, best_d

    def distance_to_walkway(self, east_m: float, north_m: float) -> float:
        """Distance to the nearest segment joining two adjacent open cells."""
        best = math.inf
        for a, b in self.edges:
            ax, ay = self.centers[a]
            bx, by = self.centers[b]
            best = min(best, _point_segment_distance(east_m, north_m, ax, ay, bx, by))
        for c in self._isolated:
            cx, cy = self.centers[c]
            best = min(best, math.hypot(east_m - cx, north_m - cy))
        return best

    def path_to_local(self, cells: Sequence[Cell]) -> List[float]:
        """Flatten a cell path to [e0, n0, e1, n1, ...] in metres."""
        flat: List[float] = []
        for cell in cells:
            east, north = self.center(cell)
            flat.extend((float(east), float(north)))
        return flat


def fix_accepted(status: int, lat: float, lon: float, require_rtk: bool = False,
                 covariance: Optional[Sequence[float]] = None,
                 covariance_type: int = COVARIANCE_TYPE_UNKNOWN,
                 max_horizontal_std_m: float = 0.0) -> Tuple[bool, str]:
    """Decide whether a NavSatFix-like reading is usable; return (ok, reason)."""
    if status <= STATUS_NO_FIX:
        return False, 'no fix'
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False, 'non-finite lat/lon'
    if require_rtk and status < STATUS_GBAS_FIX:
        return False, f'not RTK (status {status})'
    if (max_horizontal_std_m > 0.0 and covariance is not None
            and covariance_type != COVARIANCE_TYPE_UNKNOWN):
        std = math.sqrt(max(float(covariance[0]), float(covariance[4]), 0.0))
        if std > max_horizontal_std_m:
            return False, f'horizontal std {std:.2f} m > {max_horizontal_std_m:.2f} m'
    return True, ''


@dataclass
class TrackerUpdate:
    """What one position update did."""

    cell: Optional[Cell]          # last reached cell (what the discretizer publishes)
    changed: bool                 # a new cell was reached on this update
    nearest: Optional[Cell]
    nearest_dist_m: float
    walkway_dist_m: float
    off_walkway: bool = False
    bootstrapped: bool = False    # cell set from nearest at startup, not reached
    skipped_from: Optional[Cell] = None  # previous cell when the new one is not adjacent


class CellTracker:
    """Reached-cell logic with hysteresis.

    A cell counts as reached only within ``reached_radius_m`` of its centre;
    the last reached cell is kept until another one is reached.  Positions
    more than ``off_walkway_m`` from every walkway segment never change the
    cell.  With ``bootstrap_nearest`` the very first cell is the nearest one
    (if on the walkways) even when it is not yet reached.
    """

    def __init__(self, table: CellTable, reached_radius_m: float = 1.5,
                 off_walkway_m: float = 6.0, bootstrap_nearest: bool = True) -> None:
        if not (reached_radius_m > 0.0 and off_walkway_m > 0.0):
            raise ValueError('reached_radius_m and off_walkway_m must be positive')
        self.table = table
        self.reached_radius_m = reached_radius_m
        self.off_walkway_m = off_walkway_m
        self.bootstrap_nearest = bootstrap_nearest
        self.cell: Optional[Cell] = None

    def reset(self) -> None:
        """Forget the last reached cell."""
        self.cell = None

    def update(self, east_m: float, north_m: float) -> TrackerUpdate:
        """Feed one position (local metres) and return what happened."""
        if not (math.isfinite(east_m) and math.isfinite(north_m)):
            return TrackerUpdate(self.cell, False, None, math.inf, math.inf, off_walkway=True)

        nearest, dist = self.table.nearest_cell(east_m, north_m)
        walkway = self.table.distance_to_walkway(east_m, north_m)
        result = TrackerUpdate(self.cell, False, nearest, dist, walkway)
        if walkway > self.off_walkway_m:
            result.off_walkway = True
            return result

        if dist <= self.reached_radius_m:
            if nearest != self.cell:
                if self.cell is not None and not are_adjacent(self.cell, nearest):
                    result.skipped_from = self.cell
                self.cell = nearest
                result.changed = True
        elif self.cell is None and self.bootstrap_nearest:
            self.cell = nearest
            result.changed = True
            result.bootstrapped = True

        result.cell = self.cell
        return result
