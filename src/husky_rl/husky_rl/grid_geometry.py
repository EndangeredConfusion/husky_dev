"""Pure world-to-grid coordinate conversion for the RL pipeline."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class GridGeometry:
    """Describe how EKF world coordinates map onto the training grid."""

    width: int
    height: int
    cell_size_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0
    invert_y_axis: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid width and height must be positive")
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be finite and greater than zero")
        if not math.isfinite(self.origin_x_m) or not math.isfinite(self.origin_y_m):
            raise ValueError("grid origin must be finite")

    def world_to_grid(self, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        """Return the containing grid cell, or ``None`` when out of bounds."""
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            return None

        grid_x = math.floor((x_m - self.origin_x_m) / self.cell_size_m)
        raw_grid_y = math.floor((y_m - self.origin_y_m) / self.cell_size_m)
        grid_y = self.height - 1 - raw_grid_y if self.invert_y_axis else raw_grid_y

        if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            return None
        return grid_x, grid_y
