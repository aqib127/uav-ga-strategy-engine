"""Grid environment: priority zones, hazards, no-fly zones, survivor targets."""
import numpy as np
from . import config as C


class Environment:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        n = C.GRID_SIZE

        self.priority = np.zeros((n, n))
        for _ in range(6):
            r = rng.integers(5, n - 5)
            c = rng.integers(5, n - 5)
            val = rng.uniform(0.5, 1.0)
            self.priority[r:r + 8, c:c + 8] += val
        self.priority /= max(self.priority.max(), 1e-9)

        self.hazard = np.zeros((n, n))
        for _ in range(5):
            r = rng.integers(0, n - 6)
            c = rng.integers(0, n - 6)
            self.hazard[r:r + 6, c:c + 6] += rng.uniform(0.3, 1.0)
        self.hazard = np.clip(self.hazard, 0, 1)

        self.no_fly = np.zeros((n, n), dtype=bool)
        for _ in range(4):
            r = rng.integers(0, n - 5)
            c = rng.integers(0, n - 5)
            self.no_fly[r:r + 5, c:c + 5] = True

        self.targets = [(int(rng.integers(0, n)), int(rng.integers(0, n)))
                        for _ in range(10)]

    def in_bounds(self, r, c):
        return 0 <= r < C.GRID_SIZE and 0 <= c < C.GRID_SIZE

    def is_blocked(self, r, c):
        return (not self.in_bounds(r, c)) or self.no_fly[r, c]
