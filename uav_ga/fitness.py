"""Multi-objective fitness with hard-constraint penalties."""
import numpy as np
from . import config as C


class FitnessEvaluator:
    def __init__(self, env):
        self.env = env

    def _path(self, deploy, waypoints):
        return [deploy] + list(waypoints)

    def _path_length(self, pts):
        return sum(np.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(pts[:-1], pts[1:]))

    def _sample_cells(self, pts, steps=10):
        cells = []
        for a, b in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, steps):
                r = int(round(a[0] + t * (b[0] - a[0])))
                c = int(round(a[1] + t * (b[1] - a[1])))
                cells.append((r, c))
        return cells

    def _hazard_around(self, r, c, radius=2):
        """Hazard mass within `radius` cells of (r,c). Gives smooth gradient."""
        n = C.GRID_SIZE
        r0, r1 = max(0, r - radius), min(n, r + radius + 1)
        c0, c1 = max(0, c - radius), min(n, c + radius + 1)
        return float(self.env.hazard[r0:r1, c0:c1].sum())

    def evaluate(self, chrom):
        n = C.GRID_SIZE
        coverage = np.zeros((n, n))
        hazard_exposure = 0.0
        energy = 0.0
        time_total = 0.0
        detection = 0.0
        penalty = 0.0

        for i in range(C.NUM_UAVS):
            deploy, waypoints, speed, power, sensor_on = chrom.decode_uav(i)
            pts = self._path(deploy, waypoints)
            cells = self._sample_cells(pts)

            # ---- hard constraints ----
            length = self._path_length(pts)
            if length > C.MAX_FLIGHT_RANGE:
                penalty += C.PENALTY_PER_VIOLATION * (length - C.MAX_FLIGHT_RANGE) / 10
            for (r, c) in cells:
                if self.env.is_blocked(r, c):
                    penalty += C.PENALTY_PER_VIOLATION * 0.5

            # ---- coverage ----
            if sensor_on:
                for (r, c) in cells:
                    if self.env.in_bounds(r, c):
                        coverage[r, c] += power

            # ---- hazard exposure (proximity-based, smooth) ----
            for (r, c) in cells:
                if self.env.in_bounds(r, c):
                    hazard_exposure += self._hazard_around(r, c) * power

            # ---- energy & time ----
            energy += length * (0.5 + 0.5 * speed) * (0.3 + 0.7 * power)
            if speed > 0:
                time_total += length / speed

            # ---- target detection (vectorised) ----
            if sensor_on and cells:
                cell_arr = np.array(cells, dtype=float)
                tgt_arr = np.array(self.env.targets, dtype=float)
                dmat = np.hypot(
                    cell_arr[:, 0:1] - tgt_arr[None, :, 0],
                    cell_arr[:, 1:2] - tgt_arr[None, :, 1],
                )
                min_d = dmat.min(axis=0)
                detection += float(power * np.clip(1 - min_d / 5, 0, None).sum())

        # ---- normalise (reward terms in [0,1]; cost terms in [0,~0.6]) ----
        total_priority = float(self.env.priority.sum()) + 1e-9
        coverage_score = min(
            float((coverage * self.env.priority).sum() / total_priority),
            1.0,
        )
        detection_score = min(
            detection / (C.NUM_UAVS * len(self.env.targets) + 1e-9),
            1.0,
        )
        # --- relaxed cost denominators so costs sit in [0, ~0.6] ---
        energy_norm = min(energy / (C.MAX_ENERGY * C.NUM_UAVS * 2.0 + 1e-9), 1.0)
        time_norm   = min(time_total / (C.NUM_UAVS * 100.0 + 1e-9), 1.0)
        hazard_norm = min(hazard_exposure / (C.NUM_UAVS * 30.0 + 1e-9), 1.0)

        fitness = (C.W_COVERAGE * coverage_score
                   + C.W_DETECTION * detection_score
                   - C.W_ENERGY * energy_norm
                   - C.W_TIME * time_norm
                   - C.W_HAZARD * hazard_norm
                   - penalty)

        chrom.objectives = dict(
            coverage=float(coverage_score),
            detection=float(detection_score),
            energy=float(energy_norm),
            time=float(time_norm),
            hazard=float(hazard_norm),
            penalty=float(penalty),
        )
        chrom.fitness = float(fitness)
        return chrom.fitness
