"""Hybrid real/binary chromosome for a UAV fleet."""
import numpy as np
from . import config as C


class Chromosome:
    def __init__(self, genes=None):
        if genes is None:
            genes = self._random_genes()
        self.genes = np.asarray(genes, dtype=float)
        self.fitness = -1e9
        self.objectives = {}

    @staticmethod
    def _random_genes():
        n = C.GRID_SIZE
        genes = []
        for _ in range(C.NUM_UAVS):
            genes.append(np.random.uniform(0, n))
            genes.append(np.random.uniform(0, n))
            for _ in range(C.NUM_WAYPOINTS):
                genes.append(np.random.uniform(0, n))
                genes.append(np.random.uniform(0, n))
            genes.append(np.random.uniform(C.SPEED_MIN, C.SPEED_MAX))
            genes.append(np.random.uniform(C.POWER_MIN, C.POWER_MAX))
            genes.append(float(np.random.randint(0, 2)))
        return np.array(genes)

    def uav_slices(self):
        g = C.GENES_PER_UAV
        return [slice(i * g, (i + 1) * g) for i in range(C.NUM_UAVS)]

    def decode_uav(self, idx):
        s = self.genes[self.uav_slices()[idx]]
        deploy = (s[0], s[1])
        waypoints = [(s[2 + 2 * k], s[3 + 2 * k]) for k in range(C.NUM_WAYPOINTS)]
        speed = s[2 + 2 * C.NUM_WAYPOINTS]
        power = s[3 + 2 * C.NUM_WAYPOINTS]
        sensor_on = int(round(s[4 + 2 * C.NUM_WAYPOINTS])) % 2
        return deploy, waypoints, speed, power, sensor_on

    def copy(self):
        c = Chromosome(self.genes.copy())
        c.fitness = self.fitness
        c.objectives = dict(self.objectives)
        return c

    def __repr__(self):
        return f"Chromosome(len={len(self.genes)}, fit={self.fitness:.4f})"
