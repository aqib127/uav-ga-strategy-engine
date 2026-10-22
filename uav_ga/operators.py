"""Selection, crossover, mutation, repair."""
import numpy as np
from . import config as C


def tournament_select(pop, k=C.TOURNAMENT_SIZE):
    picks = np.random.choice(len(pop), size=k, replace=False)
    return max((pop[i] for i in picks), key=lambda c: c.fitness).copy()


def crossover(p1, p2):
    if np.random.rand() > C.CROSSOVER_RATE:
        return p1.copy(), p2.copy()

    g = C.GENES_PER_UAV
    child1, child2 = p1.genes.copy(), p2.genes.copy()
    mode = np.random.choice(["segment", "blend"])

    if mode == "segment":
        for i in range(C.NUM_UAVS):
            if np.random.rand() < 0.5:
                s = slice(i * g, (i + 1) * g)
                child1[s], child2[s] = child2[s].copy(), child1[s].copy()
    else:
        alpha = np.random.rand(len(p1.genes))
        child1 = alpha * p1.genes + (1 - alpha) * p2.genes
        child2 = alpha * p2.genes + (1 - alpha) * p1.genes

    from .chromosome import Chromosome
    return Chromosome(child1), Chromosome(child2)


def mutate(chrom):
    g = C.GENES_PER_UAV
    n = C.GRID_SIZE
    for i in range(C.NUM_UAVS):
        base = i * g
        if np.random.rand() < C.MUTATION_RATE:
            for k in range(1 + C.NUM_WAYPOINTS):
                idx_r = base + 2 * k
                idx_c = base + 2 * k + 1
                chrom.genes[idx_r] += np.random.normal(0, C.MUTATION_SIGMA)
                chrom.genes[idx_c] += np.random.normal(0, C.MUTATION_SIGMA)
                chrom.genes[idx_r] = np.clip(chrom.genes[idx_r], 0, n - 1)
                chrom.genes[idx_c] = np.clip(chrom.genes[idx_c], 0, n - 1)

            si = base + 2 + 2 * C.NUM_WAYPOINTS
            chrom.genes[si] = np.clip(
                chrom.genes[si] + np.random.normal(0, 0.3), C.SPEED_MIN, C.SPEED_MAX)
            pi = si + 1
            chrom.genes[pi] = np.clip(
                chrom.genes[pi] + np.random.normal(0, 0.15), C.POWER_MIN, C.POWER_MAX)

            bi = pi + 1
            if np.random.rand() < 0.1:
                chrom.genes[bi] = 1.0 - chrom.genes[bi]
    return chrom


def repair(chrom, env):
    g = C.GENES_PER_UAV
    for i in range(C.NUM_UAVS):
        base = i * g
        for k in range(1 + C.NUM_WAYPOINTS):
            r = int(round(chrom.genes[base + 2 * k]))
            c = int(round(chrom.genes[base + 2 * k + 1]))
            if env.is_blocked(r, c):
                fixed = False
                for rad in range(1, 6):
                    for dr in range(-rad, rad + 1):
                        for dc in range(-rad, rad + 1):
                            nr, nc = r + dr, c + dc
                            if env.in_bounds(nr, nc) and not env.is_blocked(nr, nc):
                                chrom.genes[base + 2 * k] = nr
                                chrom.genes[base + 2 * k + 1] = nc
                                fixed = True
                                break
                        if fixed:
                            break
                    if fixed:
                        break
    return chrom
