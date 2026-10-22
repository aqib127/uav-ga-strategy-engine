"""Random search baseline with the same evaluation budget."""
import numpy as np
from .chromosome import Chromosome
from .fitness import FitnessEvaluator


def random_search(env, budget, seed=0):
    np.random.seed(seed)
    ev = FitnessEvaluator(env)
    best, fits = None, []
    for _ in range(budget):
        c = Chromosome()
        ev.evaluate(c)
        fits.append(c.fitness)
        if best is None or c.fitness > best.fitness:
            best = c
    return best, fits
