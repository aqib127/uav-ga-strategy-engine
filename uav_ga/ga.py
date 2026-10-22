"""Main genetic algorithm loop."""
import numpy as np
from . import config as C
from .chromosome import Chromosome
from .fitness import FitnessEvaluator
from . import operators as ops


class GeneticAlgorithm:
    def __init__(self, env, seed=0):
        np.random.seed(seed)
        self.env = env
        self.eval = FitnessEvaluator(env)
        self.history = {"best": [], "avg": []}
        self.best = None

    def initialize(self):
        pop = []
        for _ in range(C.POPULATION_SIZE):
            c = Chromosome()
            self.eval.evaluate(c)
            pop.append(c)
        return pop

    def step(self, pop):
        pop.sort(key=lambda c: c.fitness, reverse=True)
        new_pop = [c.copy() for c in pop[:C.ELITISM]]
        while len(new_pop) < C.POPULATION_SIZE:
            p1 = ops.tournament_select(pop)
            p2 = ops.tournament_select(pop)
            c1, c2 = ops.crossover(p1, p2)
            c1 = ops.repair(ops.mutate(c1), self.env)
            c2 = ops.repair(ops.mutate(c2), self.env)
            self.eval.evaluate(c1)
            new_pop.append(c1)
            if len(new_pop) < C.POPULATION_SIZE:
                self.eval.evaluate(c2)
                new_pop.append(c2)
        return new_pop

    def run(self, generations=C.GENERATIONS, verbose=True):
        pop = self.initialize()
        for gen in range(generations):
            pop = self.step(pop)
            fits = [c.fitness for c in pop]
            best = max(fits)
            avg = float(np.mean(fits))
            self.history["best"].append(best)
            self.history["avg"].append(avg)
            if self.best is None or best > self.best.fitness:
                self.best = max(pop, key=lambda c: c.fitness).copy()
            if verbose and (gen % 10 == 0 or gen == generations - 1):
                print(f"Gen {gen:3d} | best={best:.4f} | avg={avg:.4f}")
        return self.best
