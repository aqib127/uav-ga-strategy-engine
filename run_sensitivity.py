"""Hyper-parameter sensitivity analysis."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga import config as C
from uav_ga.environment import Environment
from uav_ga.ga import GeneticAlgorithm

def _run(pop, mut, tour, seeds, gens):
    scores = []
    old = (C.POPULATION_SIZE, C.MUTATION_RATE, C.TOURNAMENT_SIZE)
    C.POPULATION_SIZE, C.MUTATION_RATE, C.TOURNAMENT_SIZE = pop, mut, tour
    try:
        for s in range(seeds):
            env = Environment(seed=s)
            ga = GeneticAlgorithm(env, seed=s)
            scores.append(ga.run(generations=gens, verbose=False).fitness)
    finally:
        C.POPULATION_SIZE, C.MUTATION_RATE, C.TOURNAMENT_SIZE = old
    return float(np.mean(scores))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gens", type=int, default=60)
    args = ap.parse_args()

    results = {}
    for pop in [30, 60, 100]:
        results[f"pop={pop}"] = _run(pop, 0.15, 3, args.seeds, args.gens)
    for mut in [0.05, 0.15, 0.30]:
        results[f"mut={mut}"] = _run(60, mut, 3, args.seeds, args.gens)
    for tour in [2, 3, 5]:
        results[f"tour={tour}"] = _run(60, 0.15, tour, args.seeds, args.gens)

    print(json.dumps(results, indent=2))
    json.dump(results, open("results/sensitivity.json", "w"), indent=2)

    plt.figure(figsize=(10, 4))
    plt.bar(list(results.keys()), list(results.values()))
    plt.xticks(rotation=45, ha="right"); plt.ylabel("Mean best fitness")
    plt.title("Hyper-parameter Sensitivity"); plt.tight_layout()
    plt.savefig("results/fig6_sensitivity.png", dpi=150, bbox_inches="tight")
