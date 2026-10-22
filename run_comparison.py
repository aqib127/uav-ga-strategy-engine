"""GA vs random search."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga import config as C
from uav_ga.environment import Environment
from uav_ga.ga import GeneticAlgorithm
from uav_ga.baseline import random_search

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    ga_scores, rs_scores = [], []
    for s in range(args.seeds):
        env = Environment(seed=s)
        ga = GeneticAlgorithm(env, seed=s)
        best = ga.run(verbose=False)
        ga_scores.append(best.fitness)
        budget = C.POPULATION_SIZE * C.GENERATIONS
        rb, _ = random_search(env, budget=budget, seed=s)
        rs_scores.append(rb.fitness)
        print(f"Seed {s}: GA={best.fitness:.4f}  Random={rb.fitness:.4f}")

    plt.figure(figsize=(7, 4))
    plt.bar(["GA", "Random"], [np.mean(ga_scores), np.mean(rs_scores)],
            yerr=[np.std(ga_scores), np.std(rs_scores)], capsize=5)
    plt.ylabel("Best Fitness"); plt.title("GA vs Random Search")
    plt.savefig("results/fig5_comparison.png", dpi=150, bbox_inches="tight")
    json.dump({"ga": ga_scores, "random": rs_scores},
              open("results/comparison.json", "w"), indent=2)
    print("Saved results/fig5_comparison.png and comparison.json")
