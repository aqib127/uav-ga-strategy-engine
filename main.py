"""Single GA run + figures."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_ga import config as C
from uav_ga.environment import Environment
from uav_ga.ga import GeneticAlgorithm

os.makedirs("results", exist_ok=True)

def plot_convergence(hist, path="results/fig2_convergence.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(hist["best"], label="Best")
    plt.plot(hist["avg"], label="Average")
    plt.xlabel("Generation"); plt.ylabel("Fitness")
    plt.title("Fitness Convergence"); plt.legend(); plt.grid(True)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

def plot_paths(env, chrom, path="results/fig3_paths.png"):
    plt.figure(figsize=(8, 8))
    plt.imshow(env.priority, cmap="Greens", alpha=0.6)
    plt.imshow(env.hazard, cmap="Reds", alpha=0.35)
    plt.imshow(env.no_fly, cmap="gray", alpha=0.4)
    colors = ["blue", "orange", "purple", "cyan"]
    for i in range(C.NUM_UAVS):
        deploy, wps, *_ = chrom.decode_uav(i)
        pts = [deploy] + wps
        xs = [p[1] for p in pts]; ys = [p[0] for p in pts]
        plt.plot(xs, ys, "-o", color=colors[i % len(colors)], label=f"UAV {i}")
    for (r, c) in env.targets:
        plt.plot(c, r, "k*", markersize=10)
    plt.title("Evolved UAV Flight Paths"); plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

def plot_heatmap(env, chrom, path="results/fig4_coverage.png"):
    n = C.GRID_SIZE
    cov = np.zeros((n, n))
    for i in range(C.NUM_UAVS):
        deploy, wps, speed, power, on = chrom.decode_uav(i)
        if not on: continue
        pts = [deploy] + wps
        for a, b in zip(pts[:-1], pts[1:]):
            for t in np.linspace(0, 1, 10):
                r = int(round(a[0] + t * (b[0] - a[0])))
                c = int(round(a[1] + t * (b[1] - a[1])))
                if env.in_bounds(r, c): cov[r, c] += power
    plt.figure(figsize=(8, 7))
    plt.imshow(cov, cmap="viridis")
    plt.colorbar(label="Coverage intensity")
    plt.title("Coverage Heat-map")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    env = Environment(seed=args.seed)
    ga = GeneticAlgorithm(env, seed=args.seed)
    best = ga.run()

    plot_convergence(ga.history)
    plot_paths(env, best)
    plot_heatmap(env, best)

    print("\n=== FINAL SOLUTION ===")
    print(f"Fitness: {best.fitness:.4f}")
    clean_obj = {k: float(v) for k, v in best.objectives.items()}
    print(f"Objectives: {clean_obj}")
    with open("results/summary.json", "w") as f:
        json.dump({"fitness": best.fitness,
                   "objectives": best.objectives,
                   "history": ga.history}, f, indent=2)
    print("\nPlots saved in results/")
