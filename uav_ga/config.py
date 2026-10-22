"""Central configuration for the UAV GA strategy engine."""

# ---- Grid / environment ----
GRID_SIZE = 50
NUM_UAVS = 4
NUM_WAYPOINTS = 8
START_CENTER = (GRID_SIZE // 2, 0)

# ---- Chromosome gene layout (per UAV) ----
GENES_PER_UAV = 2 + 2 * NUM_WAYPOINTS + 3     # = 21
CHROMOSOME_LENGTH = NUM_UAVS * GENES_PER_UAV  # = 84

# ---- Genetic algorithm ----
POPULATION_SIZE = 60
GENERATIONS = 100
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.15
TOURNAMENT_SIZE = 3
ELITISM = 2
MUTATION_SIGMA = 2.0

# ---- Fitness weights ----
W_COVERAGE = 0.35
W_DETECTION = 0.25
W_ENERGY = 0.15
W_TIME = 0.10
W_HAZARD = 0.15

# ---- Hard constraints ----
MAX_FLIGHT_RANGE = 120.0
MAX_ENERGY = 100.0
SPEED_MIN, SPEED_MAX = 0.5, 3.0
POWER_MIN, POWER_MAX = 0.0, 1.0
PENALTY_PER_VIOLATION = 1.0
