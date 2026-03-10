import numpy as np
import pandas as pd
from atacSim import atacDataSimulation
import itertools
import random

##################################################################
# Master controls
##################################################################

out_dir = '../three_ecDNA'

species_count = 3
# "coefficient", "venn", or "simulation"
cosegregation_type = "simulation"

# TODO: allow genes to belong to multiple species
allow_gene_overlap = False

# Allow cosegregation
allow_cosegregation = True

# For coefficient: simply the amount that follows the first species
cosegregation_coefficient = 0.5
# For venn cosegregation: value between 0 & 1, greater means more will end up cosegregating
venn_cosegregation_strength = 0.5
# For simulation: average chance to combine (using beta distribution)
average_combination_chance = 0.6
allow_self_combine = False

# Chance to add extra counts of a gene in an ecDNA species
chance_to_change = 0.1



##################################################################
# Don't Need to Change
##################################################################

gene_count_mean = 10
copy_number_initial_mean = 6
num_attempts = 10
change_distribution_param = 0.8
multinomial_mult = 1
initial_birth_scale = 0.5
death_waiting_distribution_param = 8
num_extant = 20000
num_cells_mean = 2000
capacity = [3,3,3]
sim_mult = 1.5

##################################################################

# Create fitness matrix
# is a shape (2,2,...) matrix, each entry is the number of ecDNA present times 0.1 (capping at 0.5)
def build_fitness(n):
    shape = (2,) * n
    arr = np.zeros(shape)

    for idx in np.ndindex(shape):
        ones = sum(idx)
        arr[idx] = min(0.5, ones * 0.1)

    return arr
fitness_array = build_fitness(species_count)
print(f"fitness array: {fitness_array}")

# Let gene numbers and initial values vary slightly
def init_array_random(n, center, std, min):
    arr = np.random.normal(loc=center, scale=std, size=n)
    arr = np.round(arr).astype(int)
    arr = np.clip(arr, min, None)
    return arr

# Generate venn diagram coeffs stochastically
def generate_venn(n, cosegregation_strength, seed=None):
    intersection_limiter = 1 - cosegregation_strength
    if seed is not None:
        random.seed(seed)

    # remaining capacity for each circle
    remaining = {i: 1.0 for i in range(n)}
    regions = {}

    singles = [(i,) for i in range(n)]
    intersections = []
    for r in range(2, n+1):
        intersections.extend(itertools.combinations(range(n), r))

    # shuffle intersection order
    random.shuffle(intersections)

    # initialize singles with random values (to prefer the singles)
    for s in singles:
        i = s[0]
        val = random.random()
        val = min(val, remaining[i])
        regions[s] = val
        remaining[i] -= val

    # fill intersections
    for inter in intersections:

        proposed = max(0.0, random.random() - intersection_limiter)

        # max allowed without exceeding circle capacity
        max_allowed = min(remaining[i] for i in inter)

        val = min(proposed, max_allowed)
        if val > 0 :
            regions[inter] = val
            for i in inter:
                remaining[i] -= val

    # Remove singles
    for s in singles:
        del regions[s]

    return regions

# Beta dist with mean m
def rand_with_mean(m, strength=5):
    alpha = m * strength
    beta = (1 - m) * strength
    return random.betavariate(alpha, beta)

def generate_coseg_matrix(matrix_dim, mean, allow_self_combine) :
    matrix = np.zeros((matrix_dim, matrix_dim))
    for i in range(matrix_dim) :
        for j in range(i + 1) :
            if i == j and not allow_self_combine :
                continue
            random_val = rand_with_mean(mean)
            matrix[i][j] = random_val
            matrix[j][i] = random_val
    return matrix

failures = 0
for i in range(num_attempts) :
    initial_copy_number_array = init_array_random(species_count, copy_number_initial_mean, 1, 5)
    gene_counts = init_array_random(species_count, gene_count_mean, 3, 5)

    gene_overlap = {}

    # Vary number of cells (large scale)
    num_cells =  np.random.normal(loc=num_cells_mean, scale=300)
    num_cells = np.round(num_cells).astype(int)
    num_cells = max(num_cells, 500)

    coeffs = {}
    if allow_cosegregation and cosegregation_type == 'venn' :
        coeffs = generate_venn(species_count, venn_cosegregation_strength)

    mat = np.zeros((species_count, species_count))
    if allow_cosegregation and cosegregation_type == 'simulation' :
        mat = generate_coseg_matrix(species_count, average_combination_chance, allow_self_combine)

    try:
        sim = atacDataSimulation(
            out_dir = out_dir, run_name = f'run_{i}',
            initial_copy_number_array = initial_copy_number_array,
            fitness_array = fitness_array,
            cosegregation_type = cosegregation_type,
            gene_counts = gene_counts,
            multinomial_mult = multinomial_mult,
            noise_scale = 1,
            gene_overlap = gene_overlap,
            chance_to_change = chance_to_change,
            change_distribution = np.random.geometric,
            change_distribution_param = change_distribution_param,
            initial_birth_scale = initial_birth_scale,
            death_waiting_distribution_param = death_waiting_distribution_param,
            num_extant = num_extant,
            num_cells = num_cells,
            coeffs = coeffs,
            mat = mat,
            capacity = capacity,
            sim_mult = sim_mult,
            random_seed = np.random.randint(1, 1001)
            )
        sim.run_sim()    

    except ecDNABirthDeathSimulatorError as e:
        print(f"Iteration {i}: {e}")
        failures += 1
        continue

print(f"{num_attempts - failures} / {num_attempts} Succeeded")