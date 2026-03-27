import numpy as np
import pandas as pd
from atacSim import atacDataSimulation
from collections import defaultdict
import itertools
import random
from cassiopeia.mixins.errors import ecDNABirthDeathSimulatorError
import os

##################################################################
# Master controls
##################################################################

# Number of attempts each
num_attempts = 5
out_dir_main = '../simulated_data/'

species_counts = [1,2,3,4,5]

# Proportion of each species' genes that goes into other species
# Set to 0 to prevent overlaps
gene_overlap_proportion = 0

# Chance to add extra counts of a gene in an ecDNA species
chance_to_change = 0.1

# Average number of copies of ecDNA to start with
copy_number_initial_mean = 4

# Read depth (expected number of times each gene is read by ATAC-seq). Greater makes the multinomial approximation more accurate
depth_mean = 2
depth_std = 0.4

# Always max(fitness_max, 0.1 * # distinct species). Set to 0.1 for no co-selection, and 0.
fitness_max = 0.1

##################################################################
# Don't Need to Change
##################################################################

# "coefficient", "venn", or "simulation"
cosegregation_type = "simulation"

# For coefficient: simply the amount that follows the first species
cosegregation_coefficient = 0.5
# For venn cosegregation: value between 0 & 1, greater means more will end up cosegregating
venn_cosegregation_strength = 0.5
# For simulation: average chance to combine (using beta distribution)
average_combination_chances = [0,0.2,0.4,0.6,0.8]
allow_self_combine = False

out_dir_root = f"{out_dir_main}/fmax_{fitness_max}_overlap_{gene_overlap_proportion}_extracounts_{chance_to_change}_depth_{depth_mean}"
gene_count_total = 60
gene_count_std = 0
change_distribution_param = 0.8
initial_birth_scale = 0.5
death_waiting_distribution_param = 8
num_extant = 20000
num_cells_mean = 2000
capacities = 3
sim_mult = 1.5
noise_scale = 1

##################################################################

os.makedirs(out_dir_root, exist_ok=True)

# Create fitness matrix
# is a shape (2,2,...) matrix, each entry is the number of ecDNA present times 0.1 (capping at 0.5)
def build_fitness(n, fitness_max):
    shape = (2,) * n
    arr = np.zeros(shape)

    for idx in np.ndindex(shape):
        ones = sum(idx)
        arr[idx] = min(fitness_max, ones * 0.1)

    return arr


# Let gene numbers and initial values vary slightly
def init_array_random(n, center, std, min):
    arr = np.random.normal(loc=center, scale=std, size=n)
    arr = np.round(arr).astype(int)
    min = max(min, 1)
    arr = np.clip(arr, min, None)
    return arr

# Generate venn diagram coeffs stochastically
def generate_venn(n, cosegregation_strength):
    intersection_limiter = 1 - cosegregation_strength

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

# Gets a proportion of genes from each species, and randomly assigns those to venn diagram intersections
def generate_gene_overlap(counts, overlap_prop):
    n = len(counts)
    overlap_dict = defaultdict(int)
    
    # Precompute all possible intersections of size >= 2
    all_intersections = []
    for r in range(2, n + 1):
        all_intersections.extend(itertools.combinations(range(n), r))
    
    for i, count in enumerate(counts):
        num_overlap = int(round(count * overlap_prop))
        
        # Only look at intersections from that set
        valid_intersections = [comb for comb in all_intersections if i in comb]
        
        # Assign each overlapping gene randomly
        for _ in range(num_overlap):
            chosen = random.choice(valid_intersections)
            overlap_dict[chosen] += 1
    
    return dict(overlap_dict)
##############################################################

for species_count in species_counts :
    for average_combination_chance in average_combination_chances :
        if average_combination_chance == 0 :
            allow_cosegregation = False
        gene_count_mean = gene_count_total / species_count
        fitness_array = build_fitness(species_count, fitness_max)
        print(f"fitness array: {fitness_array}")
        capacity = np.full(species_count, capacities)


        out_dir = f"{out_dir_root}/{species_count}_species_{average_combination_chance}_comb"
        os.makedirs(out_dir, exist_ok= True)

        failures = 0
        i = 0
        while i < num_attempts :
            initial_copy_number_array = init_array_random(species_count, copy_number_initial_mean, 2, 1)
            gene_counts = init_array_random(species_count, gene_count_mean, gene_count_std, 5)

            gene_overlap = {}
            if gene_overlap_proportion > 0 :
                gene_overlap = generate_gene_overlap(gene_counts, gene_overlap_proportion)

            # Vary number of cells (large scale)
            num_cells = np.random.normal(loc=num_cells_mean, scale=300)
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
                    depth_mean = depth_mean,
                    depth_std = depth_std,
                    noise_scale = noise_scale,
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
                i += 1   

            except ecDNABirthDeathSimulatorError as e:
                print(f"Iteration {i}: {e}")
                failures += 1
                continue

        print(f"{num_attempts - failures} / {num_attempts} Succeeded")