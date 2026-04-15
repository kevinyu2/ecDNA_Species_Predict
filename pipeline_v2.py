import numpy as np
import pandas as pd
from atacSim import atacDataSimulation
from collections import defaultdict
import itertools
import random
from cassiopeia.mixins.errors import ecDNABirthDeathSimulatorError
import os
import math
import argparse

##################################################################
# Master controls
##################################################################

parser = argparse.ArgumentParser(
    description="Pipeline for creating many simulations on cass",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "out_dir",
    type=str,
    help="Main output dir"
)

parser.add_argument(
    "--runs",
    type=int,
    default = 5,
    help="Number of runs for each setting"
)

parser.add_argument(
    "--species-max",
    type=int,
    default = 5,
    help="Tries all species counts up to this number"
)

parser.add_argument(
    "--overlap-prop",
    type=float,
    default = 0,
    help="Proportion of genes that will overlap"
)

parser.add_argument(
    "--change-chance",
    type=float,
    default = 0.1,
    help="Proportion of genes with extra counts"
)

parser.add_argument(
    "--init-cn",
    type=int,
    default = 4,
    help="Initial copy numbers"
)

parser.add_argument(
    "--depth",
    type=float,
    default = 1,
    help="Number of samples per actual counts"
)

parser.add_argument(
    "--depth-std",
    type=float,
    default = 0,
    help="std of depth parameter per cell"
)


parser.add_argument(
    "--fmax",
    type=float,
    default = 0.1,
    help="fitness is max(fmax, 0.1 * # distinct species)"
)

parser.add_argument(
    "--num-extant",
    type=int,
    default = 200000,
    help="Number of final cells in the simulation"
)

parser.add_argument(
    "--min-prop",
    type=float,
    default = 0.0,
    help="Reject all simulations where some ecDNA has less than this proportion"
)

args = parser.parse_args()
##################################################################


# Number of attempts each
num_attempts = args.runs
out_dir_main = args.out_dir

species_counts = np.arange(1, args.species_max + 1)

# Proportion of each species' genes that goes into other species
# Set to 0 to prevent overlaps
gene_overlap_proportion = args.overlap_prop

# Chance to add extra counts of a gene in an ecDNA species
chance_to_change = args.change_chance

# Average number of copies of ecDNA to start with
copy_number_initial_mean = args.init_cn

# Read depth (expected number of times each gene is read by ATAC-seq). Greater makes the multinomial approximation more accurate
depth_mean = args.depth
depth_std = args.depth_std

# Always max(fitness_max, 0.1 * # distinct species). Set to 0.1 for no co-selection, and 0.
fitness_max = args.fmax

num_extant = args.num_extant

# Remove any simulations where an ecDNA is in less than this proportion of sampled cells
min_ecDNA_prop = args.min_prop

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
gene_count_total = 30
gene_count_std = 0
change_distribution_param = 0.8
initial_birth_scale = 0.5
death_waiting_distribution_param = 8
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
def rand_with_mean(m, strength=15):
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
    
    num_overlap_dict = {}
    for i, count in enumerate(counts):
        # Take min because we don't want to use all of them, otherwise it's just one ecDNA really
        num_overlap_dict[i] = min(count - 1, int(math.ceil(count * overlap_prop)))
    
    done = set()

    for i, count in enumerate(counts):

        num_overlap = num_overlap_dict[i]
        # Only look at intersections from that set
        valid_intersections = [comb for comb in all_intersections if i in comb]
        
        # Assign each overlapping gene randomly
        for _ in range(num_overlap):
            # Cleanse valid intersections
            to_remove = []
            for intersect_idx, intersect in enumerate(valid_intersections) :
                for species_no in done :
                    if species_no in intersect :
                        to_remove.append(intersect_idx)
                        break 

            # remove duplicates and delete safely (reverse order)
            for idx in sorted(set(to_remove), reverse=True):
                del valid_intersections[idx]
                
            if len(valid_intersections) > 0 :

                chosen = random.choice(valid_intersections)
                overlap_dict[chosen] += 1
                for species_no in chosen :
                    num_overlap_dict[species_no] = num_overlap_dict[species_no] - 1
                    if num_overlap_dict[species_no] == 0 :
                        done.add(species_no)
    
    return dict(overlap_dict)
##############################################################

for species_count in species_counts :
    for average_combination_chance in average_combination_chances :
        if average_combination_chance <= 0 :
            allow_cosegregation = False
        else :
            allow_cosegregation = True
        gene_count_mean = gene_count_total / species_count
        fitness_array = build_fitness(species_count, fitness_max)
        print(f"fitness array: {fitness_array}")
        capacity = np.full(species_count, capacities)


        out_dir = f"{out_dir_root}/{species_count}_species_{average_combination_chance}_comb"
        os.makedirs(out_dir, exist_ok= True)

        failures = 0
        i = 0
        while i < num_attempts :
            initial_copy_number_array = init_array_random(species_count, copy_number_initial_mean, 2, 4)
            gene_counts = init_array_random(species_count, gene_count_mean, gene_count_std, 5)

            gene_overlap = {}
            if gene_overlap_proportion > 0 and species_count > 1 :
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
                    random_seed = np.random.randint(1, 1001),
                    min_ecDNA_prop = min_ecDNA_prop
                    )
                sim.run_sim() 
                i += 1   

                # Free memory
                del sim
                import gc
                gc.collect()

            except ecDNABirthDeathSimulatorError as e:
                print(f"Iteration {i}: {e}")
                failures += 1
                continue

        print(f"Succeded with {failures} aborted simulations")