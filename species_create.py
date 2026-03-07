import copy
import numpy as np
import pandas as pd
import scipy
import random
import cassiopeia as cas
import os
from itertools import combinations
from collections import defaultdict
import copy

###################################################
# Output
###################################################
out_dir = '../species_out/'
run_name = "run1"

###################################################
# ecDNA SPECIES SETTINGS
################################################### 
random_seed = 12346

# Birth/death statistics
initial_birth_scale = 0.5
death_waiting_distribution_param = 5
death_waiting_distribution = np.random.exponential(death_waiting_distribution_param)

# Number of existing cell copies at start
num_extant = 200000

# number of cells to sample
num_cells = 1000

# Number of initial copies
initial_copy_number_array = np.array([8,8,8])

# Fitness matrix, basically binary indexed: arr[i][j] denotes the fitness where i is the ecDNA status of the first ecDNA and j is the ecDNA status of the second.
# Fitness coefficients are all relative
fitness_array = np.array([
    [
        [0.0, 0.1],
        [0.1, 0.3],
    ],
    [
        [0.1, 0.2],
        [0.2, 0.3],
    ],
])

# "venn", "coefficient", or "simulation"
cosegregation_type = "venn"

# If venn :
coeffs = {
            (0,2) : 0.4,
            (0,1) : 0.25,
            (0,1,2) : 0.3}

# If cosegregation :
# How much of the copies are split purely based off of cosegregation
cosegregation = 0.5

# If simulation :
# mat[i][j] denotes the chance that species i and species j "combine"
mat =  [[0,      0.1,   0.6],
        [0.1,    0,     0.7], 
        [0.6,    0.7,   0]]
# Denotes the maximum number of attachments each species can have
capacity = [2,2,2]
# How many times the simulation runs per ecDNA
sim_mult = 1

############################################################
# Gene parameters
############################################################
# Genes per species
gene_counts = [15,19,10]

# After getting true sample counts, sample counts using multinomial
# This measures how many samples are drawn per true sample count
multinomial_mult = 1
# Add some normally distributed noise
noise_scale = 1

# Parameter to assign genes to multiple ecDNA
# dict where (a,b) : 5 means 5 overlap genes
gene_overlap = {
    (0,1) : 5
}

# Adding additional copies of genes to ecDNA species
chance_to_change = 0.2
change_distribution = np.random.geometric
change_distribution_param = 0.8

#############################################################
#############################################################
#############################################################


# Parameter to add different counts for each gene on each ecDNA (extra counts, so default is 0)
def additional_count_func(length, chance_to_change, change_distribution, change_distribution_param) :
    copies = np.zeros(length)

    # Change
    for i in range(length) :
        roll = random.random()
        if roll < chance_to_change :
            copies[i] = change_distribution(change_distribution_param)

    return copies

# Ensure dimensionality
num_ecDNA = len(gene_counts)
assert(num_ecDNA == len(initial_copy_number_array))

# Get output locations
metadata_out = f'{out_dir}/{run_name}_metadata.txt'
cellxgene_out = f'{out_dir}/{run_name}_cellxgene.tsv'
cellxgene_noiseless_out = f'{out_dir}/{run_name}_cellxgene_NONOISE.tsv'
os.makedirs(out_dir, exist_ok=True)

bd_sim = cas.sim.ecDNABirthDeathSimulator(
    birth_waiting_distribution = lambda scale: np.random.exponential(1/scale),
    initial_birth_scale = initial_birth_scale,
    death_waiting_distribution = lambda: death_waiting_distribution,
    num_extant = num_extant,
    random_seed=random_seed,
    initial_copy_number = initial_copy_number_array,
    splitting_function = lambda c, x: c+np.random.binomial(x, p=0.5), # this means ecDNA is inherited randomly
    fitness_array = fitness_array,
    cosegregation_coefficient = cosegregation,
    coeff_venn = coeffs,
    cosegregation_type = "venn",
    coeff_matrix_sim = mat,
    species_capacity = capacity,
    simulation_multiplier = sim_mult
)
ground_truth_tree = bd_sim.simulate_tree()

# subsample for cells
subsampler = cas.sim.UniformLeafSubsampler(number_of_leaves = num_cells)
ground_truth_tree = subsampler.subsample_leaves(ground_truth_tree)
counts = ground_truth_tree.cell_meta
ecDNA_species = list(counts.keys())

# Sort genes into species
gene_idx = 0
species_to_gene = defaultdict(list)
gene_to_species = defaultdict(list)
for i in range(len(ecDNA_species)) :
    while gene_counts[i] > 0 :
        curr_gene_name = "gene_" + str(gene_idx)
        gene_idx += 1

        # Account for overlaps
        accounted_for = False
        for overlap_key in gene_overlap.keys() :
            if gene_overlap[overlap_key] > 0 :
                if i in overlap_key :
                    for idx in overlap_key :
                        species_to_gene[ecDNA_species[idx]].append(curr_gene_name)
                        gene_to_species[curr_gene_name].append(ecDNA_species[idx])
                        gene_counts[idx] -= 1

                    gene_overlap[overlap_key] -= 1
                    accounted_for = True
                    break
        if not accounted_for :
            gene_counts[i] -= 1
            species_to_gene[ecDNA_species[i]].append(curr_gene_name)
            gene_to_species[curr_gene_name].append(ecDNA_species[i])

# Sort out additional copies per ecDNA
gene_to_species_mult = copy.deepcopy(gene_to_species)
additional_counts_all = {}
for species in species_to_gene.keys() :
    # Get counts of extras
    additional_counts = additional_count_func(len(species_to_gene[species]), chance_to_change, change_distribution, change_distribution_param)
    additional_counts_all[species] = additional_counts
    # Iterate through, for all above zero, append that number of the species to the dict
    for i, add in enumerate(additional_counts) :
        if add > 0:
            for copy_count in range(int(add)) :
                gene_to_species_mult[species_to_gene[species][i]].append(species)

# Match each gene to a vector of observed copy numbers
gene_cell_true_cn_dict = {}
for gene in gene_to_species_mult.keys() :
    gene_cell_true_cn_dict[gene] = np.zeros(num_cells)
    for species in gene_to_species_mult[gene] :
        gene_cell_true_cn_dict[gene] += counts[species].values


cbg_true_matrix = np.column_stack([gene_cell_true_cn_dict[g] for g in gene_cell_true_cn_dict.keys()])
cbg_noisy_matrix = np.zeros_like(cbg_true_matrix, dtype=int)

# Pick out of multinomial dist
for cell in range(num_cells):
    p = cbg_true_matrix[cell] / cbg_true_matrix[cell].sum()
    cbg_noisy_matrix[cell] = np.random.multinomial(cbg_true_matrix[cell].sum() * multinomial_mult, p)

# add gaussian noise
noise = np.random.normal(loc=0, scale=noise_scale, size=cbg_noisy_matrix.shape)
cbg_noisy_matrix = cbg_noisy_matrix + noise


# Export matrix
row_names = [f"cell_{i}" for i in range(cbg_noisy_matrix.shape[0])]
df = pd.DataFrame(cbg_noisy_matrix, columns=gene_to_species_mult.keys(), index=row_names)
df.to_csv(cellxgene_out, sep = '\t')
df = pd.DataFrame(cbg_true_matrix, columns=gene_to_species_mult.keys(), index=row_names)
df.to_csv(cellxgene_noiseless_out, sep = '\t')

# Get metadata and summary statistics
with open(metadata_out, 'w') as f :
    # the "ground truth"
    f.write("--GROUND TRUTH--\n")
    for gene, species_list in gene_to_species.items() :
        out_string = f"{gene}:"
        for species in species_list :
            out_string += f'\t{species}'
        f.write(f"{out_string}\n")
    f.write('\n')

    f.write("--SIMULATION PARAMETERS--\n")
    f.write(f"Seed:\t{random_seed}\n")
    f.write(f"Birth Param:\t{initial_birth_scale}\n")
    f.write(f"Death Param:\t{death_waiting_distribution_param}\n")
    f.write(f"Sim Start Cells:\t{num_extant}\n")
    f.write(f"Cell Count:\t{num_cells}\n")
    f.write('\n')
    f.write(f"Initial Copy Numbers:\t")
    print(initial_copy_number_array, file = f)
    print("Fitness array:", file = f)
    print(fitness_array, file = f)
    f.write('\n')
    f.write(f"Cosegregation type:\t{cosegregation_type}\n")
    if cosegregation_type == "venn" :
        f.write(f"Cosegregation coefficient:\t{cosegregation}\n")
    elif cosegregation_type == 'simuation' :
        f.write(f"Venn coefficients:\n")
        print(coeffs, file = f)
    else :
        f.write('Chance matrix:\n')
        print(mat, file = f)
        f.write('Species capacity:\t')
        print(capacity, file = f)
        f.write(f"Simulation runs per ecDNA:\t{sim_mult}\n")
    f.write('\n')

    f.write(f"Gene counts:\t{gene_counts}\n")
    f.write(f"Gene overlap:\n")
    print(gene_overlap, file = f)
    f.write(f"Additional copy chance:\t{chance_to_change}\n")
    f.write(f"Change distribution parameter:\t{change_distribution_param}\n")
    f.write('\n')

    f.write("--SIMULATED PARAMETERS--\n")
    for species, add in additional_counts_all.items() :
        print(f"Additional counts {species}:\t{add}", file = f)
    f.write('\n')

    for a, b in combinations(ecDNA_species, 2):
        correlation = scipy.stats.pearsonr(counts[a].values, counts[b].values)[0]
        f.write(f'Correlation between {a} and {b}: {correlation}\n')
