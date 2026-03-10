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
from typing import Literal


class atacDataSimulation():
    def __init__(
        self,

        ###################################################
        # Output
        ###################################################
        out_dir,
        run_name,

        ###################################################
        # ecDNA SPECIES SETTINGS
        ################################################### 
        # Number of initial copies (int array)
        initial_copy_number_array,

        # Fitness matrix, basically binary indexed: arr[i][j] denotes the fitness where i is the ecDNA status of the first ecDNA and j is the ecDNA status of the second.
        # Fitness coefficients are all relative
        fitness_array,

        # "venn", "coefficient", or "simulation"
        cosegregation_type: Literal['venn', 'coefficient', 'simulation'],

        ############################################################
        # Gene parameters
        ############################################################
        # Int list, genes per species
        gene_counts,

        # After getting true sample counts, sample counts using multinomial
        # This measures how many samples are drawn per true sample count
        multinomial_mult = 1,
        # Add some normally distributed noise
        noise_scale = 1,

        # Parameter to assign genes to multiple ecDNA
        # dict where (a,b) : 5 means 5 overlap genes
        gene_overlap = {},

        # Adding additional copies of genes to ecDNA species
        chance_to_change = 0,
        change_distribution = np.random.geometric,
        change_distribution_param = 0.8,


        ############################################################
        # Cass Simulation Settings
        ############################################################
        # How many times the simulation runs per ecDNA
        sim_mult = 1,

        random_seed = 5,

        # Birth/death statistics
        initial_birth_scale = 0.5,
        death_waiting_distribution_param = 7,

        # Number of final existing cells
        num_extant = 20000,

        # number of cells to sample
        num_cells = 1000,

        # If venn:
        # give a dict with correlation coefficients for each 
        # i.e. {(0,1) : 0.5, (0,1,2) : 0.2}
        coeffs = {}, 

        # If coefficient :
        # How much of the copies are split purely based off of cosegregation
        cosegregation = 0,

        # If simulation :
        # mat[i][j] denotes the chance that species i and species j "combine"
        mat = None,

        # Int list denoting the maximum number of attachments each species can have
        capacity = None,

    ):

    #############################################################
    #############################################################
    #############################################################

        # Ensure dimensionality
        self.num_ecDNA = len(gene_counts)
        assert(self.num_ecDNA == len(initial_copy_number_array))

        self.out_dir = out_dir
        self.run_name = run_name
        self.initial_copy_number_array = initial_copy_number_array
        self.fitness_array = fitness_array
        self.cosegregation_type = cosegregation_type
        self.gene_counts = gene_counts
        self.multinomial_mult = multinomial_mult
        self.noise_scale = noise_scale
        self.gene_overlap = gene_overlap
        self.chance_to_change = chance_to_change
        self.change_distribution = change_distribution
        self.change_distribution_param = change_distribution_param
        self.sim_mult = sim_mult
        self.random_seed = random_seed
        self.initial_birth_scale = initial_birth_scale
        self.num_extant = num_extant
        self.num_cells = num_cells
        self.coeffs = coeffs
        self.cosegregation = cosegregation
        self.mat = mat
        self.capacity = capacity
        self.death_waiting_distribution_param = death_waiting_distribution_param
        self.death_waiting_distribution = np.random.exponential(death_waiting_distribution_param)

    # Parameter to add different counts for each gene on each ecDNA (extra counts, so default is 0)
    def additional_count_func(self, length) :
        copies = np.zeros(length)

        # Change
        for i in range(length) :
            roll = random.random()
            if roll < self.chance_to_change :
                copies[i] = self.change_distribution(self.change_distribution_param)

        return copies
    
    def run_sim(self) :
        gene_counts_save = copy.deepcopy(self.gene_counts)
        gene_overlap_save = copy.deepcopy(self.gene_overlap)

        # Get output locations
        metadata_out = f'{self.out_dir}/{self.run_name}_metadata.txt'
        cellxgene_out = f'{self.out_dir}/{self.run_name}_cellxgene.tsv'
        cellxgene_noiseless_out = f'{self.out_dir}/{self.run_name}_cellxgene_NOISELESS.tsv'
        os.makedirs(self.out_dir, exist_ok=True)

        bd_sim = cas.sim.ecDNABirthDeathSimulator(
            birth_waiting_distribution = lambda scale: np.random.exponential(1/scale),
            initial_birth_scale = self.initial_birth_scale,
            death_waiting_distribution = lambda: self.death_waiting_distribution,
            num_extant = self.num_extant,
            random_seed=self.random_seed,
            initial_copy_number = self.initial_copy_number_array,
            splitting_function = lambda c, x: c+np.random.binomial(x, p=0.5),
            fitness_array = self.fitness_array,
            cosegregation_coefficient = self.cosegregation,
            coeff_venn = self.coeffs,
            cosegregation_type = self.cosegregation_type,
            coeff_matrix_sim = self.mat,
            species_capacity = self.capacity,
            simulation_multiplier = self.sim_mult
        )
        ground_truth_tree = bd_sim.simulate_tree()
        print("Tree simulation complete")

        # subsample for cells
        subsampler = cas.sim.UniformLeafSubsampler(number_of_leaves = self.num_cells)
        ground_truth_tree = subsampler.subsample_leaves(ground_truth_tree)
        counts = ground_truth_tree.cell_meta
        ecDNA_species = list(counts.keys())[0:len(self.initial_copy_number_array)]
        print(f"ecDNA Names: {ecDNA_species}")

        # Sort genes into species
        gene_idx = 0
        species_to_gene = defaultdict(list)
        gene_to_species = defaultdict(list)
        for i in range(len(ecDNA_species)) :
            while self.gene_counts[i] > 0 :
                curr_gene_name = "gene_" + str(gene_idx)
                gene_idx += 1

                # Account for overlaps
                accounted_for = False
                for overlap_key in self.gene_overlap.keys() :
                    if self.gene_overlap[overlap_key] > 0 :
                        if i in overlap_key :
                            for idx in overlap_key :
                                species_to_gene[ecDNA_species[idx]].append(curr_gene_name)
                                gene_to_species[curr_gene_name].append(ecDNA_species[idx])
                                self.gene_counts[idx] -= 1

                            self.gene_overlap[overlap_key] -= 1
                            accounted_for = True
                            break
                if not accounted_for :
                    self.gene_counts[i] -= 1
                    species_to_gene[ecDNA_species[i]].append(curr_gene_name)
                    gene_to_species[curr_gene_name].append(ecDNA_species[i])

        print("Simulating additional counts")

        # Sort out additional copies per ecDNA
        gene_to_species_mult = copy.deepcopy(gene_to_species)
        additional_counts_all = {}
        for species in species_to_gene.keys() :
            # Get counts of extras
            additional_counts = self.additional_count_func(len(species_to_gene[species]))
            additional_counts_all[species] = additional_counts
            # Iterate through, for all above zero, append that number of the species to the dict
            for i, add in enumerate(additional_counts) :
                if add > 0:
                    for copy_count in range(int(add)) :
                        gene_to_species_mult[species_to_gene[species][i]].append(species)

        # Match each gene to a vector of observed copy numbers
        gene_cell_true_cn_dict = {}
        for gene in gene_to_species_mult.keys() :
            # 2 for chromosomal copies
            gene_cell_true_cn_dict[gene] = np.full(self.num_cells, 2)
            for species in gene_to_species_mult[gene] :
                gene_cell_true_cn_dict[gene] += counts[species].values

        print("Adding noise")

        cbg_true_matrix = np.column_stack([gene_cell_true_cn_dict[g] for g in gene_cell_true_cn_dict.keys()])
        cbg_noisy_matrix = np.zeros_like(cbg_true_matrix, dtype=int)

        # Pick out of multinomial dist
        for cell in range(self.num_cells):
            row_sum = cbg_true_matrix[cell].sum()
            if row_sum != 0 :
                p = cbg_true_matrix[cell] / row_sum
                cbg_noisy_matrix[cell] = np.random.multinomial(row_sum * self.multinomial_mult, p)

        # add gaussian noise
        noise = np.random.normal(loc=0, scale=self.noise_scale, size=cbg_noisy_matrix.shape)
        cbg_noisy_matrix = cbg_noisy_matrix + noise
        cbg_noisy_matrix = np.maximum(cbg_noisy_matrix, 0)


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
            f.write(f"Seed:\t{self.random_seed}\n")
            f.write(f"Birth Param:\t{self.initial_birth_scale}\n")
            f.write(f"Death Param:\t{self.death_waiting_distribution_param}\n")
            f.write(f"Sim End Cells:\t{self.num_extant}\n")
            f.write(f"Cell Count:\t{self.num_cells}\n")
            f.write('\n')
            f.write(f"Initial Copy Numbers:\t")
            print(self.initial_copy_number_array, file = f)
            print("Fitness array:", file = f)
            print(self.fitness_array, file = f)
            f.write('\n')
            f.write(f"Cosegregation type:\t{self.cosegregation_type}\n")
            if self.cosegregation_type == "coefficient" :
                f.write(f"Cosegregation coefficient:\t{self.cosegregation}\n")
            elif self.cosegregation_type == 'venn' :
                f.write(f"Venn coefficients:\n")
                print(self.coeffs, file = f)
            else :
                f.write('Chance matrix:\n')
                print(self.mat, file = f)
                f.write('Species capacity:\t')
                print(self.capacity, file = f)
                f.write(f"Simulation runs per ecDNA:\t{self.sim_mult}\n")
            f.write('\n')

            f.write(f"Gene counts:\t{gene_counts_save}\n")
            f.write(f"Gene overlap:\n")
            print(gene_overlap_save, file = f)
            f.write(f"Additional copy chance:\t{self.chance_to_change}\n")
            f.write(f"Change distribution parameter:\t{self.change_distribution_param}\n")
            f.write('\n')

            f.write("--SIMULATED PARAMETERS--\n")
            for species, add in additional_counts_all.items() :
                print(f"Additional counts {species}:\t{add}", file = f)
            f.write('\n')

            for a, b in combinations(ecDNA_species, 2):
                correlation = scipy.stats.pearsonr(counts[a].values, counts[b].values)[0]
                f.write(f'Correlation between {a} and {b}: {correlation}\n')

        self.cbg_noisy_matrix = cbg_noisy_matrix
        self.cbg_true_matrix = cbg_true_matrix
        self.gene_counts = gene_counts_save
        self.gene_overlap = gene_overlap_save
        self.additional_counts_all = additional_counts_all