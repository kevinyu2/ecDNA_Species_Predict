########################################################################
# Runs random baseline. Basically, given k, just splits the genes randomly. Still picks the best possible jaccard!
# Usage: random_baseline.py [run dir (should have the run info in name)] [out dir (main folder, automatically generates specific name)]
########################################################################


import sys
import numpy as np
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import argparse
import time

##################################################################
# Master controls
##################################################################

parser = argparse.ArgumentParser(
    description="Pipeline for creating many simulations on cass",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)


parser.add_argument(
    "run_dir",
    type=str,
    help="Main input dir"
)

parser.add_argument(
    "out_dir",
    type=str,
    help="Main output dir"
)

parser.add_argument(
    "--give-genes",
    action="store_true",
    help="Know the genes"
)

parser.add_argument(
    "--expected-ecDNA",
    type=float,
    default= 1.0,
    help="Expected number of ecDNA species each gene is in"
)




args = parser.parse_args()

#################################################################

# If known, will do clusteirng with this
know_ecDNA = True

# directory with the data of the run
run_dir = Path(args.run_dir)
# Full location of where we print things
out_dir = args.out_dir

expectation = args.expected_ecDNA

gene_prov = args.give_genes


full_out_dir = f'{out_dir}/random_geneprov_{int(gene_prov)}_exp_{expectation}'
full_result_dir = f'{out_dir}/random_results_geneprov_{int(gene_prov)}_exp_{expectation}'



#################################################
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Run hier naive
# returns: (predicted species count, jaccard, average count err)
def random_run(out_dir, out_name, cellbygene, cellbyspecies, expectation, metadata_file, num_ecDNA, gene_prov) :
    os.makedirs(f"{out_dir}/{out_name}/", exist_ok= True)
    cellxgene_df = pd.read_csv(cellbygene, sep = '\t', index_col= 0)
    X = cellxgene_df.T
                
    # Parse metadata
    gt = defaultdict(list)
    with open(metadata_file, "r") as f:
        for line in f:
            line = line.strip()

            # stop here
            if line.startswith("--SIMULATION PARAMETERS--"):
                break

            if not line or line.startswith("--"):
                continue

            gene, species = line.split(":\t")
            for p in species.split("\t"):
                gt[p].append(gene)


    if not gene_prov :

        observed = defaultdict(list)
        # Go until each ecDNA has at least one gene
        while len(observed) < num_ecDNA :
            observed = defaultdict(list)

            # Each gene has a 1/num_ecDNA probability of going into each! (This way we can also simulate overlap)
            for gene in cellxgene_df.columns :
                for ecDNA_no in range(num_ecDNA) :
                    test = random.random()
                    if test < (1/num_ecDNA) * expectation:
                        observed[f"pred_ecDNA_{ecDNA_no + 1}"].append(gene)
    
    else :
        observed = gt

    reversed_observed = defaultdict(list)

    for key, values in observed.items():
        for v in values:
            reversed_observed[v].append(key)

    # Find out which predicted ecDNA matches to which gt ecDNA
    # Uses hungarian algorithm, with distances defined by jaccard index between gene sets
    def match_score(obs, gt):

        keys1 = list(obs.keys())
        keys2 = list(gt.keys())

        n1 = len(keys1)
        n2 = len(keys2)

        n = max(n1, n2)
        cost_matrix = np.ones((n, n))

        # fill real costs
        for i, k1 in enumerate(keys1):
            s1 = set(obs[k1])
            for j, k2 in enumerate(keys2):
                s2 = set(gt[k2])
                jaccard = len(s1 & s2) / len(s1 | s2)
                cost_matrix[i, j] = 1 - jaccard

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        mapping = {}
        new_counter = 1

        for i, j in zip(row_ind, col_ind):
            # Ignore dummy rows
            if i >= n1:
                continue

            k1 = keys1[i]
            if j < n2:
                mapping[k1] = keys2[j]
            else:
                mapping[k1] = f"NEW_ecDNA_{new_counter}"
                new_counter += 1

        # compute average jaccard only for real matches
        scores = []
        for k1, k2 in mapping.items():
            if k2.startswith("NEW_ecDNA"):
                scores.append(0)
            else:
                s1 = set(obs[k1])
                s2 = set(gt[k2])
                scores.append(len(s1 & s2) / len(s1 | s2))

        avg_jaccard = np.mean(scores)

        return mapping, avg_jaccard

    mapping, best_jaccard = match_score(observed, gt)
    reverse_mapping = {value: key for key, value in mapping.items()}

    cellxspecies_df = pd.read_csv(cellbyspecies, sep = '\t', index_col = 0)

    # When calculating usage do subtract 2
    cellbygene_temp = cellxgene_df - 2
    cellbygene_temp = cellbygene_temp.clip(lower=0)

    total_error = 0
    total_count = 0

    plt.figure()
    for species in list(cellxspecies_df.columns) :
        if species in reverse_mapping.keys() :
            random_vals = cellbygene_temp.apply(
                lambda row: np.random.choice(row.values),
                axis=1
            ).values

            total_error += ((random_vals - cellxspecies_df[species].values) ** 2).sum()
            total_count += len(random_vals)

            plt.scatter(
                random_vals,
                cellxspecies_df[species].values,
                s=1,
                alpha=0.3,
                label=species
            )
            # obs_species = reverse_mapping[species]
            # if obs_species in list(observed.keys()) :
            #     # Just trust the extra counts of the smallest one and those 1.3 times at most above it (which does not have duplicates hopefully or is on multiple ecDNA)
            #     genes = observed[obs_species]
            #     gene_sums = cellbygene_temp[genes].sum()
            #     min_gene = gene_sums.idxmin()
            #     min_value = gene_sums.min()
            #     threshold = 1.3 * min_value
            #     genes_within_range = gene_sums[gene_sums <= threshold].index.tolist()
            #     subset = cellbygene_temp[genes_within_range]
            #     avg_list = subset.mean(axis=1).values

            #     # Count the total error
            #     total_error += ((avg_list - cellxspecies_df[species].values)**2).sum()
            #     total_count += len(avg_list)
            #     plt.scatter(avg_list, cellxspecies_df[species], s = 1, alpha = 0.3, label = species)
    plt.xlabel(f"Usage")
    plt.ylabel(f"Count")
    plt.legend()
    plt.savefig(f"{out_dir}/{out_name}/{out_name}.usage_map.png")
    plt.close()

    avg_count_error = total_error / total_count


    # Make a predictions file
    with open(f"{out_dir}/{out_name}/{out_name}.predictions.txt", 'w') as f:
        f.write("--PREDICTED--\n")
        for key in reversed_observed.keys() :
            f.write(f"{key}:")
            for val in reversed_observed[key] :
                f.write(f"\t{val}")
            f.write('\n')
        f.write('\n--SIMULATION PARAMETERS--\n')
        f.write(f'Number of predicted species:\t{len(mapping.keys())}\tTrue species number:\t{len(gt.keys())}\n')
        if num_ecDNA is None :
            f.write(f'Dist cutoff:\t{threshold}\n')
        f.write(f'Best jaccard (species wise):\t{best_jaccard}\n')
        f.write(f"Mapping:\n")


    return num_ecDNA, best_jaccard, avg_count_error


run_results_dir = f"{full_result_dir}/{run_dir.name}/"
os.makedirs(run_results_dir, exist_ok=True)

# save results in pandas tsv
run_predicted_species_counts_file = f"{run_results_dir}/species_counts.tsv"
run_jaccard_file = f"{run_results_dir}/jaccard.tsv"
run_count_err_file = f"{run_results_dir}/count_err.tsv"
run_time_file = f"{run_results_dir}/runtime.tsv"

run_predicted_species_counts_list = []
run_jaccard_list = []
run_count_err_list = []
run_time_list = []

for spec_dir in Path(run_dir).glob("*"):
    # Should be in the file names
    num_ecDNA_true = int(spec_dir.name.split('_')[0])
    comb_chance = float(spec_dir.name.split('_')[2])
    print("SPEC DIR")
    print(spec_dir)


    num_ecDNA = None
    if know_ecDNA :
        num_ecDNA = num_ecDNA_true

    run_out_dir = f"{full_out_dir}/{run_dir.name}/{spec_dir.name}/"
    os.makedirs(run_out_dir, exist_ok=True)

    # Temporary dicts to turn into dataframe
    run_predicted_species_counts = {"num_ecDNA_true" : num_ecDNA_true, "comb_chance" : comb_chance}
    run_jaccard = {"num_ecDNA_true" : num_ecDNA_true, "comb_chance" : comb_chance}
    run_count_err = {"num_ecDNA_true" : num_ecDNA_true, "comb_chance" : comb_chance}
    run_time = {"num_ecDNA_true" : num_ecDNA_true, "comb_chance" : comb_chance}



    for cellbygene_path in Path(spec_dir).glob("*_cellxgene.tsv") :
        cellbygene = str(cellbygene_path)
        metadata_file = cellbygene.replace("cellxgene.tsv", "metadata.txt")
        cellbyspecies = cellbygene.replace("cellxgene.tsv", "cellxspecies.tsv")
        out_name = cellbygene_path.name.split("_cellxgene.tsv")[0]

        # Start time before function call
        start = time.time()

        # try :
        predicted_species_count, jaccard, count_err = random_run(run_out_dir, out_name, cellbygene_path, cellbyspecies, expectation, metadata_file, num_ecDNA, gene_prov)
        # except Exception as e :
        #     print(f"Error: {e}")
        #     predicted_species_count, jaccard, count_err = 0,0,0
        run_predicted_species_counts[out_name] = predicted_species_count
        run_jaccard[out_name] = jaccard
        run_count_err[out_name] = count_err
        run_time[out_name] = time.time() - start

    run_predicted_species_counts_list.append(run_predicted_species_counts)
    run_jaccard_list.append(run_jaccard)
    run_count_err_list.append(run_count_err)
    run_time_list.append(run_time)

(pd.DataFrame(run_predicted_species_counts_list)).to_csv(run_predicted_species_counts_file, index = None, sep = '\t')
(pd.DataFrame(run_jaccard_list)).to_csv(run_jaccard_file, index = None, sep = '\t')
(pd.DataFrame(run_count_err_list)).to_csv(run_count_err_file, index = None, sep = '\t')
(pd.DataFrame(run_time_list)).to_csv(run_time_file, index = None, sep = '\t')


