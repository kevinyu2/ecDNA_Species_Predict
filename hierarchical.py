########################################################################
# Runs hierarchical clustering on simulated data and generates statistics
# Usage: hierarchical.py [run dir (should have the run info in name)] [out dir (main folder, automatically generates specific name)]
########################################################################

import sys
from cnmf import cNMF
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
from scipy.cluster.hierarchy import linkage, fcluster


# If known, will do clusteirng with this
know_ecDNA = False
# If not known, will do clustering based off a distance threshold cutoff
threshold = 0.4

# directory with the data of the run
run_dir = Path(sys.argv[1])
# Full location of where we print things
out_dir = sys.argv[2]

if know_ecDNA :
    full_out_dir = f'{out_dir}/hier_countprov_1'
    full_result_dir = f'{out_dir}/hier_results_countprov_1'
else :
    full_out_dir = f'{out_dir}/hier_countprov_0_thresh_{threshold}'
    full_result_dir = f'{out_dir}/hier_results_countprov_0_thresh_{threshold}'

#################################################
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Run cNMF
# returns: (predicted species count, jaccard, average count err)
def hier_run(out_dir, out_name, cellbygene, cellbyspecies, metadata_file, threshold, num_ecDNA) :
    os.makedirs(f"{out_dir}/{out_name}/", exist_ok= True)
    cellxgene_df = pd.read_csv(cellbygene, sep = '\t', index_col= 0)
    X = cellxgene_df.T

    if num_ecDNA is not None :
        Z = linkage(X, method='average', metric='correlation')
        clusters = fcluster(Z, t=num_ecDNA, criterion='maxclust')
    else :
        Z = linkage(X, method='average', metric='correlation')
        clusters = fcluster(Z, t=threshold, criterion='distance')

    observed = defaultdict(list)
    for i in range(len(clusters)):
        observed[f"pred_ecDNA_{clusters[i]}"].append(cellxgene_df.columns[i])
    reversed_observed = defaultdict(list)

    for key, values in observed.items():
        for v in values:
            reversed_observed[v].append(key)
        gene_count = max(clusters)
                
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

        print(f"Best score: {avg_jaccard}")
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
        obs_species = reverse_mapping[species]
        if obs_species in list(observed.keys()) :
            # Just trust the extra counts of the smallest one and those 1.3 times at most above it (which does not have duplicates hopefully or is on multiple ecDNA)
            genes = observed[obs_species]
            gene_sums = cellbygene_temp[genes].sum()
            min_gene = gene_sums.idxmin()
            min_value = gene_sums.min()
            threshold = 1.3 * min_value
            genes_within_range = gene_sums[gene_sums <= threshold].index.tolist()
            subset = cellbygene_temp[genes_within_range]
            avg_list = subset.mean(axis=1).tolist()

            total_error += ((avg_list - cellxspecies_df[species])**2).sum()
            total_count += len(avg_list)
            plt.scatter(avg_list, cellxspecies_df[species], s = 1, alpha = 0.3, label = species)
    plt.xlabel(f"Usage")
    plt.ylabel(f"Count")
    plt.legend()
    plt.savefig(f"{out_dir}/{out_name}/{out_name}.usage_map.png")
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
        print(mapping, file = f)


    return gene_count, best_jaccard, avg_count_error


run_results_dir = f"{full_result_dir}/{run_dir.name}/"
os.makedirs(run_results_dir, exist_ok=True)

# save results in pandas tsv
run_predicted_species_counts_file = f"{run_results_dir}/species_counts.tsv"
run_jaccard_file = f"{run_results_dir}/jaccard.tsv"
run_count_err_file = f"{run_results_dir}/count_err.tsv"

run_predicted_species_counts_list = []
run_jaccard_list = []
run_count_err_list = []

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


    for cellbygene_path in Path(spec_dir).glob("*_cellxgene.tsv") :
        cellbygene = str(cellbygene_path)
        metadata_file = cellbygene.replace("cellxgene.tsv", "metadata.txt")
        cellbyspecies = cellbygene.replace("cellxgene.tsv", "cellxspecies.tsv")
        out_name = cellbygene_path.name.split("_cellxgene.tsv")[0]
        print("OUT NAME")
        print(out_name)

        # try :
        predicted_species_count, jaccard, count_err = hier_run(run_out_dir, out_name, cellbygene_path, cellbyspecies, metadata_file, threshold, num_ecDNA)
        # except Exception as e :
        #     print(f"Error: {e}")
        #     predicted_species_count, jaccard, count_err = 0,0,0
        run_predicted_species_counts[out_name] = predicted_species_count
        run_jaccard[out_name] = jaccard
        run_count_err[out_name] = count_err

        print("UPDATES")
        print(run_jaccard)
        print(run_predicted_species_counts)
        print(run_count_err)

    run_predicted_species_counts_list.append(run_predicted_species_counts)
    run_jaccard_list.append(run_jaccard)
    run_count_err_list.append(run_count_err)

(pd.DataFrame(run_predicted_species_counts_list)).to_csv(run_predicted_species_counts_file, index = None, sep = '\t')
(pd.DataFrame(run_jaccard_list)).to_csv(run_jaccard_file, index = None, sep = '\t')
(pd.DataFrame(run_count_err_list)).to_csv(run_count_err_file, index = None, sep = '\t')




