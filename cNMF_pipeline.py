########################################################################
# Runs cNMF on simulated data and generates statistics
# Usage: cNMF_pipeline.py [run dir (should have the run info in name)] [out dir (main folder, automatically generates specific name)]
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
import argparse

###############################################################

parser = argparse.ArgumentParser(
    description="Pipeline for testing many cNMF runs",
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
    "--iter",
    type=int,
    default = 50,
    help="Number of iterations to run cNMF"
)

parser.add_argument(
    "--know-ecDNA",
    action="store_true",
    help="Know number of species (doesn't calculate)"
)



parser.add_argument(
    "--max-species",
    type = int,
    default = 6,
    help="Max number of species to check"
)


parser.add_argument(
    "--errorw",
    type = float,
    default = 0.1,
    help="Weight of error against stability"
)

parser.add_argument(
    "--density-threshold",
    type = float,
    default = 0.1
)


args = parser.parse_args()

###############################################################

# Z score cutoff for inclusion of a gene in an ecDNA
score_cutoff = 0
# Number of NMFs to run
n_iter = args.iter

# True means we provide to the program the exact number of ecDNA
know_ecDNA = args.know_ecDNA
# Numbers to check (if num_ecDNA is none)
counts_to_check = np.arange(1,1+args.max_species)
# parameter determining importance of error in choosing the best number of ecDNA
# stability - error_w * normalzied_error
error_w = args.errorw

density_threshold = args.density_threshold

# directory with the data of the run
run_dir = Path(args.run_dir)
# Full location of where we print things
out_dir = args.out_dir

full_out_dir = f'{out_dir}/cNMF_countprov_{int(know_ecDNA)}_errorw_{error_w}'
full_result_dir = f'{out_dir}/cNMF_results_countprov_{int(know_ecDNA)}_errorw_{error_w}'


#################################################
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Run cNMF
# returns: (predicted species count, jaccard, average correlation)
def cNMF_run(out_dir, out_name, cellbygene, cellbyspecies, metadata_file, score_cutoff, n_iter, num_ecDNA, counts_to_check, error_w, density_threshold) :
    os.makedirs(out_dir, exist_ok=True)
    cnmf_obj = cNMF(output_dir=out_dir, name=out_name)
    check_one = False
    if num_ecDNA is None :
        if 1 in counts_to_check :
            cnmf_obj.prepare(counts_fn=cellbygene, tpm_fn = cellbygene, components=counts_to_check[1:], n_iter=n_iter, seed=random.randint(1,1000))
            check_one = True

        else :
            cnmf_obj.prepare(counts_fn=cellbygene, tpm_fn = cellbygene, components=counts_to_check, n_iter=n_iter, seed=random.randint(1,1000))

    else :
        cnmf_obj.prepare(counts_fn=cellbygene, components=num_ecDNA, n_iter=n_iter, seed=random.randint(1,1000))


    input_counts = pd.read_csv(cellbygene, sep = '\t', index_col = 0)
    adata = ad.AnnData(input_counts)
    cnmf_obj.save_norm_counts(adata)
    cnmf_obj.factorize(worker_i=0, total_workers=1)
    cnmf_obj.combine()
    if num_ecDNA is None :
        cnmf_obj.k_selection_plot()

    # Find best number of ecDNA using stability and error
    if num_ecDNA is None :
        npz = np.load(f"{out_dir}/{out_name}/{out_name}.k_selection_stats.df.npz", allow_pickle=True)

        k_df = pd.DataFrame(
            data=npz["data"],
            index=npz["index"],
            columns=npz["columns"]
        )

        # Include 1 (stability always at 1)
        if check_one :
            cnmf_obj_1 = cNMF(output_dir=out_dir, name=out_name)

            cnmf_obj_1.prepare(counts_fn=cellbygene, tpm_fn = cellbygene, components=1, n_iter=n_iter, seed=random.randint(1,1000))
            cnmf_obj_1.save_norm_counts(adata)
            cnmf_obj_1.factorize(worker_i=0, total_workers=1)
            cnmf_obj_1.combine()

            
            norm_counts = sc.read(cnmf_obj_1.paths['normalized_counts'])

            with np.load(cnmf_obj_1.paths['merged_spectra']%1, allow_pickle=True) as f:
                obj = pd.DataFrame(**f)
                spectra = obj
            l2_spectra = (spectra.T / np.sqrt((spectra**2).sum(axis=1))).T
            median_spectra = pd.DataFrame(l2_spectra.median(axis=0)).T
            median_spectra = (median_spectra.T / median_spectra.sum(1)).T
            rf_usages = cnmf_obj_1.refit_usage(norm_counts.X, median_spectra)
            rf_usages = pd.DataFrame(rf_usages, index=norm_counts.obs.index)
            rf_pred = rf_usages.dot(median_spectra)

            if sp.issparse(norm_counts.X):
                prediction_error = ((norm_counts.X.todense() - rf_pred) ** 2).sum().sum()
            else:
                prediction_error = ((norm_counts.X - rf_pred) ** 2).sum().sum()

            new_row = pd.DataFrame([{'k' : 1, 'local_density_threshold' : 0.5, "silhouette" : 1, "prediction_error" : prediction_error}])
            k_df = pd.concat([k_df, new_row], ignore_index=True)

        max_score = max(k_df['prediction_error'])

        k_df['normalized_prediction_error'] = (k_df['prediction_error']) / (max_score)
        k_df['score'] = k_df['silhouette'] - error_w * k_df['normalized_prediction_error']

        print(k_df)
        num_ecDNA = int(k_df.loc[k_df['score'].idxmax()]['k'])
        print(f"Number of ecDNA chosen: {num_ecDNA}")
    else :
        num_ecDNA = num_ecDNA[0]
        if num_ecDNA == 1 :
            cnmf_obj_1 = cnmf_obj


    if num_ecDNA != 1 :
        cnmf_obj.consensus(k=num_ecDNA, density_threshold=density_threshold)
        usage_df, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=num_ecDNA, density_threshold=density_threshold,  norm_usage = False)
    else :
        cnmf_obj_1.consensus(k=num_ecDNA, density_threshold=density_threshold)
        usage_df, spectra_scores, spectra_tpm, top_genes = cnmf_obj_1.load_results(K=1, density_threshold=density_threshold, norm_usage = False)

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

    # Parse cNMF spectra scores
    spectra_scores.columns = [f"pred_ecDNA_{col}" for col in spectra_scores.columns]
    spectra_tpm.columns = [f"pred_ecDNA_{col}" for col in spectra_tpm.columns]

    ecDNA_species = spectra_scores.columns
    observed = {}
    for species in ecDNA_species :
        observed[species] = []
    for i, row in spectra_scores.iterrows() :
        for species in ecDNA_species :
            if row[species] > score_cutoff :
                observed[species].append(i)

    # Find how to rescale usage in terms of tpm
    # Assumes the lowest of the used genes is 1, and takes the average of the lowest and those 1.2 times away (incase there are slight deviations)
    usage_scale = {}
    for species in ecDNA_species :
        usage_scale[species] = 1
        obs_genes = observed[species]
        if len(obs_genes) == 0:
            continue

        tpm_values = spectra_tpm.loc[obs_genes, species].values
        min_tpm = np.min(tpm_values)
        near_min = tpm_values[tpm_values <= min_tpm * 1.2]
        usage_scale[species] = np.mean(near_min)


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

    matched_observed = defaultdict(list)
    for i, row in spectra_scores.iterrows() :
        for species in ecDNA_species :
            if row[species] > score_cutoff :
                matched_observed[i].append(mapping[species])

    species_to_gene = defaultdict(list)
    for key, val in matched_observed.items() :
        for species in val :
            species_to_gene[species].append(key)

    # Calculate extra genes from spectra
    spectra_consensus = pd.read_csv(f"{out_dir}/{out_name}/{out_name}.spectra.k_{num_ecDNA}.dt_{str(density_threshold).replace('.', '_')}.consensus.txt", sep = '\t', index_col = 0)
    spectra_consensus.index = "pred_ecDNA_" + spectra_consensus.index.astype(str)
    spectra_consensus.index = spectra_consensus.index.map(mapping)

    gene_counts = spectra_consensus.copy()
    gene_counts[:] = 0
    for species, gene_list in species_to_gene.items() :
        min_val = spectra_consensus.loc[species, species_to_gene[species]].min()
        for gene in gene_list :
            gene_counts.loc[species, gene] = np.round(spectra_consensus.loc[species, gene]/min_val)

    # Get usage correlations
    usage_df.columns = [f'pred_ecDNA_{i + 1}' for i in range(len(usage_df.columns))]
    for species in ecDNA_species :
        usage_df[species] *= usage_scale[species]

    cellxspecies_df = pd.read_csv(cellbyspecies, sep = '\t', index_col = 0)
    usage_df.rename(columns=mapping, inplace=True)
    total_error = 0
    total_count = 0

    plt.figure()
    for species in list(cellxspecies_df.columns) :
        if species in list(usage_df.columns) :
            total_error += ((usage_df[species] - cellxspecies_df[species])**2).sum()
            total_count += len(usage_df[species])
            plt.scatter(usage_df[species], cellxspecies_df[species], s = 1, alpha = 0.3, label = species)
    plt.xlabel(f"Usage")
    plt.ylabel(f"Count")
    plt.legend()
    plt.savefig(f"{out_dir}/{out_name}/{out_name}.usage_map.png")
    avg_count_error = total_error / total_count

    # Make a predictions file
    with open(f"{out_dir}/{out_name}/{out_name}.predictions.txt", 'w') as f:
        f.write("--PREDICTED--\n")
        for key in spectra_scores.index :
            f.write(f"{key}:")
            for val in matched_observed[key] :
                f.write(f"\t{val}")
            f.write('\n')
        f.write('\n--SIMULATION PARAMETERS--\n')
        f.write(f'Number of predicted species:\t{len(mapping)}\tTrue species number:\t{len(gt.keys())}\n')
        f.write(f'Spectra score cutoff:\t{score_cutoff}\n')
        f.write(f'Number of iterations:\t{n_iter}\n')
        f.write(f'Best species count:\t{num_ecDNA}\n')
        f.write(f'Best jaccard (species wise):\t{best_jaccard}\n')
        f.write(f"Mapping:\n")
        print(mapping, file = f)
        f.write(f"Extra counts:\n")
        for i, row in gene_counts.iterrows() :
            f.write(f"{i}:")
            for gene in gene_counts.columns :
                if row[gene] > 1 :
                    f.write(f"\t{gene}:\t{row[gene]-1}")
            f.write('\n')

    shutil.rmtree(f"{out_dir}/{out_name}/cnmf_tmp")

    return len(usage_df.columns), best_jaccard, avg_count_error


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
        num_ecDNA = [num_ecDNA_true]

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

        # Subtract 2 for the chromosomal copies
        cellbygene_temp = pd.read_csv(cellbygene_path, sep = '\t', index_col = 0)
        cellbygene_temp = cellbygene_temp - 2
        cellbygene_temp = cellbygene_temp.clip(lower=0)
        cellbygene_temp_path = cellbygene.replace("cellxgene.tsv", "cellxgene_temp.tsv")
        cellbygene_temp.to_csv(cellbygene_temp_path, sep = '\t')

        try :
            predicted_species_count, jaccard, avg_count_error = cNMF_run(run_out_dir, out_name, cellbygene_temp_path, cellbyspecies, metadata_file, score_cutoff, n_iter, num_ecDNA, counts_to_check, error_w, density_threshold)
        except Exception as e :
            print(f"Error: {e}")
            predicted_species_count, jaccard, avg_count_error = 0,0,0
        run_predicted_species_counts[out_name] = predicted_species_count
        run_jaccard[out_name] = jaccard
        run_count_err[out_name] = avg_count_error

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




