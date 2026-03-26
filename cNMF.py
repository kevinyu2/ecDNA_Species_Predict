from cnmf import cNMF
import numpy as np
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc

out_dir = '../sample_cNMF'
out_name = "low_depth_cNMF_run0"
cellbygene = "../low_depth/run_0_cellxgene.tsv"
metadata_file = "../low_depth/run_0_metadata.txt"

# Z score cutoff for inclusion of a gene in an ecDNA
score_cutoff = 0
# Number of NMFs to run
n_iter = 100

# Set if num is known, otherwise set as None, will try to figure out
num_ecDNA = None
# Numbers to check (if num_ecDNA is none)
counts_to_check = np.arange(1,5)
# parameter determining importance of error in choosing the best number of ecDNA
# stability - error_w * normalzied_error
error_w = 0.1

#################################################################
# Run cNMF
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
        median_spectra = pd.DataFrame(spectra.median(axis=0)).T
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

    min_score = min(k_df['prediction_error'])
    max_score = max(k_df['prediction_error'])

    k_df['normalized_prediction_error'] = (k_df['prediction_error'] - min_score) / (max_score - min_score)
    k_df['score'] = k_df['silhouette'] - error_w * k_df['normalized_prediction_error']

    print(k_df)
    num_ecDNA = int(k_df.loc[k_df['score'].idxmax()]['k'])
    print(f"Number of ecDNA chosen: {num_ecDNA}")


if num_ecDNA != 1 :
    cnmf_obj.consensus(k=num_ecDNA, density_threshold=0.01)
    usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=num_ecDNA, density_threshold=0.01,  norm_usage = False)
else :
    usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj_1.load_results(K=1, density_threshold=0.01, norm_usage = False)

cnmf_obj.consensus(k=num_ecDNA, density_threshold=0.01)
usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=num_ecDNA, density_threshold=0.01)

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
ecDNA_species = spectra_scores.columns
observed = {}
for species in ecDNA_species :
    observed[species] = []
for i, row in spectra_scores.iterrows() :
    for species in ecDNA_species :
        if row[species] > score_cutoff :
            observed[species].append(i)

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
spectra_consensus = pd.read_csv(f"{out_dir}/{out_name}/{out_name}.spectra.k_{num_ecDNA}.dt_0_01.consensus.txt", sep = '\t', index_col = 0)
spectra_consensus.index = "pred_ecDNA_" + spectra_consensus.index.astype(str)
spectra_consensus.index = spectra_consensus.index.map(mapping)

gene_counts = spectra_consensus.copy()
gene_counts[:] = 0
for species, gene_list in species_to_gene.items() :
    min_val = spectra_consensus.loc[species, species_to_gene[species]].min()
    for gene in gene_list :
        gene_counts.loc[species, gene] = np.round(spectra_consensus.loc[species, gene]/min_val)

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

