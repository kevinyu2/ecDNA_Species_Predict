from cnmf import cNMF
import numpy as np
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import pandas as pd

out_dir = '../sample_cNMF'
out_name = "default_cNMF"
cellbygene = "../sample_data/run_3_cellxgene.tsv"
metadata_file = "../sample_data/run_3_metadata.txt"

# Z score cutoff for inclusion of a gene in an ecDNA
score_cutoff = 0
# Number of NMFs to run
n_iter = 100

# Set if num is known, otherwise set as None, will try to figure out
num_ecDNA = None
# parameter determining importance of error in choosing the best number of ecDNA
error_weight = 1
# stability - error_w * normalzied_error
error_w = 2

#################################################################

# Run cNMF
cnmf_obj = cNMF(output_dir=out_dir, name=out_name)
cnmf_obj.prepare(counts_fn=cellbygene, components=np.arange(2,5), n_iter=n_iter, seed=random.randint(1,1000))
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

    min_score = min(k_df['prediction_error'])
    max_score = max(k_df['prediction_error'])

    k_df['normalized_prediction_error'] = (k_df['prediction_error'] - min_score) / (max_score - min_score)
    k_df['score'] = k_df['silhouette'] - error_weight * k_df['normalized_prediction_error']

    num_ecDNA = int(k_df.loc[k_df['score'].idxmax()]['k'])
    print(f"Number of ecDNA chosen: {num_ecDNA}")


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
# TODO: hungarian algorithm when the number is wrong?
def match_score(obs, gt) :

    keys1 = list(obs.keys())
    keys2 = list(gt.keys())

    n = len(keys1)
    cost_matrix = np.zeros((n, n))

    for i, k1 in enumerate(keys1):
        s1 = set(obs[k1])
        for j, k2 in enumerate(keys2):
            s2 = set(gt[k2])
            jaccard = len(s1 & s2) / len(s1 | s2)
            cost_matrix[i, j] = 1 - jaccard

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # mapping from dict1 keys to dict2 keys
    mapping = {keys1[i]: keys2[j] for i, j in zip(row_ind, col_ind)}

    avg_jaccard = np.mean([
        len(set(obs[k1]) & set(gt[k2])) / len(set(obs[k1]) | set(gt[k2]))
        for k1, k2 in mapping.items()
    ])
    print(f"Best score: {avg_jaccard}")
    return mapping, avg_jaccard

mapping, best_jaccard = match_score(observed, gt)

matched_observed = defaultdict(list)
for i, row in spectra_scores.iterrows() :
    for species in ecDNA_species :
        if row[species] > score_cutoff :
            matched_observed[i].append(mapping[species])

with open(f"{out_dir}/{out_name}/{out_name}.predictions.txt", 'w') as f:
    f.write("--PREDICTED--\n")
    for key in spectra_scores.index :
        f.write(f"{key}:")
        for val in matched_observed[key] :
            f.write(f"\t{val}")
        f.write('\n')
    f.write('\n--SIMULATION PARAMETERS--\n')
    f.write(f'Spectra score cutoff:\t{score_cutoff}\n')
    f.write(f'Number of iterations:\t{n_iter}\n')
    f.write(f'Best species count:\t{num_ecDNA}\n')
    f.write(f'Best jaccard (species wise):\t{best_jaccard}\n')
    # TODO: gene wise jaccard?