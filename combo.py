########################################################################
# Runs hierarchical clustering on simulated data and generates statistics
# Usage: hierarchical.py [run dir (should have the run info in name)] [out dir (main folder, automatically generates specific name)]
########################################################################

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
import argparse
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import time
import heapq
from scipy.optimize import nnls


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
    "--know-ecDNA",
    action="store_true",
    help="Know number of species (doesn't calculate)"
)

parser.add_argument(
    "--dummy-distance",
    type = float,
    default = 1.3,
    help="Distance to set the dummy nodes from the rest (greater results in more likely calling num species = 1)"
) 

parser.add_argument(
    "--cNMF-threshold",
    type = float,
    default = 0.55,
    help="NNLS per cell distance threshold to choose to use cNMF"
) 

parser.add_argument(
    "--max-species",
    type = int,
    default = None,
    help="Maximum species to check if species count not known. If none, will be based on the "
) 



args = parser.parse_args()

#################################################################

# If known, will do clusteirng with this
know_ecDNA = args.know_ecDNA
# If not known, will do clustering based off a distance threshold cutoff
dummydist = args.dummy_distance

cNMF_thresh = args.cNMF_threshold

# directory with the data of the run
run_dir = Path(args.run_dir)
# fmax_0.1_overlap_0.4_extracounts_0.1_depth_1.0
if "overlap_" in args.run_dir :
    overlap = float(args.run_dir.split('overlap_')[-1].split('_')[0])
else :
    overlap = 0

# Full location of where we print things
out_dir = args.out_dir

if know_ecDNA :
    full_out_dir = f'{out_dir}/combo_countprov_1_thresh_0'
    full_result_dir = f'{out_dir}/combo_results_countprov_1_thresh_0'
else :
    full_out_dir = f'{out_dir}/combo_countprov_0_thresh_{cNMF_thresh}'
    full_result_dir = f'{out_dir}/combo_results_countprov_0_thresh_{cNMF_thresh}'

#################################################
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def silhouette_with_extra(X, labels, ddist, metric="correlation", eps=1e-5):
    n = X.shape[0]

    # Square the distances to slighly reduce the effect of double-correlation
    X =  np.sign(X) * (X ** 2)

    # Add new dummy node to help with the correlations if everything is correlated
    X_new = np.zeros((n + 1, n + 1))
    X_new[:n, :n] = X
    X_new[n, n] = 1


    D = pairwise_distances(X_new, metric=metric)

    # Reset the distance so it doesn't overblow the results for k = 1
    D += eps
    D[n, :] = ddist
    D[:, n] = ddist

    np.fill_diagonal(D, 0)

    # Add extra label for the dummy node
    labels = np.append(labels, max(labels) + 1)

    # pass it as precomputed
    silhouette = silhouette_score(D, labels, metric="precomputed")

    return silhouette


# Calculate the best number of ecDNA species to try for hierarchical clustering
# Uses hierarchical clustering, except if nothing passes some threshold, use 1
# Returns the number of ecDNA
def find_num_ecDNA(X, max_species, ddist, leeway = 0) :
    best_num = -1
    best_silhouette = -1
    silhouettes = []

    if max_species is None :
        max_species, _ = X.shape
        max_species = max_species-1

    nums_to_try = [i for i in range(1, max_species + 1)]


    for i in nums_to_try :
        Z = linkage(X, method='average', metric='euclidean')
        if i == 1 :
            clusters = np.full((X.shape[0]), 1)
        else :
            clusters = fcluster(Z, t=i, criterion='maxclust')

        silhouette = silhouette_with_extra(X, clusters, ddist, metric = 'euclidean')
        silhouettes.append(silhouette)
        if silhouette > best_silhouette :
            best_silhouette = silhouette

    
    # Allow some leeway around the silhouette score, to favor greater values wiht just slightly worse silhouette scores
    for idx, s in enumerate(silhouettes) :
        if s >= best_silhouette - leeway :
            best_num = nums_to_try[idx]

    print(f"Predicted Species Count: {best_num}")
    return best_num

def check_overlap(X, clusters, cNMF_thresh) :
    X = X - 2
    cluster_no = len(set(clusters))
    
    # Only applicable if more than 3 clusters, so just return something that will return false
    if cluster_no < 3 :
        return False

    # pairwise_errors = []

    # # Find how to normalize the errors
    # for cluster_i in range(1, cluster_no + 1):

    #     members = [
    #         X.iloc[:, cc].values
    #         for cc, k in enumerate(clusters)
    #         if k == cluster_i
    #     ]

    #     if len(members) < 2:
    #         continue

    #     for i in range(len(members)):
    #         for j in range(i + 1, len(members)):

    #             diff = members[i] - members[j]

    #             # keep only coordinates where members[i] < 1
    #             mask = members[i] < 0.5

    #             pairwise_errors.append(diff[mask])

    # # concatenate all selected differences into one vector
    # if len(pairwise_errors) > 0:
    #     pairwise_error_rmse = np.mean(np.concatenate(pairwise_errors))
    # else:
    #     pairwise_error_rmse = 0.0

    # print(f"Pairwise: {pairwise_error_rmse}")

    # The representative vector is just a sum of the vectors from each cluster
    representative_vectors = np.zeros((cluster_no, len(X)))

    gene_counts = np.zeros(cluster_no)

    # Add each vector to the representative vector
    for cluster_i in range(1, cluster_no + 1) :
        for cc, k in enumerate(clusters) :
            if k == cluster_i :
                representative_vectors[k-1] += X.iloc[:, cc].values
                gene_counts[k- 1] += 1

    rng = np.random.default_rng(42)


    cluster_means = representative_vectors / gene_counts[:, None]
 
    errors = []

    for i in range(representative_vectors.shape[0]):
        rel_test_error = 0
        for repeat in range(20):
            target = cluster_means[i]
            others = np.delete(cluster_means, i, axis=0)

            # Only use coordinates where average gene count is > 0.5
            valid_idx = np.where((target > 3))[0]

            if len(valid_idx) < 2:
                continue  # not enough points to split

            train_idx = rng.choice(
                valid_idx,
                size=max(1, len(valid_idx) // 2),
                replace=False
            )

            test_mask = np.isin(valid_idx, train_idx, invert=True)
            test_idx = valid_idx[test_mask]

            if len(test_idx) == 0:
                continue

            # Fit on train coordinates only
            coeffs, *_ = np.linalg.lstsq(
                others[:, train_idx].T,
                target[train_idx],
                rcond=None
            )


            # Predict full vector
            prediction = coeffs @ others

            # Evaluate on held-out coordinates
            rel_test_error += (
                np.linalg.norm(target[test_idx] - prediction[test_idx])
                / np.sqrt(len(test_idx))
            )

            rel_test_error /= 20

        errors.append(rel_test_error)
        print(rel_test_error)

    lowest_three = heapq.nsmallest(3, errors)

    return all(x < cNMF_thresh for x in lowest_three)
        

# Run cNMF
# returns: (predicted species count, jaccard, average count err)
def combo_run(out_dir, out_name, cellbygene, cellbyspecies, metadata_file, num_ecDNA, max_species, ddist, cNMF_thresh) :
    os.makedirs(f"{out_dir}/{out_name}/", exist_ok= True)

    print(f"Number of ecDNA (True) : {num_ecDNA_true}")
    cellxgene_df = pd.read_csv(cellbygene, sep = '\t', index_col= 0)
    X = cellxgene_df
    embed = np.corrcoef(X, rowvar=False)

    if num_ecDNA is None :
        num_ecDNA = find_num_ecDNA(embed, max_species, ddist)
        

    Z = linkage(embed, method='average', metric='correlation')
    clusters = fcluster(Z, t=num_ecDNA, criterion='maxclust')

    # Continue with hierarchical
    if not check_overlap(X, clusters, cNMF_thresh) :
        return hier_run(clusters, cellxgene_df, out_dir, out_name, cellbyspecies, metadata_file, num_ecDNA, max_species)
    
    else :
        print("Predicted Overlap")
        # TODO: actually run cNMF!
        # Here, I will estimate by just pulling it out of the cNMF results (so I don't have to rerun this right this moment. Want to make sure everything else is good first)
        
        # num_ecDNA, best_jaccard, avg_count_error

        return -1, -1, -1
    


def hier_run(clusters, cellxgene_df, out_dir, out_name, cellbyspecies, metadata_file, num_ecDNA, max_species) :
    
    observed = defaultdict(list)
    for i in range(len(clusters)):
        observed[f"pred_ecDNA_{clusters[i]}"].append(cellxgene_df.columns[i])
    reversed_observed = defaultdict(list)

    for key, values in observed.items():
        for v in values:
            reversed_observed[v].append(key)

    # TEMPORARY: this one tracks how many distinct ecDNA profiles there are within genes
    unique_gene_sets = set()
                
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
            unique_gene_sets.add(species)
            for p in species.split("\t"):
                gt[p].append(gene)

    print(f"Number of unique gene - ecDNA interactions: {len(unique_gene_sets)}")

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

    print(reverse_mapping)

    # NNLS setup: create species by gene matrix (assume no overlaps at this point)
    species_profiles = {}

    for pred_species, genes in observed.items():

        gene_sums = cellbygene_temp[genes].sum()

        min_value = gene_sums.min()
        threshold = 1.2 * min_value

        # These give us the average to divide by to get genes per species
        baseline_genes = gene_sums[gene_sums <= threshold]
        baseline_mean = baseline_genes.mean()

        profile = pd.Series(
            0.0,
            index=cellbygene_temp.columns,
            dtype=float
        )

        for gene in genes:

            profile[gene] = gene_sums[gene] / baseline_mean

        species_profiles[pred_species] = profile

    # gene by species matrix
    gene_by_species = pd.DataFrame(species_profiles)
    A = gene_by_species.values

    pred_species_usage = []

    for cell in cellbygene_temp.index:

        b = cellbygene_temp.loc[cell].values

        x, residual = nnls(A, b)

        pred_species_usage.append(x)

    pred_species_df = pd.DataFrame(
        pred_species_usage,
        index=cellbygene_temp.index,
        columns=gene_by_species.columns
    )

    # print(pred_species_df.head())
    # print(cellxspecies_df.head())

    pred_species_df.rename(columns=mapping, inplace=True)
    total_error = 0
    total_count = 0


    plt.figure()
    for species in list(cellxspecies_df.columns) :
        if species in list(pred_species_df.columns) :
            total_error += ((pred_species_df[species] - cellxspecies_df[species])**2).sum()
            total_count += len(pred_species_df[species])
            plt.scatter(pred_species_df[species], cellxspecies_df[species], s = 1, alpha = 0.3, label = species)

    plt.xlabel(f"Usage")
    plt.ylabel(f"Count")
    plt.legend()
    plt.savefig(f"{out_dir}/{out_name}/{out_name}.usage_map.png")
    avg_count_error = total_error / total_count

    # plt.figure()
    # for species in list(cellxspecies_df.columns) :
    #     if species in reverse_mapping.keys() :
    #         obs_species = reverse_mapping[species]
    #         if obs_species in list(observed.keys()) :
    #             # Just trust the extra counts of the smallest one and those 1.3 times at most above it (which does not have duplicates hopefully or is on multiple ecDNA)
    #             genes = observed[obs_species]
    #             gene_sums = cellbygene_temp[genes].sum()
    #             min_gene = gene_sums.idxmin()
    #             min_value = gene_sums.min()
    #             threshold = 1.3 * min_value

    #             baseline_genes = gene_sums[gene_sums <= threshold]
    #             baseline_mean = baseline_genes.mean()
    #             genes_within_range = gene_sums[gene_sums <= threshold].index.tolist()
    #             subset = cellbygene_temp[genes_within_range]
    #             avg_list = subset.mean(axis=1).values

    #             # Count the total error
    #             total_error += ((avg_list - cellxspecies_df[species])**2).sum()
    #             total_count += len(avg_list)
    #             plt.scatter(avg_list, cellxspecies_df[species].values, s = 1, alpha = 0.3, label = species)
    # plt.xlabel(f"Usage")
    # plt.ylabel(f"Count")
    # plt.legend()
    # plt.savefig(f"{out_dir}/{out_name}/{out_name}.usage_map.png")
    # plt.close()

    # avg_count_error = total_error / total_count

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

num_predicted_overlap = 0
total_done = 0

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

        start = time.time()

        # try :
        predicted_species_count, jaccard, count_err = combo_run(run_out_dir, out_name, cellbygene_path, cellbyspecies, metadata_file, num_ecDNA, args.max_species, dummydist, cNMF_thresh)
        # TODO: remove this
        if predicted_species_count == -1 :
            num_predicted_overlap += 1
        total_done += 1
        # if overlap > 0 and predicted_species_count == -1 :
        #     predicted_species_count = num_ecDNA_true
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

print(f"Overlap: {overlap}, Predicted: {num_predicted_overlap}/{total_done}")


(pd.DataFrame(run_predicted_species_counts_list)).to_csv(run_predicted_species_counts_file, index = None, sep = '\t')
(pd.DataFrame(run_jaccard_list)).to_csv(run_jaccard_file, index = None, sep = '\t')
(pd.DataFrame(run_count_err_list)).to_csv(run_count_err_file, index = None, sep = '\t')
(pd.DataFrame(run_time_list)).to_csv(run_time_file, index = None, sep = '\t')



