import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import math
from collections import defaultdict

run_out_dir = "../const_sim_out_withnaive"
plot_out_dir = "./const_sim_out_full_figs"

# What to title these in the plots!
folder_to_name = {
    "cNMF_results_countprov_0_errorw_0.35" : "cNMF",
    "combo_results_countprov_0_thresh_0.55" : "Combo",
    "naive_results_countprov_0_thresh_0.75" : "Naive",
    "hier_results_countprov_0_ddist_1.3" : "Hier"
}



def get_runtimes(run_out_dir, folder_to_name) :
    runtimes = defaultdict(list)

    # Give an order to the iteration
    priority = {
        "naive": 0,
        "hier": 1,
        "cNMF": 2,
        "combo": 3,
    }

    sorted_keys = sorted(
        folder_to_name,
        key=lambda k: next(
            priority[p]
            for p in priority
            if k.startswith(p)
        )
    )

    # For combo
    combo_need_cNMF = set()
    cNMF_runtimes_save = {}
    
    # Iterate through all results
    for iter, folder_name in enumerate(sorted_keys):
        result_dir = Path(f"{run_out_dir}/{folder_name}")
        print(f"Processing {result_dir}")
    
        # Get metadata embedded in the folder name
        method, _, _, countprov, _, val = result_dir.name.split('_')
        countprov = bool(int(countprov))
        val = float(val)
    
        for inner_dir in result_dir.glob("*") :
    
            _, fmax, _, overlap, _, extra_counts, _, depth = inner_dir.name.split('_')
            depth = float(depth)
            overlap = float(overlap)
            fmax = float(fmax)
            extra_counts = float(extra_counts)

            # Just get the lines that need cNMF
            # TODO: remove if you ever change combo
            if method == "combo" : 
                species_counts_df = pd.read_csv(f"{inner_dir}/species_counts.tsv", sep = '\t')         
                for row_idx, row in species_counts_df.iterrows() :
                    for col_idx, col in enumerate(species_counts_df.columns) :
                        if "run_" in col and 'unique' not in col :  
                            if row[col] == -1 :
                                combo_need_cNMF.add((fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col))



                
            runtime_df = pd.read_csv(f"{inner_dir}/runtime.tsv", sep = '\t')         
           
            for row_idx, row in runtime_df.iterrows() :
                for col_idx, col in enumerate(runtime_df.columns) :
                    if "run_" in col and 'unique' not in col :

                        # For combo, remove later
                        if method == "cNMF" :
                            cNMF_runtimes_save[(fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)] = row[col]

                        # Add the cNMF time to this
                        if method == "combo" and (fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) in combo_need_cNMF :
                            runtimes[f"{folder_to_name[folder_name]} with cNMF"].append(cNMF_runtimes_save[(fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)] + row[col])
                        elif method == "combo" :
                            runtimes[f"{folder_to_name[folder_name]} without cNMF"].append(row[col])
                        else :
                            runtimes[folder_to_name[folder_name]].append(row[col])

    return runtimes

runtimes = get_runtimes(run_out_dir, folder_to_name)



# Get rid of NaNs
clean_runtimes = {}

for k, v in runtimes.items():
    cleaned = []

    for x in v:
        try:
            x = float(x)
            if np.isfinite(x):   # removes NaN, inf, -inf
                cleaned.append(x)
        except:
            pass

    clean_runtimes[k] = cleaned
runtimes = clean_runtimes

box_data = []
labels = []

for k, v in runtimes.items():
    box_data.append(v)
    labels.append(k)


fig, ax = plt.subplots(figsize=(10, 6))

plt.yscale('log')

bp = ax.boxplot(
    box_data,
    tick_labels=labels,
)

plt.xticks(rotation=45)
plt.ylabel("Seconds (log)")
plt.title("Runtime of Methods on Simulated Data (30 Genes, ~2000 cells)")
plt.tight_layout()

plt.savefig(f"{plot_out_dir}/runtimes_log.png")
plt.close()



fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(
    box_data,
    tick_labels=labels,
)

plt.xticks(rotation=45)
plt.ylabel("Seconds")
plt.title("Runtime of Methods on Simulated Data (30 Genes, ~2000 cells)")
plt.tight_layout()

plt.savefig(f"{plot_out_dir}/runtimes.png")
