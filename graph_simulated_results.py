import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

#############################################################################

run_out_dir = Path("/orcd/data/ki/001/lab/jones/kyu06/cass_data/five_species_out/")
plot_out_dir = "/orcd/data/ki/001/lab/jones/kyu06/cass_data/five_species_figs/"
sim_dir = "/orcd/data/ki/001/lab/jones/kyu06/cass_data/five_species"

# Should have this be the simulated one
# options: "mean_ecDNA_prop", "min_ecDNA_prop"
x = "min_ecDNA_prop"
# None if you are doing 2D plot
# Some options : "num_ecDNA_true", "comb_chance" (cosegregation), "countprov", "fmax", "overlap", 
# "extra_counts", "depth", "threshold", "errorw"
x2 = None

# Should be count_err, jaccard, or species_counts
y = "jaccard"

# Variable name : list of things to allow
consts = {"countprov" : [True], "overlap" : [0]}


# Plot only points that have the correct species count (as if that is wrong most stats are quite bad)
use_only_correct_species = True

##############################################################################

os.makedirs(plot_out_dir, exist_ok = True)


# Formulate output name automatically
consts_str = ""
for key, val in consts.items() :
    val_str = ""
    for v in val :
        val_str += f",{v}"
    val_str = val_str[1:]
    consts_str += f"_{key}-{val_str}"
plot_name = f"{plot_out_dir}/{y}_{x}{consts_str}"

if use_only_correct_species :
    plot_name += "_onlycorr"
plot_name += ".png"

pd.set_option('display.max_columns', None)

x_list = []
y_list = []
if x2 is not None :
    x2_list = []
else :
    x2_list = None
labels = []

wrong_species = set()

for result_dir in run_out_dir.glob("*results*") :
    print(result_dir)

    # Get metadata embedded in the folder name
    method, _, _, countprov, _, val = result_dir.name.split('_')
    countprov = bool(countprov)
    val = float(val)

    for inner_dir in result_dir.glob("*") :

        _, fmax, _, overlap, _, extra_counts, _, depth = inner_dir.name.split('_')


        depth = float(depth)
        overlap = float(overlap)
        fmax = float(fmax)
        extra_counts = float(extra_counts)

        species_counts_df = pd.read_csv(f"{inner_dir}/species_counts.tsv", sep = '\t')

        # Get correct species data (fill in dict with wrong species)
        if use_only_correct_species :
            for row_idx, row in species_counts_df.iterrows() :
                for col in species_counts_df.columns :
                    if "run_" in col :
                        if row['num_ecDNA_true'] != row[col] :
                            wrong_species.add((method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col))
    

        if y == "count_err" :
            df = pd.read_csv(f"{inner_dir}/count_err.tsv", sep = '\t')
        elif y == "jaccard" :
            df = pd.read_csv(f"{inner_dir}/jaccard.tsv", sep = '\t')
        else :
            df = pd.read_csv(f"{inner_dir}/species_counts.tsv", sep = '\t')
            # If species counts, make it binary right or wrong
            for rowidx, row in df.iterrows() :
                for col in df.columns :
                    if "run_" in col :
                        df.loc[rowidx, col] = int(row['num_ecDNA_true'] == row[col])
    

        
        if method == "hier" :
            df['method'] = "hier"
            df['threshold'] = val
        elif method == "cNMF" :
            df['method'] = "cNMF"
            df['errorw'] = val
        else :
            print("ERROR: unknown method (only hier and cNMF known)")
            exit(0)
        df['countprov'] = countprov
        df['fmax'] = fmax
        df['overlap'] = overlap
        df['extra_counts'] = extra_counts
        df['depth'] = depth

        # Add points
        for i, row in df.iterrows() :
            num_true = row['num_ecDNA_true']
            comb_chance = row['comb_chance']
            for col in df.columns :
                if "run_" in col :            
                    metadata_file = f"{sim_dir}/fmax_{fmax}_overlap_{overlap}_extracounts_{extra_counts}_depth_{depth}/{num_true}_species_{comb_chance:g}_comb/{col}_metadata.txt"
                    min_ecDNA = 100
                    total_ecDNA = 0
                    with open(metadata_file, "r") as metaf:
                        for line in metaf :
                            # Line with ecDNA percentage
                            if 'Percent of cells with ecDNA_' in line :
                                percent = float(line.rstrip().split(':\t')[-1])
                                total_ecDNA += percent
                                min_ecDNA = min(min_ecDNA, percent)
                            if 'Depths' in line :
                                break
                                
                    # Add items to lists
                    y_list.append(row[col])
                    if x2 is not None :
                        x2_list.append(row[x2])
                    if x == "mean_ecDNA_prop" :
                        x_list.append(total_ecDNA/num_true)
                    elif x == "min_ecDNA_prop" :
                        x_list.append(min_ecDNA)
                    labels.append(row["method"])
                        

if x2 is None :
    for lab in set(labels):
        xs = [xi for xi, l in zip(x_list, labels) if l == lab]
        ys = [yi for yi, l in zip(y_list, labels) if l == lab]
        plt.scatter(xs, ys, label=lab, alpha = 0.2)
    
    plt.ylabel(y)
    plt.xlabel(x)

plt.legend()
print(f"Outputting to {plot_name}")
plt.savefig(plot_name)

                    
                    