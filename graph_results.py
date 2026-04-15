import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

#############################################################################

run_out_dir = Path("/orcd/data/ki/001/lab/jones/kyu06/cass_data/five_species_out/")
plot_out_dir = "/orcd/data/ki/001/lab/jones/kyu06/cass_data/five_species_figs/"
# plot_out_dir = "~/test/"

# Some options : "num_ecDNA_true", "comb_chance" (cosegregation), "countprov", "fmax", "overlap", 
# "extra_counts", "depth", "threshold", "errorw"
x = "depth"

# If provided (not none), will make a grid like plot with this on the y
x2 = None

# Should be count_err, jaccard, or species_counts
y = "count_err"

# Variable name : list of things to allow
consts = {"countprov" : [True], 'overlap' : [0]}

# Line plot instead of scatter, with the mean (and hopefully min max bars). Only for one x, if x2 provided, does nothing
use_mean = True
# Plot only points that have the correct species count (as if that is wrong most stats are quite bad)
use_only_correct_species = True
# Plot only points that have the correct jaccard for count error
use_only_correct_jaccard = True

##############################################################################

os.makedirs(plot_out_dir, exist_ok = True)

# Improve graph
colors = {"Hier" : "blue", "cNMF" : "red"}
label_dict = {"num_ecDNA_true" : "Number of True ecDNA Species",
              "comb_chance" : "Cosegregation Strength",
              "countprov" : "Species Number Known",
              "fmax" : "Maximum Selection Coefficient",
              "overlap" : "Proportion Genes Overlapped",
              "extra_counts" : "Chance for Extra Copies of Gene",
              "depth" : "Relative Read Depth",
              "threshold" : "Hierarchical Threshold",
              "errorw" : "cNMF Error Score Weight",
              "count_err" : "ecDNA Count Error Per Cell",
              "jaccard" : "Average Jaccard of Genes",
              "species_counts" : "Proportion Species Number Correct"}

# Formulate output name automatically
x2str = ""
if x2 is not None :
    x2str = f"_{x2}"
consts_str = ""
for key, val in consts.items() :
    val_str = ""
    for v in val :
        val_str += f",{v}"
    val_str = val_str[1:]
    consts_str += f"_{key}-{val_str}"
plot_name = f"{plot_out_dir}/{y}_{x}{x2str}{consts_str}"
if use_mean :
    plot_name += "_mean"
if use_only_correct_species :
    plot_name += "_onlycorr"
plot_name += ".png"

pd.set_option('display.max_columns', None)

count_err_dfs = []
jaccard_dfs = []
species_counts_dfs = []
wrong_species = set()
wrong_jaccard = set()

# Iterate through all results
for result_dir in run_out_dir.glob("*results*") :
    print(f"Processing {result_dir}")

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

        count_err_df = pd.read_csv(f"{inner_dir}/count_err.tsv", sep = '\t')
        jaccard_df = pd.read_csv(f"{inner_dir}/jaccard.tsv", sep = '\t')
        species_counts_df = pd.read_csv(f"{inner_dir}/species_counts.tsv", sep = '\t')

        # Add metadata
        for name, df in [("sc", species_counts_df), ("jac", jaccard_df), ("ce", count_err_df)] :
            # Get wrong species data and populate the correlation
            if name == "sc" :
                df['corr_prop'] = 0
                for row_idx, row in df.iterrows() :
                    wrong_count = 0
                    num_runs = 0
                    for col in df.columns :
                        if "run_" in col :
                            num_runs += 1
                            if row['num_ecDNA_true'] != row[col] :
                                wrong_count += 1
                                wrong_species.add((method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col))
                    df.loc[row_idx, 'corr_prop'] = (num_runs - wrong_count)/num_runs
            
            elif name == "jac" :
                # Get wrong jaccard info
                # Note: currently only supports average jaccard rather than a binary right / wrong
                for row_idx, row in df.iterrows() :
                    for col in df.columns :
                        if "run_" in col :
                            if row[col] < 1 :
                                wrong_jaccard.add((method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col))
                            # if in wrong species, remove
                            if use_only_correct_species and (method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) in wrong_species :
                                df.loc[row_idx, col] = np.nan
            
            elif name == "ce" :
                # Get wrong jaccard info
                for row_idx, row in df.iterrows() :
                    for col in df.columns :
                        if "run_" in col :
                            # remove based on wrong species and jaccard
                            if use_only_correct_species and (method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) in wrong_species :
                                df.loc[row_idx, col] = np.nan
                            if use_only_correct_jaccard and (method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) in wrong_jaccard :
                                df.loc[row_idx, col] = np.nan
                                
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

        count_err_dfs.append(count_err_df)
        jaccard_dfs.append(jaccard_df)
        species_counts_dfs.append(species_counts_df)

count_err_full_df = pd.concat(count_err_dfs, ignore_index=True)
jaccard_full_df = pd.concat(jaccard_dfs, ignore_index=True)
species_counts_full_df = pd.concat(species_counts_dfs, ignore_index=True)

# For three variable grid plots, extract 3 cols
def extract_points(df, row_cols, x_col, y_col):
    x_plot, y_plot, z_plot = [], [], []

    for _, row in df.iterrows():
        for rcol in row_cols:
            if pd.notna(row[rcol]):
                z_plot.append(row[rcol])
                x_plot.append(row[x_col])
                y_plot.append(row[y_col])
                

    return x_plot, y_plot, z_plot

# For three variable grid plots
def build_grid_from_points(x_plot, y_plot, z_plot, x_unique, y_unique, fill_value=-1):
    x_map = {val: idx for idx, val in enumerate(x_unique)}
    y_map = {val: idx for idx, val in enumerate(y_unique)}

    # Total z axis
    Z = np.full((len(y_unique), len(x_unique)), fill_value, dtype=float)
    # Number of items in Z axis
    counts = np.zeros((len(y_unique), len(x_unique)), dtype=int)

    for x_val, y_val, z_val in zip(x_plot, y_plot, z_plot):
        i = y_map[y_val]
        j = x_map[x_val]

        # If first value, just assign
        if counts[i, j] == 0:
            Z[i, j] = z_val
        else:
            # Accumulate for now, then divide to get mean
            Z[i, j] += z_val

        counts[i, j] += 1

    # Turn into mean
    Z_mean = np.divide(Z, counts, where=counts > 0)
    Z_mean[counts == 0] = fill_value

    return Z_mean, counts

if y == "count_err" :
    df_to_use = count_err_full_df.copy()
elif y == "jaccard" :
    df_to_use = jaccard_full_df.copy()
elif y == "species_counts" :
    df_to_use = species_counts_full_df.copy()


# Use the consts to cut down on the dataframe
for key, val in consts.items() :
    df_to_use = df_to_use.loc[df_to_use[key].isin(val)]



# Adds mean and number of runs to count
def get_mean(df_to_use) :
    df_to_use["mean"] = 0
    df_to_use["num_runs"] = 0
    for rowidx, row in df_to_use.iterrows() :
        total_num = 0
        total_sum = 0
        for col in df_to_use.columns :
            if "run_" in col : 
                if not pd.isna(row[col]) :
                    total_num += 1
                    total_sum += row[col]
        
        if total_num > 0 :
            df_to_use.loc[rowidx, "mean"] = total_sum / total_num
            df_to_use.loc[rowidx, "num_runs"] = total_num
    return df_to_use

# Determine which columns are actually getting graphed
if y == "species_counts" :
    row_cols = ["corr_prop"]
if use_mean and x2 is None and y != "species_counts":
    df_to_use = get_mean(df_to_use)
    row_cols = ["mean"]
else :
    row_cols = [col for col in df_to_use.columns if col.startswith("run_")] 


hier = df_to_use.loc[df_to_use["method"] == "hier"]
cNMF = df_to_use.loc[df_to_use["method"] == "cNMF"]



if x2 is None :
    for df_name, method_name in [(hier, "Hier"), (cNMF, "cNMF")] :
        x_plot = []
        y_plot = []
        lab_plot = []
        lab_max = -1

        # Add non NaN values to plot
        if not use_mean : 
            for i, row in df_name.iterrows() :
                for rcol in row_cols :
                    if pd.isna(row[rcol]) == False :
                        y_plot.append(row[rcol])
                        x_plot.append(row[x])
            plt.scatter(x_plot, y_plot, label=method_name, color = colors[method_name], alpha=0.4)
            plt.legend()


        # Also include how many there are
        else :
            for i, row in df_name.iterrows() :
                for rcol in row_cols :
                    if pd.isna(row[rcol]) == False :
                        y_plot.append(row[rcol])
                        x_plot.append(row[x])
                        lab_plot.append(row["num_runs"])
                        lab_max = max(lab_max, row["num_runs"])

            # Sort lists for line plot
            combined = list(zip(x_plot, y_plot, lab_plot))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            x_plot, y_plot, lab_plot = zip(*combined_sorted)
            plt.scatter(x_plot, y_plot, label=method_name, color = colors[method_name], alpha=0.4, s = 50 * np.array(lab_plot)/lab_max)
            
            # Size legend
            legend_vals = [lab_max * 0.25, lab_max * 0.5, lab_max * 0.75, lab_max]
            
            legend_sizes = [v * 50 / lab_max for v in legend_vals]
            
            handles = [
                plt.scatter([], [], s=s, color='gray', alpha=0.4)
                for s in legend_sizes
            ]
            
            labels = [f"{int(v)}" for v in legend_vals]
            
            plt.legend(handles, labels, title="Number of Runs")

  
    plt.xlabel(label_dict[x])
    plt.ylabel(label_dict[y])
    plt.title(f"{label_dict[y]} cNMF vs Hierarchical" , fontsize=16)

# Grid plot             
else :

    # extract x, y, z from cNMF and hier
    x1, y1, z1 = extract_points(cNMF, row_cols, x, x2)
    x2_, y2_, z2 = extract_points(hier, row_cols, x, x2)

    # union the axes
    x_unique = np.unique(np.concatenate([x1, x2_]))
    y_unique = np.unique(np.concatenate([y1, y2_]))

    # Build grids
    Z1, count1 = build_grid_from_points(x1, y1, z1, x_unique, y_unique, fill_value=-1)
    Z2, count2 = build_grid_from_points(x2_, y2_, z2, x_unique, y_unique, fill_value=-1)

    # Allow some wiggle room in color (otherwise breaks when all are perfect)
    vmin = min(Z1.min(), Z2.min()) - 0.001
    vmax = max(Z1.max(), Z2.max()) + 0.001

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    im1 = axes[0].imshow(Z1, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("cNMF")


    im2 = axes[1].imshow(Z2, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Hierarchical")

    # Horizontal and vertical axis (just one vertical, shared)
    axes[0].set_xlabel(label_dict[x])
    axes[0].set_ylabel(label_dict[x2])
    axes[1].set_xlabel(label_dict[x])

    for ax in axes:
        ax.set_xticks(range(len(x_unique)))
        ax.set_xticklabels(x_unique)
        ax.set_yticks(range(len(y_unique)))
        ax.set_yticklabels(y_unique)


    for i in range(len(y_unique)):
        for j in range(len(x_unique)):
            # Z values (top-left, bigger)
            if count1[i, j] > 0:
                axes[0].text(j - 0.3, i + 0.3,
                            f"{Z1[i,j]:.2f}",
                            ha='left', va='top',
                            color='white', fontsize=10, fontweight='bold')

                axes[0].text(j + 0.3, i - 0.3,
                            f"n={count1[i,j]}",
                            ha='right', va='bottom',
                            color='white', fontsize=7)

            if count2[i, j] > 0:
                axes[1].text(j - 0.3, i + 0.3,
                            f"{Z2[i,j]:.2f}",
                            ha='left', va='top',
                            color='white', fontsize=10, fontweight='bold')

                axes[1].text(j + 0.3, i - 0.3,
                            f"n={count2[i,j]}",
                            ha='right', va='bottom',
                            color='white', fontsize=7)
                
    # Reserve space on the right for colorbar
    fig.subplots_adjust(right=0.85)  # leave 15% for colorbar

    fig.suptitle(f"{label_dict[y]} cNMF vs Hierarchical" , fontsize=16)

    # Place colorbar in the reserved space
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im1, cax=cbar_ax, label="Value")

print(f"Outputting to {plot_name}")
plt.savefig(plot_name)
