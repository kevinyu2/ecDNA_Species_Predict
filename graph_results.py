import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

#############################################################################

run_out_dir = Path("../pipeline_out")
plot_out_dir = "../pipeline_out"

x = "comb_chance"

# If provided (not none), will make a grid like plot with this on the y
x2 = "depth"

# Should be count_err, jaccard, or species_counts
y = "count_err"

# Variable name : list of things to allow
consts = {"countprov" : [True], "depth" : [2]}

# Line plot instead of scatter, with the mean (and hopefully min max bars)
use_mean = True
# Plot only points that have the correct species count (as if that is wrong most stats are quite bad)
use_only_correct_species = True

##############################################################################

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

for result_dir in run_out_dir.glob("*results*") :

    # Get metadata embedded in the folder name
    method, _, _, countprov, _, val = result_dir.name.split('_')
    countprov = bool(countprov)
    val = float(val)

    for inner_dir in result_dir.glob("*") :
        print(inner_dir)

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
            else :
                df["mean"] = df.filter(like="run_").mean(axis=1)

                # Mean but only for those with right species count
                df["correct_species_mean"] = np.nan
                for row_idx, row in df.iterrows() :
                    total_sum = 0
                    num_runs_correct = 0
                    for col in df.columns :
                        if "run_" in col :
                            # Check if in wrong species
                            if (method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) not in wrong_species :
                                num_runs_correct += 1
                                total_sum += row[col]
                    if num_runs_correct > 0 :
                        df.loc[row_idx, "correct_species_mean"] = total_sum / num_runs_correct

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

# For three variable grid plots
def extract_points(df, row_cols, x_col, y_col):
    x_plot, y_plot, z_plot = [], [], []

    for _, row in df.iterrows():
        for rcol in row_cols:
            if pd.notna(row[rcol]):
                z_plot.append(row[rcol])
                x_plot.append(row[x_col])
                y_plot.append(row[y_col])

    return x_plot, y_plot, z_plot

def build_grid_from_points(x_plot, y_plot, z_plot, x_unique, y_unique, fill_value=-1):
    x_map = {val: idx for idx, val in enumerate(x_unique)}
    y_map = {val: idx for idx, val in enumerate(y_unique)}

    Z = np.full((len(y_unique), len(x_unique)), -1.0)

    for x_val, y_val, z_val in zip(x1, y1, z1):
        Z[y_map[y_val], x_map[x_val]] = z_val

    return Z

if y == "count_err" :
    df_to_use = count_err_full_df.copy()
elif y == "jaccard" :
    df_to_use = jaccard_full_df.copy()
elif y == "species_counts" :
    df_to_use = species_counts_full_df.copy()

if y == "count_err" or y == "jaccard" :        
    # Use the consts to cut down on the dataframe
    for key, val in consts.items() :
        df_to_use = df_to_use.loc[df_to_use[key].isin(val)]

    hier = df_to_use.loc[df_to_use["method"] == "hier"]
    cNMF = df_to_use.loc[df_to_use["method"] == "cNMF"]

    if use_mean :
        if use_only_correct_species :
            row_cols = ["correct_species_mean"]
        else :
            row_cols = ["mean"]
            
    else :
        row_cols = [col for col in df_to_use.columns if col.startswith("run_")] 

    
    if x2 is None :
        # Add non NaN values to plot
        x_plot = []
        y_plot = []
        for i, row in hier.iterrows() :
            for rcol in row_cols :
                if pd.isna(row[rcol]) == False :
                    y_plot.append(row[rcol])
                    x_plot.append(row[x])

        if use_mean :
            # Sort lists for line plot
            combined = list(zip(x_plot, y_plot))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            x_plot, y_plot = zip(*combined_sorted)
            plt.plot(x_plot, y_plot, label="hier", alpha=0.4)
        else :
            plt.scatter(x_plot, y_plot, label="hier", alpha=0.4)
        x_plot = []
        y_plot = []

        
        for i, row in cNMF.iterrows() :
            for rcol in row_cols :
                if pd.isna(row[rcol]) == False :
                    y_plot.append(row[rcol])
                    x_plot.append(row[x])

        if use_mean :
            combined = list(zip(x_plot, y_plot))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            x_plot, y_plot = zip(*combined_sorted)
            plt.plot(x_plot, y_plot, label="cNMF", alpha=0.4)
        else :
            plt.scatter(x_plot, y_plot, label="cNMF", alpha=0.4)
        plt.legend()
        plt.xlabel(x)
        plt.ylabel(y)

    # Grid plot             
    else :
        if use_only_correct_species :
            row_cols = ["correct_species_mean"]
        else :
            row_cols = ["mean"]


        # extract x, y, z from cNMF and hier
        x1, y1, z1 = extract_points(cNMF, row_cols, x, x2)
        x2_, y2_, z2 = extract_points(hier, row_cols, x, x2)

        # union tje axes
        x_unique = np.unique(np.concatenate([x1, x2_]))
        y_unique = np.unique(np.concatenate([y1, y2_]))

        # Build grids
        Z1 = build_grid_from_points(x1, y1, z1, x_unique, y_unique, fill_value=-1)
        Z2 = build_grid_from_points(x2_, y2_, z2, x_unique, y_unique, fill_value=-1)


        vmin = min(Z1.min(), Z2.min()) - 0.001
        vmax = max(Z1.max(), Z2.max()) + 0.001

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        im1 = axes[0].imshow(Z1, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("cNMF")


        im2 = axes[1].imshow(Z2, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("hier")

        axes[0].set_xlabel(x)
        axes[0].set_ylabel(x2)

        axes[1].set_xlabel(x)

        for ax in axes:
            ax.set_xticks(range(len(x_unique)))
            ax.set_xticklabels(x_unique)
            ax.set_yticks(range(len(y_unique)))
            ax.set_yticklabels(y_unique)


        # Add cell annotations
        for i in range(len(y_unique)):
            for j in range(len(x_unique)):
                axes[0].text(j, i, f"{Z1[i,j]:.2f}", ha='center', va='center', color='white')
                axes[1].text(j, i, f"{Z2[i,j]:.2f}", ha='center', va='center', color='white')
        # Reserve space on the right for colorbar
        fig.subplots_adjust(right=0.85)  # leave 15% for colorbar

        fig.suptitle(f"{y} cNMF vs Hierarchical" , fontsize=16)

        # Place colorbar in the reserved space
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax, label="Value")

    print(f"Outputting to {plot_name}")
    plt.savefig(plot_name)






elif y == "species_counts" :
    
    # Use the consts to cut down on the dataframe
    for key, val in consts.items() :
        df_to_use = df_to_use.loc[df_to_use[key].isin(val)]

    print(df_to_use)

    hier = df_to_use.loc[df_to_use["method"] == "hier"]
    cNMF = df_to_use.loc[df_to_use["method"] == "cNMF"]

    row_cols = ['corr_prop']

    if x2 is None :

        # Add non NaN values to plot
        x_plot = []
        y_plot = []
        for i, row in hier.iterrows() :
            for rcol in row_cols :
                if pd.isna(row[rcol]) == False :
                    y_plot.append(row[rcol])
                    x_plot.append(row[x])
    
        combined = list(zip(x_plot, y_plot))
        combined_sorted = sorted(combined, key=lambda x: x[0])
        x_plot, y_plot = zip(*combined_sorted)
        plt.plot(x_plot, y_plot, label="hier", alpha=0.4)

        x_plot = []
        y_plot = []
        for i, row in cNMF.iterrows() :
            for rcol in row_cols :
                if pd.isna(row[rcol]) == False :
                    y_plot.append(row[rcol])
                    x_plot.append(row[x])
        if use_mean :
            combined = list(zip(x_plot, y_plot))
            combined_sorted = sorted(combined, key=lambda x: x[0])
            x_plot, y_plot = zip(*combined_sorted)
            plt.plot(x_plot, y_plot, label="cNMF", alpha=0.4)
        else :
            plt.scatter(x_plot, y_plot, label="cNMF", alpha=0.4)
        plt.legend()
        plt.xlabel(x)
        plt.ylabel(y)
        
        plt.show()


    # Grid plot             
    else :

        # extract x, y, z from cNMF and hier
        x1, y1, z1 = extract_points(cNMF, row_cols, x, x2)
        x2_, y2_, z2 = extract_points(hier, row_cols, x, x2)

        # union tje axes
        x_unique = np.unique(np.concatenate([x1, x2_]))
        y_unique = np.unique(np.concatenate([y1, y2_]))

        # Build grids
        Z1 = build_grid_from_points(x1, y1, z1, x_unique, y_unique, fill_value=-1)
        Z2 = build_grid_from_points(x2_, y2_, z2, x_unique, y_unique, fill_value=-1)


        vmin = min(Z1.min(), Z2.min()) - 0.001
        vmax = max(Z1.max(), Z2.max()) + 0.001

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        im1 = axes[0].imshow(Z1, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title("cNMF")


        im2 = axes[1].imshow(Z2, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title("hier")

        axes[0].set_xlabel(x)
        axes[0].set_ylabel(x2)

        axes[1].set_xlabel(x)

        for ax in axes:
            ax.set_xticks(range(len(x_unique)))
            ax.set_xticklabels(x_unique)
            ax.set_yticks(range(len(y_unique)))
            ax.set_yticklabels(y_unique)

        # Add cell annotations
        for i in range(len(y_unique)):
            for j in range(len(x_unique)):
                axes[0].text(j, i, f"{Z1[i,j]:.2f}", ha='center', va='center', color='white')
                axes[1].text(j, i, f"{Z2[i,j]:.2f}", ha='center', va='center', color='white')
        # Reserve space on the right for colorbar
        fig.subplots_adjust(right=0.85)  # leave 15% for colorbar

        fig.suptitle(f"{y} cNMF vs Hierarchical" , fontsize=16)

        # Place colorbar in the reserved space
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im1, cax=cbar_ax, label="Value")

    print(f"Outputting to {plot_name}")
    plt.savefig(plot_name)






