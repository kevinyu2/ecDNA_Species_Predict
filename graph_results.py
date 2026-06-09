import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import math


run_out_dir = "../const_sim_out_withnaive"
plot_out_dir = "./const_sim_out_full_figs"

# What to title these in the plots!
folder_to_name = {
    "cNMF_results_countprov_0_errorw_0.25" : "cNMF (errorw = 0.25)",
    "combo_results_countprov_0_thresh_0.55" : "Combo (threshold = 0.55)",
    "naive_results_countprov_0_thresh_0.75" : "Naive (threshold = 0.75)",
    "hier_results_countprov_0_ddist_1.3" : "Hier (dummy weight = 1.3)"
}

# Unique settings for each plot
settings = [{"x" : "depth", "x2" : "comb_chance", "y" : "species_counts", "consts" : {"overlap" : [0]}, "plot_title" : "Percentage with Species Number Correct (No Overlap)"},
            {"x" : "overlap", "x2" : "comb_chance", "y" : "species_counts", "consts" : {"num_ecDNA_true" : [2,3,4,5]}, "plot_title" : "Percentage with Species Number Correct (2+ True Species)"},
            {"x" : "depth", "x2" : "comb_chance", "y" : "jaccard", "consts" : {"overlap" : [0], "num_ecDNA_true" : [2,3,4,5]}, "plot_title" : "Average Jaccard Given Species Number Correct (No Overlap, 2+ True Species)"},
            {"x" : "overlap", "x2" : "comb_chance", "y" : "jaccard", "consts" : {"num_ecDNA_true" : [2,3,4,5]}, "plot_title" : "Average Jaccard Given Species Number Correct (2+ True Species)"},
            {"x" : "depth", "x2" : "comb_chance", "y" : "count_err", "consts" : {"overlap" : [0], "num_ecDNA_true" : [2,3,4,5]}, "plot_title" : "Average Per Cell Error Given Jaccard = 1 (No Overlap, 2+ True Species)"},
            {"x" : "overlap", "x2" : "comb_chance", "y" : "count_err", "consts" : {"num_ecDNA_true" : [2,3,4,5]}, "plot_title" : "Average Per Cell Error Given Jaccard = 1 (2+ True Species)"},
            {"x" : "num_ecDNA_true", "x2" : "comb_chance", "y" : "species_counts", "consts" : {"overlap" : [0]}, "plot_title" : "Percentage with Species Number Correct (No Overlap)"},
            {"x" : "num_ecDNA_true", "x2" : "comb_chance", "y" : "jaccard", "consts" : {"overlap" : [0]}, "plot_title" : "Average Jaccard Given Species Number Correct (No Overlap)"},
            {"x" : "num_ecDNA_true", "x2" : "comb_chance", "y" : "count_err", "consts" : {"overlap" : [0]}, "plot_title" : "Average Per Cell Error Given Jaccard = 1 (No Overlap)"},
            {"x" : "num_ecDNA_true", "x2" : "num_ecDNA_predicted", "y" : "species_counts", "consts" : {"overlap" : [0]}, "plot_title" : "Predicted vs Correct Species Count (No Overlap)"},
            {"x" : "num_ecDNA_true", "x2" : "num_ecDNA_predicted", "y" : "species_counts", "consts" : {"overlap" : [0], "depth" : [0.5, 1.0, 1.5, 2.0], "comb_chance" : [0.0, 0.2, 0.4, 0.6]}, "plot_title" : "Predicted vs Correct Species Count (No Overlap, Depth > 0.25, Cosegregation < 0.8)"},
            {"x" : "num_ecDNA_true", "x2" : "num_ecDNA_predicted", "y" : "species_counts", "consts" : {"overlap" : [0.4]}, "plot_title" : "Predicted vs Correct Species Count (0.4 Overlap)"},
            {"x" : "num_ecDNA_true", "x2" : "num_ecDNA_predicted", "y" : "species_counts", "consts" : {"overlap" : [0.4], "depth" : [0.5, 1.0, 1.5, 2.0], "comb_chance" : [0.0, 0.2, 0.4, 0.6]}, "plot_title" : "Predicted vs Correct Species Count (Overlap 0.4, Depth > 0.25, Cosegregation < 0.8)"},
            {"x" : "num_ecDNA_true", "x2" : "num_ecDNA_predicted", "y" : "species_counts", "consts" : {"overlap" : [0.4], "comb_chance" : [0]}, "plot_title" : "Predicted vs Correct Species Count (0.4 Overlap, Coseg 0)"}
            ]





#############################################################################
def graph_results(
    run_out_dir,
    plot_out_dir,

    count_err_full_df,
    jaccard_full_df,
    species_counts_full_df,
    # Some options : "num_ecDNA_true", "comb_chance" (cosegregation), "countprov", "fmax", "overlap", 
    # "extra_counts", "depth", "threshold", "errorw"
    x = "num_ecDNA_true",
    
    # If provided (not none), will make a grid like plot with this on the y
    # If x = "num_ecDNA_true", x2 = "num_ecDNA_predicted", and y = "species_counts", will show true vs predicted species counts
    x2 = "comb_chance",
    
    # Should be count_err, jaccard, or species_counts
    y = "species_counts",
    
    # Variable name : list of things to allow
    consts = {"countprov" : [False], "overlap" : [0]},
    
    # Line plot instead of scatter, with the mean (and hopefully min max bars). Only for one x, if x2 provided, does nothing
    use_mean = True,
    # Plot only points that have the correct species count (as if that is wrong most stats are quite bad)
    use_only_correct_species = True,
    # Plot only points that have the correct jaccard for count error
    use_only_correct_jaccard = True,
    
    # Improve graph
    # TODO: allow customization for which methods we use and naming conventions
    colors = {"hier" : "blue", "cNMF" : "red", "naive" : "green", "combo" : "purple"},

    # If none, will attempt to calculate by itself
    plot_title = None
    
) :
    run_out_dir = Path(run_out_dir)
    ##############################################################################
    
    os.makedirs(plot_out_dir, exist_ok = True)
    
    label_dict = {"num_ecDNA_true" : "Number of True ecDNA Species",
                  "comb_chance" : "Cosegregation Strength",
                  "countprov" : "Species Number Known",
                  "fmax" : "Maximum Selection Coefficient",
                  "overlap" : "Proportion Genes Overlapped",
                  "extra_counts" : "Chance for Extra Copies of Gene",
                  "depth" : "Insertions Per Copy Number (Saturation)",
                  "threshold" : "Hierarchical Threshold",
                  "errorw" : "cNMF Error Score Weight",
                  "count_err" : "ecDNA Count Error Per Cell",
                  "jaccard" : "Average Jaccard of Genes",
                  "species_counts" : "Proportion Species Number Correct",
                  "num_ecDNA_predicted" : "Predicted ecDNA Species Count"
                  }
    
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
    
    
    
    # For three variable grid plots, extract 3 cols
    def extract_points(df, row_cols, x_col, y_col):
        x_plot, y_plot, z_plot, nums = [], [], [], []
        
    
        for _, row in df.iterrows():
            for rcol in row_cols:
                if pd.notna(row[rcol]):
                    z_plot.append(row[rcol])
                    x_plot.append(row[x_col])
                    y_plot.append(row[y_col])
                    if "corr_prop" in row_cols:
                        nums.append(row['corr_num'])
    
    
        return x_plot, y_plot, z_plot, nums
    
    # For three variable grid plots
    def build_grid_from_points(x_plot, y_plot, z_plot, x_unique, y_unique, fill_value, nums):
    
        x_map = {val: idx for idx, val in enumerate(x_unique)}
        y_map = {val: idx for idx, val in enumerate(y_unique)}
    
        # Total z axis
        Z = np.full((len(y_unique), len(x_unique)), fill_value, dtype=float)
        # Number of items in Z axis
        counts = np.zeros((len(y_unique), len(x_unique)), dtype=int)
        final_counts = np.zeros((len(y_unique), len(x_unique)), dtype=int)
    
        it = 0
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
            
            # If looking at species counts, add based on how many runs there were
            if len(nums) > 0 :
                final_counts[i,j] += nums[it]
            else :
                final_counts[i,j] += 1
                
            it += 1
            
        # Turn into mean
        Z_mean = np.divide(Z, counts, where=counts > 0)
        Z_mean[counts == 0] = fill_value
    
        return Z_mean, final_counts
    
    def build_truevspred(df, displaymax = 6) :
        runcols = [c for c in df.columns if c.startswith("run_") and not c.endswith('unique')]
        max_pred = min(displaymax, int(df[runcols].max().max()))
        max_true = min(displaymax, int(df["num_ecDNA_true"].max()))
        Z = np.zeros((max_pred, max_true))
        for idx, row in df.iterrows() :
            for colname in runcols :
                if pd.isna(row[colname]) or pd.isna(row["num_ecDNA_true"]):
                    continue         
                rowidx = min(displaymax, int(row[colname])) - 1
                colidx = min(displaymax, int(row["num_ecDNA_true"])) - 1

                Z[rowidx][colidx] += 1
        
        return Z
    
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
                if "run_" in col and 'unique' not in col : 
                    if not pd.isna(row[col]) :
                        total_num += 1
                        total_sum += row[col]
            
            if total_num > 0 :
                df_to_use.loc[rowidx, "mean"] = total_sum / total_num
                df_to_use.loc[rowidx, "num_runs"] = total_num
        return df_to_use
    
    # Determine which columns are actually getting graphed
    if x2 == "num_ecDNA_predicted" and x == "num_ecDNA_true" and y == "species_counts" :
        row_cols = ["TRUEVSPRED"]
    elif y == "species_counts":
        row_cols = ["corr_prop"]
    elif use_mean and x2 is None :
        df_to_use = get_mean(df_to_use)
        row_cols = ["mean"]
    else :
        row_cols = [col for col in df_to_use.columns if col.startswith("run_")] 
    
    
    method_names = df_to_use['mname'].unique()

    dfs = []
    titles = []
    for mname in method_names :
        dfs.append(df_to_use.loc[df_to_use["mname"] == mname])
        titles.append(mname)
    
    
    if x2 is None :
        print("Currently not supporting x2 == None")
        exit(0)
        # for df_name, method_name in [(hier, "Hier"), (cNMF, "cNMF")] :
        #     x_plot = []
        #     y_plot = []
        #     lab_plot = []
        #     lab_max = -1
    
        #     # Add non NaN values to plot
        #     if not use_mean : 
        #         for i, row in df_name.iterrows() :
        #             for rcol in row_cols :
        #                 if pd.isna(row[rcol]) == False :
        #                     y_plot.append(row[rcol])
        #                     x_plot.append(row[x])
        #         plt.scatter(x_plot, y_plot, label=method_name, color = colors[method_name], alpha=0.4)
        #         plt.legend()
    
    
        #     # Also include how many there are
        #     else :
        #         for i, row in df_name.iterrows() :
        #             for rcol in row_cols :
        #                 if pd.isna(row[rcol]) == False :
        #                     y_plot.append(row[rcol])
        #                     x_plot.append(row[x])
        #                     lab_plot.append(row["num_runs"])
        #                     lab_max = max(lab_max, row["num_runs"])
    
        #         # Sort lists for line plot
        #         combined = list(zip(x_plot, y_plot, lab_plot))
        #         combined_sorted = sorted(combined, key=lambda x: x[0])
        #         x_plot, y_plot, lab_plot = zip(*combined_sorted)
        #         plt.scatter(x_plot, y_plot, label=method_name, color = colors[method_name], alpha=0.4, s = 50 * np.array(lab_plot)/lab_max)
                
        #         # Size legend
        #         legend_vals = [lab_max * 0.25, lab_max * 0.5, lab_max * 0.75, lab_max]
                
        #         legend_sizes = [v * 50 / lab_max for v in legend_vals]
                
        #         method_handles = [
        #             plt.scatter([], [], color=colors[m], alpha=0.4, label=m)
        #             for m in method_names
        #         ]            
        #         handles = method_handles + [
        #             plt.scatter([], [], s=s, color='gray', alpha=0.4)
        #             for s in legend_sizes
        #         ]
                
        #         labels = method_names + [f"{int(v)}" for v in legend_vals]
                
        #         plt.legend(handles, labels, title="Number of Runs")
    
      
        # plt.xlabel(label_dict[x])
        # plt.ylabel(label_dict[y])
        # if plot_title is None :
        #     plt.title(f"{label_dict[y]} cNMF vs Hierarchical" , fontsize=16)
        # else :
        #     plt.title(plot_title, fontsize = 16)
    
    # Grid plot             
    else :

        # Get axes
        n_plots = len(dfs)
        ncols = min(3, n_plots)
        if n_plots % 2 == 0 and n_plots % 3 != 0 :
            ncols = 2
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 5 * nrows),
            sharex=True,
            sharey=True
        )

        axes = np.atleast_1d(axes).ravel()

        all_Z = []
        all_counts = []

        all_x = []
        all_y = []

        if 'TRUEVSPRED' not in row_cols :

            # Get the axis buckets
            for df in dfs:
                xv, yv, zv, nums = extract_points(df, row_cols, x, x2)

                all_x.append(xv)
                all_y.append(yv)

            x_unique = np.unique(np.concatenate(all_x))
            y_unique = np.unique(np.concatenate(all_y))

            # Build Z
            for df in dfs:
                xv, yv, zv, nums = extract_points(df, row_cols, x, x2)

                Z, counts = build_grid_from_points(
                    xv, yv, zv,
                    x_unique,
                    y_unique,
                    -1,
                    nums
                )

                all_Z.append(Z)
                all_counts.append(counts)


        else :
            rows = 0
            cols = 0
            # First run to get the borders
            for df in dfs :
                Z = build_truevspred(df)

                rows = max(rows, Z.shape[0])
                cols = max(cols, Z.shape[1])

            # now fill in the actual with padding
            for df in dfs :
                Z = build_truevspred(df)

                Z_pad = np.zeros((rows, cols), dtype=int)
                Z_pad[:Z.shape[0], :Z.shape[1]] = Z

                all_Z.append(Z_pad)
                all_counts.append(None)

                x_unique = range(1, cols + 1)
                y_unique = range(1, rows + 1)

    
        # Allow some wiggle room in color (otherwise breaks when all are perfect)
        vmin = min(Z.min() for Z in all_Z) - 0.001
        vmax = max(Z.max() for Z in all_Z) + 0.001


        for ax, Z, counts, title in zip(axes, all_Z, all_counts, titles):

            im = ax.imshow(
                Z,
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(title)

            ax.set_xticks(range(len(x_unique)))
            ax.set_xticklabels(x_unique)

            ax.set_yticks(range(len(y_unique)))
            ax.set_yticklabels(y_unique)

            # add labels
            if "TRUEVSPRED" not in row_cols:
                for i in range(len(y_unique)):
                    for j in range(len(x_unique)):
                        if counts[i, j] > 0:
                            ax.text(
                                j - 0.3, i + 0.3,
                                f"{Z[i,j]:.2f}",
                                ha="left", va="top",
                                color="white",
                                fontsize=10,
                                fontweight="bold"
                            )

                            ax.text(
                                j + 0.3, i - 0.3,
                                f"n={counts[i,j]}",
                                ha="right", va="bottom",
                                color="white",
                                fontsize=7
                            )
            else:
                for i in range(len(y_unique)):
                    for j in range(len(x_unique)):
                        ax.text(
                            j - 0.3,
                            i + 0.3,
                            f"{Z[i,j]}",
                            ha="left",
                            va="top",
                            color="white",
                            fontsize=10,
                            fontweight="bold"
                        )
        
        # Hide unused axes (like if there are 5)
        for ax in axes[n_plots:]:
            ax.set_visible(False)

        # Common labels
        for ax in axes[:n_plots]:
            ax.set_xlabel(label_dict[x])

        for ax in axes[::ncols]:
            ax.set_ylabel(label_dict[x2])

        fig.subplots_adjust(right=0.88)

        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Value")


        if plot_title is None:
            fig.suptitle(
                f"{label_dict[y]} Comparison",
                fontsize=16
            )
        else:
            fig.suptitle(
                plot_title,
                fontsize=16
            )
            
    print(f"Outputting to {plot_name}")
    plt.savefig(plot_name)

def get_dfs(run_out_dir, folder_to_name, use_only_correct_species = True, use_only_correct_jaccard = True) :

    count_err_dfs = []
    jaccard_dfs = []
    species_counts_dfs = []
    wrong_species = set()
    wrong_jaccard = set()

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
    cNMF_log = {}
    
    # Iterate through all results
    for folder_name in sorted_keys:
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
    
            count_err_df = pd.read_csv(f"{inner_dir}/count_err.tsv", sep = '\t')
            jaccard_df = pd.read_csv(f"{inner_dir}/jaccard.tsv", sep = '\t')
            species_counts_df = pd.read_csv(f"{inner_dir}/species_counts.tsv", sep = '\t')
            
    
            # Add metadata
            for name, df in [("sc", species_counts_df), ("jac", jaccard_df), ("ce", count_err_df)] :

                df['method'] = method
                df['val'] = val
                df['mname'] = folder_to_name[folder_name]
                df['countprov'] = countprov
                df['fmax'] = fmax
                df['overlap'] = overlap
                df['extra_counts'] = extra_counts
                df['depth'] = depth

                # Get wrong species data and populate the correlation
                if name == "sc" :
                    df['corr_prop'] = 0.0
                    df['corr_num'] = 0
                    for row_idx, row in df.iterrows() :
                        wrong_count = 0
                        num_runs = 0
                        for col_idx, col in enumerate(df.columns) :
                            if "run_" in col and 'unique' not in col :
                                
                                # For combo, remove later
                                if method == "cNMF" :
                                    cNMF_log[("sc", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)] = row[col]
                                if method == "combo" and row[col] == -1 :
                                    
                                    df.iloc[row_idx, col_idx] = cNMF_log[("sc", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)]
                                    row[col] = cNMF_log[("sc", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)]


                                num_runs += 1
                                if row['num_ecDNA_true'] != row[col] :
                                    wrong_count += 1
                                    wrong_species.add((method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col))
                                    
                        df.loc[row_idx, 'corr_num'] = num_runs
                        df.loc[row_idx, 'corr_prop'] = (num_runs - wrong_count)/num_runs
                
                elif name == "jac" :
                    # Get wrong jaccard info
                    # Note: currently only supports average jaccard rather than a binary right / wrong
                    for row_idx, row in df.iterrows() :
                        for col_idx, col in enumerate(df.columns) :
                            if "run_" in col and 'unique' not in col :
                                # For combo, remove later
                                if method == "cNMF" :
                                    cNMF_log[("jac", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)] = row[col]
                                if method == "combo" and row[col] == -1 :
                                    
                                    df.iloc[row_idx, col_idx] = cNMF_log[("jac", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)]

                                    row[col] = cNMF_log[("jac", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)]


                                if row[col] < 1 :
                                    wrong_jaccard.add((method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col))
                                # if in wrong species, remove
                                if use_only_correct_species and (method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) in wrong_species :
                                    df.loc[row_idx, col] = np.nan

                                
                
                elif name == "ce" :
                    # Get wrong jaccard info
                    for row_idx, row in df.iterrows() :
                        for col_idx, col in enumerate(df.columns) :
                            if "run_" in col and 'unique' not in col :

                                # For combo, remove later
                                if method == "cNMF" :
                                    cNMF_log[("ce", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)] = row[col]
                                if method == "combo" and row[col] == -1 :
                                    df.iloc[row_idx, col_idx] = cNMF_log[("ce", fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col)]


                                # remove based on wrong species and jaccard
                                if use_only_correct_species and (method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) in wrong_species :
                                    df.loc[row_idx, col] = np.nan
                                if use_only_correct_jaccard and (method, val, fmax, overlap, extra_counts, depth, row['num_ecDNA_true'], row['comb_chance'], col) in wrong_jaccard :
                                    df.loc[row_idx, col] = np.nan

    
            count_err_dfs.append(count_err_df)
            jaccard_dfs.append(jaccard_df)
            species_counts_dfs.append(species_counts_df)
    
    count_err_full_df = pd.concat(count_err_dfs, ignore_index=True)
    jaccard_full_df = pd.concat(jaccard_dfs, ignore_index=True)
    species_counts_full_df = pd.concat(species_counts_dfs, ignore_index=True)

    return count_err_full_df, jaccard_full_df, species_counts_full_df

ce, j, sc = get_dfs(run_out_dir, folder_to_name)

for args in settings :
    graph_results(run_out_dir, plot_out_dir, ce, j, sc, **args)

