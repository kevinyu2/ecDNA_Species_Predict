# Species Simulation and Testing

This directory contains scripts used to create ecDNA species simulations
and test/validate species deconvolution methods. 
NOT meant to be used outside of methods development.

To use, download https://github.com/kevinyu2/Cassiopeia_ecDNA_sim first

## Files


### Simulation

```atacSim.py``` : class that builds onto cassiopeia simulation to generate resulting scATAC-seq data

```pipeline_v2.py``` : runs data simulation. Do not use ```pipeline.py```, this has since been depreciated


### Deconvolution

```hierarchical2.py``` : calls the hierarchical method on the data created. Do not use ```hierarchical.py```, this has since been depreciated

```naive_hier.py``` : calls the naive hierarchical method on the data created

```cNMF_pipeline.py``` : calls the cNMF method on the data created

```combo.py``` : calls the combo method on the data created

### Plotting

```graph_results.py``` : main heatmaps

```graph_runtime.py``` : plots runtimes of deconvolution methods

### Extra

```species_deconvolution.py``` : full list of methods to use beyond the sim. 
implemented in https://github.com/JonesCompBioLab/scamp

```coseg_strength_corr.py``` : plot correlation vs cosegregation strength of simulation


## Examples

### Simulation


```bash
python pipeline_v2.py $OUT_DIR \
    --runs 5 \
    --species-max 5 \
    --overlap-prop $OVERLAP_PROP \
    --depth $DEPTH_VALUE  \
    --coseg-type simulation \
    --sim-mult 1.4 \
    --const-comb \
    --test-coseg \
    --total-genes 20
```

For 5 runs of 1-5 species, ```OVERLAP_PROP``` overlap, ```DEPTH_VALUE``` depth (usually 0.25-2), simulation cosegregation (recommended), 20 total genes.
```const-comb``` means non-varying cosegregation values (use unless testing correlation as a function of cosegregation)
```--test-coseg``` means only having two species cosegregate. If not, lower levels of cosegregation are suggested

### Deconvolution

```bash
python (cNMF_pipeline.py, combo.py, hierarchical2.py, OR naive_hier.py) \
    "$INPUT_DIR" \
    "$OUT_DIR" 
```
Input should be the result of simulation, i.e. ```./fmax_0.1_overlap_0.4_extracounts_0.1_depth_1.5```

### Plotting

Edit the following lines in ```graph_results.py``` :


```run_out_dir = [TODO]
plot_out_dir = [TODO]

# What to title these in the plots!
folder_to_name = {
    "cNMF_results_countprov_0_errorw_0.25" : "cNMF (errorw = 0.25)",
    "combo_results_countprov_0_thresh_0.55" : "Combo (threshold = 0.55)",
    "naive_results_countprov_0_thresh_0.75" : "Naive (threshold = 0.75)",
    "hier_results_countprov_0_ddist_1.3" : "Hier (dummy weight = 1.3)"
}
```

```run_out_dir``` is the ```OUT_DIR``` from the deconvolution scripts. This script
will plot all results in this folder. ```plot_out_dir``` is where the plots are saved.
```folder_to_name``` should be edited, with keys denoting which folders within ```OUT_DIR``` to use,
and values denoting the label of that method in the plots.