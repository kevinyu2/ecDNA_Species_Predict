# Species Simulation and Testing

This directory contains scripts used to create ecDNA species simulations
and test/validate species deconvolution methods. 
NOT meant to be used outside of methods development.

To use, download https://github.com/kevinyu2/Cassiopeia_ecDNA_sim first

## Files


### Simulation

```atacSim.py``` : class that builds onto cassiopeia simulation to generate resulting scATAC-seq data

```pipeline_v2.py``` : runs data simulation. Do not use ```pipeline.py```, this has since been depreciated

```bash
python $SCRIPT $OUT_DIR \
    --runs 5 \
    --species-max 5 \
    --overlap-prop $OVERLAP_PROP \
    --depth $DEPTH_VALUE  \
    --coseg-type simulation \
    --sim-mult 1.4 \
    --const-comb \
    --test-coseg \
    --total-genes 20

### Deconvolution

```hierarchical2.py``` : calls the hierarchical method on the data created. Do not use ```hierarchical.py```, this has since been depreciated

```naive_hier.py``` : calls the naive hierarchical method on the data created

```cNMF_pipeline.py``` : calls the cNMF method on the data created

```bash
python your_script.py \
    "$INPUT_DIR" \
    "$OUT_DIR" \
    --errorw 0.25


```combo.py``` : calls the combo method on the data created

### Plotting

```graph_results.py``` : main heatmaps

```graph_runtime.py``` : plots runtimes of deconvolution methods

### Extra

```species_deconvolution.py``` : full list of methods to use beyond the sim. 
implemented in https://github.com/JonesCompBioLab/scamp

```coseg_strength_corr.py``` : plot correlation vs cosegregation strength of simulation


