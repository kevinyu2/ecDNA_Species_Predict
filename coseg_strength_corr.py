import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import ast
from collections import defaultdict

#############################################################################


sim_dir = "/orcd/data/ki/001/lab/jones/kyu06/cass_data/venn_sim"
plot_out_dir = "/orcd/data/ki/001/lab/jones/kyu06/cass_data/five_species_figs/"

species_max = 5

##############################################################################


# Tracks number of species each time
species_counts = []
# Our y axis
correlations = []
# X axis
strengths = []

# Automatically Detects Mode
mode = "unknown"

for result_dir in Path(sim_dir).glob("*") :
    print(result_dir)

    # These are named by species number and comb chance
    for inner_dir in result_dir.glob("*") :

        species_num, _, comb_chance, _ = inner_dir.name.split('_')
        species_num = int(species_num)
        comb_chance = float(comb_chance)

        # Doesn't make sense for species number = 1
        if species_num > 1 and species_num <= species_max :

            # These are named by run number
            for metadata_file in inner_dir.glob("*metadata.txt") :
                
                with open(metadata_file, "r") as metaf :

                    # Tracker to know when to do venn
                    nextline_venn = False
                    nextline_mat = False
                    venn_dict = {}
                    matrix = []
                    
                    curr_row = []

                    for line in metaf :
                        if nextline_venn :
                            start = line.find('{')
                            end = line.rfind('}') + 1
                            dict_part = line[start:end]

                            # Convert to actual dict
                            data = ast.literal_eval(dict_part)
                            venn_dict = defaultdict(float, data)

                            nextline_venn = False
                        
                        if nextline_mat :
                            # End of the line
                            if "Species capacity" in line :
                                nextline_mat = False
                            else :
                                old_line = line.rstrip()
                                
                                line = line.replace('[', '').replace(']', '')
                                line = line.strip()
                                for x in line.split() :
                                    curr_row.append(float(x))
                                    

                                if old_line.endswith(']') :
                                    
                                    matrix.append(curr_row)
                                    curr_row = []




                        # Add the actual results
                        if "Correlation between" in line :
                            species_1 = int(line.split("ecDNA_")[1].split(" ")[0])
                            species_2 = int(line.split("ecDNA_")[2].split(":")[0])

                            correlation = float(line.split(' ')[-1])
                            cosegregation = 0
                            # Venn mode
                            if len(venn_dict) > 0 :
                                for key, value in venn_dict.items() :
                                    if species_1 in key and species_2 in key :
                                        cosegregation += value
                            
                            # Simulation mode
                            if matrix != []:
                                # print(species_1)
                                # print(species_2)
                                # print(matrix)
                                cosegregation = matrix[species_1][species_2]

                            species_counts.append(species_num)
                            correlations.append(correlation)
                            strengths.append(cosegregation)


                        # Lines signalling important lines
                        if 'Venn coefficients' in line :
                            mode = "Venn"
                            nextline_venn = True
                        if 'Chance matrix' in line :
                            mode = "Simulation"
                            nextline_mat = True
                            


# print(species_counts)
# print(correlations)
# print(strengths)

out_loc = f"{plot_out_dir}/coseg_correlations_{mode}.png"


plt.scatter(strengths, correlations, c = species_counts, cmap = "viridis", alpha = 0.3, s = 10)
plt.colorbar(label = "Number of ecDNA Species")
plt.xlabel("Cosegregation Strength")
plt.ylabel("Pairwise Correlations")
plt.title(f"Cosegregation Strength ({mode}) vs. Observed Correlations")

plt.savefig(out_loc)
print(f"Saved to {out_loc}")