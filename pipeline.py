import numpy as np
import pandas as pd
from atacSim import atacDataSimulation

sim = atacDataSimulation(
    out_dir = '../species_out', run_name = 'run2',
    initial_copy_number_array = np.array([8,8,8]),
    fitness_array = np.array([
        [
            [0.0, 0.1],
            [0.1, 0.3],
        ],
        [
            [0.1, 0.2],
            [0.2, 0.3],
        ],
    ]),
    cosegregation_type = "venn",
    gene_counts = [10,10,10],
    multinomial_mult = 1,
    noise_scale = 1,
    gene_overlap = {(0,1) : 3},
    chance_to_change = 0.1,
    change_distribution = np.random.geometric,
    change_distribution_param = 0.8,
    initial_birth_scale = 0.5,
    death_waiting_distribution = np.random.exponential(7),
    num_extant = 20000,
    num_cells = 1000,
    coeffs = {
        (0,2) : 0.4,
        (0,1) : 0.25,
        (0,1,2) : 0.3}
    
    )

sim.run_sim()