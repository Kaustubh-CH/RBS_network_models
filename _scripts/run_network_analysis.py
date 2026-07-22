''' Analyze all _data.pkl or _data.json files in a target directory recursively.
    - .npy files are saved in the same directory as the _data.pkl file.
'''
from RBS_network_models.utils.network_analysis import *

# OPTIONS = [
#     1, # new, under development
#     2 # old, but working
#     ]

selected_option = 1
if selected_option == 1:
    # imports ===================================================================================================
    #TARGET_DIR = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_FxHET/250702/sensitivity_analysis/run0000'
    TARGET_DIR = '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/sorted/well000'
    CONV_PATH = '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/conv_params.py'

    kwargs={
        # search parameters
        'target_dir': TARGET_DIR,
        'exclude': ['_archive',], # exclude any paths that contain these strings
        #'source': 'simulated',  # 'simulated' or 'experimental'
        'source': 'experimental',  
        # convolution params
        'conv_path': CONV_PATH,  # path to the conversion parameters file
        
        # analysis options
        #'compute_spike_metrics': True,  # whether to compute spike metrics
        #'compute_burst_metrics': True,  # whether to compute burst metrics
        #'classify_units': True,  # whether to classify units
        #'compute_dynamic_time_warping': False,  # whether to compute dynamic time war
        #'compute_summary_metrics': True,  # whether to compute summary metrics
        
        # fitness options
        # 'compute_fitness': False,  # whether to compute fitness
        # 'fitness_schema': None,  # fitness schema to use
        # 'reference_data_path': None,  # path to the reference data file for fitness computation
        
        # runtime
        #'parallel': True,  # whether to run in parallel
        'parallel': False,  # while debugging, set to False to run sequentially
        #'procs': 16,  # number of processes to use for parallel processing
        'procs': 256, # number of processes to use for parallel processing - 1 full node
        
        #'mpi': True,  # whether to use MPI for parallel processing
        #'slurm': True,  # whether to use SLURM for parallel processing
        
        # verbosity
        #'verbose': True,  # whether to print verbose output
        
    }

    if __name__ == "__main__":
        run_network_analysis(**kwargs)
        # Note: This script is intended to be run as a standalone script.
        # If you are running this in a Jupyter notebook or an interactive environment,
        # you may want to comment out the following line to prevent the script from exiting.
        

# elif selected_option == 2:
#     from netpyne import sim
#     import os
#     from datetime import datetime as dt
#     import json
#     from RBS_network_models.sim_analysis import process_simulation_v3
#     from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_WT.src.conv_params import conv_params, mega_params
#     #from RBS_network_models.sensitivity_analysis import prepare_permuted_sim_v2
#     # Notes ===================================================================================================

#     reference_data_path = (
#         '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/network_analysis/well005/metrics.npy'
#         )
#     sim_data_path = '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/sensitivity_analyses/2025-03-13_propVelocity_3_data_65s/_propVelocity_3/_propVelocity_3_data.pkl'
#     sim.load(sim_data_path) # load simulation data
#     saveFolder = sim.cfg.saveFolder # get save folder from sim.cfg

#     kwargs = {
#         'sim_data_path': sim_data_path,
#         'reference_data_path': reference_data_path,
#         'conv_params': conv_params,
#         'mega_params': mega_params,
#         'simData': sim.allSimData,
#         'popData': sim.net.allPops,
#         'cellData': sim.net.allCells,
#         'output_dir': saveFolder,
#         #'fitnessFuncArgs': fitnessFuncArgs,
#         'DEBUG_MODE': False,
#     }

#     process_simulation_v3(kwargs)

