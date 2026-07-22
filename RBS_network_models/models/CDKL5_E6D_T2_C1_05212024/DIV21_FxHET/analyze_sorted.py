'''
This script uses the extract_network_features module to perform network analyis on a targeted raw data file and then extract features from the network to later pass to batch simulations.
'''
# Notes: =====================================================================
'''
    # before # aw 2025-02-26 11:52:29
    # - NOTE: this works...on login node in NERSC.
    # -- But I think I remember that it didn't work on local machine. laptop.
    # -- TODO: test on local machine.
    # - [about raw_data_paths list of paths]
    # -- NOTE: this is a list of paths to raw data files that you want to extract features from this is useful for batch processing.
    # -- NOTE: Also, if parent dirs are provided, each path will be searched recursively for .h5 files
'''

# Imports =====================================================================
import os
from RBS_network_models import extract_features as ef
from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.conv_params import conv_params, mega_params
# /RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/conv_params.py
# Paths =============================================================================
sorted_data_dirs = [
    # '/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/sorted/well005',
    #'/global/homes/a/adammwea/pscratch/z_analyzed_data/CDKL5-E6D_T2_C1_05212024/CDKL5-E6D_T2_C1_05212024/240611/M08029/Network/000091/sorted/well001',
    '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/sorted/well000',
    # '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/sorted/well001',
    # '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/sorted/well002',
    # '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/sorted/well003',
    # '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/sorted/well004',
    # '/pscratch/sd/k/ktub1999/networkSimulations/z_analyzed_data/sorted/well005'

                    ] # NOTE: this is a list of paths to sorted data files that you want to extract features from.
output_dirs = [sorted_dir.replace('sorted', 'network_analysis') for sorted_dir in sorted_data_dirs
               ] # NOTE: this is a list of output directories for each network analysis of each sorted data file.

# Parallelism =============================================================================
'''check available cores'''
print("Number of cores available: ", os.cpu_count())
#max_workers = 128 # aw 2025-02-24 04:07:33 - I got an odd error trying to use 256 cores... just going to use 128 for now.

# Main =============================================================================
'''main'''
kwargs = {
    'sorted_data_dirs': sorted_data_dirs,
    'output_dirs': output_dirs,
    'conv_params': conv_params,
    'mega_params': mega_params,
    # 'max_workers': 32, # 1/4 node
    #'max_workers': 64, # full node
    'max_workers': os.cpu_count(), # use all available cores
    #'max_workers': max_workers, # use all available cores
    #'max_workers': 100, # number of parallel processes to use
    'max_duration': 20, # Specify some limit in seconds to only plot a portion of the data
    'plot_wfs': False, # plot waveforms while classifying neurons
    #'plot_wfs': True, # plot waveforms while classifying neurons
    #'debug_mode': True, #default is False, set to True to reduce units and bursts processed for quicker debugging
}

feature_data = ef.analyze_network_data(**kwargs)
print("Network Analysis Complete.")

# Perlmutter =============================================================================
'''
#run in interactive node
salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=1 --image=adammwea/axonkilo_docker:v7
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
python /global/homes/a/adammwea/workspace/repos/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_WT/analyze_sorted.py
'''