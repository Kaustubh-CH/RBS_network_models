import matplotlib; matplotlib.use('Agg')
import os
from netpyne import sim
import json
import numpy as np
#===================================================================================================
try:
    # Get and test MPI rank
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    rank = mpi_rank
    print("Initiating Rank:", rank) 
except ImportError:
    print("mpi4py not found, running in serial mode.")
    rank = 0
    mpi_size = 1

#===================================================================================================

#set paths to simulation configuration files
simCfg = 'cfg.py'
netParams = 'netParams.py'

# initialize simConfig and netParams objects
simConfig, netParams = sim.readCmdLineArgs(
    simConfigDefault=simCfg,
    netParamsDefault=netParams,
)

# Run simulation
sim.createSimulateAnalyze(simConfig = simConfig, netParams = netParams)

# # After simulation, create spike_times_by_unit and save it
# if rank == 0:  # Only do this on the master rank
#     try:
#         # Get the saved data path
#         if hasattr(sim.cfg, 'saveFolder') and hasattr(sim.cfg, 'simLabel'):
#             data_file = os.path.join(sim.cfg.saveFolder, sim.cfg.simLabel + '_data.pkl')
            
#             # Load the saved data
#             import pickle
#             with open(data_file, 'rb') as f:
#                 data = pickle.load(f)
            
#             # Extract spike data
#             if 'simData' in data and 'spkt' in data['simData'] and 'spkid' in data['simData']:
#                 spike_times = np.array(data['simData']['spkt']) / 1000  # Convert to seconds
#                 spike_ids = np.array(data['simData']['spkid'])
                
#                 # Create spike_times_by_unit dictionary
#                 spike_times_by_unit = {}
#                 for unit_id in np.unique(spike_ids):
#                     spike_times_by_unit[int(unit_id)] = spike_times[spike_ids == unit_id]
                
#                 # Add spike_times_by_unit and sampling_rate to simData
#                 data['simData']['spike_times_by_unit'] = spike_times_by_unit
#                 data['simData']['spike_times'] = spike_times
                
#                 # Add sampling_rate - for simulated data, this is computed from dt
#                 # sampling_rate = 1000.0 / dt (where dt is in ms)
#                 if hasattr(sim.cfg, 'dt'):
#                     sampling_rate = 1000.0 / sim.cfg.dt  # Convert from ms to Hz
#                 else:
#                     sampling_rate = 'Not available - dt not found in simConfig'
                
#                 data['simData']['sampling_rate'] = sampling_rate
                
#                 # Save back to file
#                 with open(data_file, 'wb') as f:
#                     pickle.dump(data, f)
                
#                 print(f"Successfully added spike_times_by_unit and sampling_rate to {data_file}")
#                 if isinstance(sampling_rate, (int, float)):
#                     print(f"Sampling rate: {sampling_rate} Hz")
#             else:
#                 print("Warning: No spike data found in simData")
#     except Exception as e:
#         print(f"Error adding spike_times_by_unit and sampling_rate: {e}")
#         import traceback
#         traceback.print_exc()