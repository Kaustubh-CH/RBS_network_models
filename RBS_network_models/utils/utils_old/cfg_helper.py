from netpyne import specs
import random
import os
import json

'''Helper functions'''
import inspect
#from DIV21.utils.sim_helper import import_module_from_path
#from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import import_module_from_path

import importlib
def import_module_from_path(module_path):
    #module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("module.name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def handle_params_import(path=None, cfg=None):
    '''Handle the import of the evolutionary parameter space
    
    Note: This function is necessary because the evolutionary parameter space 
    is handled differently when running simulations directly or from a batchRun script.
    '''
    print('Attempting to import evolutionary parameters...')
    try:
        from __main__ import params  # Import evolutionary parameters from main i.e. batchRun script
        print('Evolutionary parameters imported successfully.')
    except ImportError:
        import warnings
        warnings.simplefilter('always')
        
        print('Could not import params from __main__. Attempting to import from path...')
        # warning, importing params in this way will import the evolutionary parameter space,
        # importantly, it won't be importing the params selected by evol config.
        # this is normal if you are running a single simulation, or otherwise testing the cfg script
        # but if you are running a batch simulation, you should be importing the params from __main__
        
        warning_message = ('\n'
                           'Importing params from path. '
                           'This is normal if you are running a single simulation, '
                           'or otherwise testing the cfg script. If you are running '
                           'a batch simulation, you should be '
                           'importing the params from __main__')
        warnings.warn(warning_message)
        
        assert path is not None, 'Path to evolutionary parameter space not provided.'
        assert cfg is not None, 'cfg object not provided.'
        
        params = import_module_from_path(path)
        params = params.params
        
        # cycle through params, if any are ranges of values, randomly select one between the range
        print('Randomizing parameters within specified ranges...')
        for key, value in params.items():
            #check if list with 2 values
            if isinstance(value, list) and len(value) == 2:
                #check if range
                if value[0] < value[1]:
                    params[key] = random.uniform(value[0], value[1])
                
        print('Adding params to cfg object...')
        for param in params:
            setattr(cfg, param, params[param]) 
        
        print('Evolutionary parameters imported successfully.')
    return params, cfg
            
  
    #     # from RBS_network_simulations.modules.simulation_config.evolutionary_parameter_space import params
	# 	# from RBS_network_simulations.optimization_scripts.batchRun_evol import rangify_params # use rangify_params from batchRun script to treat params as if it were imported from batchRun script
	# 	# params = rangify_params(params)
	# return params

def get_cell_numbers_from_fitness_target_script(target_script_path=None):
    fitnessFuncArgs = import_module_from_path(target_script_path)
    fitnessFuncArgs = fitnessFuncArgs.fitnessFuncArgs
    
    num_excite = fitnessFuncArgs['features']['num_excite']
    num_inhib = fitnessFuncArgs['features']['num_inhib']
    
    return num_excite, num_inhib

    # depreacated methods to import the fitness function arguments - jic I need reference
        #from RBS_network_simulations.optimization_scripts.batchRun_evol import import_module_from_path
        #from workspace.RBS_network_simulations.workspace.optimization_projects.CDKL5_DIV21_dep._2_batchrun_optimization.batch_run_helper import import_module_from_path
        #from workspace.RBS_network_simulations._archive.temp_user_args import USER_fitness_target_script
        #from optimization_projects.CDKL5_DIV21.scripts.batch_scripts.temp_user_args import USER_fitness_target_script 

        #import_module_from_path(USER_fitness_target_script, 'fitnessFuncArgs') #dynamically import fitnessFuncArgs from USER_fitness_target_script defined as python scripts so that we can optimize different data

def print_cfg_options():
    print(
        '''Simulation configuration options:

        'simcfg': {
            'duration': 1000.0,  # Total simulation time in milliseconds (ms).
            'tstop': 1000.0,  # End time of the simulation in ms; typically set equal to 'duration'.
            'dt': 0.025,  # Integration time step in ms, determining the resolution of the simulation.
            'hParams': {  # Parameters for NEURON's 'h' module:
                'celsius': 6.3,  # Temperature in degrees Celsius for the simulation.
                'v_init': -65.0,  # Initial membrane potential in millivolts (mV).
                'clamp_resist': 0.001  # Resistance value for voltage clamps.
            },
            'coreneuron': False,  # If True, uses CoreNEURON for faster simulations; False uses standard NEURON.
            'dump_coreneuron_model': False,  # If True, exports the model in a format compatible with CoreNEURON.
            'random123': False,  # Enables the Random123 random number generator for reproducible random sequences.
            'cache_efficient': False,  # Activates CVode's 'cache_efficient' option to optimize performance on multiple cores.
            'gpu': False,  # Enables GPU execution in CoreNEURON if set to True.
            'cvode_active': False,  # Activates the variable time step integration method CVode when True.
            'use_fast_imem': False,  # Utilizes CVode's 'fast_imem' to record membrane currents efficiently when True.
            'cvode_atol': 0.001,  # Sets the absolute error tolerance for CVode integration.
            'seeds': {  # Seeds for randomization to ensure reproducibility:
                'conn': 1,  # Seed for network connectivity.
                'stim': 1,  # Seed for input stimulation.
                'loc': 1,  # Seed for cell locations.
                'cell': 1  # Seed for cell and synaptic mechanism parameters.
            },
            'rand123GlobalIndex': None,  # Global index for all instances of Random123; None if not specified.
            'createNEURONObj': True,  # If True, creates a runnable network in NEURON during instantiation.
            'createPyStruct': True,  # If True, creates a simulator-independent Python structure of the network.
            'addSynMechs': True,  # Adds synaptic mechanisms to the model when True.
            'includeParamsLabel': True,  # Includes labels of parameter rules that created cells, connections, or stimulations.
            'gatherOnlySimData': False,  # If True, gathers only simulation data, omitting network and cell data to reduce gathering time.
            'compactConnFormat': False,  # Uses a compact list format for connections instead of dictionaries when True.
            'connRandomSecFromList': True,  # Selects a random section (and location) from a list even when 'synsPerConn=1' if True.
            'distributeSynsUniformly': True,  # Distributes synapses uniformly across sections when True; otherwise, places one synapse per section.
            'pt3dRelativeToCellLocation': True,  # Makes cell 3D points relative to the cell's x, y, z location when True.
            'invertedYCoord': True,  # Inverts the y-axis coordinate to represent depth (0 at the top) when True.
            'allowSelfConns': False,  # Permits connections from a cell to itself if True.
            'allowConnsWithWeight0': True,  # Allows connections with zero weight when True.
            'oneSynPerNetcon': True,  # Creates one individual synapse object for each NetCon when True; otherwise, the same synapse can be shared.
            'saveCellSecs': True,  # Saves all section information for each cell when True; setting to False reduces time and space but prevents re-simulation.
            'saveCellConns': True,  # Saves all connection information for each cell when True; setting to False reduces time and space but prevents re-simulation.
            'timing': True,  # Displays timing information for each process when True.
            'saveTiming': False,  # Saves timing data to a pickle file if True.
            'printRunTime': False,  # Prints the run time at specified intervals (in seconds) when set to a positive value.
            'printPopAvgRates': False,  # Prints average firing rates for populations after the run when True.
            'printSynsAfterRule': False,  # Prints the total number of connections after each connection rule is applied when True.
            'verbose': False,  # Displays detailed messages during simulation when True.
            'progressBar': 2,  # Controls the display of the progress bar: 0: no progress bar; 1: progress bar without leaving the previous output; 2: progress bar leaving the previous output.
            'recordCells': [],  # Specifies which cells to record traces from (e.g., 'all', a specific cell ID, or a population label).
            'recordTraces': {},  # Dictionary defining which traces to record.
            'recordCellsSpikes': -1,  # Indicates which cells to record spike times from (-1 to record from all cells).
            'recordStim': False,  # Records spikes of cell stimulations when True.
            'recordLFP': [],  # List of 3D locations to record local field potentials (LFP).
            'recordDipole': False,  # Records dipoles using the LFPykit method when True.
            'recordDipolesHNN': False,  # Records dipoles using the HNN method when True.
            'saveLFPCells': False,  # Stores LFP generated individually by each cell when True.
            'saveLFPPops': False,  # Stores LFP generated individually by each population when True.
            'saveDipoleCells': False,  # Stores dipole generated individually by each cell when True.
            'saveDipolePops': False,  # Stores dipole generated individually by each population when True.
            'recordStep': 0.1,  # Step size in ms to save data (e.g., voltage traces, LFP, etc.).
            'recordTime': True,  # Records time step of recording when True.
            'simLabel': '',  # Name of the simulation (used as filename if none provided).
            'saveFolder': '',  # Path where to save output data.
            'filename': 'model_output',  # Name of the file to save model output (if omitted, then 'saveFolder' + 'simLabel' is used).
            'saveDataInclude': ['netParams', 'netCells', 'netPops', 'simConfig', 'simData'],  # Specifies which data to include when saving.
            'timestampFilename': False,  # Adds a timestamp to the filename to avoid overwriting when True.
            'savePickle': False, # Save params, network and sim output to pickle file
            'saveJson': False,  # Save data in JSON format.
            'saveMat': False,  # Save data in .mat format.
            'saveCSV': False,  # Save data in .csv format.
            'saveDpk': False,  # Save data in .dpk format.
            'saveHDF5': False,  # Save data in .hdf5 format.
            'saveDat': False,  # Save data in .dat format.
            'backupCfgFile': None,  # Path to backup configuration file.
            'validateNetParams': False,  # Validate network parameters when True.
            'analysis': {}  # Dictionary of analysis functions to run.
        }
        '''
            

        '''all possible options for simcfg, copied during batch evol optimization
        'simcfg' : {
            'duration': 1000.0, 
            'tstop': 1000.0, 
            'dt': 0.025, 
            'hParams': {
                celsius: 6.3, 
                v_init: -65.0, 
                clamp_resist: 0.001
                }, 
            'coreneuron': False, 
            'dump_coreneuron_model': False, 
            'random123': False, 
            'cache_efficient': False, 
            'gpu': False, 
            'cvode_active': False, 
            'use_fast_imem': False, 
            'cvode_atol': 0.001, 
            'seeds': {
                conn: 1, 
                stim: 1, 
                loc: 1, 
                cell: 1
                }, 
            'rand123GlobalIndex': None, 
            'createNEURONObj': True, 
            'createPyStruct': True, 
            'addSynMechs': True, 
            'includeParamsLabel': True, 
            'gatherOnlySimData': False, 
            'compactConnFormat': False, 
            'connRandomSecFromList': True, 
            'distributeSynsUniformly': True, 
            'pt3dRelativeToCellLocation': True, 
            'invertedYCoord': True, 
            'allowSelfConns': False, 
            'allowConnsWithWeight0': True, 
            'oneSynPerNetcon': True, 
            'saveCellSecs': True, 
            'saveCellConns': True, 
            'timing': True, 
            'saveTiming': False, 
            'printRunTime': False, 
            'printPopAvgRates': False, 
            'printSynsAfterRule': False, 
            'verbose': False, 
            'progressBar': 2, 
            'recordCells': [], 
            'recordTraces': {}, 
            'recordCellsSpikes': -1, 
            'recordStim': False, 
            'recordLFP': [], 
            'recordDipole': False, 
            'recordDipolesHNN': False, 
            'saveLFPCells': False, 
            'saveLFPPops': False, 
            'saveDipoleCells': False, 
            'saveDipolePops': False, 
            'recordStep': 0.1, 
            'recordTime': True, 
            'simLabel': '', 'saveFolder': '', 
            'filename': 'model_output', 
            'saveDataInclude': [
                'netParams', 'netCells', 'netPops', 'simConfig', 'simData'], 
            'timestampFilename': False, 
            'savePickle': False, 
            'saveJson': False, 
            'saveMat': False, 
            'saveCSV': False, 
            'saveDpk': False,
            'saveHDF5': False,
            'saveDat': False,
            'backupCfgFile': None,
            'validateNetParams': False,
            'analysis': {}, 
        }
        '''
    )
    