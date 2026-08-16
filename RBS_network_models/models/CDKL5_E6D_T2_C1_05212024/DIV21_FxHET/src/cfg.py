from netpyne import specs
from RBS_network_models.utils.utils_old.cfg_helper import import_module_from_path
import os
import numpy as np
import random
from pathlib import Path

# NOTES ===============================================================

# version control
#version = 0.0 # prior to 28Dec2024
#version = 1.0 # major updates on 28Dec2024
#version = 2.0 # # aw 2025-02-04 13:15:09
version = 3.0 # aw 2025-03-12 10:04:04 - updated to use latest feature data file instead of middle prepared funcArgs

# Versions ============================================================
if version == 3.0:
    # globals *******************************************************
    global experimental_features

    # subroutines ***************************************************
    def import_evol_params():
        try:
            from __main__ import params
        except:
            import warnings
            warnings.simplefilter('always')
            
            print('Could not import params from __main__. Attempting to import from path...')
            warning_message = ('\n'
                                'Importing params from src. '
                                'This is normal if you are running a single simulation, '
                                'or otherwise testing the cfg script. If you are running '
                                'a batch simulation, you should be '
                                'importing the params from __main__')
            warnings.warn(warning_message)
            
            #from RBS_network_models.CDKL5.DIV21.src.evol_params import paramsZ
            from RBS_network_models.models.CDKL5_E6D_T2_C1_05212024.DIV21_FxHET.src.evol_params import params
            
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
    
    # main cfg script ***********************************************
    # iterate through feature data files (.py) in the features directory - get the latest one
    # import using import_module_from_path
    
    # features data path
    # Priority:
    #  1) FEATURE_DATA_PATH env var (set by src/batch.py from run_batch reference_data_paths)
    #  2) __main__.feature_data_path if available (single-sim/manual runs)
    feature_data_path = os.environ.get('FEATURE_DATA_PATH', None)
    if not feature_data_path:
        try:
            from __main__ import feature_data_path as _main_feature_data_path
            feature_data_path = _main_feature_data_path
        except Exception:
            feature_data_path = None

    if not feature_data_path:
        raise ValueError(
            "feature_data_path is not set. Provide reference_data_paths in run_batch.py "
            "(preferred), or set FEATURE_DATA_PATH in the environment."
        )

    feature_data_path = str(Path(feature_data_path).expanduser().resolve())
    assert os.path.exists(feature_data_path), f'{feature_data_path} not found'
    
    # Initialize simulation configuration
    cfg = specs.SimConfig()
    
    #add data from fitness_targets to cfg -> useful in preparing netParams
    cfg.locations_known = True
    cfg.features_path = feature_data_path
    if 'experimental_features' not in globals():
        import h5py
        experimental_features = {}
        with h5py.File(cfg.features_path, 'r') as _f:
            for _key in _f.keys():
                if _key.startswith('unit_'):
                    _uid_str = _key[len('unit_'):]
                    try:
                        _uid = int(_uid_str)
                    except ValueError:
                        _uid = _uid_str
                    experimental_features[_uid] = dict(_f[_key].attrs)
    
    # unit locations - classify from cell_type attribute ('excitatory' / 'inhibitory')
    cfg.inhib_units = [uid for uid, attrs in experimental_features.items() if attrs.get('cell_type') == 'inhibitory']
    cfg.excit_units = [uid for uid, attrs in experimental_features.items() if attrs.get('cell_type') == 'excitatory']
    cfg.num_excite = len(cfg.excit_units)
    cfg.num_inhib = len(cfg.inhib_units) 

    # Import evolutionary parameters
    import_evol_params()
    
    # Pharmacological scaling factors
    # 1.0 = no drug effect (baseline), 0.0 = full receptor/channel block
    # These can be overridden from batch configs to simulate drug conditions:
    #   AP5+NBQX  -> scale_AMPA=0, scale_NMDA=0
    #   AP5       -> scale_NMDA=0
    #   Bicuculline / Gabazine -> scale_GABA=0
    #   4AP       -> scale_K=0
    cfg.scale_AMPA = getattr(cfg, 'scale_AMPA', 1.0)
    cfg.scale_NMDA = getattr(cfg, 'scale_NMDA', 1.0)
    cfg.scale_GABA = getattr(cfg, 'scale_GABA', 1.0)
    cfg.scale_K = getattr(cfg, 'scale_K', 1.0)

    # set simulation duration
    # T_target_s is selected per-recording when the experimental target h5 is
    # created (see _scripts/create_experimental_target.py:compute_T_target).
    # batch.py reads it from the target h5 and exports it via the T_TARGET_S
    # environment variable; we pick it up here so each fitting run uses the
    # window length matched to its target. Falls back to 20 s if absent.
    import os as _os
    _T_target_env = _os.environ.get('T_TARGET_S', None)
    cfg.duration_seconds = float(_T_target_env) if _T_target_env else 20.0

    # Network cool down period - allows network to stabilize before analysis
    # This time is added to simulation duration but excluded from metric computation
    cfg.network_cool_down = 5.0  # Cool down period in seconds (set to 0 to disable)

    # set simulation configuration
    cfg.duration = (cfg.duration_seconds + cfg.network_cool_down) * 1e3  # Duration of the simulation, in ms (includes cool down)
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = [
        'simData', 
        'simConfig', 
        'netParams', 
        'netCells', 
        'netPops'
        ]  # Data to save
    cfg.saveJson = False  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}

    # Select cells for recording - just select random cell from cfg.excit_units and cfg.inhib_units - lists of gids
    E_cells = random.sample(cfg.excit_units, min(2, cfg.num_excite))
    I_cells = random.sample(cfg.inhib_units, min(2, cfg.num_inhib))
    assert all([x in cfg.excit_units for x in E_cells]), 'E_cells contains gids not in excitatory population'
    assert all([x in cfg.inhib_units for x in I_cells]), 'I_cells contains gids not in inhibitory population'
    #cfg.recordCells = [('E', E_cells), ('I', I_cells)]

    # --- OPT-IN soma voltage recording (RBS_RECORD_CELLS=<n per population>) ---
    # OFF by default (recordCells stays []), because turning it on changes the
    # size of every trial pkl and studies already in flight should not start
    # writing a different amount of data halfway through.
    #
    # A POOL, not a pair. The report wants two excitatory and two inhibitory
    # traces where the E and I cells are genuinely CONNECTED, but connectivity
    # is not known here -- the network is not built until netParams runs, and
    # whether any given E-I pair connects depends on 'exp(-dist_3D/probLengthConst)*prob'
    # (netParams.py:244), i.e. on distance. Picking exactly 2+2 here would give
    # a connected pair only by luck. So record a small pool and let the report
    # pick a verified-connected pair out of it post hoc, reading conns[].preGid
    # from the saved network (cfg.saveCellConns is already True).
    #
    # Indices are POP-RELATIVE ('E', i) tuples, not gids. cfg.excit_units /
    # cfg.inhib_units hold EXPERIMENTAL unit ids, which are not the simulation's
    # gid numbering -- feeding those to recordCells would record the wrong cells.
    _n_rec = int(os.environ.get('RBS_RECORD_CELLS', 0))
    if _n_rec > 0:
        _ne = min(_n_rec, cfg.num_excite)
        _ni = min(_n_rec, cfg.num_inhib)
        _eidx = sorted(random.sample(range(cfg.num_excite), _ne))
        _iidx = sorted(random.sample(range(cfg.num_inhib), _ni))
        cfg.recordCells = [('E', i) for i in _eidx] + [('I', i) for i in _iidx]
        print(f"RBS_RECORD_CELLS={_n_rec}: recording soma_voltage from "
              f"{_ne} E + {_ni} I cells (pop-relative indices) "
              f"at recordStep={cfg.recordStep} ms")

    #testing new params
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    cfg.cache_efficient = True
    cfg.cvode_active = True
    cfg.use_fast_imem = True
    cfg.allowSelfConns = True
    cfg.oneSynPerNetcon = False

    #new new params
    cfg.validateNetParams = True
    #cfg.validateDataSaveOptions = True
    cfg.verbose = False
    #cfg.verbose = True
    
    # success message
    print('cfg.py script completed successfully.')   
elif version == 2.0:
    import random
    
    # iterate through feature data files (.py) in the features directory - get the latest one
    # import using import_module_from_path
    feature_data_path = '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/features'
    feature_data_path = os.path.abspath(feature_data_path)
    feature_data_files = [f for f in os.listdir(feature_data_path) if f.endswith('.py')]
    feature_data_files = sorted(feature_data_files, key=lambda x: os.path.getmtime(os.path.join(feature_data_path, x)))
    feature_data_file = feature_data_files[-1]
    feature_data_file = os.path.join(feature_data_path, feature_data_file)
    feature_data_file = os.path.abspath(feature_data_file)
    fitness_targets = import_module_from_path(feature_data_file)
    
    # Initialize simulation configuration
    cfg = specs.SimConfig()
    #cfg.verbose = True # Show detailed messages
    
    #add data from fitness_targets to cfg -> useful in preparing netParams
    cfg.locations_known = True
    cfg.unit_locations = fitness_targets.fitnessFuncArgs['targets']['unit_locations']
    cfg.inhib_units = fitness_targets.fitnessFuncArgs['targets']['inhib_units']
    cfg.excit_units = fitness_targets.fitnessFuncArgs['targets']['excit_units']    
    num_excite = fitness_targets.fitnessFuncArgs['features']['num_excite']
    num_inhib = fitness_targets.fitnessFuncArgs['features']['num_inhib']    

    # Import evolutionary parameters
    def import_evol_params():
        try:
            from __main__ import params
        except:
            import warnings
            warnings.simplefilter('always')
            
            print('Could not import params from __main__. Attempting to import from path...')
            warning_message = ('\n'
                                'Importing params from src. '
                                'This is normal if you are running a single simulation, '
                                'or otherwise testing the cfg script. If you are running '
                                'a batch simulation, you should be '
                                'importing the params from __main__')
            warnings.warn(warning_message)
            
            from RBS_network_models.CDKL5.DIV21.src.evol_params import params
            
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
    import_evol_params()
    
    # set simulation duration
    #cfg.duration_seconds = 1  # Duration of the simulation, in seconds
    cfg.duration_seconds = 15  # Duration of the simulation, in seconds
    
    cfg.addNetStim = False  # Whether to add network stimulation

    if cfg.addNetStim:
        cfg.NetStim1 = {
            'pop': 'E',  # Target excitatory population
            'cellConds': {},  # Apply to all cells in pop
            'ynorm': [0, 1],  # Full range
            'sec': 'soma',
            'loc': 0.5,
            'synMech': 'exc',
            'synMechWeightFactor': [1.0],
            'start': 0,  # Start immediately
            'interval': 1000.0 / 10.0,  # 10 Hz average
            'noise': 1.0,  # Poisson noise (randomness)
            'number': 1e9,  # Very large number (continuous)
            'weight': 0.01,  # Weight of stimulation
            'delay': 0
        }
    # set simulation configuration
    cfg.duration = cfg.duration_seconds * 1e3  # Duration of the simulation, in ms
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = [
        'simData', 
        'simConfig', 
        'netParams', 
        'netCells', 
        'netPops'
        ]  # Data to save
    cfg.saveJson = False  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}

    # Select cells for recording
    E_cells = random.sample(range(num_excite), min(2, num_excite))
    I_cells = random.sample(range(num_inhib), min(2, num_inhib))
    cfg.num_excite = num_excite
    cfg.num_inhib = num_inhib
    cfg.recordCells = [('E', E_cells), ('I', I_cells)]

    #testing new params
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    cfg.cache_efficient = True
    cfg.cvode_active = True
    cfg.use_fast_imem = True
    cfg.allowSelfConns = True
    cfg.oneSynPerNetcon = False

    #new new params
    cfg.validateNetParams = True
    #cfg.validateDataSaveOptions = True
    cfg.verbose = False
    #cfg.verbose = True

    # success message
    print('cfg.py script completed successfully.') 
elif version == 1.0:
    import random
    #import DIV21.src.fitness_targets as fitness_targets
    import RBS_network_models.CDKL5.DIV21.src.fitness_targets as fitness_targets
    num_excite = fitness_targets.fitnessFuncArgs['features']['num_excite']
    num_inhib = fitness_targets.fitnessFuncArgs['features']['num_inhib']
    
    # Initialize simulation configuration
    cfg = specs.SimConfig()
    cfg.verbose = True # Show detailed messages

    # Import evolutionary parameters
    def import_evol_params():
        try:
            from __main__ import params
        except:
            import warnings
            warnings.simplefilter('always')
            
            print('Could not import params from __main__. Attempting to import from path...')
            # warning, importing params in this way will import the evolutionary parameter space,
            # importantly, it won't be importing the params selected by evol config.
            # this is normal if you are running a single simulation, or otherwise testing the cfg script
            # but if you are running a batch simulation, you should be importing the params from __main__
            
            warning_message = ('\n'
                                'Importing params from src. '
                                'This is normal if you are running a single simulation, '
                                'or otherwise testing the cfg script. If you are running '
                                'a batch simulation, you should be '
                                'importing the params from __main__')
            warnings.warn(warning_message)
            
            # assert path is not None, 'Path to evolutionary parameter space not provided.'
            # assert cfg is not None, 'cfg object not provided.'
            
            # params = import_module_from_path(path)
            #params = params.params
            #from DIV21.src.evol_params import params
            from RBS_network_models.CDKL5.DIV21.src.evol_params import params
            
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
    import_evol_params()
    
    cfg.duration_seconds = 1  # Duration of the simulation, in seconds
    cfg.duration = cfg.duration_seconds * 1e3  # Duration of the simulation, in ms
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = [
        'simData', 
        'simConfig', 
        'netParams', 
        'netCells', 
        'netPops'
        ]  # Data to save
    cfg.saveJson = False  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}

    # Select cells for recording
    E_cells = random.sample(range(num_excite), min(2, num_excite))
    I_cells = random.sample(range(num_inhib), min(2, num_inhib))
    cfg.num_excite = num_excite
    cfg.num_inhib = num_inhib
    cfg.recordCells = [('E', E_cells), ('I', I_cells)]

    #testing new params
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    cfg.cache_efficient = True
    cfg.cvode_active = True
    cfg.use_fast_imem = True
    cfg.allowSelfConns = True
    cfg.oneSynPerNetcon = False

    #new new params
    cfg.validateNetParams = True
    #cfg.validateDataSaveOptions = True
    cfg.verbose = False
    #cfg.verbose = True

    # success message
    print('cfg.py script completed successfully.')    
elif version == 0.0:
    from cfg_helper import *
    #from netpyne.batchtools import specs

    # Path to runtime kwargs
    cfg_runtime_kwargs_path = './cfg_kwargs.json'
    cfg_runtime_kwargs_path = os.path.abspath(cfg_runtime_kwargs_path)

    # Initialize simulation configuration
    cfg = specs.SimConfig()
    cfg.verbose = True # Show detailed messages

    # Import runtime kwargs
    assert os.path.exists(cfg_runtime_kwargs_path), f'{cfg_runtime_kwargs_path} not found'
    with open(cfg_runtime_kwargs_path, 'r') as f:
        cfg_kwargs = json.load(f)
    cfg.duration_seconds = cfg_kwargs['duration_seconds']
    cfg.target_script_path = cfg_kwargs['target_script_path']

    # Import evolutionary parameters
    cfg.params_script_path = cfg_kwargs['param_script_path']
    # cfg.params = handle_params_import(cfg.params_script_path, cfg)
    params = handle_params_import(cfg.params_script_path, cfg)
    #including the above line makes the json non-serializable, just leave it out for now.

    cfg.duration = cfg.duration_seconds * 1e3  # Duration of the simulation, in ms
    cfg.cache_efficient = True  # Use CVode cache_efficient option to optimize load on many cores
    cfg.dt = 0.025  # Internal integration timestep to use
    cfg.recordStep = 0.1  # Step size in ms to save data (e.g., V traces, LFP, etc)
    cfg.saveDataInclude = ['simData', 'simConfig', 'netParams', 'netCells', 'netPops']  # Data to save
    cfg.saveJson = True  # Save data in JSON format
    cfg.printPopAvgRates = [100, cfg.duration]  # Print population average rates
    cfg.savePickle = True  # Save params, network and sim output to pickle file

    # Record traces
    cfg.recordTraces['soma_voltage'] = {"sec": "soma", "loc": 0.5, "var": "v"}
    print('Recording soma voltage traces.')

    # Select cells for recording
    num_excite, num_inhib = get_cell_numbers_from_fitness_target_script(target_script_path=cfg.target_script_path)
    #numExcitatory, numInhibitory = USER_num_excite, USER_num_inhib
    E_cells = random.sample(range(num_excite), min(2, num_excite))
    I_cells = random.sample(range(num_inhib), min(2, num_inhib))
    cfg.num_excite = num_excite
    cfg.num_inhib = num_inhib
    cfg.recordCells = [('E', E_cells), ('I', I_cells)]
    print(f'Cells selected for recording: {cfg.recordCells}')

    #testing new params
    #cfg.coreneuron = True
    #cfg.dump_coreneuron_model = True
    cfg.cache_efficient = True
    cfg.cvode_active = True
    cfg.use_fast_imem = True
    cfg.allowSelfConns = True
    cfg.oneSynPerNetcon = False

    #new new params
    cfg.validateNetParams = True
    #cfg.validateDataSaveOptions = True
    cfg.verbose = True

    # success message
    print('cfg.py script completed successfully.')    