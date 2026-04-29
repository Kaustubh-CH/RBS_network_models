''' 

batch.py - Batch run script for Organoid_RTT_R270X_DIV112_WT

'''
# imports ==========================================================================================
import os
from netpyne import specs
from netpyne.batch import Batch
#RBS_network_models/utils/utils_old/batch_helper.py
from RBS_network_models.utils.utils_old.batch_helper import rangify_params, get_num_nodes, get_cores_per_node, get_tasks_per_node
from RBS_network_models.utils.utils_old.cfg_helper import import_module_from_path
# from RBS_network_models.fitnessFunc_claude import fitnessFunc_claude as fitnessFunc
from RBS_network_models.fitnessFunc_v2 import fitnessFunc_v2 as fitnessFunc
# from RBS_network_models.fitnessFunc import fitnessFunc_v3 as fitnessFunc

from RBS_network_models.utils.utils_old.helper import indent_decrease, indent_increase
import numpy as np
from pathlib import Path
from netpyne import sim
from copy import deepcopy
import json
import h5py

# functions ==========================================================================================
def batchEvol_v2(**kwargs):
    '''     
    Evolutionary algorithm optimization of a network using NetPyNE
    To run locally: mpiexec -np [num_cores] nrniv -mpi batchRun.py
    To run in interactive mode:
        salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --tasks-per-node=32 --cpus-per-task=4 --image=adammwea/netsims_docker:v1
    '''
    # subfunctions ============================================================================================
    
    def init_params(params):
        params = kwargs.get('parameter_space', None)
        if params is None: raise ValueError("parameter_space must be provided in kwargs")
        params = rangify_params(params)
        return params
    
    def load_reference_data_paths(kwargs):
        reference_data_list = []
        reference_data_paths = kwargs.get('reference_data_paths', None)
        #reference_data_paths = [os.path.abspath(path) for path in reference_data_paths] # convert reference paths to abs paths if they are not already
        reference_data_paths = [Path(path).expanduser().resolve() for path in reference_data_paths]
        if reference_data_paths is None: raise ValueError("reference_data_paths must be provided in kwargs")
        for path in reference_data_paths:
            if not os.path.exists(path): raise ValueError(f"Reference data path does not exist: {path}")
            else: 
                path_str = path.resolve().__str__()
                file_ext = path.suffix.lower()
                
                print(f'Loading reference data from {path_str}...')
                
                if file_ext == '.h5' or file_ext == '.hdf5':
                    # Load HDF5 file (similar to fitnessFunc_claude.py)
                    print(f'  Detected HDF5 format: {file_ext}')
                    try:
                        with h5py.File(path_str, 'r') as f:
                            # Just validate it can be opened - actual loading happens in fitness function
                            print(f'  Validated HDF5 file with groups: {list(f.keys())}')
                        reference_data_list.append(path_str)
                        print('  HDF5 reference data path stored.')
                    except Exception as e:
                        raise ValueError(f"Error validating HDF5 file {path_str}: {e}")
                        
                elif file_ext == '.npy':
                    # Load numpy file
                    print(f'  Detected numpy format: {file_ext}')
                    try:
                        # Validate it can be loaded
                        data = np.load(path_str, allow_pickle=True)
                        print(f'  Validated numpy file with shape/type: {type(data)}')
                        reference_data_list.append(path_str)
                        print('  Numpy reference data path stored.')
                    except Exception as e:
                        raise ValueError(f"Error loading numpy file {path_str}: {e}")
                        
                else:
                    print(f'  Warning: Unknown file format {file_ext}, storing path anyway')
                    reference_data_list.append(path_str)
                    
        kwargs['reference_data_list'] = reference_data_list
        
        # load reference data into global
        reference_data_path = reference_data_list[0] # HACK: for now, this only works for one reference data set - i feel we may want to change this in the future
        kwargs['reference_data_path'] = reference_data_path
        kwargs['feature_data_path'] = reference_data_path

        # Make feature path available to cfg.py (both local import and spawned MPI ranks)
        os.environ['FEATURE_DATA_PATH'] = str(reference_data_path)
        print(f"Set FEATURE_DATA_PATH={os.environ['FEATURE_DATA_PATH']}")
        
        return kwargs
    
    def init_fitnessFunc_args(**kwargs):
        # setting up fitnessFuncArgs
        print('setting up fitness function arguments...')
        conv_params = kwargs.get('conv_params', None)
        mega_params = kwargs.get('mega_params', None)
        if conv_params is None: raise ValueError("conv_params must be provided in kwargs")
        if mega_params is None: raise ValueError("mega_params must be provided in kwargs")
        fitnessFuncArgs = {
            #'reference_data': reference_data,
            'conv_params': conv_params,
            'mega_params': mega_params,
            'plot_sim': kwargs.get('plot_sim', False),
            'reference_data_path': kwargs.get('reference_data_path', None),
            'excit_units': kwargs.get('excit_units', []),
            'inhib_units': kwargs.get('inhib_units', []),
            'batching': True,
            
            # compute_network_metrics args
            'try_load': True, # try to load .npy metrics file if it exists
            'run_parallel': True,
            #'max_workers': 4, #NOTE: this should match the number of cores per node, would probably risk oversubscribing the node if set too high
            #'max_workers': 8, #NOTE: this should match the number of cores per node, would probably risk oversubscribing the node if set too high
            
            # nvm these run in series for now, just use 128 
            'max_workers': 128, 
            'burst_sequencing': True,
            'use_v2_burst_scoring': kwargs.get('use_v2_burst_scoring', True),
            #fitness schema
            'fit_schema': kwargs.get('fit_schema', None),
            'seed_fitness': kwargs.get('seed_fitness', None), # necessary for fitness function to do N-factorial
            
            # Network cool down parameter - excludes this period from metric computation
            'network_cool_down': kwargs.get('network_cool_down', 0.0),
        }
        return fitnessFuncArgs  
    
    def init_batch_attributes(kwargs):
        # simulation max iteration options -- max iterations before stopping generation
        time_sleep = kwargs.get('time_sleep', 5)  # seconds
        max_wait_min = kwargs.get('max_wait_min', 25 * 5)  # minutes
        default_maxiter_wait = int(max_wait_min * 60 / max(time_sleep, 1))
        maxiter_wait = kwargs.get('maxiter_wait', default_maxiter_wait)
        #b.batchLabel = 'evol' #NOTE: if left unset, batchLabel will be set to datetime at runtime
        
        # pop size options
        #pop_size = 512
        #pop_size = 256        
        #pop_size = 128
        # pop_size = 2
        pop_size = kwargs.get('pop_size', 10)

        # num elites options
        #num_elites = 50
        #num_elites = 75
        #num_elites = 128
        #num_elites = 4
        #num_elites = 16
        num_elites = kwargs.get('num_elites', 32)

        kwargs.update({
            'time_sleep': time_sleep,
            'max_wait_min': max_wait_min,
            'maxiter_wait': maxiter_wait,
            'pop_size': pop_size,
            'num_elites': num_elites,
        })
        print(f"Batch wait settings: time_sleep={time_sleep}s, max_wait_min={max_wait_min} min, maxiter_wait={maxiter_wait}")
        return kwargs
    
    def get_seed_cfgs(params, **kwargs):
        seeds_iterable = []
        fitness_data = []
        seeds = kwargs.get('seeds', None)

        # assert seeds is a list of paths to seed cfg files
        if seeds is None:
            return None, None
        # if seeds is None: raise ValueError("seeds must be provided in kwargs")
        
        # reverse the order of seeds, so that the newest seeds are used first
        #seeds = seeds[::-1] # NOTE: newer seeds are at the top now...

        # get number of elites, only use this many seeds if less than the number of seeds
        # num_elites = kwargs.get('num_elites', None)
        # if num_elites is None: raise ValueError("num_elites must be provided in kwargs")
        # if len(seeds) > num_elites:
        #     seeds = seeds[:num_elites]
        #     print(f'Using {num_elites} seeds: {seeds}')

        # limit to pop size instead #aw 2025-05-19 17:49:46
        pop_size = kwargs.get('pop_size', None)
        if pop_size is None: raise ValueError("pop_size must be provided in kwargs")
        if len(seeds) > pop_size:
            seeds = seeds[:pop_size]
            print(f'Using {pop_size} seeds: {seeds}')

        # format the seeds into a list of lists, where each list is a candidate - as netpyne expects  
        evol_params = params.copy()
        for i, seed in enumerate(seeds):
            #load simcfg from seed
            seed_iterable = []
            
            # check if _cfg.json is at the end of the seed path, if not, append it
            if not seed.endswith('_cfg.json'):
                seed = seed + '_cfg.json'

            # assert seed is a valid path
            if not os.path.exists(seed):
                #raise ValueError(f"Seed path does not exist: {seed}")
                print(f"WARNING: Seed path does not exist: {seed}, skipping...")
                continue
                
            # load the seed configuration
            try:
                print(f'Loading seed configuration from {seed}...')
                sim.loadSimCfg(seed, setLoaded=True)
                simcfg = deepcopy(sim.cfg.todict())
                sim.clearAll()  # clear the sim object to avoid memory issues
                fitpath = seed.replace('cfg', 'fitness')
            except Exception as e:
                print(f'Error loading seed configuration from {seed}: {e}')
                sim.clearAll()  # clear the sim object to avoid memory issues
                continue  # skip this seed if there was an error

            # load fitness data
            try:
                if os.path.exists(fitpath):
                    print(f'Loading fitness data from {fitpath}...')
                    with open(fitpath, 'r') as f:
                        fit = json.load(f)
                    #simcfg['fitness'] = fitness_data
                else:
                    print(f'Fitness data file does not exist: {fitpath}, initializing empty fitness data.')
                    fit = {}
            except Exception as e:
                print(f'Error loading fitness data from {fitpath}: {e}')
                fit = {}


            for key in evol_params:
                if key in simcfg:
                    seed_iterable.append(simcfg[key])
                else:
                    upper_bound = evol_params[key]['values'][1]
                    lower_bound = evol_params[key]['values'][0]
                    random_value = np.random.uniform(lower_bound, upper_bound)
                    seed_iterable.append(random_value)
            seeds_iterable.append(seed_iterable)
            fitness_data.append(fit)
        if len(seeds_iterable) == 0: seeds_iterable = None
        return seeds_iterable, fitness_data    
        
    def init_runCfg(b, **kwargs):
        # set run configuration 
        run_Cfg_script_path = kwargs.get('runCfg_script_path', None)
        if run_Cfg_script_path is None: raise ValueError("runCfg_script_path must be provided in kwargs")       

        # Use srun only when inside a Slurm job/allocation
        in_slurm = bool(os.environ.get('SLURM_JOB_ID'))
        mpi_command = 'MPICH_GPU_SUPPORT_ENABLED=0 srun -N 1' if in_slurm else 'MPICH_GPU_SUPPORT_ENABLED=0'
        print(f"Detected Slurm job: {in_slurm}. Using mpiCommand='{mpi_command}'")

        b.runCfg = {
            'type': 
                'mpi_direct', 
                #'hpc_slurm', #TODO: not sure if this is really an option
                #'mpi_bulletin', #TODO: not sure if this is really an option
            'script': run_Cfg_script_path,
            'mpiCommand': 
                #'',
                
                #'mpirun',
                
                #'shifter --image=adammwea/netsims_docker:v1'
                mpi_command,
                # # bind to socket
                # ' --cpu-bind=verbose,cores'
                # ' --hint=multithread' # enable multithreading on each core
                # ' --cores-per-task=4' # set number of cores per task
                #,
            'nrnCommand': 'nrniv',
            'nodes': 1,                             
                # NOTE: Importantly, these are the number of nodes to use for each simulation, I think. 
                # So if I want to put 4 simulations on each node, 2 per socket.
                # nodes should be set to 1, and tasks_per_node should be set to cores_per_node / 4                                        
            #'coresPerNode': 16,
            #'coresPerNode': 4, #i.e., 4 mpi tasks per sim, @256 cands per gen, @1 cpu per task = 1024 cores. 4 nodes, each with 256 logical cores, allows 1024 cores to be used.
            
            # aw 2025-04-22 13:17:37 4 is too slow. going to try more. allow srun commands to queue
            #'coresPerNode': 16, #i.e., 4 mpi tasks per sim, @256 cands per gen, @1 cpu per task = 1024 cores. 4 nodes, each with 256 logical cores, allows 1024 cores to be used.
            
            # # aw 2025-04-23 03:37:46 lets try maximizing for 1 sim / socket (i.e. 64 tasks per node)
            'coresPerNode': 128,
            
            'reservation': None,
            #'skip': False, #if rerunning, skip if output files already exist
            'skip': True, #if rerunning, skip if output files already exist
            }
        
        # for key in kwargs, replace b.runCfg[key] = kwargs[key] if matching key exists in b.runCfg
        for key in kwargs:
            if key in b.runCfg:
                b.runCfg[key] = kwargs[key]
                print(f'Overriding b.runCfg.{key} = {kwargs[key]}')
                
        # return batch object
        return b
    
    def initialize_batch(params, kwargs):
        indent_increase()
        
        # init file paths, modify as needed
        cfgFile_path = str(Path('/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/cfg.py').expanduser().resolve())
        netParams_path = str(Path('/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models/models/CDKL5_E6D_T2_C1_05212024/DIV21_FxHET/src/netParams.py').expanduser().resolve())

        # Load network_cool_down parameter from cfg file
        print('Loading network_cool_down parameter from cfg...')
        cfg_module = import_module_from_path(cfgFile_path)
        network_cool_down = getattr(cfg_module.cfg, 'network_cool_down', 0.0)
        kwargs['network_cool_down'] = network_cool_down
        kwargs['excit_units'] = list(getattr(cfg_module.cfg, 'excit_units', []))
        kwargs['inhib_units'] = list(getattr(cfg_module.cfg, 'inhib_units', []))
        print(f'network_cool_down = {network_cool_down} seconds')
        print(f"cfg excit/inhib counts = {len(kwargs['excit_units'])}/{len(kwargs['inhib_units'])}")

        # init batch object
        print('initializing batch object...')
        b = Batch(cfgFile=cfgFile_path, netParamsFile=netParams_path, params=params)
        batchLabel = kwargs.get('batchLabel', None)
        if batchLabel:
            b.batchLabel = batchLabel
        b.method = 'optuna' # set method to evolutionary algorithm
        
        # #init fitness function args
        # print('initializing fitness function arguments...')
        fitnessFuncArgs = init_fitnessFunc_args(**kwargs)      
        
        # apply kwargs to batch object
        print('setting batch object attributes...')
        kwargs=init_batch_attributes(kwargs)
        
        #convert seeds into interable list of params
        print('setting seeds configs...')
        seeds_iterable, seed_fitness = get_seed_cfgs(params, **kwargs)
        if seeds_iterable is not None:
            kwargs['seed_fitness'] = seed_fitness

        #init fitness function args
        print('initializing fitness function arguments...')
        fitnessFuncArgs = init_fitnessFunc_args(**kwargs)           
        
        # evolutionary algorithm configuration
        print('setting evolutionary algorithm configuration...')
        # test_simulated_data_path = '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Jan12_20s_noSeed_v4/batch_runs/batch_2026-01-12_spiking_only/gen_6/gen_6_cand_25_data.pkl'
        # # fitnessFuncArgs['reference_data_path'] = '/pscratch/sd/k/ktub1999/networkSimulatons_Sonnet/experimental_data.h5'
        # fitnessFuncArgs['sim_data_path'] = test_simulated_data_path
        # fitnessFuncArgs['batching'] = False
        # fitnessFuncArgs['try_load'] = False
        # #should be bad
        # import pdb; pdb.set_trace()
        # fit_value1 = fitnessFunc( simulated_data_path=test_simulated_data_path,**fitnessFuncArgs)
        # test_simulated_data_path = '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Jan12_20s_noSeed_v4/batch_runs/batch_2026-01-12_spiking_only/gen_7/gen_7_cand_57_data.pkl'
        # fitnessFuncArgs['sim_data_path'] = test_simulated_data_path
        # #should be good
        # fit_value2 = fitnessFunc( simulated_data=test_simulated_data_path,**fitnessFuncArgs)
        # test_simulated_data_path = '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Jan12_20s_noSeed_v3/batch_runs/batch_2026-01-12_spiking_only/gen_54/gen_54_cand_9_data.pkl'
        # fitnessFuncArgs['sim_data_path'] = test_simulated_data_path
        # #should be best
        # fit_value3 = fitnessFunc( simulated_data=test_simulated_data_path,**fitnessFuncArgs)
        # print(f'Test fitness values: bad={fit_value1}, good={fit_value2}, best={fit_value3}')
        mode = 'optuna'
        if mode == 'asd':
            b.method = 'asd'
            b.optimCfg = {
                'fitnessFunc': fitnessFunc,
                'fitnessFuncArgs': fitnessFuncArgs,
                'maxFitness': 1000,
                'maxiters': 1000,
                'maxtime': 3600 * 24,
                'stepsize': 0.1,
                'sinc': 2,
                'sdec': 2,
                'pinc': 2,
                'pdec': 2,
                'maxiter_wait': 10,
                'time_sleep': 5,
                'popsize': 1,
            }
        if mode =='optuna':
            b.optimCfg = {
            'fitnessFunc': fitnessFunc,
            'fitnessFuncArgs': fitnessFuncArgs,
            'maxFitness': 10000,
            'maxiters': kwargs.get('maxiters', 1000),          # total number of trials
            'maxtime': 3600 * 24,      # 8 hour budget
            'maxiter_wait': kwargs.get('maxiter_wait', 100),
            'time_sleep': kwargs.get('time_sleep', 15),
            'direction': 'minimize',
            }
        elif seeds_iterable is not None:
            b.evolCfg = {
                'evolAlgorithm': 'custom',
                'fitnessFunc': fitnessFunc,
                'fitnessFuncArgs': fitnessFuncArgs,
                'pop_size': kwargs.get('pop_size', 16),
                'num_elites': kwargs.get('num_elites', 4),
                'mutation_rate': 0.5,
                'crossover': 0.5,
                'maximize': False,
                'max_generations': 1000,
                'time_sleep': kwargs.get('time_sleep', 5),
                'maxiter_wait': kwargs.get('maxiter_wait', 10),
                'defaultFitness': 1000,
                'seeds': seeds_iterable, #requires params to put candidates in correct order,
                #'startGeneration': 8, #NOTE: dont used this. 
            }
        else:
            b.evolCfg = {
                'evolAlgorithm': 'custom',
                'fitnessFunc': fitnessFunc,
                'fitnessFuncArgs': fitnessFuncArgs,
                'pop_size': kwargs.get('pop_size', 16),
                'num_elites': kwargs.get('num_elites', 4),
                'mutation_rate': 0.5,
                'crossover': 0.5,
                'maximize': False,
                'max_generations': 1000,
                'time_sleep': kwargs.get('time_sleep', 5),
                'maxiter_wait': kwargs.get('maxiter_wait', 10),
                'defaultFitness': 1000,
            }
        # init runcfg
        print('setting run configuration...')
        b = init_runCfg(b, **kwargs)

        # append tag to batch label for easy identification if desired
        tag = kwargs.get('tag', None)
        print(f'tag = {tag}')
        if tag is not None and not b.batchLabel.endswith(f'_{tag}'):
            b.batchLabel = b.batchLabel + f'_{tag}'
                
        # set save folder
        batchFolder = kwargs.get('batchFolder', None)
        if batchFolder is None: raise ValueError("batchFolder must be provided in kwargs")
        b.saveFolder = os.path.join(batchFolder, b.batchLabel)
        os.makedirs(b.saveFolder, exist_ok=True)
        print(f'b.saveFolder = {b.saveFolder}')

        # import sys
        # sys.exit()
        
        # return batch object
        print('batch object initialized.')
        indent_decrease()
        return b
        
    # globals ==========================================================================================
    global reference_data
    
    # main ==========================================================================================
    indent_increase()
    
    #parameters space to explore
    print('rangifying parameters...')
    params = kwargs.get('parameter_space', None)
    params = init_params(params)
    
    # load reference data paths
    print('loading reference data paths...')
    kwargs = load_reference_data_paths(kwargs)
    
    ## create batch object
    print('initializing batch object...')
    b = initialize_batch(params, kwargs)

    ## pre-enqueue seed trials into Optuna study (if using optuna with seeds)
    seeds = kwargs.get('seeds', None)
    if b.method == 'optuna' and seeds is not None:
        import optuna
        # build the study with the same name/storage that NetPyNE will use
        study_name = b.batchLabel
        storage = f'sqlite:///{b.saveFolder}/{b.batchLabel}_storage.db'
        direction = b.optimCfg.get('direction', 'minimize')
        study = optuna.create_study(
            study_name=study_name, storage=storage,
            load_if_exists=True, direction=direction,
        )
        # get param labels in the same order NetPyNE will use
        paramLabels = [x['label'] for x in b.params]
        # get seed cfgs using the same logic as evolCfg seeds
        seeds_iterable, _ = get_seed_cfgs(params, **kwargs)
        if seeds_iterable is not None:
            print(f'Enqueueing {len(seeds_iterable)} seed trial(s) into Optuna study...')
            for seed_vals in seeds_iterable:
                seed_dict = {str(label): float(val) for label, val in zip(paramLabels, seed_vals)}
                study.enqueue_trial(seed_dict)
            print(f'Seed trials enqueued into {storage}')
        del study  # close study so NetPyNE can open it cleanly

    ## run batch
    print('running batch...')
    b.run()
    
    # end indentation
    indent_decrease()

def batchEvol(feature_path, **kwargs):
    ''' 
    
    Evolutionary algorithm optimization of a network using NetPyNE
    To run locally: mpiexec -np [num_cores] nrniv -mpi batchRun.py
    To run in interactive mode:
        salloc -A m2043 -q interactive -C cpu -t 04:00:00 --nodes=2 --tasks-per-node=32 --cpus-per-task=4 --image=adammwea/netsims_docker:v1
    '''
    #parameters space to explore
    ## network
    # from .evol_params import params
    # from .fitness_targets import fitnessFuncArgs
    # from ....fitnessFunc import fitnessFunc
    
    ## format params so all values are lists of length 2, min and max values
    from RBS_network_models.CDKL5.DIV21.src.evol_params import params
    from RBS_network_models.fitnessFunc import fitnessFunc
    feature_module = import_module_from_path(feature_path)
    fitnessFuncArgs = feature_module.fitnessFuncArgs
    params = rangify_params(params)
    
    ## create batch object
    b = Batch(
        cfgFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/cfg.py',
        netParamsFile='/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/netParams.py',
        #cfg=None,
        #netParams=None,
        params=params,
        #groupedParams=None,
        #initCfg=None,
        #seed=None,
    )
    
    ## set batch object attributes
    time_sleep = 5 # seconds
    #max_wait = 30 # minutes
    max_wait = 20 # minutes
    maxiter_wait = max_wait * 60 / time_sleep # convert to number of iterations
    #b.batchLabel = 'evol' #NOTE: if left unset, batchLabel will be set to datetime at runtime
    from RBS_network_models.CDKL5.DIV21.src.conv_params import conv_params
    from RBS_network_models.CDKL5.DIV21.src.conv_params import mega_params
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'fitnessFunc': fitnessFunc,
        'fitnessFuncArgs': {
            **fitnessFuncArgs,
            'conv_params': conv_params,
            'mega_params': mega_params, 
            'reference_data_path': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/Organoid_RTT_R270X/DIV112_WT/network_metrics/Organoid_RTT_R270X_pA_pD_B1_d91_250107_M07297_Network_000028_network_metrics_well005.npy',
            'plot_sim': False,
            #'plot_sim': True,
            },
        #'pop_size': 8,
        #'pop_size': 128,
        '#pop_size': 256,
        #'pop_size': 196,
        'num_elites': 50, 
        #'num_elites': 1,
        'mutation_rate': 0.5,
        'crossover': 0.5,
        'maximize': False,
        'max_generations': 1000,
        'time_sleep': time_sleep,
        'maxiter_wait': maxiter_wait,
        'defaultFitness': 1000,
        #pass list of paths in seed_dir to seed the population
        #seed_dir = kwargs['seed_dir']
        #'seeds': get_seed_cfgs(kwargs['seed_dir'], params), #requires params to put candidates in correct order
    }
    #b.initCfg = {}
    b.method = 'evol'
    #b.mpiCommandDefault = 'mpiexec'
    #b.optimCfg = {}
    
    ## set run configuration
    # tasks_per_node = get_cores_per_node() // 4 #NOTE: I think this is the number of tasks to run on each node
    nodes_per_core = get_cores_per_node()
    #tasks_per_sim = get_tasks_per_node() // 4
    mpi_tasks_per_node = 64
    mpi_tasks_per_sim = mpi_tasks_per_node // 4
    
    b.runCfg = {
        'type': 
            'mpi_direct', 
            #'hpc_slurm', #TODO: not sure if this is really an option
            #'mpi_bulletin', #TODO: not sure if this is really an option
        #'script': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/CDKL5/DIV21/src/init.py',
        'script': '/pscratch/sd/a/adammwea/workspace/RBS_network_models/RBS_network_models/Organoid_RTT_R270X/DIV112_WT/src/init.py',
        'mpiCommand': '',
            
            # 'mpirun',
            
            # 'srun'
            # # bind to socket
            # ' --cpu-bind=verbose,cores'
            # ' --hint=multithread' # enable multithreading on each core
            # ' --cores-per-task=4' # set number of cores per task
            # ,
            
        'nrnCommand': 'nrniv',
        #'nodes': get_num_nodes(),
        'nodes': 1,                             # NOTE: Importantly, these are the number of nodes to use for each simulation, I think. 
                                                # So if I want to put 4 simulations on each node, 2 per socket.
                                                # nodes should be set to 1, and tasks_per_node should be set to cores_per_node / 4                                        
        #'coresPerNode': mpi_tasks_per_sim, #NOTE: I think this basically translates to mpi tasks per node
        #'coresPerNode': 1,
        'coresPerNode': 4,
        'reservation': None,
        'skip': False, #if rerunning, skip if output files already exist
        }
    
    # for key in kwargs, replace b.runCfg[key] = kwargs[key] if matching key exists in b.runCfg
    for key in kwargs:
        #print (f'kwargs: {key} = {kwargs[key]}')
        if key in b.runCfg:
        #if hasattr(b.runCfg, key):
            b.runCfg[key] = kwargs[key]
            #setattr(b.runCfg, key, kwargs[key])
            print(f'Overriding b.runCfg.{key} = {kwargs[key]}')
            #print(f'Overriding b.runCfg.{key} = {getattr(b.runCfg, key)}')
            
    b.saveFolder = f'/pscratch/sd/a/adammwea/workspace/RBS_network_models/data/CDKL5/DIV21/batch_runs/{b.batchLabel}'
    # b.seed = None #NOTE: I think this is for getting identical random numbers if rerunning the same batch
    
    # To debug the batch script without running the full optimization, you can uncomment the following line:
    # import sys
    # sys.exit()
    
    ## run batch
    b.run()

def batchOptuna(**kwargs):
    print('not implemented yet')
    pass