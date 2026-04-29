import os
import sys
import shutil
import datetime
import json
import pandas as pd
from netpyne import sim
import dill
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
# ===============================================================

'''batchRun helper functions (newer)'''
def process_seed(seed, params, verbose):
    candidate = []
    try:
        assert os.path.exists(seed), f'Seed file {seed} does not exist'
        if '_cfg' in seed:
            if verbose:
                print('Loading file', seed)
            with open(seed, 'rb') as f:
                if verbose:
                    print('Loading seed file...')
                cfg = dill.load(f)
        else:
            cfg = sim.loadSimCfg(seed, setLoaded=False)
        
        for key, value in params.items():
            if hasattr(cfg, key):
                candidate.append(getattr(cfg, key))
            else:
                random_val = np.random.uniform(value[0], value[1])
                candidate.append(random_val)
        
    except Exception as e:
        if verbose:
            print(f"Error processing {seed}: {e}")
        return None
    
    return candidate

def get_seed_cfgs(seed_dir, params, num_workers=4, verbose=True):
    # check number of available cpus
    import os
    num_cpus = os.cpu_count()
    
    try:
        seed_list = [os.path.join(seed_dir, f) for f in os.listdir(seed_dir)]
        # if len(seed_list) == 0:
        #     raise ValueError(f'No seed files found in {seed_dir}')
        # if len(seed_list) < num_workers:
        #     num_workers = len(seed_list)
        # if num_workers > num_cpus:
        #     num_workers = num_cpus
        # if num_workers is -1 or num_workers is None:
        #     num_workers = num_cpus
        # if num_workers is 0:
        #     num_workers = 1
        assert num_workers < len(seed_list), f'num_workers ({num_workers}) must be less than the number of seed files ({len(seed_list)})'
        assert num_workers < num_cpus, f'num_workers ({num_workers}) must be less than the number of available cpus ({num_cpus})'
        
        
        with Pool(processes=num_workers) as pool:
            if verbose:
                candidates = pool.starmap(process_seed, [(seed, params, verbose) for seed in seed_list])
            else:
                candidates = list(tqdm(pool.starmap(process_seed, [(seed, params, verbose) for seed in seed_list]), total=len(seed_list), desc="Processing Seeds"))
        
        return [c for c in candidates if c is not None]  # Filter out None values if any error occurred
    except Exception as e:
        if verbose:
            print(f"Error in get_seed_cfgs: {e}")
        return None

def rangify_params(params):
    for key, value in params.items():
        if isinstance(value, list):
            if len(value) == 2:
                params[key] = [min(value), max(value)]
            # else:
            #     params[key] = [min(value), max(value), value[2]]
        elif isinstance(value, int) or isinstance(value, float):
            params[key] = [value, value]
            
    assert all(isinstance(value, list) for value in params.values()), 'All values in params must be lists'
    return params

def get_seed_cfgs_old(seed_dir, params):
    try:
        seed_list = [os.path.join(seed_dir, f) for f in os.listdir(seed_dir)]
        candidates = []
        for seed in seed_list:
            if '_cfg' in seed:
                #load pkl file directly
                candidate = []
                assert os.path.exists(seed), f'Seed file {seed} does not exist'
                print('Loading file', seed)
                with open(seed, 'rb') as f:
                    print('Loading seed file...')
                    cfg = dill.load(f)
                for key, value in params.items():
                    if hasattr(cfg, key):
                        candidate.append(getattr(cfg, key))
                    elif not hasattr(cfg, key):
                        random_val = np.random.uniform(value[0], value[1]) ## if the key is not in the cfg, generate a random value - since this format is position sensitive
                        candidate.append(random_val)
                candidates.append(candidate)
            else:
                candidate = []
                assert os.path.exists(seed), f'Seed file {seed} does not exist'
                cfg = sim.loadSimCfg(seed, setLoaded=False)
                for key, value in params.items():
                    if hasattr(cfg, key):
                        candidate.append(getattr(cfg, key))
                    elif not hasattr(cfg, key):
                        random_val = np.random.uniform(value[0], value[1])
                        candidate.append(random_val)
                candidates.append(candidate)
    except:
        candidates = None
        #seed_list = {} #TODO: this might be the correct way to handle this... not sure. Verify.
    return candidates

def detect_allocation_slurm():
    cores_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))  # Default to 1 core per node if not in SLURM
    number_of_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))  # Default to 1 node if not in SLURM
    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", cores_per_node))  # Default tasks per node = cores
    total_tasks = number_of_nodes * ntasks_per_node
    return cores_per_node, number_of_nodes, ntasks_per_node, total_tasks

import os
def detect_allocation_local():
    #nodes = 1
    cores_per_node = os.cpu_count() #logical cores, physical cores = cores_per_node//2
    number_of_nodes = 1
    ntasks_per_node = 1
    total_tasks = 1
    return cores_per_node, number_of_nodes, ntasks_per_node, total_tasks

def get_num_nodes():
    # Detect SLURM environment or use defaults
    if 'SLURM_JOB_NUM_NODES' in os.environ:
        cores_per_node, number_of_nodes, ntasks_per_node, total_tasks = detect_allocation_slurm()
    else:
        cores_per_node, number_of_nodes, ntasks_per_node, total_tasks = detect_allocation_local()
    return number_of_nodes

def get_cores_per_node():
    # Detect SLURM environment or use defaults
    if 'SLURM_JOB_NUM_NODES' in os.environ:
        cores_per_node, number_of_nodes, ntasks_per_node, total_tasks = detect_allocation_slurm()
    else:
        cores_per_node, number_of_nodes, ntasks_per_node, total_tasks = detect_allocation_local()
    return cores_per_node

def get_tasks_per_node():
    # Detect SLURM environment or use defaults
    if 'SLURM_JOB_NUM_NODES' in os.environ:
        cores_per_node, number_of_nodes, ntasks_per_node, total_tasks = detect_allocation_slurm()
    else:
        cores_per_node, number_of_nodes, ntasks_per_node, total_tasks = detect_allocation_local()
    return ntasks_per_node

'''batchRun helper functions'''

'''reimplemented'''   
def save_batch_cfg_to_file(batch_cfg, **kwargs):
    # batch_cfg_file = os.path.join(kwargs['run_path'], 'batch_cfg.py')
    # with open(batch_cfg_file, 'w') as f:
    #     f.write(f'batch_cfg = {batch_cfg}')
    # print(f'batch_cfg saved to {batch_cfg_file}')
    
    # import json
    # batch_cfg_file = os.path.join(kwargs['run_path'], 'batch_cfg.json')
    # with open(batch_cfg_file, 'w') as f:
    #     json.dump(batch_cfg, f, indent=4)
    # print(f'batch_cfg saved to {batch_cfg_file}')
    
    # Save batch_cfg in run_path as pkl
    batch_cfg_file = os.path.join(kwargs['run_path'], 'batch_cfg.pkl')
    with open(batch_cfg_file, 'wb') as f:
        dill.dump(batch_cfg, f)
    print(f'batch_cfg saved to {batch_cfg_file}')
    
def save_batch_config_to_file(USER_vars, **kwargs):
    # Save USER_vars in run_path as JSON
    USER_run_path = USER_vars['USER_run_path']
    with open(f'{USER_run_path}/temp_user_args.py', 'w') as f:
        json.dump(USER_vars, f, cls=batchcfgEncoder, indent=4)

def mock_command_example(**kwargs):

        type=kwargs.get('type', 'mpi_direct')
    
        # Copy pasted (and slightly modifed) from batch/utils.py
        if type == 'mpi_bulletin':
            # ----------------------------------------------------------------------
            # MPI master-slaves
            # ----------------------------------------------------------------------
            pc.submit(runJob, nrnCommand, script, cfgSavePath, netParamsSavePath, jobPath, pc.id())
            print('-' * 80)
        else:
            # ----------------------------------------------------------------------
            # MPI job command
            # ----------------------------------------------------------------------

            if mpiCommand == '':
                command = '%s %s simConfig=%s netParams=%s ' % (nrnCommand, script, cfgSavePath, netParamsSavePath)
            else:
                command = '%s -n %d %s -python -mpi %s simConfig=%s netParams=%s ' % (
                    mpiCommand,
                    numproc,
                    nrnCommand,
                    script,
                    cfgSavePath,
                    netParamsSavePath,
                )

            # ----------------------------------------------------------------------
            # run on local machine with <nodes*coresPerNode> cores
            # ----------------------------------------------------------------------
            if type == 'mpi_direct':
                #executer = '/bin/bash'
                executer = executor
                jobString = jobStringMPIDirect(custom, folder, command)
            # ----------------------------------------------------------------------
            # Create script to run on HPC through slurm
            # ----------------------------------------------------------------------
            elif type == 'hpc_slurm':
                executer = 'sbatch'
                jobString = jobStringHPCSlurm(
                    jobName,
                    allocation,
                    walltime,
                    nodes,
                    coresPerNode,
                    jobPath,
                    email,
                    reservation,
                    custom,
                    folder,
                    command,
                )

def configure_command(**kwargs):
    ''' Build mpi command'''
    preset_configs = kwargs.get('preset_configs', None)
    from datetime import datetime
    import os
    
    # Check that run_path is not None
    run_path = kwargs.get('run_path', None)
    assert run_path is not None, 'run_path must be specified'
    
    #get preset config
    interactive_node = preset_configs.get('interactive_node', False)
    sbatch = preset_configs.get('sbatch', False)
    login_node = preset_configs.get('login_node', False)
    local = preset_configs.get('local', False)
    
    # select from preset configs
    if interactive_node:    
        # User arguments
        shifter_command = 'shifter --image=adammwea/netsims_docker:v1'
        #shifter_command = 'shifter --image=adammwea/netpyneshifter:v5'
        init_script = 'modules/simulation_config/init.py'
        init_script = os.path.abspath(init_script)
        

        # Detect SLURM environment or use defaults
        alloc_cores_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))  # Default to 1 core per node if not in SLURM
        alloc_number_of_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))  # Default to 1 node if not in SLURM
        alloc_ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", alloc_cores_per_node))  # Default tasks per node = cores
        
        #print allocation info
        print(f'alloc_cores_per_node: {alloc_cores_per_node}')
        print(f'alloc_number_of_nodes: {alloc_number_of_nodes}')
        print(f'alloc_ntasks_per_node: {alloc_ntasks_per_node}')
        
        kwargs['alloc_cores_per_node'] = alloc_cores_per_node
        kwargs['alloc_number_of_nodes'] = alloc_number_of_nodes
        kwargs['alloc_ntasks_per_node'] = alloc_ntasks_per_node
        
        #total_tasks = number_of_nodes * ntasks_per_node
        #ntasks_per_node = 256
        #tasks_per_node = int(ntasks_per_node/2) #use half the tasks per node to get a single simulation on a single socket
        
        # best_case = True
        # if best_case:
        #     cores_per_node = 128
        
        #something I'm trying - not sure if it will work
        #pop_size = kwargs.get('pop_size', 100)
        #cores_per_srun = cores_per_node//pop_size
        #tasks_per_srun = cores_per_srun #run the whole generation in parallel
        #if tasks_per_node < 1: tasks_per_node = 1
        #cpu_per_task_override = 4
        
        custom_kwargs = {
            'nodes_per_command': 1,
            #'sims_per_node': 1,
            'sims_per_node': 4,
            #'ntasks_per_node': 16,
            'ntasks_per_sim': 4,
            #'cores_per_task': 16,
            #'cores_per_task': 4,
            #'physical_cores_per_node': 128, # this is unique to perlmutter
            'logical_cores_available_per_node': 256, # this is unique to perlmutter
        }
        
        nodes_per_sim = custom_kwargs['nodes_per_command']
        cores_per_sim = custom_kwargs['logical_cores_available_per_node'] // custom_kwargs['sims_per_node']
        cores_per_task = cores_per_sim // custom_kwargs['ntasks_per_sim']
        tasks_per_sim = custom_kwargs['ntasks_per_sim']
        print(f'nodes_per_sim: {nodes_per_sim}')
        print(f'cores_per_sim: {cores_per_sim}')
        print(f'cores_per_task: {cores_per_task}')
        print(f'tasks_per_sim: {tasks_per_sim}')
        logical_cores_per_node = custom_kwargs['logical_cores_available_per_node']
        assert cores_per_sim*tasks_per_sim == logical_cores_per_node, 'cores_per_sim*tasks_per_sim must equal logical_cores_per_node'
        
        # put in kwargs
        kwargs['nodes_per_sim'] = nodes_per_sim
        kwargs['cores_per_sim'] = cores_per_sim
        kwargs['cores_per_task'] = cores_per_task
        kwargs['tasks_per_sim'] = tasks_per_sim
        available_cores = cores_per_sim*tasks_per_sim
        
        
        print(f'cores_per_sim*tasks_per_sim: {cores_per_sim*tasks_per_sim}')
        #wait 10 seconds
        #import time
        #time.sleep(10)
                
        command_kwargs = {
            'type': 'mpi_direct',
            'script': init_script,
            #'mpiCommand': 'srun',
            #'mpiCommand': f'srun',
            #'coresPerNode': 4,
            #'coresPerNode': 32,
            'coresPerNode': tasks_per_sim,
            #'mpiCommand': f'srun -N 1', # -n {coresPerNode} gets added in command building
            'mpiCommand': f'srun -N {nodes_per_sim}', # -n {coresPerNode} gets added in command building
            #'nrnCommand': f'--cpus-per-task=1 --hint=nomultithread --cpu-bind=cores {shifter_command} nrniv', #get single simulation on a single socket
            
            #'nrnCommand': f'--cpus-per-task={cpu_per_task_override} --hint=multithread --cpu-bind=cores '
            'nrnCommand': f'--cpus-per-task={cores_per_task} --hint=nomultithread --cpu-bind=cores '
                            f'{shifter_command} '
                            #'nvprof --profile-child-processes '
                            'nrniv -threads', #-mpi and -python are added in command building
            
            #'coresPerNode': cores_per_node,
            #'coresPerNode': tasks_per_srun,
            
            'reservation': None,
            'skip': True, #skip set to true avoids repeating the same simulations ... need to verify if it skips whole generations or just individual simulations
            #'nodes': number_of_nodes,
            #'nodes': 1,
            'nodes': custom_kwargs['nodes_per_command'],
        }
        
        # Extract command kwargs
        mpiCommand = command_kwargs['mpiCommand']
        #numproc = command_kwargs['coresPerNode']*command_kwargs['nodes']
        numproc = command_kwargs['coresPerNode']
        nrnCommand = command_kwargs['nrnCommand']
        script = command_kwargs['script']
        
        # Build output folder and fitness target script paths
        mock_gen_dir = 'gen_i'
        mock_cfgsavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_cfg.json')
        mock_netparamssavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_netParams.json')
        cfgSavePath = mock_cfgsavepath
        netParamsSavePath = mock_netparamssavepath
        
        #this directly mimics the code in netpyne/batch/utils.py
        # ----------------------------------------------------------------------
        # MPI job command
        # ----------------------------------------------------------------------
        if mpiCommand == '':
            command = '%s %s simConfig=%s netParams=%s ' % (nrnCommand, script, cfgSavePath, netParamsSavePath)
            #print(f'Example Command:\n{command}')
        else:
            command = '%s -n %d %s -python -mpi %s simConfig=%s netParams=%s ' % (
                mpiCommand,
                numproc,
                nrnCommand,
                script,
                cfgSavePath,
                netParamsSavePath,
            )
            print(f'Example Command:\n{command}')
            print('command generated')
    elif sbatch:
        implemented=False
        assert implemented, 'sbatch not implemented yet'
    elif login_node:
        # User arguments
        shifter_command = 'shifter --image=adammwea/netsims_docker:v1'
        #shifter_command = 'shifter --image=adammwea/netpyneshifter:v5'
        init_script = 'modules/simulation_config/init.py'
        init_script = os.path.abspath(init_script)
        

        # Detect SLURM environment or use defaults
        cores_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))  # Default to 1 core per node if not in SLURM
        number_of_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))  # Default to 1 node if not in SLURM
        ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", cores_per_node))  # Default tasks per node = cores
        total_tasks = number_of_nodes * ntasks_per_node
        ntasks_per_node = 256
        tasks_per_node = int(ntasks_per_node/2) #use half the tasks per node to get a single simulation on a single socket
        
        #dummy variables for testing
        nodes_per_sim = 1
        cores_per_sim = 128
        cores_per_task = 16
        tasks_per_sim = 8
        
        kwargs['nodes_per_sim'] = nodes_per_sim
        kwargs['cores_per_sim'] = cores_per_sim
        kwargs['cores_per_task'] = cores_per_task
        kwargs['tasks_per_sim'] = tasks_per_sim
        
        command_kwargs = {
            'type': 'mpi_direct',
            'script': init_script,
            #'mpiCommand': 'srun',
            #'mpiCommand': f'srun',
            'mpiCommand': f'', 
            'nrnCommand': f'{shifter_command} nrniv', #get single simulation on a single socket
            #'coresPerNode': cores_per_node,
            #'coresPerNode': tasks_per_node,
            #'reservation': None,
            #'skip': USER_skip,
            #'nodes': number_of_nodes,
            'nodes': 1,
        }
        
        # Extract command kwargs
        mpiCommand = command_kwargs['mpiCommand']
        #numproc = command_kwargs['coresPerNode']*command_kwargs['nodes']
        nrnCommand = command_kwargs['nrnCommand']
        script = command_kwargs['script']
        
        # Build output folder and fitness target script paths
        mock_gen_dir = 'gen_i'
        mock_cfgsavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_cfg.json')
        mock_netparamssavepath = os.path.join(run_path, mock_gen_dir, f'{mock_gen_dir}_cand_j_netParams.json')
        cfgSavePath = mock_cfgsavepath
        netParamsSavePath = mock_netparamssavepath
        
        #this directly mimics the code in netpyne/batch/utils.py
        # ----------------------------------------------------------------------
        # MPI job command
        # ----------------------------------------------------------------------
        if mpiCommand == '':
            command = '%s %s simConfig=%s netParams=%s ' % (nrnCommand, script, cfgSavePath, netParamsSavePath)
        else:
            command = '%s -n %d %s -python -mpi %s simConfig=%s netParams=%s ' % (
                mpiCommand,
                numproc,
                nrnCommand,
                script,
                cfgSavePath,
                netParamsSavePath,
            )
    elif local:
        implemented=False
        assert implemented, 'local not implemented yet'
    else: 
        raise ValueError('No preset config selected')    
    
    #consolidate command kwargs
    kwargs.update(command_kwargs)
        
    # print/return command
    mock_command = command
    #print(f'Example Command:\n{mock_command}')    
    return mock_command, kwargs

def setup_environment_wrapper(verbose = False):
    from pprint import pprint
    import setup_environment as setup_environment
    setup_environment.set_pythonpath()
    import sys
    
    if verbose:
        #print sys path
        pprint(sys.path)
    #sys.exit(0)
    return sys.path

def add_output_path_to_kwargs(output_folder_name, fitness_target_script, outside_of_repo = False, **kwargs):
    #import setup_environment as setup_environment
    
    #workspace_path = setup_environment.get_git_root()
    fitness_target_script = os.path.abspath(fitness_target_script)
    #step out of workspace_path to avoid writing to the repository
    if outside_of_repo:
        workspace_path = os.path.dirname(workspace_path)
    output_folder_path = os.path.join(workspace_path, output_folder_name) # Output folder path for all runs
    output_folder_path = os.path.abspath(output_folder_path)
    
    kwargs['output_path'] = output_folder_path
    kwargs['fitness_target_script'] = fitness_target_script
    #kwargs['fitness_target_script'] = fitness_target_script
    
    return kwargs

def check_for_existing_runs(output_path, current_date):
    '''Check for existing runs in the output path'''
    # try:
    #     existing_runs = [run for run in os.listdir(output_path) if run.startswith(current_date)]
    # except:
    #     existing_runs = []
    
    try:
        existing_runs = [run for run in os.listdir(output_path)]
    except:
        existing_runs = []
        
    # Find the highest run number for the day
    if existing_runs:
        highest_run_number = max(int(run.split('_Run')[1].split('_')[0]) for run in existing_runs)
    else:
        highest_run_number = 0
    return existing_runs, highest_run_number

def add_run_path_to_kwargs(**kwargs):
    '''Build the output run path for the current run'''
    import datetime
    
    '''init'''
    rmtree = False
    mkrundir = False
    
    #extract current kwargs
    #output_path = kwargs.get('output_path', None)
    overwrite_run = kwargs.get('overwrite_run', False)
    continue_run = kwargs.get('continue_run', False)
    continue_latest_run = kwargs.get('continue_latest_run', False)
    label = kwargs.get('label', 'default')
    output_path = kwargs.get('output_path', None)
    assert output_path is not None, 'output_path must be specified'
    
    #initialize current_date
    current_date = datetime.datetime.now().strftime('%y%m%d')     # Get current date in YYMMDD format
    existing_runs, highest_run_number = check_for_existing_runs(output_path, current_date)     # Get list of existing runs for the day

    # Increment the run number for the new run
    new_run_number = highest_run_number + 1
    prev_run_number = new_run_number - 1

    # Update run_name with the new format
    run_name = f'{current_date}_Run{new_run_number}_{label}'
    prev_run_name = f'{current_date}_Run{prev_run_number}_{label}'

    # Get unique run path
    run_path = os.path.join(output_path, run_name)
    prev_run_path = os.path.join(output_path, prev_run_name)
    
    #get latest run path - which may have run on a different day
    for dirs in os.listdir(output_path):
        if label in dirs:
            run_path = os.path.join(output_path, dirs)
            break
    latest_run_path = run_path
    lasted_run_name = os.path.basename(latest_run_path)
    

    # Check if run exists and if it should be overwritten or continued
    if overwrite_run or continue_run:
        if prev_run_name in existing_runs:
            if overwrite_run and os.path.exists(prev_run_path):
                #print(f'Overwriting previous run at: {prev_run_path}')
                # print(f'Deleting previous run at: {prev_run_path}')
                #shutil.rmtree(prev_run_path)
                rmtree = True
                #assert that prev_run_path has been deleted
                #assert not os.path.exists(prev_run_path), f'Previous run at {prev_run_path} was not deleted'
                run_path = prev_run_path
            elif continue_latest_run:
                print(f'Continuing latest run at: {latest_run_path}')
                run_path = latest_run_path
            elif continue_run and os.path.exists(prev_run_path):
                #print(f'Continuing previous run at: {prev_run_path}')
                run_path = prev_run_path
    elif os.path.exists(prev_run_path) and os.path.isdir(prev_run_path) and not os.listdir(prev_run_path):
        rmtree = True #delete empty directory even if not overwriting, avoid creating new directory unnecessarily
        run_path = prev_run_path

    # Create the new run directory if it does not exist
    if not os.path.exists(run_path):
        #print(f'Creating new run directory at: {run_path}')
        #os.makedirs(run_path)
        mkrundir = True
        
    #update kwargs
    kwargs['run_path'] = run_path
    kwargs['rmtree'] = rmtree
    kwargs['mkrundir'] = mkrundir
    kwargs['prev_run_path'] = prev_run_path

    return kwargs

def init_output_paths_as_needed(rmtree, mkrundir, **kwargs):
    '''Initialize output paths as needed'''
    import shutil
    import time
    
    #extract current kwargs
    output_path = kwargs.get('output_path', None)
    prev_run_path = kwargs.get('prev_run_path', None)
    run_path = kwargs.get('run_path', None)
    
    #remove previous run if rmtree is True
    if rmtree:
        print(f'Overwriting previous run at: {prev_run_path}')
        print(f'Deleting previous run at: {prev_run_path}')
        shutil.rmtree(prev_run_path)
        #delay for a few seconds to allow the system to catch up
        time.sleep(2)
        assert not os.path.exists(prev_run_path), f'Previous run at {prev_run_path} was not deleted'
        
    if mkrundir:
        print(f'Creating new run directory at: {run_path}')
        os.makedirs(run_path, exist_ok=True)
        time.sleep(2)
        assert os.path.exists(run_path), f'New run directory at {output_path} was not created'
    
    return

## Function to serialize the batch_config dictionary
import json
class batchcfgEncoder(json.JSONEncoder):
    def default(self, obj):
        if callable(obj):
            return str(obj)
        return super().default(obj)

## Function to initialize batch config
def init_batch_cfg(USER_vars, **kwargs):
    '''
    Generate Config
    '''
    
    # Configure global user variables
    USER_run_path = USER_vars.get('USER_run_path', None)
    USER_fitness_target_script = USER_vars.get('USER_fitness_target_script', None)
    USER_seed_evol = USER_vars.get('USER_seed_evol', False)
    USER_method = USER_vars.get('USER_method', None)
    USER_cfg_file = USER_vars.get('USER_cfg_file', None)
    USER_netParamsFile = USER_vars.get('USER_netParamsFile', None)
    USER_runCfg_type = USER_vars.get('USER_runCfg_type', None)
    USER_init_script = USER_vars.get('USER_init_script', None)
    USER_mpiCommand = USER_vars.get('USER_mpiCommand', None)
    USER_nrnCommand = USER_vars.get('USER_nrnCommand', None)
    USER_nodes = USER_vars.get('USER_nodes', None)
    USER_cores_per_node = USER_vars.get('USER_coresPerNode', None)
    USER_skip = USER_vars.get('USER_skip', None)
    USER_pop_size = USER_vars.get('USER_pop_size', None)
    USER_num_elites = USER_vars.get('USER_num_elites', None)
    USER_mutation_rate = USER_vars.get('USER_mutation_rate', None)
    USER_crossover = USER_vars.get('USER_crossover', None)
    USER_max_generations = USER_vars.get('USER_max_generations', None)
    USER_time_sleep = USER_vars.get('USER_time_sleep', None)
    USER_maxiter_wait = USER_vars.get('USER_maxiter_wait', None)

    # Import fitness function and arguments
    fitnessFuncArgs = import_module_from_path(USER_fitness_target_script) # dynamically import fitnessFuncArgs from USER_fitness_target_script defined as python scripts so that we can optimize different data
    fitnessFuncArgs = fitnessFuncArgs.fitnessFuncArgs
    
    #from optimization_projects.CDKL5_DIV21.scripts.batch_scripts.calculate_fitness_v3 import fitnessFunc
    preset_configs = kwargs.get('preset_configs', None)
    login_node = preset_configs.get('login_node', False)
    interactive_node = preset_configs.get('interactive_node', False)
    sbatch = preset_configs.get('sbatch', False)
    local = preset_configs.get('local', False)
    
    # Configure global user variables
    #for debugging
    interactive_node = True
    login_node = False
    #debugging
    # if login_node or local:
    #         from optimization_projects.CDKL5_DIV21.scripts.batch_scripts.calculate_fitness_vCurrent import fitnessFunc
    # elif interactive_node or sbatch:
    #         from optimization_projects.CDKL5_DIV21.scripts.batch_scripts.fitness_helper import submit_fitness_job as fitnessFunc
    # else:
    #     raise ValueError('No preset config selected')
    from optimization_projects.CDKL5_DIV21.scripts.batch_scripts.calculate_fitness_vCurrent import fitnessFunc
    
    # Load HOF seeds if USER_seed_evol is True
    seed_paths = kwargs.get('seed_paths', None)
    #HOF_seeds = get_HOF_seeds(**kwargs) if USER_seed_evol else None
    seed_cfgs = get_seed_cfgs(**kwargs) if USER_seed_evol else None

    batch_config_options = {
        "run_path": USER_run_path,
        'batchLabel': os.path.basename(USER_run_path),
    }

    batch_config = {
        'batchLabel': batch_config_options['batchLabel'],
        'saveFolder': batch_config_options['run_path'],
        'method': USER_method,
        'cfgFile': USER_cfg_file,
        'netParamsFile': USER_netParamsFile,
        'runCfg': {
            'type': USER_runCfg_type,
            'script': USER_init_script,
            'mpiCommand': USER_mpiCommand,
            'nrnCommand': USER_nrnCommand,
            'nodes': USER_nodes,
            'coresPerNode': USER_cores_per_node,
            'reservation': None,
            'skip': USER_skip,
        },
        'evolCfg': {
            'evolAlgorithm': 'custom',
            'fitnessFunc': fitnessFunc,
            'fitnessFuncArgs': {**fitnessFuncArgs, 'pop_size': USER_pop_size},
            'pop_size': USER_pop_size,
            'num_elites': USER_num_elites,
            'mutation_rate': USER_mutation_rate,
            'crossover': USER_crossover,
            'maximize': False,
            'max_generations': USER_max_generations,
            'time_sleep': USER_time_sleep,
            'maxiter_wait': USER_maxiter_wait,
            'defaultFitness': 1000,
            'seeds': seed_cfgs,
        }
    }

    batch_run_path = batch_config['saveFolder']

    # Save batch_config in run_path as JSON    
    # with open(f'{batch_run_path}/batch_config.json', 'w') as f:
    #     json.dump(batch_config, f, cls=batchcfgEncoder, indent=4)

    # Validate batch_config before running
    assert 'method' in batch_config, 'method must be specified in batch_config'
    assert 'skip' in batch_config['runCfg'], 'skip must be specified in batch_config'
    assert 'batchLabel' in batch_config, 'batchLabel must be specified in batch_config'

    return batch_config

## Function to run the batch
def batchRun(batch_config=None, **kwargs):
    global params #make it global so it can be accessed by the cfg.py file easily
    
    '''Main function to run the batch'''
    assert batch_config is not None, 'batch_config must be specified'  # Ensure batch_config is provided    
    #params = evolutionary_parameter_space.params # Get parameter space from user-defined file
    params = kwargs.get('param_space', None)
    params = rangify_params(params) # Convert parameter space to ranges
    from netpyne.batch import Batch
    batch = Batch(params=params) # Create Batch object with parameters to modify

    # Set attributes from batch_config to batch object
    for key, value in batch_config.items():
        if hasattr(batch, key):
            setattr(batch, key, value)

    # Prepare method-specific parameters
    batch.runCfg = batch_config['runCfg']
    if 'evolCfg' in batch_config:
        batch.evolCfg = batch_config['evolCfg']

    # Run the batch
    batch.run()

def pre_run_checks(USER_vars, **kwargs):
    from pprint import pprint
    
    print("\n" + "="*50)
    print("PRE-RUN CHECKS")
    print("="*50 + "\n")
    
    def alert_user(*args, color='yellow'):
        colors = {
            'yellow': '\033[93m',
            'red': '\033[91m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        for arg in args:
            print(f"{colors[color]}{arg}{colors['reset']}")
    
    def print_dict(d, indent=0, color='yellow'):
        for key, value in d.items():
            if isinstance(value, dict):
                alert_user(' ' * indent + f'{key}:', color=color)
                print_dict(value, indent + 4, color)
            else:
                alert_user(' ' * indent + f'{key}: {value}', color=color)
    
    #print all USER_vars in yellow, as a list
    for key, value in USER_vars.items():
        alert_user(f'{key}: {value}', color='yellow')
    print('')  # newline
    
    # Print batch config in yellow
    batch_cfg = kwargs.get('batch_cfg', None)
    if batch_cfg:
        alert_user('Batch config:', color='yellow')
        print_dict(batch_cfg)
        print('')  # newline
    #
    import os   
    output_path = kwargs.get('output_path', None)
    assert output_path is not None, 'output_folder_path must be specified'
    alert_user(f'Output path: {output_path}', color='white')
    print('') #newline    
    
    run_path = kwargs.get('run_path', None)
    assert run_path is not None, 'run_path must be specified'
    alert_user(f'Run path: {run_path}', color='white')
    print('') #newline
    
    example_command = kwargs.get('example_command', None)
    alert_user(f'Example command:\n{example_command}', color='white')
    print('') #newline
    
    run_path = kwargs.get('run_path')
    rmtree = kwargs.get('rmtree')
    mkrundir = kwargs.get('mkrundir')
    overwrite = kwargs.get('overwrite')
    continue_run = kwargs.get('continue_run')
    run_path_exists = os.path.exists(run_path)   
    alert_user(f"Flags:\n- rmtree={rmtree}\n- mkrundir={mkrundir}\n- overwrite={overwrite}\n- continue_run={continue_run}", color='white')
    print('') #newline
    
    if rmtree and overwrite:
        assert run_path_exists, 'run_path must exist to use rmtree - this is flagged incorrectly by the program. please check the code.'
        alert_user('Overwrite is set to True, and rmtree has been flagged by the program to do this properly.',
                   'This will delete the entire run_path and all of its contents.', color='red')
    if rmtree and not overwrite:
        assert run_path_exists, 'run_path must exist to use rmtree - this is flagged incorrectly by the program. please check the code.'
        alert_user('Overwrite is set to False, and rmtree has been flagged.',
                   'This is probably because the current run_path is empty and removing it is an easy way to deal with it before initiating the run.',
                   'If the run_path is not empty, this will delete the entire run_path and all of its contents.', color='yellow')
    if mkrundir and run_path_exists:
        alert_user('The run_path already exists.',
                   'This is probably because the run_path is being reused from a previous run.',
                   'If this is not the case, the run_path will be deleted and recreated.', color='yellow')
    if mkrundir and not run_path_exists:
        alert_user('The run_path does not exist.',
                   'This is probably because the run_path is being created for the first time.',
                   'If this is not the case, the run_path will be created.', color='yellow')
    if continue_run and run_path_exists:
        alert_user('Continue run is set to True and the run_path already exists.',
                   'Make sure that you mean to select these options.',
                   'Running the command will continue the run from the latest run in the output folder.', color='yellow')
    
    print("\n" + "="*50)
    print("END OF PRE-RUN CHECKS")
    print("="*50 + "\n")
    
    # Wait for user input before running
    try:
        user_input = input('Press Enter to run the command, or "c" to cancel: ')
        if user_input.lower() == 'c':
            print('Command execution cancelled.')
            sys.exit()
    except KeyboardInterrupt:
        print('\nCommand execution cancelled by user.')
        sys.exit()
    
## Function to get HOF seeds
# def get_seed_cfgs(**kwargs):
#     seed_paths = kwargs.get('seed_paths', None)
#     assert seed_paths is not None, 'seed_paths must be specified'
    
#     '''Load seeds from Hall of Fame'''
    
#     #extract current pop_size
#     pop_size = kwargs.get('pop_size', None)
#     assert pop_size is not None, 'pop_size must be specified'
    
#     seed_data_paths = [seed_path['path'] for seed_path in seed_paths if seed_path['seed'] is True]
#     #seed_cfg_paths = [seed_path['path'].replace('_data.json', '_cfg.json') for seed_path in seed_paths if seed_path['seed'] is True]
    
#     seeds = []
#     for seed_path in seed_data_paths:
#         cfg_path = seed_path.replace('_data.json', '_cfg.json')
#         with open(cfg_path, 'r') as f:
#             seed = json.load(f)
#         seed_cfg = seed['simConfig']
        
#         #only keep overlap with USER_evelo_param_space.py
#         #from workspace.optimization_projects.CDKL5_DIV21_dep.parameter_spaces._241202_adjusted_evol_params import params
#         params = kwargs.get('param_space', None)        
#         parameter_space = params
#         seed_cfg = {
#             key: parameter_space[key][0] if isinstance(parameter_space[key], list) and parameter_space[key][0] == parameter_space[key][1] 
#             else seed_cfg[key] for key in parameter_space if key in seed_cfg
#         }
        
#         seed_cfg = list(seed_cfg.values()) #get rid of keys, just make list of values
#         seed_cfg = [float(val) for val in seed_cfg] #make sure all values are floats
#         seeds.append(seed_cfg)
#         print(f'Successfully loaded seed from {cfg_path}')
#         if len(seeds) >= pop_size: break
        
#     print(f'Loaded {len(seeds)} seeds')
#     assert len(seeds) > 0, 'No seeds loaded'
#     return seeds
        
def save_global_kwargs_to_file(**kwargs):
    """
    Saves provided keyword arguments to a temporary Python file
    for later use in simulations.
    """
    try:
        # Extract the 'run_path' from the kwargs
        run_path = kwargs.get('run_path')
        if not run_path:
            raise ValueError("The 'run_path' argument is required.")

        # Define paths for the temporary and final files
        script_path = os.path.dirname(os.path.realpath(__file__))
        temp_global_kwargs_path = os.path.join(script_path, 'temp_global_kwargs.py')
        global_kwargs_path = os.path.join(run_path, 'temp_global_kwargs.py')

        # Remove existing temporary file if it exists
        if os.path.exists(temp_global_kwargs_path):
            os.remove(temp_global_kwargs_path)

        # Write kwargs to the temporary file
        with open(temp_global_kwargs_path, "w") as f:
            for key, value in kwargs.items():
                if isinstance(value, str):
                    if '\n' in value:
                        f.write(f"{key} = '''{value}'''\n")
                    else:
                        f.write(f"{key} = '{value}'\n")
                elif isinstance(value, (bool, int, float)):
                    f.write(f"{key} = {value}\n")
                else:
                    # For complex types, convert to a string representation
                    f.write(f"{key} = {repr(value)}\n")

        # Ensure the run directory exists
        os.makedirs(run_path, exist_ok=True)

        # Copy the temporary file to the run directory
        shutil.copy(temp_global_kwargs_path, global_kwargs_path)

    except Exception as e:
        print(f"Error in save_global_kwargs_to_file: {e}")
        raise

def save_global_kwargs_to_json(**kwargs):
    """
    Saves provided keyword arguments to a JSON file
    for later use in simulations.
    """
    try:
        # Extract the 'run_path' from the kwargs
        run_path = kwargs.get('run_path')
        if not run_path:
            raise ValueError("The 'run_path' argument is required.")

        # Define paths for the final JSON file
        os.makedirs(run_path, exist_ok=True)
        global_kwargs_path = os.path.join(run_path, 'global_kwargs.json')

        # Write kwargs to the JSON file with indentation
        with open(global_kwargs_path, "w") as f:
            json.dump(kwargs, f, indent=4)

        print(f"Saved global kwargs to {global_kwargs_path}")

    except Exception as e:
        print(f"Error in save_global_kwargs_to_json: {e}")
        raise
    
def save_global_kwargs_to_files(**kwargs):
    """
    Saves provided keyword arguments to both a pickle file and a Python file
    for later use in simulations.
    """
    try:
        # Extract the 'run_path' from the kwargs
        run_path = kwargs.get('run_path')
        if not run_path:
            raise ValueError("The 'run_path' argument is required.")

        # Define paths for the files
        os.makedirs(run_path, exist_ok=True)
        global_kwargs_pickle_path = os.path.join(run_path, 'global_kwargs.pkl')
        global_kwargs_py_path = os.path.join(run_path, 'global_kwargs.py')

        # Save as a pickle file
        with open(global_kwargs_pickle_path, "wb") as pickle_file:
            pickle.dump(kwargs, pickle_file)
        print(f"Saved global kwargs to pickle file: {global_kwargs_pickle_path}")

        # Save as a Python file with a dictionary
        with open(global_kwargs_py_path, "w") as py_file:
            py_file.write("# This file is auto-generated. Do not edit manually.\n")
            py_file.write("global_kwargs = {\n")
            for key, value in kwargs.items():
                if isinstance(value, str):
                    py_file.write(f"    '{key}': '''{value}''',\n" if '\n' in value else f"    '{key}': '{value}',\n")
                elif isinstance(value, (bool, int, float)):
                    py_file.write(f"    '{key}': {value},\n")
                else:
                    py_file.write(f"    '{key}': {repr(value)},\n")
            py_file.write("}\n")
        print(f"Saved global kwargs to Python file: {global_kwargs_py_path}")

    except Exception as e:
        print(f"Error in save_global_kwargs_to_files: {e}")
        raise
  
        
        
    
    # print(f'Loading Hall of Fame from {USER_HOF}')
    # assert os.path.exists(USER_HOF), f'USER_HOF file not found: {USER_HOF}'
    # seeded_HOF_cands = pd.read_csv(USER_HOF).values.flatten()
    # seeded_HOF_cands = [cfg.replace('_data', '_cfg') for cfg in seeded_HOF_cands]
    # seeded_HOF_cands = [os.path.abspath(f'./{cfg}') for cfg in seeded_HOF_cands]
    # for cfg in seeded_HOF_cands:
    #     if 'NERSC/NERSC' in cfg: 
    #         seeded_HOF_cands[seeded_HOF_cands.index(cfg)] = cfg.replace('NERSC/NERSC', 'NERSC')
    #     else: continue

    # seeds = []
    # for cfg in seeded_HOF_cands:
    #     if not os.path.exists(cfg): cfg = None #check if file exists, else set to None       
    #     if cfg is not None:    
            
    #         # open cfg file and extract simConfig
    #         with open(cfg, 'r') as f:
    #             seed = json.load(f)
    #         seed = seed['simConfig']
            
    #         #only keep overlap with USER_evelo_param_space.py
    #         seed = {
    #             key: evolutionary_parameter_space[key][0] if evolutionary_parameter_space[key][0] == evolutionary_parameter_space[key][1] 
    #             else seed[key] for key in evolutionary_parameter_space if key in seed
    #         }

    #         seed = list(seed.values()) #get rid of keys, just make list of values
    #         seed = [float(val) for val in seed] #make sure all values are floats
    #         seeds.append(seed)
    #         print(f'Successfully loaded seed from {cfg}')
    #         if len(seeds) >= USER_pop_size: break
    #     else: continue

    # print(f'Loaded {len(seeds)} seeds from Hall of Fame')
    # assert len(seeds) > 0, 'No seeds loaded from Hall of Fame'
    # return seeds

def save_global_kwargs_to_pickle(**kwargs):
    """
    Saves provided keyword arguments to a pickle file in the current working directory.
    """
    try:
        # Define the path for the pickle file in the current working directory
        # pickle_file_path = os.path.join(os.getcwd(), 'global_kwargs.pkl')
        
        # Define the path for the pickle file in the script directory
        script_path = os.path.dirname(os.path.realpath(__file__))
        pickle_file_path = os.path.join(script_path, 'temp_global_kwargs.pkl')

        # Save the keyword arguments to the pickle file
        with open(pickle_file_path, "wb") as pickle_file:
            pickle.dump(kwargs, pickle_file)

        print(f"Saved global kwargs to pickle file: {pickle_file_path}")
        
        return pickle_file_path

    except Exception as e:
        print(f"Error in save_global_kwargs_to_pickle: {e}")
        raise
# ______________________________________________________________________________________________________________________

