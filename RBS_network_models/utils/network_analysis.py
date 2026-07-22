# import MEA_Analysis.NetworkAnalysis_aw.compute_network_metrics as cnm
import MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis as cnm
import RBS_network_models.utils.netpyne_helpers as nph
import os
# import MEA_Analysis.NetworkAnalysis_aw.network_metrics_helper as nmh
import numpy as np
from multiprocessing import Process, Queue, Value, Lock, cpu_count
from pathlib import Path
from threading import Thread
from netpyne import sim
from rich.console import Console
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, TimeRemainingColumn
import time

import spikeinterface as si
import glob 

# === Globals ===
completed_tasks = Value('i', 0)
failed_tasks = Value('i', 0)
lock = Lock()
console = Console()

# === Functions ===

def import_conv_params(conv_path):
    """
    Import convolution parameters from a specified path.
    
    Parameters:
    conv_path (str): Path to the convolution parameters file.
    
    Returns:
    dict: convolution parameters.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("conv_params", conv_path)
    conv_params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv_params)
    
    #conv_params = conv_params.conv_params  # Assuming the conversion parameters are stored in a variable named 'conv_params' in the file.
    #mega_params = conv_params.mega_params  # Assuming the mega parameters are stored in a variable named 'mega_params' in the file.
    
    burst_params = conv_params.burst_params if hasattr(conv_params, 'burst_params') else None
    hyperburst_params = conv_params.hyperburst_params if hasattr(conv_params, 'hyperburst_params') else None
    
    if burst_params is None:
        raise ValueError("burst_params is None, but it should be defined in the convolution parameters file.")
    if hyperburst_params is None:
        raise ValueError("hyperburst_params is not None, but it should be defined in the convolution parameters file.")
    
    return burst_params, hyperburst_params

def _get_data_files(target_dir, exclude=None, **kwargs):
    """
    Get all _data.pkl or _data.json files in the target directory recursively.
    """
    source = kwargs.get('source', 'simulated')
        
    if source == 'experimental':
            # Look for spikeinterface_recording.json
            if kwargs.get('recursive', True):
                pattern = os.path.join(target_dir, '**', 'spikeinterface_recording.json')
            else:
                pattern = os.path.join(target_dir, 'spikeinterface_recording.json')
            data_files = glob.glob(pattern, recursive=True)
    else:
            pkl_files = nph.get_sim_pkl_paths(target_dir)
            json_files = nph.get_sim_json_paths(target_dir)
            data_files = pkl_files + json_files
        
    if exclude:
            data_files = [f for f in data_files if not any(excl in f for excl in exclude)]
    return data_files
def _unpack_network_metrics_args(data_file, **kwargs):
    """
    Unpack the arguments for network metrics computation from the data file and kwargs.
    """
    
    # load simulation data components needed for network metrics computation
    allSimData, allPops, allCells = nph.load_sim_data(data_file)
    
    # define the output directory based on the data file path - save data in the same directory as the _data.pkl file
    output_dir = os.path.dirname(data_file)
    
    # check if conv_path is provided in kwargs, if not raise an error
    conv_path = kwargs.get('conv_path', None)
    if conv_path is None: 
        raise ValueError("conv_path must be defined prior to unpacking the network analysis parameters.")
    
    # import convolution parameters from the specified path
    burst_params, hyperburst_params = import_conv_params(kwargs.get('conv_path', None))
    
    # source must be defined in kwargs for network analysis to proceed
    source = kwargs.get('source', None)
    if source is None:
        raise ValueError("source must be defined prior to running the network analysis.")
    
    # repack the arguments into a dictionary for network metrics computation
    nm_kwargs = {
        # raw data
        'sim_data_path': data_file,
        'source': source,  # 'simulated' or 'experimental'
        #'conv_path': kwargs.get('conv_path', None),
        'burst_params': burst_params,
        'hyperburst_params': hyperburst_params,
        'simData': allSimData,
        'popData': allPops,
        'cellData': allCells,
        
        # output directory
        'output_dir': output_dir,
        
        # runtime options
        'parallel': kwargs.get('parallel', False),
        
        # analysis options
        'compute_spike_metrics': kwargs.get('compute_spike_metrics', True),
        'compute_burst_metrics': kwargs.get('compute_burst_metrics', True),
        'classify_units': kwargs.get('classify_units', False),
        'compute_dynamic_time_warping': kwargs.get('compute_dynamic_time_warping', False),
        'compute_summary_metrics': kwargs.get('compute_summary_metrics', True),
    }

    return nm_kwargs

def _submit_slurm_jobs(data_files, **kwargs):
    """
    Submit SLURM jobs for each data file.
    """
    # This function should implement the logic to submit SLURM jobs for each data file.
    # For now, we will just print the files that would be processed.
    for data_file in data_files:
        
        allSimData, allPops, allCells = nph.load_sim_data(data_file)
        output_dir = os.path.dirname(data_file)
        conv_params
        swargs = {
            'sim_data_path': data_file,
            'conv_path': kwargs.get('conv_path', None),
            'simData': allSimData,
            'popData': allPops,
            'cellData': allCells,
            'output_dir': output_dir,         
        }
        raise NotImplementedError("SLURM job submission is not implemented yet.")

def prep_cnm_kwargs(data_file, **kwargs):
    """
    Format the arguments for network metrics computation from the data file and kwargs.
    """
    # init
    print(f"Preparing network metrics arguments for {data_file}...")
    source = kwargs.get('source', 'simulated')
    
    if source == 'experimental':
        # Experimental Data Loading Logic
        assert os.path.exists(data_file), f"Data file {data_file} does not exist."
        folder_path = os.path.dirname(data_file)
        print(f"Output directory set to {folder_path}.")
        
        # Load Recording
        try:
            recording = si.load_extractor(data_file)
        except Exception as e:
            raise ValueError(f"Failed to load recording from {data_file}: {e}")

        # Load Sorting (expecting 'sorter_output' folder next to recording json)
        sorter_output_path = os.path.join(folder_path, 'sorter_output')
        if os.path.exists(sorter_output_path):
            try:
                # Try reading as Kilosort output
                sorting = si.read_kilosort(sorter_output_path)
            except:
                try:
                    # Try loading as SI object
                    sorting = si.load_extractor(sorter_output_path)
                except Exception as e:
                     raise ValueError(f"Failed to load sorting from {sorter_output_path}: {e}")
        else:
            raise ValueError(f"Could not find 'sorter_output' directory in {folder_path}")

        # Create Sorting Analyzer (required for metrics)
        # We create it in memory; it will compute templates/waveforms on demand if needed by cnm
        print("Creating SortingAnalyzer...")
        sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")

        # Import convolution parameters
        conv_path = kwargs.get('conv_path', None)
        if conv_path is None: 
            raise ValueError("conv_path must be defined.")
        burst_params, hyperburst_params = import_conv_params(conv_path)
        
        # Construct kwargs for compute_network_metrics
        nm_kwargs = {
            'source': 'experimental',
            'sim_data_path': data_file, # kept for reference
            'recording_object': recording,
            'sorting_object': sorting,
            'sorting_analyzer': sorting_analyzer,
            'burst_params': burst_params,
            'hyperburst_params': hyperburst_params,
            'output_dir': folder_path,
            'parallel': kwargs.get('parallel', False),
            'compute_spike_metrics': kwargs.get('compute_spike_metrics', True),
            'compute_burst_metrics': kwargs.get('compute_burst_metrics', True),
            'classify_units': kwargs.get('classify_units', False),
            'compute_dynamic_time_warping': kwargs.get('compute_dynamic_time_warping', False),
            'compute_summary_metrics': kwargs.get('compute_summary_metrics', True),
        }
        return nm_kwargs
    else:
        
        # load simulation data components needed for network metrics computation
        assert os.path.exists(data_file), f"Data file {data_file} does not exist."
        assert data_file.endswith(('.pkl', '.json')), f"Data file {data_file} must be a .pkl or .json file."
        allSimData, allPops, allCells = nph.load_sim_data(data_file)
        print(f"Loaded simulation data from {data_file}.")
        
        # define the output directory based on the data file path - save data in the same directory as the _data.pkl file
        output_dir = os.path.dirname(data_file)    
        print(f"Output directory set to {output_dir}.")
        
        # check if conv_path is provided in kwargs, if not raise an error
        conv_path = kwargs.get('conv_path', None)
        if conv_path is None: 
            raise ValueError("conv_path must be defined prior to unpacking the network analysis parameters.")
        print(f"Using convolution parameters from {conv_path}.")
        
        # import convolution parameters from the specified path
        burst_params, hyperburst_params = import_conv_params(kwargs.get('conv_path', None))
        print(f"Imported burst and hyperburst parameters.")
        
        # format raw spiking data
        print(f"Formatting raw spiking data...")
        t = allSimData.t.copy()
        spkt = allSimData.spkt.copy()
        spkid = allSimData.spkid.copy()
        if len(spkt) == 0:
            print(f"Warning: No spikes found in {data_file}.")
            raise ValueError(f"No spikes found in {data_file}. Cannot compute network metrics.")
        time_vector = np.array(t) / 1000 # convert to seconds
        spike_times = np.array(spkt) / 1000 # convert to seconds
        spike_times_by_unit = {int(i): spike_times[spkid == i] for i in np.unique(spkid)}
        print(f"Formatted raw spiking data with {len(spike_times)} spikes across {len(spike_times_by_unit)} units.")
        
        # parse known classes for units, if any
        unit_pops = {}
        for pop in allPops:
            for cell in allCells:
                if 'tags' in cell and 'pop' in cell['tags']:
                    if cell['tags']['pop'] == pop:
                        unit_pops[cell['gid']] = pop
                
            
        # repack the arguments into a dictionary for network metrics computation
        nm_kwargs = {
            
            #kwargs
            #**kwargs,
            
            # info for reference
            'sim_data_path': data_file,
            'source': 'netpyne',  # 'simulated'
            
            # save path
            #'save_path'
            
            # convolution params
            'burst_params': burst_params,
            'hyperburst_params': hyperburst_params,
            
            # raw data (required)
            'spkt': spike_times,
            'spkt_by_unit': spike_times_by_unit,
            't': time_vector,
            
            # raw data (optional)
            't_unit': 's',  # time unit is seconds
            'unit_pops': unit_pops,  # dictionary to store unit pops, e.g., {'unit1': 'E', 'unit2': 'I', ...}
            
            # output directory
            'output_dir': output_dir,
            
            # runtime options
            'parallel': kwargs.get('parallel', False),
            'try_load': kwargs.get('try_load', True),  # whether to try loading existing results
            
            # analysis options
            'compute_spike_metrics': kwargs.get('compute_spike_metrics', True),
            'compute_burst_metrics': kwargs.get('compute_burst_metrics', True),
            'classify_units': kwargs.get('classify_units', False),
            'compute_dynamic_time_warping': kwargs.get('compute_dynamic_time_warping', False),
            'compute_summary_metrics': kwargs.get('compute_summary_metrics', True),
        }


        return nm_kwargs

def _process_sequential(data_files, **kwargs):
    """
    Process each data file sequentially.
    """
    print(f"Processing {len(data_files)} files sequentially...")
    for data_file in data_files:
        
        #nm_kwargs = _unpack_network_metrics_args(data_file, **kwargs)
        nm_kwargs = prep_cnm_kwargs(data_file, **kwargs)
        print(f"Processing {data_file}...")
        cnm.compute_network_metrics(nm_kwargs)  # Call the network metrics computation function

# === Worker process ===
def _worker(queue, completed_tasks, failed_tasks, lock, max_workers, kwargs):
    while not queue.empty():
        #
        try:
            data_file = queue.get_nowait() # Get the next file from the queue
        except:
            break
        
        #
        try:
            nm_kwargs = prep_cnm_kwargs(data_file, **kwargs)
            print(f"Processing {data_file}...")
            nm_kwargs['max_workers'] = max_workers  # Set max workers for parallel processing
            print(f"Computing network metrics for {data_file} with max_workers={nm_kwargs['max_workers']}...")
            cnm.compute_network_metrics(nm_kwargs)  # Call the network metrics computation function
            print(f"Successfully processed {data_file}.")
            
            with lock:
                completed_tasks.value += 1
            #break  # Exit the loop on successful processing
        
        except Exception as e:
            console.print(f"[bold red]Error processing {data_file}: {e}")
            #continue
            with lock:
                failed_tasks.value += 1
            #break  # Exit the loop on error to avoid further processing

# === Monitor CLI progress ===
def _monitor_progress(total_tasks, completed_tasks, failed_tasks, lock):
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing", total=total_tasks)
        while not progress.finished:
            time.sleep(10)
            with lock:
                done = completed_tasks.value + failed_tasks.value
            progress.update(task, completed=done)
            if done >= total_tasks:
                break

# === Main entry ===
def _run_manual_process_pool(data_files, procs=None, **kwargs):
                    
    total_tasks = len(data_files)
    console.print(f"[bold blue]Found {total_tasks} simulation files to process.")
    # import sys
    # sys.exit(0)

    if total_tasks == 0:
        console.print("[bold red]No valid simulation files found.")
        return

    queue = Queue()
    for path in data_files:
        queue.put(path)

    num_sockets = min(8, cpu_count() // 2)
    console.print(f"[green]Launching {num_sockets} parent processes (each with up to 64 child workers).")
    
    max_workers = procs//num_sockets if procs else 4 # default to 4 workers per socket
    assert max_workers < procs, f"max_workers ({max_workers}) must be less than total processes ({procs})."
    console.print(f"[green]Using {max_workers} workers per process.")

    monitor_thread = Thread(target=_monitor_progress, args=(total_tasks, completed_tasks, failed_tasks, lock))
    monitor_thread.start()

    processes = []
    for _ in range(num_sockets):
        p = Process(target=_worker, args=(queue, completed_tasks, failed_tasks, lock, max_workers, kwargs))
        p.daemon = False  # Key: must be non-daemonic
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    monitor_thread.join()
    
    # print final results
    with lock:
        console.print(f"[bold green]Completed tasks: {completed_tasks.value}")
        console.print(f"[bold red]Failed tasks: {failed_tasks.value}")
        if failed_tasks.value > 0:
            console.print("[bold red]Some tasks failed. Please check the logs for details.")
    
    return "Completed"

def _process_in_parallel(data_files, **kwargs):
    
    #max_workers = kwargs.get('max_workers', None)
    #procs = kwargs.get('procs', 16)  # number of processes to use for parallel processing
    _run_manual_process_pool(data_files, **kwargs)
    
def run_network_analysis(target_dir, **kwargs):
    """
    Compute network analysis on all simulation data files in the target directory recursively.
    - .npy files are saved in the same directory as the _data.pkl file.
    - looks for _data.pkl or _data.json files in the target directory.
    """
    
    # collect all data files in the target directory
    data_files=_get_data_files(target_dir, **kwargs)
    console.print(f"[bold blue]Found {len(data_files)} simulation data files in {target_dir}.")
    #
    parallel = kwargs.get('parallel', False)
    if parallel:
        mpi = kwargs.get('mpi', False)
        if mpi:
            raise ValueError("MPI parallel processing is not supported yet. Please set 'mpi' to True or False.")
            # slurm = kwargs.get('slurm', False)
            # if slurm:
            #     # Handle SLURM-specific logic
            #     _submit_slurm_jobs(data_files, **kwargs)
            #     pass
            # else:
            #     raise ValueError("Non-SLURM mpi processing is not supported yet. Please set 'slurm' to True or False.")
            #     _submit_mpi_jobs(data_files, **kwargs)
        else:
            #raise ValueError("Non-MPI parallel processing is not supported yet. Please set 'mpi' to True or False.")
            _process_in_parallel(data_files, **kwargs)
    else:
        # simple sequential processing
        #raise ValueError("Sequential processing is not supported yet. Please set 'parallel' to True.")
        _process_sequential(data_files, **kwargs)
            
    #return network_data
    print(f"Network analysis completed for {len(data_files)} files in {target_dir}.")