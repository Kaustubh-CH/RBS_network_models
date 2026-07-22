import pickle
import numpy as np
import glob
import os
import json
import sys
from matplotlib.backends.backend_pdf import PdfPages

# Add the path to MEA_Analysis module
sys.path.insert(0, '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/RBS_network_models')

# Import the network plotting function
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_summary_v3 as plot_network_summary

# --- 1. Configuration ---
# The batch directory containing all generations
batch_dir = '/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Jan05_20s_noSeed/batch_runs/batch_2026-01-05_spiking_only'

# Specify which generations to process (e.g., [0, 1, 2] or range(0, 10))
generations_to_process = range(0, 17)  # Process gen_0 through gen_16

# Output directory for plots
output_dir = os.path.join(batch_dir, 'batch_plots_network_summary')
os.makedirs(output_dir, exist_ok=True)

print(f"Processing generations: {list(generations_to_process)}")
print(f"Batch directory: {batch_dir}")
print(f"Output directory: {output_dir}")

# Plotting parameters
limit_seconds = None  # Set to a number to limit the time window, or None for full duration

# --- 2. Helper Functions ---
def load_fitness_data(fitness_file):
    """Load fitness data from JSON file."""
    try:
        with open(fitness_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {fitness_file}: {e}")
        return None

def extract_fitness_score(fitness_data):
    """Extract overall fitness score from fitness data."""
    if fitness_data is None:
        return None
    
    # Try different possible fitness value locations
    if 'fitness' in fitness_data:
        return fitness_data['fitness']
    elif 'total_fitness' in fitness_data:
        return fitness_data['total_fitness']
    elif 'score' in fitness_data:
        return fitness_data['score']
    else:
        # If no overall score, calculate average of all 'fit' values
        fit_values = []
        def extract_fit_recursive(d):
            if isinstance(d, dict):
                if 'fit' in d:
                    fit_values.append(d['fit'])
                for v in d.values():
                    extract_fit_recursive(v)
        extract_fit_recursive(fitness_data)
        if fit_values:
            return np.mean(fit_values)
    return None

def convert_spike_data_to_network_metrics(sim_data, fitness_data=None):
    """
    Convert spike data from pickle file to network_metrics format expected by plot_network_summary.
    
    Args:
        sim_data: Dictionary containing 'spkt' (spike times) and 'spkid' (spike GIDs)
        fitness_data: Optional fitness data dictionary
    
    Returns:
        network_metrics: Dictionary in the format expected by plot_network_summary
    """
    
    # Extract spike times and IDs
    spk_times = np.array(sim_data['spkt'])
    spk_gids = np.array(sim_data['spkid'])
    
    # Get duration
    duration = sim_data.get('t', [-1])[-1]
    if duration == -1:
        duration = np.max(spk_times) if len(spk_times) > 0 else 1000
    
    # Create a minimal network_metrics structure
    # This structure mimics what extract_features.py expects
    network_metrics = {
        'spike_times': spk_times,
        'spike_gids': spk_gids,
        'duration_ms': duration,
        'sampling_frequency': 1000.0,  # Assuming 1kHz sampling for simulation data
        
        # Classification output - basic structure
        # You may need to adjust this based on your actual cell population ranges
        'classification_output': {
            'include_units': np.unique(spk_gids).tolist(),
            'classified_units': {},
        },
        
        # Convolution analysis output - basic structure
        'convolution_analysis_output': {
            'spike_times_ms': spk_times,
            'spike_gids': spk_gids,
            'duration_ms': duration,
        },
        
        # Sorting output info
        'sorting_output': 'simulated_data',
    }
    
    # Add basic classification for excitatory/inhibitory
    # Assuming GIDs 0-100 are excitatory, 101-200 are inhibitory
    # Adjust these ranges based on your simulation setup
    for gid in np.unique(spk_gids):
        if gid <= 100:
            network_metrics['classification_output']['classified_units'][gid] = {
                'desc': 'excit',
                'cluster': 0
            }
        else:
            network_metrics['classification_output']['classified_units'][gid] = {
                'desc': 'inhib',
                'cluster': 1
            }
    
    # Add fitness data if available
    if fitness_data is not None:
        network_metrics['fitness_data'] = fitness_data
        network_metrics['fitness_score'] = extract_fitness_score(fitness_data)
    
    return network_metrics

# --- 3. Main Processing Loop ---
all_generation_fitness = {}  # Store fitness data for all generations

for gen_num in generations_to_process:
    gen_dir = os.path.join(batch_dir, f'gen_{gen_num}')
    
    if not os.path.exists(gen_dir):
        print(f"Warning: Generation {gen_num} directory not found: {gen_dir}")
        continue
    
    # Find all pickle and fitness files
    pkl_files = sorted(glob.glob(os.path.join(gen_dir, "*_data.pkl")))
    fitness_files = sorted(glob.glob(os.path.join(gen_dir, "*_fitness.json")))
    
    print(f"\n{'='*60}")
    print(f"Processing Generation {gen_num}")
    print(f"Found {len(pkl_files)} data files and {len(fitness_files)} fitness files")
    print(f"{'='*60}")
    
    # Create output directory for this generation
    gen_output_dir = os.path.join(output_dir, f'gen_{gen_num}')
    os.makedirs(gen_output_dir, exist_ok=True)
    
    # Store fitness data for this generation
    gen_fitness_data = {}
    
    # Load all fitness data for this generation
    import re
    for fitness_file in fitness_files:
        basename = os.path.basename(fitness_file)
        # Extract candidate number from filename
        match = re.search(r'cand_(\d+)', basename)
        if match:
            cand_num = int(match.group(1))
            fitness_data = load_fitness_data(fitness_file)
            fitness_score = extract_fitness_score(fitness_data)
            if fitness_score is not None:
                gen_fitness_data[cand_num] = fitness_score
    
    # Store in all generations dict
    if gen_fitness_data:
        all_generation_fitness[gen_num] = gen_fitness_data
    
    # Create a mapping from candidate number to fitness file for quick lookup
    cand_to_fitness_file = {}
    for fitness_file in fitness_files:
        basename = os.path.basename(fitness_file)
        match = re.search(r'cand_(\d+)', basename)
        if match:
            cand_num = int(match.group(1))
            cand_to_fitness_file[cand_num] = fitness_file

    # Process activity data
    for file_name in pkl_files:
        try:
            # Extract candidate info from filename
            basename = os.path.basename(file_name)
            match = re.search(r'cand_(\d+)', basename)
            cand_num = match.group(1) if match else 'unknown'
            
            # Load full fitness data for this candidate
            fitness_data_full = None
            fitness_score = None
            if cand_num != 'unknown':
                cand_num_int = int(cand_num)
                if cand_num_int in cand_to_fitness_file:
                    fitness_data_full = load_fitness_data(cand_to_fitness_file[cand_num_int])
                    fitness_score = extract_fitness_score(fitness_data_full)
            
            print(f"  Processing candidate {cand_num}..." + (f" (fitness: {fitness_score:.2f})" if fitness_score is not None else " (no fitness)"))
            
            # Load Data
            with open(file_name, 'rb') as f:
                data = pickle.load(f)

            if 'simData' in data:
                sim_data = data['simData']
            else:
                sim_data = data

            # Check if spikes exist
            if 'spkt' not in sim_data or len(sim_data['spkt']) == 0:
                print(f"    Warning: No spikes in {basename}")
                continue

            # Convert spike data to network_metrics format
            network_metrics = convert_spike_data_to_network_metrics(sim_data, fitness_data_full)
            
            # Define output paths for different plot types
            # 3-panel plot (full duration)
            plot_3p_full = os.path.join(gen_output_dir, f'cand_{cand_num}_3panel_full.pdf')
            
            # 3-panel plots with different time windows
            plot_3p_35s = os.path.join(gen_output_dir, f'cand_{cand_num}_3panel_35s.pdf')
            plot_3p_140s = os.path.join(gen_output_dir, f'cand_{cand_num}_3panel_140s.pdf')
            
            # 2-panel plots
            plot_2p_full = os.path.join(gen_output_dir, f'cand_{cand_num}_2panel_full.pdf')
            plot_2p_35s = os.path.join(gen_output_dir, f'cand_{cand_num}_2panel_35s.pdf')
            
            # Generate plots using plot_network_summary
            print(f"    Generating 3-panel plots...")
            
            # Full duration 3-panel
            kwargs_3p_full = {
                'bursting_plot_path': None,
                'bursting_fig_path': None,
                'mode': '3p',
                'y_lim': (0, 25),
                'x_lim': (0, limit_seconds) if limit_seconds else None,
                'plot_class': True,
                'output_dir': gen_output_dir,
            }
            plot_network_summary(network_metrics, **kwargs_3p_full)
            
            # 35s window 3-panel
            kwargs_3p_35s = {
                'bursting_plot_path': None,
                'bursting_fig_path': None,
                'mode': '3p',
                'y_lim': (0, 25),
                'x_lim': (0, 35),
                'plot_class': True,
            }
            plot_network_summary(network_metrics, **kwargs_3p_35s)
            
            # 140s window 3-panel
            kwargs_3p_140s = {
                'bursting_plot_path': None,
                'bursting_fig_path': None,
                'mode': '3p',
                'y_lim': (0, 25),
                'x_lim': (0, 140),
                'plot_class': True,
            }
            plot_network_summary(network_metrics, **kwargs_3p_140s)
            
            print(f"    Generating 2-panel plots...")
            
            # Full duration 2-panel
            kwargs_2p_full = {
                'bursting_plot_path': None,
                'bursting_fig_path': None,
                'mode': '2p',
                'y_lim': (0, 25),
                'x_lim': (0, limit_seconds) if limit_seconds else None,
                'plot_class': True,
            }
            plot_network_summary(network_metrics, **kwargs_2p_full)
            
            # 35s window 2-panel
            kwargs_2p_35s = {
                'bursting_plot_path': None,
                'bursting_fig_path': None,
                'mode': '2p',
                'y_lim': (0, 25),
                'x_lim': (0, 35),
                'plot_class': True,
            }
            plot_network_summary(network_metrics, **kwargs_2p_35s)
            
            print(f"    Plots saved to {gen_output_dir}")
            
        except Exception as e:
            print(f"    Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "="*60)
print("All processing complete!")
print(f"Output directory: {output_dir}")
print("="*60)
