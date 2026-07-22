import argparse
import glob
import numpy as np
import os
import json
from multiprocessing import Pool
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_metrics_v2

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze and plot network metrics.")
    parser.add_argument('--batch-paths', nargs='+', required=True,
                        help='List of batch directories to search for _metrics.npy files.')
    parser.add_argument('--top-n', type=int, default=16,
                        help='Number of top fitness files to include.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes to use. Set to 1 to disable parallelism.')
    parser.add_argument('--y-lim', type=float, nargs=2, default=[0, 16],
                        help='Y-axis limits for plotting.')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel plotting. Ignored if num-workers == 1.')
    return parser.parse_args()

def load_metrics_file(fpath):
    metrics = np.load(fpath, allow_pickle=True).item()
    print(f'Loaded {fpath}')
    return metrics

def get_metrics_files(batch_paths):
    metrics_files = []
    for batch_path in batch_paths:
        metrics_files += glob.glob(f'{batch_path}/**/*_metrics.npy', recursive=True)
    return metrics_files

def sort_by_fitness(metrics_files, top_n):
    fitness_values = []
    valid_files = []

    for fpath in metrics_files:
        json_path = fpath.replace('_metrics.npy', '_fitness.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                fit_value = data.get('fit', 1000)
                fitness_values.append(fit_value)
                valid_files.append(fpath)

    sorted_idx = np.argsort(fitness_values)
    top_files = np.array(valid_files)[sorted_idx][:top_n]
    top_fitness = np.array(fitness_values)[sorted_idx][:top_n]

    return top_files.tolist(), top_fitness.tolist()

def main():
    debug = False  # 🔁 Flip to True when debugging

    # uncomment this to run in debug mode
    # debug = True  # 🔁 Flip to True when debugging
    # debug_args = [
    #     '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-25_normBRandFRR_optBLandAmps_db'
    # #    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios',
    # #    '--output_dir',
    # #    '/global/homes/a/adammwea/pscratch/z_simulated_data/CDKL5-E6D_T2_C1_05212024/DIV21_WT/batch_runs/batch_2025-05-20_BRandFRratios/drift',
    #     '--workers', '12',
    # #    '--no_jitter',
    # ]

    
    args = parse_args()

    if args.num_workers == 1:
        args.parallel = False

    metrics_files = get_metrics_files(args.batch_paths)
    selected_files, fitness_vals = sort_by_fitness(metrics_files, args.top_n)

    print(f"Selected top {len(selected_files)} metrics files based on fitness.")

    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            npy_list = pool.map(load_metrics_file, selected_files)
    else:
        npy_list = list(map(load_metrics_file, selected_files))

    plot_kwargs = {'y_lim': tuple(args.y_lim)}
    plot_network_metrics_v2(npy_list, plot_kwargs, parallel=args.parallel, num_workers=args.num_workers)
    print("Done plotting network metrics")

# This block enables the script to be run interactively or as a module
if __name__ == "__main__":
    main()
# /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Jan12_20s_noSeed_v3/batch_runs/batch_2026-01-12_spiking_only