#!/usr/bin/env python3
"""
Load NetPyNE simulation data and visualize neuron locations, cell types, and connections.
"""

import numpy as np
import pickle
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_sim_data(filepath):
    """
    Load network data from NetPyNE output pickle file.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if 'net' not in data or 'cells' not in data['net']:
        raise ValueError("Invalid format: 'net' or 'cells' not found in pickle data.")
    
    cells = data['net']['cells']
    
    neuron_data = {
        'gid': [],
        'pop': [],
        'x': [],
        'y': [],
        'z': [],
        'conns': []
    }
    
    # Process cells
    for i, cell in enumerate(cells):
        gid = cell.get('gid', i)
        tags = cell.get('tags', {})
        
        neuron_data['gid'].append(gid)
        neuron_data['pop'].append(tags.get('pop', 'Unknown'))
        neuron_data['x'].append(tags.get('x', np.nan))
        neuron_data['y'].append(tags.get('y', np.nan))
        neuron_data['z'].append(tags.get('z', np.nan))
        
        # Save connections where this cell is POST-synaptic
        if 'conns' in cell:
            for conn in cell['conns']:
                pre_gid = conn.get('preGid')
                # NetPyNE typically uses integers for GIDs, but external inputs might be strings/symbols
                if hasattr(pre_gid, '__int__') or isinstance(pre_gid, (int, str)):
                    neuron_data['conns'].append({
                        'preGid': pre_gid,
                        'postGid': gid,
                        'weight': conn.get('weight', 1.0),
                        'synMech': conn.get('synMech', ''),
                        'label': conn.get('label', '')
                    })
                    
    return neuron_data

def visualize_network(neuron_data, output_path=None, plot_conns=True):
    gids = np.array(neuron_data['gid'])
    pops = np.array(neuron_data['pop'])
    x = np.array(neuron_data['x'])
    y = np.array(neuron_data['y'])
    conns = neuron_data['conns']
    
    # Map GID to index for coordinates
    gid_to_idx = {gid: idx for idx, gid in enumerate(gids)}
    
    exc_mask = pops == 'E'
    inh_mask = pops == 'I'
    unk_mask = ~(exc_mask | inh_mask)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    num_plotted_lines = 0
    # Plot connections
    if plot_conns and len(conns) > 0:
        lines = []
        colors = []
        print(f"Processing {len(conns)} connections...")
        for conn in conns:
            pre_gid = conn['preGid']
            post_gid = conn['postGid']
            
            # Ensure both pre and post are internal cells (have valid coordinates)
            if pre_gid in gid_to_idx and post_gid in gid_to_idx:
                p1 = (x[gid_to_idx[pre_gid]], y[gid_to_idx[pre_gid]])
                p2 = (x[gid_to_idx[post_gid]], y[gid_to_idx[post_gid]])
                lines.append([p1, p2])
                num_plotted_lines += 1
                
                # Determine connection color
                # In NetPyNE, label often has 'E->I' or just pop info, or synMech
                # Here we simplify: if label string contains 'E', maybe blue; else check synMech
                # To be safe, look at presynaptic pop to determine color 
                pre_idx = gid_to_idx[pre_gid]
                pre_pop = pops[pre_idx]
                if pre_pop == 'E':
                    colors.append('blue')
                elif pre_pop == 'I':
                    colors.append('red')
                else:
                    colors.append('gray')
                    
        print(f"Plotting {num_plotted_lines} internal connections...")
        # Using a very low alpha to prevent oversaturation
        lc = LineCollection(lines, colors=colors, alpha=0.02, linewidths=0.5)
        ax.add_collection(lc)
        
    # Plot neurons
    ax.scatter(x[exc_mask], y[exc_mask], c='blue', alpha=0.8, s=50, label='Excitatory (E)', edgecolor='w', zorder=5)
    ax.scatter(x[inh_mask], y[inh_mask], c='red', alpha=0.8, s=50, label='Inhibitory (I)', edgecolor='w', zorder=5)
    if unk_mask.sum() > 0:
        ax.scatter(x[unk_mask], y[unk_mask], c='gray', alpha=0.8, s=50, label='Unknown', edgecolor='w', zorder=5)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    title = 'Simulated Network Topology'
    if plot_conns:
        title += f' ({num_plotted_lines} connections)'
    ax.set_title(title)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'datalim')
    
    # Make sure text labels indicating gid are visible if network is very small
    if len(gids) <= 30:
        for gid, px, py in zip(gids, x, y):
            ax.text(px, py+2, str(gid), fontsize=8, ha='center', va='bottom', zorder=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Total neurons: {len(gids)}")
    print(f"Excitatory (E): {exc_mask.sum()} ({exc_mask.sum()/len(gids)*100:.1f}%)")
    print(f"Inhibitory (I): {inh_mask.sum()} ({inh_mask.sum()/len(gids)*100:.1f}%)")
    if unk_mask.sum() > 0:
        print(f"Unknown: {unk_mask.sum()} ({unk_mask.sum()/len(gids)*100:.1f}%)")
    print(f"Total internal connections plotted: {num_plotted_lines}")
    
def main():
    parser = argparse.ArgumentParser(description='Visualize NetPyNE simulated network topology.')
    parser.add_argument('--input', type=str, required=True, help='Input NetPyNE pickle (.pkl) file')
    parser.add_argument('--output', type=str, default='network_layout.png', help='Output image file path')
    parser.add_argument('--no-conns', action='store_true', help='Skip plotting connections')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: File {input_file} not found")
        return
        
    print(f"Loading simulated network from {input_file}...")
    neuron_data = load_sim_data(str(input_file))
    
    visualize_network(neuron_data, args.output, plot_conns=not args.no_conns)
    
if __name__ == '__main__':
    main()
'''
module load conda
conda activate preshifter
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts/visualize_sim_network.py \
  --input /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_large_limited_tau_2_w1_focus_InhFR/batch_runs/batch_2026-04-02_spiking_only/gen_39/trial_39_data.pkl \
  --output /pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/CDKL5_seed_large_limited_tau_2_w1_focus_InhFR/batch_runs/batch_2026-04-02_spiking_only/gen_39/sim_network_layout.png
  '''