#!/usr/bin/env python3
"""
Load experimental features and visualize neuron locations with cell type color-coding.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def explore_hdf5(filepath):
    """
    Explore the structure of the HDF5 file to understand available data.
    """
    print("="*60)
    print("Exploring HDF5 File Structure")
    print("="*60)
    
    with h5py.File(filepath, 'r') as f:
        print("\nTop-level groups/datasets:")
        for key in list(f.keys())[:5]:
            print(f"  {key}")
        
        # Look at first unit as example
        unit_keys = [k for k in f.keys() if k.startswith('unit_')]
        if unit_keys:
            first_unit = unit_keys[0]
            print(f"\nExample unit ({first_unit}):")
            print(f"  Datasets: {list(f[first_unit].keys())}")
            print(f"  Attributes: {list(f[first_unit].attrs.keys())}")
            
            # Print some attribute values
            for attr in list(f[first_unit].attrs.keys())[:5]:
                val = f[first_unit].attrs[attr]
                print(f"    {attr}: {val} ({type(val).__name__})")
    
    print()


def load_neuron_data(filepath):
    """
    Load neuron data from HDF5 file.
    
    Returns:
        Dictionary with unit_id, cell_type, and any location coordinates
    """
    neuron_data = {
        'unit_id': [],
        'cell_type': [],
        'x': [],
        'y': [],
        'z': [],
        'depth': [],
        'firing_rate': [],
    }
    
    with h5py.File(filepath, 'r') as f:
        unit_keys = sorted([k for k in f.keys() if k.startswith('unit_')])
        
        for unit_key in unit_keys:
            unit_id = int(unit_key.replace('unit_', ''))
            unit_grp = f[unit_key]
            
            neuron_data['unit_id'].append(unit_id)
            neuron_data['cell_type'].append(unit_grp.attrs.get('cell_type', 'unknown'))
            neuron_data['firing_rate'].append(unit_grp.attrs.get('firing_rate', 0))
            
            # Load location coordinates
            neuron_data['x'].append(unit_grp.attrs.get('loc_x', np.nan))
            neuron_data['y'].append(unit_grp.attrs.get('loc_y', np.nan))
            neuron_data['z'].append(unit_grp.attrs.get('loc_z', unit_grp.attrs.get('depth', np.nan)))
            neuron_data['depth'].append(unit_grp.attrs.get('depth', np.nan))
    
    return neuron_data


def report_duplicate_locations(neuron_data, decimals=6):
    """
    Find neurons sharing the same (x, y) location and print their unit IDs.

    Parameters
    ----------
    neuron_data : dict
        Output of load_neuron_data().
    decimals : int
        Rounding precision before comparing float coordinates.
    """
    unit_ids = np.array(neuron_data['unit_id'])
    x = np.array(neuron_data['x'], dtype=float)
    y = np.array(neuron_data['y'], dtype=float)
    
    valid = np.isfinite(x) & np.isfinite(y) 
    if valid.sum() == 0:
        print("\nNo valid (x, y) coordinates found for duplicate-location check.")
        return

    coords = np.column_stack([
        np.round(x[valid], decimals=decimals),
        np.round(y[valid], decimals=decimals),
    ])
    valid_ids = unit_ids[valid]

    loc_to_units = {}
    for uid, coord in zip(valid_ids, coords):
        key = tuple(coord.tolist())
        loc_to_units.setdefault(key, []).append(int(uid))

    duplicates = {loc: uids for loc, uids in loc_to_units.items() if len(uids) > 1}

    print("\n" + "=" * 60)
    print("Duplicate location check")
    print("=" * 60)
    if not duplicates:
        print("No duplicate neuron locations found.")
        return

    print(f"Found {len(duplicates)} duplicated location(s):")
    for loc, uids in duplicates.items():
        print(f"  location={loc} -> unit_ids={uids}")


def visualize_schematic(neuron_data, output_path=None):
    """
    Generate a schematic visualization with a subset of neurons,
    random connections, and space for parameters, saved as PDF.
    """
    # 1. Sub-sample neurons
    x = np.array(neuron_data['x'])
    y = np.array(neuron_data['y'])
    cell_types = np.array(neuron_data['cell_type'])
    
    has_xy = ~(np.isnan(x) | np.isnan(y))
    valid_idx = np.where(has_xy)[0]
    
    exc_idx = [i for i in valid_idx if cell_types[i] == 'excitatory']
    inh_idx = [i for i in valid_idx if cell_types[i] == 'inhibitory']
    
    # Sample subset to avoid crowding a 2x1 plot
    n_exc = min(50, len(exc_idx))
    n_inh = min(16, len(inh_idx))
    
    np.random.seed(42)
    sampled_exc = np.random.choice(exc_idx, n_exc, replace=False) if n_exc > 0 else []
    sampled_inh = np.random.choice(inh_idx, n_inh, replace=False) if n_inh > 0 else []
    sampled_all = np.concatenate([sampled_exc, sampled_inh]).astype(int)
    
    # Make the plot 2inches x 1 inch, but give more space for parameters
    # The figure is wider (4.5 x 1.5) so the scatter plot is roughly 2x1 on the left,
    # reserving the right half for text.
    fig = plt.figure(figsize=(4.5, 1.5))
    ax = fig.add_axes([0.05, 0.1, 0.45, 0.8])
    
    if len(sampled_all) > 0:
        sx = x[sampled_all]
        sy = y[sampled_all]
        stypes = cell_types[sampled_all]
        
        # Add random connections between Inh and Exc
        if n_exc > 0 and n_inh > 0:
            for inh_i in sampled_inh:
                n_conns = np.random.randint(2, 5)
                targets = np.random.choice(sampled_exc, min(n_conns, len(sampled_exc)), replace=False)
                for tg in targets:
                    ax.plot([x[inh_i], x[tg]], [y[inh_i], y[tg]], color='red', alpha=0.4, linewidth=0.5, zorder=1)
                    
            for exc_i in sampled_exc:
                # Add a few exc-exc random connections too, to make it look like a network
                if np.random.random() < 0.3:
                    n_conns = np.random.randint(1, 3)
                    targets = np.random.choice(sampled_exc, min(n_conns, len(sampled_exc)), replace=False)
                    for tg in targets:
                        ax.plot([x[exc_i], x[tg]], [y[exc_i], y[tg]], color='blue', alpha=0.2, linewidth=0.5, zorder=1)
                        
        # Scatter the sampled neurons
        exc_mask = stypes == 'excitatory'
        inh_mask = stypes == 'inhibitory'
        
        ax.scatter(sx[exc_mask], sy[exc_mask], c='blue', alpha=0.8, s=15, label='Exc', zorder=2)
        ax.scatter(sx[inh_mask], sy[inh_mask], c='red', alpha=0.8, s=15, label='Inh', zorder=2)
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add text area for parameters on the right space
    text_ax = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    text_ax.axis('off')
    
    param_text = (
        "Variable Parameters:\n"
        "- scale_AMPA: \n"
        "- scale_NMDA: \n"
        "- scale_GABA: \n"
        "- Inh FR target: "
    )
    text_ax.text(0, 0.5, param_text, fontsize=8, va='center', ha='left', family='monospace')
    
    if not output_path:
        output_path = "schematic_network.pdf"
        
    if not output_path.lower().endswith('.pdf'):
        output_path = output_path.rsplit('.', 1)[0] + '.pdf'
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved schematic visualization to {output_path}")
    plt.close()


def visualize_neurons(neuron_data, output_path=None):
    """
    Visualize neuron locations with cell type color-coding.
    """
    # Convert lists to arrays
    x = np.array(neuron_data['x'])
    y = np.array(neuron_data['y'])
    z = np.array(neuron_data['z'])
    cell_types = np.array(neuron_data['cell_type'])
    firing_rates = np.array(neuron_data['firing_rate'])
    
    # Check if we have valid coordinates
    has_xy = ~(np.isnan(x) | np.isnan(y))
    has_z = ~np.isnan(z)
    
    print(f"Units with valid x,y: {has_xy.sum()}")
    print(f"Units with valid z: {has_z.sum()}")
    
    if has_xy.sum() == 0:
        print("Warning: No valid x,y coordinates found. Using unit_id as x coordinate...")
        x = np.arange(len(neuron_data['unit_id']))
        y = np.arange(len(neuron_data['unit_id']))
        has_xy = np.ones(len(x), dtype=bool)
    
    # Create color mapping
    color_map = {'excitatory': 'blue', 'inhibitory': 'red', 'unknown': 'gray'}
    colors = [color_map.get(ct, 'gray') for ct in cell_types]
    
    # Create plots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 2D scatter (x-y plane) if we have data
    if has_xy.sum() > 0:
        ax1 = fig.add_subplot(131)
        
        exc_mask = cell_types == 'excitatory'
        inh_mask = cell_types == 'inhibitory'
        
        ax1.scatter(x[exc_mask], y[exc_mask], c='blue', alpha=0.6, s=50, label='Excitatory')
        ax1.scatter(x[inh_mask], y[inh_mask], c='red', alpha=0.6, s=50, label='Inhibitory')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Neuron Locations (X-Y Plane)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Firing rate vs cell type
    ax2 = fig.add_subplot(132)
    
    exc_fr = firing_rates[exc_mask] if exc_mask.sum() > 0 else []
    inh_fr = firing_rates[inh_mask] if inh_mask.sum() > 0 else []
    
    data_to_plot = []
    labels = []
    if len(exc_fr) > 0:
        data_to_plot.append(exc_fr)
        labels.append('Excitatory')
    if len(inh_fr) > 0:
        data_to_plot.append(inh_fr)
        labels.append('Inhibitory')
    
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor('blue' if label == 'Excitatory' else 'red')
            patch.set_alpha(0.6)
    
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_title('Firing Rate Distribution by Cell Type')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cell type counts
    ax3 = fig.add_subplot(133)
    
    exc_count = (cell_types == 'excitatory').sum()
    inh_count = (cell_types == 'inhibitory').sum()
    unk_count = (cell_types == 'unknown').sum()
    
    counts = []
    cell_labels = []
    colors_bar = []
    
    if exc_count > 0:
        counts.append(exc_count)
        cell_labels.append('Excitatory')
        colors_bar.append('blue')
    if inh_count > 0:
        counts.append(inh_count)
        cell_labels.append('Inhibitory')
        colors_bar.append('red')
    if unk_count > 0:
        counts.append(unk_count)
        cell_labels.append('Unknown')
        colors_bar.append('gray')
    
    ax3.bar(cell_labels, counts, color=colors_bar, alpha=0.6)
    ax3.set_ylabel('Count')
    ax3.set_title('Neuron Count by Cell Type')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for i, (label, count) in enumerate(zip(cell_labels, counts)):
        ax3.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Total neurons: {len(neuron_data['unit_id'])}")
    print(f"Excitatory: {exc_count} ({exc_count/len(neuron_data['unit_id'])*100:.1f}%)")
    print(f"Inhibitory: {inh_count} ({inh_count/len(neuron_data['unit_id'])*100:.1f}%)")
    if unk_count > 0:
        print(f"Unknown: {unk_count} ({unk_count/len(neuron_data['unit_id'])*100:.1f}%)")
    
    if exc_mask.sum() > 0:
        print(f"\nExcitatory firing rate: {firing_rates[exc_mask].mean():.2f} ± {firing_rates[exc_mask].std():.2f} Hz")
    if inh_mask.sum() > 0:
        print(f"Inhibitory firing rate: {firing_rates[inh_mask].mean():.2f} ± {firing_rates[inh_mask].std():.2f} Hz")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize neuron locations with cell type color-coding',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, default='experimental_features.h5',
                        help='Input HDF5 file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image file path')
    parser.add_argument('--explore', action='store_true',
                        help='Explore HDF5 structure and exit')
    parser.add_argument('--schematic', action='store_true',
                        help='Generate a small schematic visualization with parameters text instead of full plots')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: File {input_file} not found")
        return
    
    if args.explore:
        explore_hdf5(str(input_file))
        return
    
    print(f"Loading experimental features from {input_file}...")
    neuron_data = load_neuron_data(str(input_file))
    
    print(f"Loaded {len(neuron_data['unit_id'])} neurons")

    # Print neurons that share identical locations
    report_duplicate_locations(neuron_data)
    
    if args.schematic:
        visualize_schematic(neuron_data, args.output)
    else:
        visualize_neurons(neuron_data, args.output)


if __name__ == '__main__':
    main()
# python3 /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts/visualize_neurons.py --input /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_002_well004.h5 --output  '/pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts/location_cdkl5_2_w4.png'