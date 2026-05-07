#!/usr/bin/env python3
"""
Create experimental features HDF5 from spike times and metrics.

This script:
1. Loads spike times from .npy file
2. Loads metrics from metrics_curated.xlsx
3. Loads network parameters from network_results.json
4. Classifies units based on firing rate (top 20% as inhibitory, rest as excitatory)
5. Creates a features dictionary and saves to HDF5
"""

import numpy as np
import pandas as pd
import h5py
import json
import argparse
import sys
from pathlib import Path

# Convergence-window helpers live next door in feature_window.py.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def load_spike_times(spike_times_path):
    """
    Load spike times from npy file.
    
    Args:
        spike_times_path: Path to spike_times.npy file
        
    Returns:
        Dictionary with unit_id as keys and spike time arrays as values
    """
    spike_times = np.load(spike_times_path, allow_pickle=True).item()
    return spike_times


def load_metrics(metrics_path):
    """
    Load metrics from Excel file.
    
    Args:
        metrics_path: Path to metrics_curated.xlsx file
        
    Returns:
        DataFrame with unit metrics
    """
    df = pd.read_excel(metrics_path)
    # The first column (Unnamed: 0) is actually the unit_id
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'unit_id'})
    return df


def load_network_results(network_results_path):
    """
    Load network results from JSON file.
    
    Args:
        network_results_path: Path to network_results.json file
        
    Returns:
        Dictionary with network parameters
    """
    with open(network_results_path, 'r') as f:
        network_results = json.load(f)
    return network_results


def classify_units(df):
    """
    Classify units based on firing rate.
    Top 20% firing rate neurons are classified as inhibitory, rest as excitatory.
    
    Args:
        df: DataFrame with unit metrics including firing_rate column
        
    Returns:
        DataFrame with added 'cell_type' column
    """
    # Calculate the 80th percentile threshold
    threshold = df['firing_rate'].quantile(0.8)
    
    # Create classification
    df['cell_type'] = df['firing_rate'].apply(
        lambda x: 'inhibitory' if x >= threshold else 'excitatory'
    )
    
    print(f"  Classification threshold (80th percentile): {threshold:.3f} Hz")
    
    return df


def compute_T_target(spike_times, cell_types, T_grid=None, schema_path=None):
    """
    Pick the smallest grid window length T (excluding T_full) whose rate features
    minimise a chosen schema's weighted absolute deviation from the same features
    measured on the full recording. Returns a dict with T_target_s + diagnostics.

    Reuses the helpers in feature_window.py so the math is exactly what
    the diagnostic plots show.
    """
    # Lazy import — keeps create_experimental_target usable without matplotlib
    # if --no_auto_T is set.
    from feature_window import (
        compute_curve, load_schema, schema_fitness_curve, build_grid,
        DEFAULT_T_GRID, DEFAULT_FOCUS_SCHEMA,
    )

    T_full = max(
        (float(np.asarray(t).max()) for t in spike_times.values() if len(t)),
        default=0.0,
    )
    if T_full <= 0:
        raise ValueError("No spikes — cannot compute T_target.")

    grid = build_grid(T_grid or DEFAULT_T_GRID, T_full)
    rows = compute_curve(spike_times, cell_types, grid)

    sp = schema_path or DEFAULT_FOCUS_SCHEMA
    print(f"   Decision schema: {sp}")
    fs = load_schema(sp)
    out = schema_fitness_curve(fs, rows)
    if out is None:
        print("   ! Decision schema has no rate-error metrics with non-zero weight; "
              "falling back to T_full.")
        return {
            "T_target_s": float(T_full),
            "T_full_s": float(T_full),
            "method": "fallback_T_full",
            "schema": str(sp),
        }

    fitness, terms = out
    T_arr = np.array([r["T_s"] for r in rows], dtype=float)
    f_arr = np.array(fitness, dtype=float)
    best_idx = int(np.argmin(f_arr[:-1])) if len(f_arr) > 1 else 0
    return {
        "T_target_s": float(T_arr[best_idx]),
        "T_full_s": float(T_full),
        "method": "schema_argmin",
        "schema": str(sp),
        "best_fitness": float(f_arr[best_idx]),
        "T_grid": [float(t) for t in T_arr],
        "fitness_per_T": [(float(t), float(f)) for t, f in zip(T_arr, f_arr)],
        "n_rate_terms": int(len(terms)),
    }


def truncate_and_recompute(spike_times, df, T_target):
    """Cap each unit's spike train at t <= T_target and refresh the per-unit
    metrics that depend on duration (firing_rate, num_spikes). Quality metrics
    that come from the sorter (snr, presence_ratio, ...) are intrinsic and
    left alone."""
    truncated = {}
    for u, t in spike_times.items():
        arr = np.asarray(t)
        truncated[u] = arr[arr <= T_target]

    df = df.copy()
    df['num_spikes'] = df['unit_id'].apply(lambda u: int(len(truncated.get(u, []))))
    df['firing_rate'] = df['num_spikes'] / float(T_target)
    return truncated, df


def deduplicate_units_by_location(df, spike_times, use_z=False, decimals=6):
    """
    Remove duplicate-location units, keeping the one with highest firing rate.

    Duplicate key is (loc_x, loc_y) by default, or (loc_x, loc_y, loc_z) if
    use_z=True. Coordinates are rounded to `decimals` before grouping.

    Args:
        df: DataFrame with columns unit_id, firing_rate, loc_x, loc_y (+ optional loc_z)
        spike_times: Dict[unit_id, np.ndarray]
        use_z: Whether to include z in duplicate key
        decimals: Decimal precision used for location matching

    Returns:
        filtered_df, filtered_spike_times, dropped_unit_ids
    """
    required_cols = ['unit_id', 'firing_rate', 'loc_x', 'loc_y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Cannot deduplicate: missing required column '{col}' in metrics file")

    work = df.copy()
    if use_z and 'loc_z' not in work.columns:
        print("  Warning: --deduplicate_use_z requested but 'loc_z' not found; using x,y only")
        use_z = False

    # Keep only rows with valid coordinates/firing rate for dedup logic
    key_cols = ['loc_x', 'loc_y'] + (['loc_z'] if use_z else [])
    for c in key_cols:
        work[c] = pd.to_numeric(work[c], errors='coerce')
    work['firing_rate'] = pd.to_numeric(work['firing_rate'], errors='coerce')
    valid_mask = work[key_cols + ['firing_rate']].notna().all(axis=1)

    valid_df = work[valid_mask].copy()
    invalid_df = work[~valid_mask].copy()

    # Rounded location keys for robust float matching
    rounded_cols = []
    for c in key_cols:
        rc = f"__{c}_rounded"
        valid_df[rc] = valid_df[c].round(decimals)
        rounded_cols.append(rc)

    # Sort so highest firing-rate is kept per location key
    valid_df = valid_df.sort_values(by='firing_rate', ascending=False)
    keep_valid = valid_df.drop_duplicates(subset=rounded_cols, keep='first')

    # Keep all invalid rows unchanged (cannot safely deduplicate without coords/rate)
    kept_df = pd.concat([keep_valid.drop(columns=rounded_cols), invalid_df], ignore_index=True)

    original_ids = set(df['unit_id'].tolist())
    kept_ids = set(kept_df['unit_id'].tolist())
    dropped_ids = sorted(original_ids - kept_ids)

    if dropped_ids:
        print(f"  Deduplication removed {len(dropped_ids)} units with lower firing rate at duplicate locations")
        print(f"  Dropped unit IDs: {dropped_ids[:30]}" + (" ..." if len(dropped_ids) > 30 else ""))
    else:
        print("  No duplicate locations found to remove")

    # Filter spike_times so dropped units are not stored in HDF5
    filtered_spike_times = {uid: st for uid, st in spike_times.items() if uid in kept_ids}

    return kept_df, filtered_spike_times, dropped_ids


def create_features_dict(spike_times, df, network_results):
    """
    Create the features dictionary.
    
    Args:
        spike_times: Dictionary of spike times per unit
        df: DataFrame with unit metrics
        network_results: Dictionary with network parameters
        
    Returns:
        Dictionary with structure: feat[unit_id] = {spike_times: ..., features: ...}
    """
    features = {}
    
    # Convert dataframe to dict indexed by unit_id for easy lookup
    df_dict = df.set_index('unit_id').to_dict('index')
    
    # Process each unit that has spike times
    for unit_id in spike_times.keys():
        features[unit_id] = {}
        
        # Add spike times
        features[unit_id]['spike_times'] = spike_times[unit_id]
        
        # Add metrics from xlsx if this unit exists in the metrics
        if unit_id in df_dict:
            for column, value in df_dict[unit_id].items():
                features[unit_id][column] = value
        else:
            print(f"  Warning: Unit {unit_id} has spike times but no metrics in xlsx")
    
    # Check for units in xlsx that don't have spike times
    units_without_spikes = set(df_dict.keys()) - set(spike_times.keys())
    if units_without_spikes:
        print(f"  Warning: {len(units_without_spikes)} units in xlsx have no spike times")
    
    # Store network results as a separate entry
    features['_network_results'] = network_results
    
    return features


def save_to_hdf5(features, output_path):
    """
    Save features dictionary to HDF5 file.
    
    Args:
        features: Dictionary with all unit features
        output_path: Path to output HDF5 file
    """
    with h5py.File(output_path, 'w') as f:
        # Save network results if present
        if '_network_results' in features:
            grp = f.create_group('network_results')
            network_results = features['_network_results']
            
            # Save network results as attributes or datasets
            for key, value in network_results.items():
                if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                    grp.attrs[key] = value
                elif isinstance(value, (list, dict)):
                    grp.attrs[key] = json.dumps(value)
                elif isinstance(value, np.ndarray):
                    grp.create_dataset(key, data=value)
                else:
                    try:
                        grp.attrs[key] = str(value)
                    except:
                        print(f"  Warning: Could not save network result '{key}'")
        
        # Save each unit's features
        for unit_id, unit_features in features.items():
            if unit_id == '_network_results':
                continue
                
            # Create group for this unit
            unit_grp = f.create_group(f'unit_{unit_id}')
            
            for feature_name, feature_value in unit_features.items():
                if feature_name == 'spike_times':
                    # Save spike times as dataset
                    unit_grp.create_dataset('spike_times', data=feature_value)
                else:
                    # Save other features as attributes
                    try:
                        # Convert pandas/numpy types to native Python types
                        if pd.isna(feature_value):
                            unit_grp.attrs[feature_name] = 'NaN'
                        elif isinstance(feature_value, (np.integer, np.int64, np.int32)):
                            unit_grp.attrs[feature_name] = int(feature_value)
                        elif isinstance(feature_value, (np.floating, np.float64, np.float32)):
                            unit_grp.attrs[feature_name] = float(feature_value)
                        elif isinstance(feature_value, str):
                            unit_grp.attrs[feature_name] = feature_value
                        elif isinstance(feature_value, (int, float, bool)):
                            unit_grp.attrs[feature_name] = feature_value
                        else:
                            # Try converting to string as fallback
                            unit_grp.attrs[feature_name] = str(feature_value)
                    except Exception as e:
                        print(f"  Warning: Could not save feature '{feature_name}' for unit {unit_id}: {e} (type: {type(feature_value)})")


def main():
    parser = argparse.ArgumentParser(
        description='Create experimental features HDF5 from spike times and metrics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--spike_times', type=str, required=True, 
                        help='Path to spike_times.npy file')
    parser.add_argument('--metrics', type=str, required=True,
                        help='Path to metrics_curated.xlsx file')
    parser.add_argument('--network_results', type=str, required=True,
                        help='Path to network_results.json file')
    parser.add_argument('--output', type=str, default='experimental_features.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--deduplicate_locations', action='store_true',
                        help='Remove duplicate-location units by keeping the highest firing-rate unit')
    parser.add_argument('--deduplicate_use_z', action='store_true',
                        help='Use (loc_x, loc_y, loc_z) for duplicate matching (default: x,y only)')
    parser.add_argument('--deduplicate_decimals', type=int, default=6,
                        help='Rounding precision for location matching during deduplication')
    parser.add_argument('--T_target', type=float, default=None,
                        help='Manually override the target window length (s). '
                             'If omitted, auto-detect via convergence sweep.')
    parser.add_argument('--decision_schema', type=str, default=None,
                        help='Schema file used to pick T_target (default: schema_v2_equal.py).')
    parser.add_argument('--T_grid', type=float, nargs='+', default=None,
                        help='Window-length grid in seconds (default: 10 20 30 45 60 90 120 180 240).')
    parser.add_argument('--no_auto_T', action='store_true',
                        help='Disable T_target selection and truncation; keep the full recording.')

    args = parser.parse_args()
    
    print("="*60)
    print("Creating Experimental Target Features")
    print("="*60)
    
    # Load data
    print("\n1. Loading spike times...")
    spike_times = load_spike_times(args.spike_times)
    print(f"   Loaded spike times for {len(spike_times)} units")
    unit_ids = sorted(spike_times.keys())
    print(f"   Unit ID range: {min(unit_ids)} - {max(unit_ids)}")
    
    print("\n2. Loading metrics...")
    df = load_metrics(args.metrics)
    print(f"   Loaded metrics for {len(df)} units")
    print(f"   Features: {list(df.columns)}")
    
    print("\n3. Loading network results...")
    network_results = load_network_results(args.network_results)
    print(f"   Loaded {len(network_results)} network parameters")
    
    # Classify units
    print("\n4. Classifying units...")
    df = classify_units(df)
    excitatory_count = (df['cell_type'] == 'excitatory').sum()
    inhibitory_count = (df['cell_type'] == 'inhibitory').sum()
    print(f"   Excitatory: {excitatory_count} ({excitatory_count/len(df)*100:.1f}%)")
    print(f"   Inhibitory: {inhibitory_count} ({inhibitory_count/len(df)*100:.1f}%)")

    # Optional deduplication by location
    if args.deduplicate_locations:
        print("\n5. Deduplicating units by location...")
        df, spike_times, dropped_ids = deduplicate_units_by_location(
            df,
            spike_times,
            use_z=args.deduplicate_use_z,
            decimals=args.deduplicate_decimals,
        )
        print(f"   Remaining units in metrics: {len(df)}")
        print(f"   Remaining units in spike_times: {len(spike_times)}")

    # T_target selection + spike-train truncation
    print("\n6. Determining target window length T_target_s...")
    T_info = None
    if args.no_auto_T and args.T_target is None:
        print("   --no_auto_T set; skipping window selection (full recording retained).")
    else:
        if args.T_target is not None:
            T_info = {"T_target_s": float(args.T_target), "method": "manual_override",
                      "schema": None}
            print(f"   Using manual T_target = {args.T_target} s")
        else:
            cell_types = dict(zip(df['unit_id'], df['cell_type']))
            T_info = compute_T_target(
                spike_times, cell_types,
                T_grid=args.T_grid,
                schema_path=args.decision_schema,
            )
            print(f"   Auto-selected T_target = {T_info['T_target_s']} s "
                  f"(schema = {Path(T_info['schema']).name}, "
                  f"weighted Δ = {T_info.get('best_fitness', 'n/a')})")
        T_target = T_info["T_target_s"]
        print(f"   Truncating spike trains to t <= {T_target} s and refreshing "
              f"firing_rate / num_spikes in the per-unit attrs.")
        spike_times, df = truncate_and_recompute(spike_times, df, T_target)
        # Persist the decision into network_results so it lands in the h5.
        network_results['T_target_s'] = float(T_target)
        network_results['T_target_diagnostics'] = T_info

    # Create features dictionary
    print("\n7. Creating features dictionary...")
    features = create_features_dict(spike_times, df, network_results)
    # Subtract 1 for the _network_results entry
    num_units = len(features) - 1
    print(f"   Created features for {num_units} units")
    
    # Save to HDF5
    print(f"\n8. Saving to {args.output}...")
    save_to_hdf5(features, args.output)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    
    # Print example of how to load the data
    print("\nTo load the data:")
    print(f"  import h5py")
    print(f"  with h5py.File('{args.output}', 'r') as f:")
    print(f"      # List all units")
    print(f"      units = [k for k in f.keys() if k.startswith('unit_')]")
    print(f"      # Load spike times for first unit")
    print(f"      unit_id = units[0].replace('unit_', '')")
    print(f"      spike_times = f[units[0]]['spike_times'][:]")
    print(f"      # Load features")
    print(f"      cell_type = f[units[0]].attrs['cell_type']")
    print(f"      firing_rate = f[units[0]].attrs['firing_rate']")


if __name__ == '__main__':
    main()

'''
python /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/_scripts/create_experimental_target.py \
  --spike_times      /pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/results_CDKL5_02_full/sd/k/ktub1999/networkSimulations/experimental_data/well000/spike_times.npy \
  --metrics          /pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/results_CDKL5_02_full/sd/k/ktub1999/networkSimulations/experimental_data/well000/metrics_curated.xlsx \
  --network_results  /pscratch/sd/k/ktub1999/networkSimulations/MEA_Analysis/results_CDKL5_02_full/sd/k/ktub1999/networkSimulations/experimental_data/well000/network_results.json \
  --deduplicate_locations \
  --output           /pscratch/sd/k/ktub1999/networkSimulations/RBS_network_models/processed_experimental_targets/CDKL5_002_well000.h5

    '''