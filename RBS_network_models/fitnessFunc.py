
# Imports =========================================================================================
import os
import json
import numpy as np
import traceback
from .fitnessCalcs import * # import all comparison helper functions
import time
import os
import subprocess
import numpy as np
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import compute_network_metrics
from netpyne import sim
from matplotlib import pyplot as plt

# Functions =========================================================================================
def the_scoring_function_asymmetric_parabola(val, target_val, maxFitness, weight,
                                              min_val=None, max_val=None,
                                              plot_fit_curve=False, fit_curve_path=None):
    # import numpy as np
    # import matplotlib.pyplot as plt

    fit_curve_title = os.path.basename(fit_curve_path) if fit_curve_path is not None else None
    fit_curve_title = fit_curve_title.replace('.png', '') if fit_curve_title is not None else None

    if val is None:
        return maxFitness

    default_min = False
    default_max = False

    # Assign default bounds if missing
    if min_val is None:
        min_val = target_val - 200*weight
        default_min = True
    if max_val is None:
        max_val = target_val + 200*weight
        default_max = True

    # Ensure bounds are ordered correctly
    if min_val > max_val:
        min_val, max_val = max_val, min_val
        default_min, default_max = default_max, default_min

    # Coefficients for asymmetric parabolas
    a_left = maxFitness / ((target_val - min_val) ** 2 / weight) if target_val != min_val else float('inf')
    a_right = maxFitness / ((max_val - target_val) ** 2 / weight) if target_val != max_val else float('inf')

    def compute_score(x):
        if x <= target_val:
            return min(a_left * (x - target_val) ** 2, maxFitness)
        else:
            return min(a_right * (x - target_val) ** 2, maxFitness)

    if plot_fit_curve:
        #os.makedirs(fit_curve_dir, exist_ok=True) if fit_curve_dir is not None else None
        os.makedirs(os.path.dirname(fit_curve_path), exist_ok=True) if fit_curve_path is not None else None

        # Range for plotting
        span = max(abs(val - target_val), abs(max_val - min_val)) * 1.2
        x_range = np.linspace(target_val - span, target_val + span, 500)
        y_range = [compute_score(x) for x in x_range]

        plt.figure(figsize=(8, 4))
        plt.plot(x_range, y_range, label="Parabolic Scoring Curve")
        #plt.axvline(target_val, color='green', linestyle='--', label='Target Value (Score=0)')
        plt.axvline(target_val, color='green', linestyle='--', label='Target Value ({:.2f}, Score=0)'.format(target_val))
        score_val = compute_score(val)
        #plt.axvline(val, color='red', linestyle=':', label='Input Value (Score={:.2f})'.format(score_val))
        #plt.axvline(score_val, color='red', linestyle=':', label='Input Value ({:.2f}, Score={:.2f})'.format(val, score_val))

        # lets do input value as a red dot
        plt.scatter(val, score_val, color='red', label='Input Value ({:.2f}, Score={:.2f})'.format(val, score_val), zorder=5)

        #plt.axvline(val, color='red', linestyle=':', label='Actual Value')

        if default_min:
            plt.axvline(min_val, color='blue', linestyle='-.', label='Default Min Bound')
        else:
            plt.axvline(min_val, color='blue', linestyle='-.', label='Min Bound')

        if default_max:
            plt.axvline(max_val, color='purple', linestyle='-.', label='Default Max Bound')
        else:
            plt.axvline(max_val, color='purple', linestyle='-.', label='Max Bound')

        plt.xlabel("Input Value")
        plt.ylabel("Score")
        #plt.title("Asymmetric Parabolic Fitness Scoring Curve")
        plt.title(f"Fitness Scoring Curve: {fit_curve_title}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if fit_curve_path is not None:
            plt.savefig(fit_curve_path, dpi=300)
            print(f'Saved fit curve to {fit_curve_path}')
            plt.savefig(fit_curve_path.replace('.png', '.svg'))
            print(f'Saved fit curve to {fit_curve_path.replace(".png", ".svg")}')
        plt.close()

    return compute_score(val)

def fitnessFunc_v3(simulated_data, reference_data_path, **kwargs): #def fitnessFunc_v2(simulated_data, experimental_data, **kwargs):
    """
    Main logic of the calculate_fitness function.
    Ensures the function does not crash and always returns a fitness value.
    """    
    # subfunctions =================================================================
    def calculate_fitness(simulated_data, experimental_data, fit_schema=None, fitness_dict={}, path=None, parent_key=None, **kwargs):
        """
        Calculate the fitness value based on the simulated and experimental data.
        """
        # subfunctions =========================================================
        def init_skip_keys():
            # skip keys of non-interest
            skip_keys = [
                'data',
                'note',                        
                ]
            return skip_keys
        
        def handle_inclusion_dep(abs_path):
            # include_keys = {
            #     'source': False,
            #     'trimmed': False,
            #     'spiking_data':{
            #         'spike_times': False,
            #         'spiking_times_by_unit': False,
                    
            #         'spiking_metrics_by_unit': False,
            #         # 2025-05-08 22:52:48 - removing unit metrics from fitness scoring for now
            #         # 'spiking_metrics_by_unit': {
            #         #     'int':{
            #         #         'num_spikes': False,
            #         #         'wf_metrics': False,
            #         #         'fr': True,
            #         #         'isi': True,
            #         #         'spike_times': False,
            #         #         },
            #         #     },
                    
            #         'frs': True,
            #         'i_frs': True,
            #         'e_frs': True,
            #         'u_frs': False,
            #         'isi': True,
            #         'i_isi': True,
            #         'e_isi': True,
            #         'u_isi': False,
            #         },
            #     'sim_data_path': False,
            #     'timeVector': False,
            #     'sampling_rate': False,
            #     'gids': False,
            #     'unit_ids': False,

            #     'bursting_data': False,
            #     # 2025-05-08 23:04:26 - in the case of bursting data, this wont be fit at all for now.
            #     # 'bursting_data':{
            #     #     'ax': False,
            #     #     'convolved_data': False, # maybe I could include some of this... but I think its pretty much captured by other metrics
            #     #     'unit_metrics': False,
            #     #     'unit_metrics': {
            #     #         'int':{
            #     #             'burst_id': False,
            #     #             'quiet_id': False,
            #     #             'bursts': False,
            #     #             'quiets': False,
            #     #             'burst_durations': False, # these should be in summary metrics or something... right? # aw 2025-04-28 11:35:07 yes, these are in burst_metrics
            #     #             'quiet_durations': False,
            #     #             'burst_part_rate': True,
            #     #             'quiet_part_rate': True,
            #     #             'burst_part_perc': True,
            #     #             'fr': {
            #     #                 'in_burst': True,
            #     #                 'out_burst': True,
            #     #                 },
            #     #             'isi': {
            #     #                 'in_burst': True,
            #     #                 'out_burst': True,
            #     #                 },
            #     #             'spike_counts': {
            #     #                 'in_burst': True,
            #     #                 'out_burst': True,
            #     #                 },
            #     #             'fano_factor': {
            #     #                 'in_burst': True,
            #     #                 'out_burst': True,
            #     #                 },
            #     #             },
            #     #         },
            #     #     'burst_metrics': {
            #     #         'num_bursts': False,
            #     #         'burst_rate': True,
            #     #         'burst_ids': False,
            #     #         'ibi': True,
            #     #         'burst_amp': True,
            #     #         'burst_duration': True,
            #     #         'burst_parts': False,
            #     #         'num_units_per_burst': True,
            #     #         'in_burst_fr': True,                        
            #     #         },
            #     #     'warnings': False,
            #     # },

            #     'mega_bursting_data':{
            #         'ax': False,
            #         'convolved_data': False, # maybe I could include some of this... but I think its pretty much captured by other metrics
                    
            #         'unit_metrics': False,
            #         # 2025-05-08 22:53:26 - removing unit metrics from fitness scoring for now
            #         # 'unit_metrics': {
            #         #     'int':{
            #         #         'burst_id': False,
            #         #         'quiet_id': False,
            #         #         'bursts': False,
            #         #         'quiets': False,
            #         #         'burst_durations': False, # these should be in summary metrics or something... right?
            #         #         'quiet_durations': False,
            #         #         'burst_part_rate': True,
            #         #         'quiet_part_rate': True,
            #         #         'burst_part_perc': True,
            #         #         'fr': {
            #         #             'in_burst': True,
            #         #             'out_burst': True,
            #         #             },
            #         #         'isi': {
            #         #             'in_burst': True,
            #         #             'out_burst': True,
            #         #             },
            #         #         'spike_counts': {
            #         #             'in_burst': True,
            #         #             'out_burst': True,
            #         #             },
            #         #         'fano_factor': {
            #         #             'in_burst': True,
            #         #             'out_burst': True,
            #         #             },
            #         #         },
            #         #     },

            #         'burst_metrics': {
            #             'num_bursts': False,
            #             'burst_rate': True,
            #             'burst_ids': False,
            #             'ibi': True,
            #             'burst_amp': True,
            #             'burst_duration': True,
            #             'burst_parts': False,
            #             'num_units_per_burst': True,
            #             'in_burst_fr': True,                        
            #             },
            #         'warnings': False,
            #     },

            #     'HFBursting_metrics': {
            #         'reg_in_mega_by_mega': False,
            #         'HFB_count': True,
            #         'HFB_rate': True,
            #         'HFB_presence_fraction': True,
            #         'avg_reg_burst_duration_in_mega': True,
            #         'reg_IBI_within_mega': True,              
            #     },


            #     'unit_types': False,
            #     'unit_locations': False,
            #     'simData': False,
            #     'popData': False,
            #     'cellData': False,
            # }


            ## aw 2025-05-16 10:00:19 lets try only including mega burst rate
            include_keys = {
                'source': False,
                'trimmed': False,
                
                #'spiking_data':False,
                # 2025-05-19 17:46:57 - including spiking data now after optimizing burst rate.
                'spiking_data':{
                    'spike_times': False,
                    'spiking_times_by_unit': False,
                    
                    'spiking_metrics_by_unit': False,
                    # 2025-05-08 22:52:48 - removing unit metrics from fitness scoring for now
                    # 'spiking_metrics_by_unit': {
                    #     'int':{
                    #         'num_spikes': False,
                    #         'wf_metrics': False,
                    #         'fr': True,
                    #         'isi': True,
                    #         'spike_times': False,
                    #         },
                    #     },
                    
                    'frs': False,
                    'i_frs': False,
                    'e_frs': False,
                    'u_frs': False,
                    'EI_fr_ratios': True,
                    'isi': False,
                    'i_isi': False,
                    'e_isi': False,
                    'u_isi': False,
                    },
                'sim_data_path': False,
                'timeVector': False,
                'sampling_rate': False,
                'gids': False,
                'unit_ids': False,

                'sim_data_path': False,
                'timeVector': False,
                'sampling_rate': False,
                'gids': False,
                'unit_ids': False,

                # bursting data
                'bursting_data': False,
                
                # hyper bursting data
                'mega_bursting_data':{
                    'ax': False,
                    'convolved_data': False, # maybe I could include some of this... but I think its pretty much captured by other metrics
                    'unit_metrics': False,
                    'burst_metrics': {
                        'num_bursts': False,
                        'burst_rate': True,
                        'burst_ids': False,
                        'ibi': False,
                        'burst_amp': False,
                        'burst_duration': False,
                        'burst_parts': False,
                        'num_units_per_burst': False,
                        'in_burst_fr': False,                        
                        },
                    'warnings': False,
                },
                
                #'HFBursting_metrics'
                'HFBursting_metrics': False,

                'unit_types': False,
                'unit_locations': False,
                'simData': False,
                'popData': False,
                'cellData': False,
            }
            
            # logic ==============================================================
            # manage inclusions
            #include = include_keys.get(key, False)
            #include_key = include_keys.get(key, False)
            
            # split abs_path into parts if it contains '.'
            if '.' in abs_path:
                # split into parts, replace any ints with 'int'
                parts = abs_path.split('.')
                parts = ['int' if part.isdigit() else part for part in parts]
                
                # dig into the dictionary
                for part in parts: # dig into the dictionary
                    if part in include_keys:
                        include_keys = include_keys[part]
                    else:
                        include_key = False
                    
                    # break if include_keys is a boolean
                    if isinstance(include_keys, bool):
                        include = include_keys
                        return include
                
                if isinstance(include_keys, dict):
                    include = True
                    return include
            else:
                include_key = include_keys[abs_path]
                if isinstance(include_key, dict): include = True
                elif isinstance(include_key, bool): include = include_key
                return include
        
        def handle_inclusion(abs_path, fit_schema):
            
            include_keys = fit_schema # aw 2025-05-20 15:44:59 now defining this dynamically as part of the model and batching process outside of the function.
            
            # defaults
            include = False
            max_val = None
            min_val = None
            normalize = False
            weight = 1

            # logic ==============================================================
            # manage inclusions
            #include = include_keys.get(key, False)
            #include_key = include_keys.get(key, False)
            
            # split abs_path into parts if it contains '.'
            if '.' in abs_path:
                # split into parts, replace any ints with 'int'
                parts = abs_path.split('.')
                parts = ['int' if part.isdigit() else part for part in parts]
                
                # dig into the dictionary
                for part in parts: # dig into the dictionary
                    if part in include_keys:
                        include_keys = include_keys[part]
                    else:
                        include_key = False
                    
                    # break if include_keys is a boolean
                    if isinstance(include_keys, bool):
                        include = include_keys
                        return include, max_val, min_val, normalize, weight
                    # else:
                    #     raise ValueError(f'Unknown key {part} in include_keys.')
                
                if isinstance(include_keys, dict):
                    if 'include' in include_keys:
                        include = include_keys['include']
                        if 'max_val' in include_keys:
                            max_val = include_keys['max_val']
                        if 'min_val' in include_keys:
                            min_val = include_keys['min_val']
                        if 'normalize' in include_keys:
                            normalize = include_keys['normalize']
                        if 'weight' in include_keys:
                            weight = include_keys['weight']
                        return include, max_val, min_val, normalize, weight
                    else:
                        include = True
                        return include, max_val, min_val, normalize, weight
            else:
                include_key = include_keys[abs_path]
                if isinstance(include_key, dict): include = True
                elif isinstance(include_key, bool): include = include_key
                return include, max_val, min_val, normalize, weight

        def handle_nans_and_infs(score, simulated_data, target, key):
            # handle NaN scores
            if np.isnan(score):
                # print(f'NaN score for key {key}.')
                # fitness_dict[key]['fit'] = 1000
                if np.isnan(simulated_data[key]) and not np.isnan(target):
                    # if simulated data is NaN and target is not, set score to 1000
                    #fitness_dict[key]['fit'] = 1000
                    score = 1000
                elif not np.isnan(simulated_data[key]) and np.isnan(target):
                    # if simulated data is not NaN and target is, set score to 1000
                    #fitness_dict[key]['fit'] = 1000
                    score = 1000
                elif np.isnan(simulated_data[key]) and np.isnan(target):
                    # if both are NaN, set score to 1
                    score = 1
                elif not np.isnan(simulated_data[key]) and not np.isnan(target):
                    if simulated_data[key] == target:
                        # if both are equal, set score to 1
                        score = 1
                    else:
                        print(f'Unknown NaN case for key {key}.')
                else:
                    print(f'Unknown NaN case for key {key}.')
            elif np.isinf(np.abs(score)):
                print(f'Infinite score for key {key}.')
                
            return score  
        
        def handle_spike_times_comparison(simulated_data, experimental_data, key, fitness_dict, plot_fit_curve=False, fit_curve_dir=None, **ckwargs):
            if path.endswith('spiking_times_by_unit'):
                def compare_spiking_times_by_unit(simulated_data, experimental_data, key, fitness_dict):
                    # Import necessary libraries for VP distance
                    import quantities as pq
                    from neo.core import SpikeTrain
                    from elephant.spike_train_dissimilarity import victor_purpura_distance
                    
                    # Get spike times - both are now single 1D numpy arrays
                    sim_spikes_array = simulated_data[key]  # 1D numpy array of spike times
                    exp_spikes_array = experimental_data[key]  # 1D numpy array of spike times
                    
                    # Check if arrays are empty
                    if len(sim_spikes_array) == 0 and len(exp_spikes_array) == 0:
                        # No spikes in either dataset - perfect match
                        return 0.0
                    
                    if len(sim_spikes_array) == 0 or len(exp_spikes_array) == 0:
                        # One dataset has no spikes - maximum penalty
                        return 1000.0
                    
                    # Determine time range - find max spike time across both arrays
                    t_stop = max(sim_spikes_array.max(), exp_spikes_array.max())
                    
                    # Convert to Neo SpikeTrain objects
                    sim_train = SpikeTrain(sim_spikes_array * pq.ms, t_stop=t_stop * pq.ms)
                    exp_train = SpikeTrain(exp_spikes_array * pq.ms, t_stop=t_stop * pq.ms)
                    
                    # Calculate VP distance
                    q_value = 0.1 / pq.ms  # Cost parameter - adjust as needed
                    distance = victor_purpura_distance([sim_train, exp_train], q_value)[0, 1]
                    
                    return distance
                
                vp_distance = compare_spiking_times_by_unit(simulated_data, experimental_data, key, fitness_dict)
                
                # Extract parameters from ckwargs (same as handle_float_or_int_comparison)
                max_val = ckwargs.get('max_val', None)
                min_val = ckwargs.get('min_val', 0)  # VP distance minimum is 0
                weight = ckwargs.get('weight', 0.5)
                normalize = ckwargs.get('normalize', False)
                
                # score - VP distance is already a distance metric, so lower is better
                target = 0  # best possible distance is 0
                maxFitness = 1000  # maximum fitness score
                
                # Create fit curve path if needed
                abs_path_underscored = abs_path.replace('.', '_')
                fit_curve_path = os.path.join(fit_curve_dir, f'{abs_path_underscored}_fit_curve.png') if fit_curve_dir is not None else None
                
                # Use scoring function with VP distance
                score = vp_distance
                # score = the_scoring_function_asymmetric_parabola(vp_distance, target, maxFitness, weight, min_val=min_val, max_val=max_val, plot_fit_curve=plot_fit_curve, fit_curve_path=fit_curve_path)
                score = handle_nans_and_infs(score, simulated_data, target, key)
                
                # Apply normalization if requested (same logic as handle_float_or_int_comparison)
                # if normalize:
                #     print(f'Normalizing score for key {key}.')
                #     seed_fitness = kwargs.get('seed_fitness', None)
                #     fit_list = []
                #     if seed_fitness is not None:
                #     for seed in seed_fitness:
                #         # parse abs_path to get series of keys to navigate the dictionary
                #         keys = abs_path.split('.')
                #         try:
                #             for k in keys:
                #                 seed = seed[k]
                #             fit_list.append(seed['fit'])
                #         except Exception as e:
                #             pass
                #     if len(fit_list) == 0:
                #         print(f'No fitness values found for key {key}. Unable to normalize score. Using raw score.')
                #         pass
                #     else:                    
                #         max_fit = max(fit_list)
                #         if max_fit > 1000: max_fit = 1000 # cap max fitness to 1000
                #         normalized_score = score / max_fit * 1000 # normalize score to max fitness within seeds and scale to 1000
                #         print(f'Raw score for key {key}: {score}')
                #         score = normalized_score
                #         print(f'Normalized score for key {key}: {score}')
                
                fitness_dict[key]['fit'] = score
                fitness_dict[key]['vp_distance'] = vp_distance
                fitness_dict[key]['value'] = vp_distance
                fitness_dict[key]['target'] = target

                return fitness_dict
            else:
                return {} 

        def handle_list_comparison(simulated_data, experimental_data, key, fitness_dict, plot_fit_curve=False,plot_fit_curve_dir =None):
            # compare lists
            
                
            if 'participating_units' in key:
                def compare_participating_units(simulated_data, experimental_data, key, fitness_dict):
                    # compare the set of both lists - get percent overlap
                    simulated_set = set(simulated_data[key])
                    experimental_set = set(experimental_data[key])
                    overlap = len(simulated_set.intersection(experimental_set))
                    percent_overlap = overlap / len(experimental_set)
                    return percent_overlap
                
                percent_overlap = compare_participating_units(simulated_data, experimental_data, key, fitness_dict)
                
                # score
                target = 1 # best possible overlap is 1
                weight = 1 # penalty for deviation from target
                maxFitness = 1000 # maximum fitness score
                min_val = 0 # minimum value allowed
                max_val = 1
                #score = the_scoring_function(percent_overlap, target, weight, maxFitness, min_val=min_val, max_val=max_val)
                #score = the_scoring_function_quadratic_smooth_sigmoid(percent_overlap, target, maxFitness, weight, min_val=min_val, max_val=max_val)
                #score = the_scoring_function_blended_sigmoid(percent_overlap, target, maxFitness, weight, min_val=min_val, max_val=max_val)
                #score = the_scoring_function_bounded_parabola(percent_overlap, target, maxFitness, weight, min_val=min_val, max_val=max_val)
                score = the_scoring_function_asymmetric_parabola(percent_overlap, target, maxFitness, weight, min_val=min_val, max_val=max_val)
                score = handle_nans_and_infs(score, simulated_data, target, key)
                fitness_dict[key]['fit'] = score
                
                #return fitness_dict
            else:
                raise ValueError(f'Unknown list for key {key}.')
            # elif any(exclude_logic):
            #     pass
            # else:
            #     raise ValueError(f'Unknown list for key {key}.')
            
            return fitness_dict
        
        def handle_numpy_array_comparison(simulated_data, experimental_data, key, fitness_dict, plot_fit_curve=False, fit_curve_dir=None):
            # # compare numpy arrays
            # print(f'Comparing numpy array for key {key}.')
            # #abs_path = f'{path}.{key}'
            # dtw_list = ['time', 'bursts', 'quiets', 'sequence_stack']
            
            #
            dtw_compare_keys = [
                # 'spike_times',
                'spiking_times_by_unit',
                'spiking_metrics_by_unit',
                'bursts',
                'quiets',
                
            ]
            
            if any(item in abs_path for item in dtw_compare_keys):
            #if 'times' in abs_path:
                #start = time.time()
                if len(simulated_data[key]) > 0 and len(experimental_data[key]) > 0:
                    # distance
                    distance = event_times_dtw_distance(simulated_data[key], experimental_data[key])
                    fitness_dict[key]['distance'] = distance
                    
                    #score
                    target = 0 # best possible distance is the same
                    weight = 1 # penalty for deviation from target
                    maxFitness = 1000 # maximum fitness score
                    min_val = 0 # minimum value allowed
                    max_val = 1000 # TODO: set max val based on experimental data. 
                                    #May need to work into regular network_metrics to make this easy
                    #score = the_scoring_function(distance, target, weight, maxFitness, min_val=min_val, max_val=max_val)
                    score = the_scoring_function_quadratic_smooth_sigmoid(distance, target, maxFitness, weight, min_val=min_val, max_val=max_val)
                    fitness_dict[key]['fit'] = score
                    
                    # handle NaN scores
                    if np.isnan(score):
                        print(f'NaN score for key {key}.')
                        fitness_dict[key]['fit'] = 1000
                    
                else:
                    #continue
                    raise ValueError(f'Empty numpy array for key {key}.')
                #time_elapsed = time.time() - start
                #print(f'DTW distance: {distance}')
                #print(f'DTW time elapsed: {time_elapsed} seconds.')
                
            else:
                raise ValueError(f'Unknown numpy array for key {key}.')
                
            return fitness_dict
        
        def handle_float_or_int_comparison(simulated_data, experimental_data, key, fitness_dict, plot_fit_curve=False, fit_curve_dir=None, **ckwargs):

            max_val = ckwargs.get('max_val', None)
            min_val = ckwargs.get('min_val', None)
            weight = ckwargs.get('weight', 1)
            normalize = ckwargs.get('normalize', False)
            
            # compute score
            target = experimental_data[key]
            maxFitness = 1000 # maximum fitness score
            #fit_curve_path = os.path.join(fit_curve_dir, f'{key}_fit_curve.png') if fit_curve_dir is not None else None
            abs_path_underscored = abs_path.replace('.', '_')
            fit_curve_path = os.path.join(fit_curve_dir, f'{abs_path_underscored}_fit_curve.png') if fit_curve_dir is not None else None

            #TODO: implement normalize later # 2025-05-20 16:15:36
            #aw 2025-05-22 19:12:02 finished below
            # if simulated_data[key] is None:
            #     print(f'No simulated data for key {key}. Setting score to 1000.')
            #     score = 1000
            # if 'EI_spike_ratios' in abs_path_underscored:
            #     print(f'Key {key} is EI_spike_ratios. Setting score to 1000.')
            #     #score = 1000
            #     if key == 'min':
            #         print(f'Key {key} is min. Setting score to 1000.')
            score = the_scoring_function_asymmetric_parabola(simulated_data[key], target, maxFitness, weight, min_val=min_val, max_val=max_val, plot_fit_curve=plot_fit_curve, fit_curve_path=fit_curve_path)
            score = handle_nans_and_infs(score, simulated_data, target, key)

            if normalize:
                print(f'Normalizing score for key {key}.')
                seed_fitness = kwargs.get('seed_fitness', None)
                fit_list = []
                for seed in seed_fitness:
                    # parse abs_path to get series of keys to navigate the dictionary
                    keys = abs_path.split('.')
                    try:
                        for k in keys:
                            seed = seed[k]
                        fit_list.append(seed['fit'])
                    except Exception as e:
                        pass
                if len(fit_list) == 0:
                    print(f'No fitness values found for key {key}. Unable to normalize score. Using raw score.')
                    pass
                else:                    
                    max_fit = max(fit_list)
                    if max_fit > 1000: max_fit = 1000 # cap max fitness to 1000
                    normalized_score = score / max_fit * 1000 # normalize score to max fitness within seeds and scale to 1000
                    print(f'Raw score for key {key}: {score}')
                    score = normalized_score
                    print(f'Normalized score for key {key}: {score}')                           
                    
            #update fitness dict and return
            fitness_dict[key]['fit'] = score
            fitness_dict[key]['value'] = simulated_data[key]
            fitness_dict[key]['target'] = target
            #fitness_dict[key]['max_val'] = max_val
            #fitness_dict[key]['min_val'] = min_val
            #fitness_dict[key]['weight'] = weight
            #fitness_dict[key]['normalize'] = normalize            
            return fitness_dict
        
        def handle_string_comparison(simulated_data, experimental_data, key, fitness_dict):
            # compare strings
            str_list = []
            if any(item in abs_path for item in str_list):
                # simple comparison
                if simulated_data[key] == experimental_data[key]: 
                    fitness_dict[key]['fit'] = 1
                else: 
                    fitness_dict[key]['fit'] = 1000
            else:
                print(simulated_data[key])
                #raise ValueError(f'Unknown string for key {key}.')
                
            return fitness_dict
        
        def handle_unit_mapping(path, key, kwargs):
            
            # define abs_path
            if path is not None: abs_path = f'{path}.{key}'
            else: abs_path = key
            
            # init paths, they may be updated later as needed
            exp_id = None
            sim_id = None
            exp_path = abs_path
            sim_path = abs_path
            
            # modulate path as needed if it contains 'by_unit'
            if 'by_unit' in abs_path:
                #check if any part is a number
                parts = abs_path.split('.')  # split abs_path by .
                success_int = False
                for i, part in enumerate(parts):
                    try:
                        int(part)
                        #exp_id = int(part)
                        sim_id = int(part)
                        success_int = True
                        break
                    except:
                        pass
                
                # if a number was found, replace it with the mapped value
                if success_int:                
                    # make sure experimental_to_simulated_mapping is in kwargs
                    # experimental_to_simulated_mapping = kwargs.get('experimental_to_simulated_mapping', {})
                    # assert experimental_to_simulated_mapping is not None, 'No experimental to simulated mapping found in kwargs.'
                    
                    # reconstruct abs_path, repalce int with mapped int, get int value, get mapped value, replace in abs_path
                    #sim_id = experimental_to_simulated_mapping.get(exp_id, None)
                    # exp_path = abs_path
                    # sim_path = abs_path.replace(str(exp_id), str(sim_id)) if sim_id is not None else abs_path
                    
                    # had it backwards above.
                    simulated_to_experimental_mapping = kwargs.get('simulated_to_experimental_mapping', {})
                    assert simulated_to_experimental_mapping is not None, 'No simulated to experimental mapping found in kwargs.'
                    exp_transform = simulated_to_experimental_mapping.get(sim_id, None)
                    exp_id = exp_transform['closest_exp_unit_id'] if exp_transform is not None else None
                    assert exp_id is not None, f'No experimental id found for simulated id {sim_id}.'                    
                    exp_path = abs_path.replace(str(sim_id), str(exp_id)) if exp_id is not None else abs_path
                    sim_path = abs_path
                                
            # get the last item in sim and exp paths to get the keys
            if sim_path is not None: 
                try: sim_key = int(sim_path.split('.')[-1])
                except: sim_key = sim_path.split('.')[-1]
            if exp_path is not None: 
                try: exp_key = int(exp_path.split('.')[-1])
                except: exp_key = exp_path.split('.')[-1]
            
            return key, exp_key, sim_key, exp_id, sim_id, abs_path, exp_path, sim_path
                
        # main logic =============================================================
        
        # decide to plot_fit_curve or not
        plot_fit_curve = kwargs.get('plot_fit_curve', False)
        fitness_save_path = kwargs.get('fitness_save_path', None)
        # remove ness.json at the end - make that a folder - this is where we put fit_curve plots
        fit_curve_dir = fitness_save_path.replace('_fitness.json', '_fit_curves') if fitness_save_path is not None else None

        # init include keys and skip keys
        skip_keys = init_skip_keys()
        #include_keys = init_include_keys()
        
        # update path
        if parent_key is not None and path is not None: 
            path += f'.{parent_key}'
            #fitness_dict[parent_key] = {}
        elif parent_key is not None and path is None: 
            path = parent_key
            #fitness_dict[parent_key] = {}
        elif parent_key is None and path is not None: pass
        else: path = None
        #if path is not None: print(f'Path: {path}')
        
        # loop through all paths in both dictionaries and compare the values of the keys
        for key in simulated_data:
            if key in skip_keys: continue # skip keys of non-interest          
            if key =='spiking_times_by_unit':
                print('here')
            if key in experimental_data:
                
                # handle unit mapping
                key, exp_key, sim_key, exp_id, sim_id, abs_path, exp_path, sim_path = handle_unit_mapping(path, key, kwargs)
                if exp_id is not None and sim_id is None: continue # dont score units if not mapped               
                assert key==sim_key, f'Key mismatch: {key} != {sim_key}' # quality check
                
                #include = handle_inclusion(abs_path)
                try:
                    include, max_val, min_val, normalize, weight = handle_inclusion(abs_path, fit_schema)
                    if not include: continue # only include keys of interest 
                except:
                    print(f'Key {key} not found in fitness schema. Skipping key.')
                    continue 
                
                # update fit dict
                if key not in fitness_dict:
                    fitness_dict[sim_key] = {}
                
                #if isinstance(simulated_data[key], dict):
                if isinstance(simulated_data[sim_key], dict):
                    fitness_dict[sim_key] = calculate_fitness(simulated_data[sim_key], experimental_data[exp_key], fit_schema, fitness_dict[sim_key], path=path, parent_key=sim_key, **kwargs)
                elif isinstance(simulated_data[sim_key], list):
                    fitness_dict = handle_list_comparison(simulated_data, experimental_data, key, fitness_dict, plot_fit_curve=plot_fit_curve, plot_fit_curve_dir=fit_curve_dir)
                elif isinstance(simulated_data[sim_key], np.ndarray): 
                    # continue
                    ckwargs = {}
                    ckwargs['max_val'] = max_val
                    ckwargs['min_val'] = min_val
                    ckwargs['normalize'] = True
                    ckwargs['weight'] = weight
                    ckwargs['seed_fitness'] = kwargs.get('seed_fitness', {})
                    fitness_dict = handle_spike_times_comparison(simulated_data, experimental_data, key, fitness_dict, plot_fit_curve=plot_fit_curve, fit_curve_dir=fit_curve_dir, **ckwargs)
                    #fitness_dict = handle_numpy_array_comparison(simulated_data, experimental_data, key, fitness_dict)
                elif isinstance(simulated_data[sim_key], (float, int, np.int64, np.float64)):
                    ckwargs = {}
                    ckwargs['max_val'] = max_val
                    ckwargs['min_val'] = min_val
                    ckwargs['normalize'] = normalize
                    ckwargs['weight'] = weight
                    ckwargs['seed_fitness'] = kwargs.get('seed_fitness', {})
                    fitness_dict = handle_float_or_int_comparison(simulated_data, experimental_data, key, fitness_dict, plot_fit_curve=plot_fit_curve, fit_curve_dir=fit_curve_dir, **ckwargs)
                elif isinstance(simulated_data[sim_key], str): 
                    continue
                    #fitness_dict = handle_string_comparison(simulated_data, experimental_data, key, fitness_dict)
                elif isinstance(simulated_data[sim_key], set):
                    continue # ignore sets for now
                    #raise ValueError(f'Unknown data type for key {key}.')
                elif isinstance(simulated_data[sim_key], tuple):
                    continue # ignore tuples for now
                    #raise ValueError(f'Unknown data type for key {key}.')
                elif simulated_data[sim_key] is None:
                    if experimental_data[exp_key] is not None:
                        fitness_dict[sim_key]['fit'] = 1000 # if simulated data is None, set score to 1000
                        fitness_dict[sim_key]['value'] = None
                        fitness_dict[sim_key]['target'] = experimental_data[exp_key]
                    elif experimental_data[exp_key] is None:
                        fitness_dict[sim_key]['fit'] = 0
                        fitness_dict[sim_key]['value'] = None
                        fitness_dict[sim_key]['target'] = None
                    else:
                        fitness_dict[sim_key]['fit'] = 1000 # if experimental data is None, set score to 1000
                        fitness_dict[sim_key]['value'] = None
                        fitness_dict[sim_key]['target'] = experimental_data[exp_key]
                        fitness_dict[sim_key]['warnings'] = f'Some error occurred here'
                    #continue
                    #raise ValueError(f'Key {key} is None.')
                else:
                    # other data types
                    try: 
                        print(type(simulated_data[key]))
                    except: 
                        print(f'Unknown data type for key {key}.')
                    raise ValueError(f'Unknown data type for key {key}.')
            else:
                #print(f'Key {key} not found in experimental data.')
                pass
            
        # # recurse through all keys in fitness_dict and delete empty dictionaries
        # for key in fitness_dict:
        #     if fitness_dict[key] == {}: del fitness_dict[key]
        
        # return fitness dictionary
        return fitness_dict
    
    def clean_fitness_dict(fitness_dict, path=None, parent_key=None):
       
        # Create a list of keys to delete to avoid modifying the dictionary while iterating
        keys_to_delete = [key for key in fitness_dict if isinstance(fitness_dict[key], dict) and not fitness_dict[key]]

        # Delete empty dictionaries
        for key in keys_to_delete:
            #if key in unscored_keys:
            del fitness_dict[key] # if key is unscored, delete it
            # else:
            #     fitness_dict[key] = {'fit': 1000} # if key is scored, set it to 1000 if empty

        # Recurse through remaining dictionary values
        for key, value in list(fitness_dict.items()):  # Use list() to avoid RuntimeError due to dict size change
            if path is not None: abs_path = f'{path}.{key}'
            else: abs_path = key
            
            if isinstance(value, dict):
                clean_fitness_dict(value, path=abs_path, parent_key=key)  # Recurse into nested dictionaries
                # If the nested dictionary becomes empty after recursion, delete it as well
                if not value:
                    #if key in unscored_keys:
                    del fitness_dict[key] # if key is unscored, delete it
                    # else:
                    #     fitness_dict[key] = {'fit': 1000}
                    
        # return cleaned fitness dictionary            
        return fitness_dict

    def get_avg_fitness_weighted_v2(fitness_dict):
        """
        Recursively folds up the fitness values with normalization across different metric types.
        Normalizes fit values within each metric category before computing weighted averages.
        This ensures that metrics with different ranges (e.g., VP distance 0-100 vs parabola scores 0-1000)
        contribute equally to the final fitness.
        """
        
        def collect_leaf_fits(d, abs_path='', fit_groups=None):
            """Collect all leaf fit values and group them by metric type."""
            if fit_groups is None:
                fit_groups = {
                    'spike_times': [],  # VP distances
                    'scalar_metrics': []  # Other metrics (frs, burst_amp, etc.)
                }
            
            if 'fit' in d and not any(isinstance(v, dict) and 'fit' in v for v in d.values()):
                # This is a leaf node with a fit value
                fit_val = d['fit']
                if isinstance(fit_val, (int, float)) and not np.isnan(fit_val):
                    # Categorize based on path
                    if 'spiking_times_by_unit' in abs_path:
                        fit_groups['spike_times'].append((abs_path, fit_val))
                    else:
                        fit_groups['scalar_metrics'].append((abs_path, fit_val))
            
            # Recurse into child dictionaries
            for key, value in d.items():
                if isinstance(value, dict):
                    new_path = f'{abs_path}.{key}' if abs_path else key
                    collect_leaf_fits(value, new_path, fit_groups)
            
            return fit_groups
        
        def normalize_fits(fit_groups):
            """Normalize fit values within each group to [0, 1] range."""
            normalized_map = {}
            
            for group_name, fits in fit_groups.items():
                if not fits:
                    continue
                
                # Extract just the values
                values = [f[1] for f in fits]
                paths = [f[0] for f in fits]
                
                # Find max value in this group
                max_val = max(values)
                
                # Normalize to [0, 1]
                if max_val > 0:
                    for path, val in zip(paths, values):
                        normalized_map[path] = val / max_val
                else:
                    for path, val in zip(paths, values):
                        normalized_map[path] = 0.0
            
            return normalized_map
        
        def fold_up_normalized(d, abs_path='', normalized_map=None):
            """Fold up using normalized values."""
            if 'fit' in d and not any(isinstance(v, dict) and 'fit' in v for v in d.values()):
                # Leaf node - use normalized value if available
                if abs_path in normalized_map:
                    normalized_val = normalized_map[abs_path]
                    return normalized_val, 1
                else:
                    # Fallback to original value
                    fit_val = d['fit']
                    if isinstance(fit_val, (int, float)) and not np.isnan(fit_val):
                        return fit_val / 1000.0, 1  # Normalize to [0,1] assuming max 1000
                    else:
                        return 0.0, 0
            
            total_sum = 0.0
            total_count = 0
            
            for key, value in d.items():
                if isinstance(value, dict):
                    new_path = f'{abs_path}.{key}' if abs_path else key
                    child_sum, child_count = fold_up_normalized(value, new_path, normalized_map)
                    total_sum += child_sum
                    total_count += child_count
            
            if total_count > 0:
                weighted_avg = total_sum / total_count
                # Store as normalized value (0-1 range)
                d['fit_normalized'] = weighted_avg
                # Also store scaled back to 0-1000 for compatibility
                d['fit'] = min(weighted_avg * 1000.0, 1000.0)
                return total_sum, total_count
            
            return 1.0, 0  # Return normalized max (1.0) if no fits found
        
        try:
            # Step 1: Collect all leaf fit values and group by type
            fit_groups = collect_leaf_fits(fitness_dict)
            
            # Step 2: Normalize within each group
            normalized_map = normalize_fits(fit_groups)
            
            # Step 3: Fold up using normalized values
            total_sum, total_count = fold_up_normalized(fitness_dict, '', normalized_map)
            
            # Compute average from normalized values (0-1 range)
            if total_count > 0:
                avg_fitness_normalized = total_sum / total_count
                # Scale normalized average (0-1) back to (0-1000)
                avg_fitness = avg_fitness_normalized * 1000.0
            else:
                avg_fitness = 1000.0
            
            # Cap at 1000
            if avg_fitness > 1000.0:
                avg_fitness = 1000.0
            
            fitness_dict['fit'] = avg_fitness
            fitness_dict['fit_normalized'] = avg_fitness / 1000.0
            
        except Exception as e:
            traceback.print_exc()
            print(f"Error in normalized fitness calculation: {e}")
            avg_fitness = 1000.0
        
        return avg_fitness, fitness_dict

    def get_avg_fitness_weighted(fitness_dict):
        """
        Recursively folds up the fitness values in a nested dictionary.
        Computes weighted averages from the deepest level up, where the weight is the number
        of contributing leaf 'fit' values. Stores the computed average at each node.
        """

        def fold_up(d, abs_path=None):
            if 'fit' in d:
                fit_val = d['fit']
                if isinstance(fit_val, (int, float)) and not np.isnan(fit_val):
                    return fit_val, 1
                else:
                    return 0.0, 0  # ignore invalid or NaN

            total_sum = 0.0
            total_count = 0

            for key, value in d.items():
                if isinstance(value, dict):
                    # track pathing just for debugging
                    if abs_path is not None: abs_path = f'{abs_path}.{key}'
                    else: abs_path = key
                    
                    # recurse into child dictionaries
                    child_sum, child_count = fold_up(value, abs_path)
                    total_sum += child_sum
                    total_count += child_count

            if total_count > 0:
                weighted_avg = total_sum / total_count
                
                # sometimes the weighted average can be greater than 1000.0, so cap it
                if weighted_avg > 1000.0:
                    weighted_avg = 1000.0

                d['fit'] = weighted_avg  # store computed fitness at this level
                return total_sum, total_count

            # No fitness values found, assign default
            return 1000.0, 0

        try:
            total_sum, total_count = fold_up(fitness_dict)
            
            # # ensure three main keys are present
            main_fits = [
                        'spiking_data', 
                        #'bursting_data', 2025-05-08 23:22:12 - removing bursting data from fit for now
                        'mega_bursting_data']
            for key in main_fits:
                if key not in fitness_dict: 
                    fitness_dict[key] = {}
                if 'fit' not in fitness_dict[key]: 
                    fitness_dict[key]['fit'] = 1000  
            
            #avg_fitness = total_sum / total_count if total_count else 1000.0
            # aw 2025-04-23 01:05:10 on second thought, equally weight the three main keys, top level
            spiking_data_fit = fitness_dict['spiking_data']['fit']
            #bursting_data_fit = fitness_dict['bursting_data']['fit']
            mega_bursting_data_fit = fitness_dict['mega_bursting_data']['fit']
            # avg_fitness = (spiking_data_fit + bursting_data_fit + mega_bursting_data_fit) / 3.0
            # avg_fitness = mega_bursting_data_fit
            
            only_spiking = True
            if not only_spiking:
                avg_fitness = (spiking_data_fit + mega_bursting_data_fit) / 2.0
            else:
                avg_fitness = spiking_data_fit

            # aw 2025-04-23 03:34:59 temporarily prioritize bursting_metrics over spiking_metrics
            #avg_fitness = (spiking_data_fit + 2.0 * bursting_data_fit + 2.0 * mega_bursting_data_fit) / 5.0
            #avg_fitness = (spiking_data_fit + 2.0 * mega_bursting_data_fit) / 3.0

            # cap the average fitness to 1000.0
            if avg_fitness > 1000.0:
                avg_fitness = 1000.0
            
            fitness_dict['fit'] = avg_fitness  # store computed fitness at this level
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error folding up fitness values: {e}")
            avg_fitness = 1000.0  # default fallback

        #avg_fitness = fitness_dict.get('fit', 1000.0)  # get the final average fitness
        # at top level
        #total_sum, total_count = fold_up(fitness_dict)
        #avg_fitness = total_sum / total_count if total_count else 1000.0
        return avg_fitness, fitness_dict  # return the average fitness and the updated dictionary

    def get_avg_fitness(fitness_dict):
        """
        Recursively folds up the fitness values in a nested dictionary.
        It computes averages from the deepest level up and assigns an averaged 
        fitness value at each level.
        """
        
        #abs_path = ''

        def fold_up(fitness_dict, abs_path=''):
            # Check if current level contains 'fit' values
            if 'fit' in fitness_dict:
                return fitness_dict['fit'], 1  # Return fitness and count 1 comparison
            
            # Collect fitness values from child dictionaries
            fitness_values = []
            num_comparisons = 0  # Track number of fitness values aggregated

            for key, value in fitness_dict.items():
                if isinstance(value, dict):
                    prev_path = abs_path
                    abs_path = f'{abs_path}.{key}'
                    child_fitness, child_count = fold_up(value, abs_path)
                    abs_path = prev_path
                    fitness_values.append(child_fitness)
                    #fitness_dict[key]['count'] = child_count  # Store count of comparisons
                    num_comparisons += child_count

            # If we collected fitness values, compute weighted average
            if fitness_values:
                avg_fitness = np.nanmean(fitness_values)
                #fitness_dict['count'] = num_comparisons  # Store count of comparisons
                fitness_dict['fit'] = avg_fitness  # Store computed fitness at this level
                return avg_fitness, num_comparisons

            #return 0, 0  # If no fitness values exist, return 0 (or another default)
            return 1000, 0 #fitness_dict  # If no fitness values exist, return 1000 (or another default)

        try:
            avg_fitness, _ = fold_up(fitness_dict)  # Start recursion
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error folding up fitness values: {e}")
            avg_fitness = 1000  # Default error value

        return avg_fitness, fitness_dict
    
    def get_sim_data_from_call_stack(kwargs):
        """
        Get the candidate and job path from the call stack.
        """
        #simData = kwargs.get('simData', None)
        #if simData is None: raise ValueError('No simulated data found in kwargs.')
        
        # get candidate and job path from call stack
        from .utils.utils_old.extract_simulated_data import get_candidate_and_job_path_from_call_stack, retrieve_sim_data_from_call_stack, extract_data_of_interest_from_sim
        candidate_path, candidate_label = get_candidate_and_job_path_from_call_stack() #NOTE this part specifcally only works in batch processing, not in single candidate processing
        fitness_save_path = f'{candidate_path}_fitness.json'
        pkl_path = f'{candidate_path}_data.pkl'
        cfg_path = f'{candidate_path}_cfg.json'
        metrics_path = f'{candidate_path}_metrics.npy'
        
        # load simulated data
        sim.load(pkl_path)
        simData = sim.allSimData.todict().copy()
        popData = sim.net.allPops.copy()
        cellData = sim.net.allCells.copy()        
        simCfg = sim.cfg.todict().copy()
        netparams = sim.net.params.todict().copy()
        
        # update kwargs
        kwargs.update({
            'simData': simData,
            'cellData': cellData,
            'popData': popData,
            'simCfg': simCfg,
            'netParams': netparams,
            'simLabel': candidate_label,
            'data_file_path': pkl_path,
            'cfg_file_path': cfg_path,
            'fitness_save_path': fitness_save_path, #output
            'metrics_save_path': metrics_path, #output
        })
        
        # 
        print(f'data retrieved from {candidate_path}')
        return kwargs
    
    def get_sim_data_from_simData_pkl(kwargs):
        """
        Get the candidate and job path from simulated data object.
        """        
        # load simulated data pkl 
        sim_data_path = kwargs.get('sim_data_path', None)
        assert sim_data_path is not None, 'No simulated data path found in kwargs.'
        sim.load(sim_data_path)
        
        # setup paths
        candidate_path = sim_data_path.replace('_data.pkl', '')
        candidate_label = os.path.basename(candidate_path)
        fitness_save_path = f'{candidate_path}_fitness.json'
        pkl_path = f'{candidate_path}_data.pkl'
        cfg_path = f'{candidate_path}_cfg.json'
        metrics_path = f'{candidate_path}_metrics.npy'
        
        # load simulated data
        #sim.load(pkl_path)
        simData = sim.allSimData.todict().copy()
        popData = sim.net.allPops.copy()
        cellData = sim.net.allCells.copy()        
        simCfg = sim.cfg.todict().copy()
        netparams = sim.net.params.todict().copy()
        
        # update kwargs
        kwargs.update({
            'simData': simData,
            'cellData': cellData,
            'popData': popData,
            'simCfg': simCfg,
            'netParams': netparams,
            'simLabel': candidate_label,
            'data_file_path': pkl_path,
            'cfg_file_path': cfg_path,
            'fitness_save_path': fitness_save_path, #output
            'metrics_save_path': metrics_path, #output
        })
        
        # return kwargs
        print(f'data retrieved from {candidate_path}')
        return kwargs
    
    def map_unit_locations(simulated_metrics, experimental_metrics, **kwargs):
        # TODO: MAP SIMULATED UNITS TO EXPERIMENTAL UNITS BY XY COORDINATES
        experimental_unit_locations = experimental_metrics['unit_locations']
        simulated_unit_locations = simulated_metrics['unit_locations']
        
        simulated_to_experimental_mapping = {}
        for sim_unit_id, sim_loc in simulated_unit_locations.items():
            closest_exp_unit_id = None
            closest_distance = float('inf')
            for exp_unit_id, exp_loc in experimental_unit_locations.items():
                distance = np.linalg.norm(np.array(sim_loc) - np.array(exp_loc))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_exp_unit_id = exp_unit_id
                if distance == 0:
                    closest_exp_unit_id = exp_unit_id
                    break
            if distance != 0: print(f'Warning: No exact match for unit {sim_unit_id}. Closest match is {closest_exp_unit_id} with distance {closest_distance}.')
            simulated_to_experimental_mapping[sim_unit_id] = {
                'closest_exp_unit_id': closest_exp_unit_id,
                'distance': closest_distance
            }
            
        # reorient
        experimental_to_simulated_mapping = {v['closest_exp_unit_id']: k for k, v in simulated_to_experimental_mapping.items()}
            
        kwargs['simulated_to_experimental_mapping'] = simulated_to_experimental_mapping
        kwargs['experimental_to_simulated_mapping'] = experimental_to_simulated_mapping
        
        # return
        return kwargs
    
    def get_sim_data(kwargs):
        # check if batching - if so, kwargs need to be expanded to include all simulated data instead of just simData
        if kwargs.get('batching', False):
            kwargs = get_sim_data_from_call_stack(kwargs)
        else:
            kwargs = get_sim_data_from_simData_pkl(kwargs)
        return kwargs
    
    def get_simulated_metrics(kwargs):
        # try loading simulated data if its already been computed
        simulated_metrics = None
        metrics_save_path = kwargs.get('metrics_save_path', None)
        if kwargs.get('try_load', False):
            if os.path.exists(metrics_save_path):
                simulated_metrics = np.load(metrics_save_path, allow_pickle=True).item()
                print(f'Loaded metrics from {metrics_save_path}')
            
        # if simulated metrics is still None, compute them
        if simulated_metrics is None:
            simulated_metrics = compute_network_metrics(**kwargs)
            
        # if simulated metrics is still none, there was an error
        if simulated_metrics is None:
            return None
        
        # save simulated metrics to .npy file
        if metrics_save_path is not None: # save metrics to .npy file
            np.save(metrics_save_path, simulated_metrics)
            print(f'Saved metrics to {metrics_save_path}')
            
        return simulated_metrics
    
    def break_deals(fitness_dict, simulated_metrics, kwargs):
        """
        Deal breaker function to handle errors and return a high fitness value.
        """
        # import traceback
        # print('Error in fitness calculation. Returning high fitness value.')
        # traceback.print_exc()
        # fitness_save_path = kwargs.get('fitness_save_path', None)
        
        # init
        deal_broken = False

        # unpack
        mega_bursting_data = fitness_dict.get('mega_bursting_data', {})
        burst_metrics = mega_bursting_data.get('burst_metrics', {})
        spike_data = fitness_dict.get('spiking_data', {})

        # baseline and burst amplitude must both be below 1000
        baseline_fit = mega_bursting_data.get('baseline', {}).get('fit', 1000)
        burst_amp_fit = burst_metrics.get('burst_amp', {}).get('fit', 1000)

        #fr and ei ratio fits
        frs_fit = spike_data.get('frs', {}).get('fit', 1000)
        EI_ratio_fit = spike_data.get('EI_fr_ratios', {}).get('fit', 1000)

        # TODO: migrate this into metrics later, #HACK
        # # aw 2025-05-27 13:09:02 done, going to leave this in here for now and just remove some of the deal breakers
        sim_spike_data = simulated_metrics.get('spiking_data', {})
        e_frs = sim_spike_data.get('e_frs', {}).get('data', [])
        i_frs = sim_spike_data.get('i_frs', {}).get('data', [])

        # get number of non-nan values in e_frs and i_frs
        num_e_frs = np.count_nonzero(~np.isnan(e_frs))
        num_i_frs = np.count_nonzero(~np.isnan(i_frs))

        # 
        if num_e_frs == 0 or num_i_frs == 0:
            deal_broken = True
            print(f'Warning: No excitatory or inhibitory firing rates found. Deal broken.')
            return deal_broken

        # total expected number of excitatory and inhibitory units
        tot = len(e_frs) + len(i_frs)
        total_firing = num_e_frs + num_i_frs

        #
        spiking_metrics_by_unit = sim_spike_data.get('spiking_metrics_by_unit', {})
        Espikes = 0
        Ispikes = 0
        for unit, unit_type in simulated_metrics.get('unit_types', {}).items():
            if unit_type == 'E':
                # if unit not in e_frs: e_frs.append(unit)
                num_spikes = spiking_metrics_by_unit.get(unit, {}).get('num_spikes', 0)
                Espikes += num_spikes
            elif unit_type == 'I':
                #if unit not in i_frs: i_frs.append(unit)
                num_spikes = spiking_metrics_by_unit.get(unit, {}).get('num_spikes', 0)
                Ispikes += num_spikes
        Espikes_per_unit = Espikes / num_e_frs if num_e_frs > 0 else 0
        Ispikes_per_unit = Ispikes / num_i_frs if num_i_frs > 0 else 0

        # if Espikes_per_unit > Ispikes_per_unit:
        #     print(f'Warning: More excitatory spikes per unit ({Espikes_per_unit}) than inhibitory spikes per unit ({Ispikes_per_unit}). Deal broken.')
        #     deal_broken = True
        #     return deal_broken
        
        # if num_e_frs < num_i_frs:
        #     # if there are more inhibitory units than excitatory units, then the deal is broken
        #     print(f'Warning: More inhibitory units ({num_i_frs}) than excitatory units ({num_e_frs}). Deal broken.')
        #     deal_broken = True
        #     return deal_broken

        # if total_firing/tot < 0.5:
        #     # if the total firing rate is less than 50% of the expected total firing rate, then the deal is broken
        #     print(f'Warning: Total units firing ({total_firing}) is less than 50% of expected total units ({tot}). Deal broken.')
        #     deal_broken = True
        #     return deal_broken

        # amplitude and baseline must both be below 1000
        # if baseline_fit < 1000 and burst_amp_fit < 1000:
        #     pass
        # else:
        #     deal_broken = True
        #     print(f'Warning: Baseline fit ({baseline_fit}) or burst amplitude fit ({burst_amp_fit}) is above 1000. Deal broken.')
        #     return deal_broken

        # if burst_amp_fit < 1000, baseline_fit must also be below 1000
        # accepting baseline_fit < 1000 even if burst_amp_fit is above 1000
        # if burst_amp_fit < 1000:
        #     try: assert baseline_fit < 1000, f'Baseline fit ({baseline_fit}) is above 1000.'
        #     except AssertionError as e:
        #         print(f'Warning: {e}. Deal broken.')
        #         deal_broken = True
        #         return deal_broken
        # else:
        #     pass
        
        # frs and EI ratio must both be below 1000
        # if frs_fit < 1000 and EI_ratio_fit < 1000:
        #     pass
        # else:
        #     deal_broken = True
        #     return deal_broken


        # test conns, reject any simulation having a neuron with zero cons
        cell_data = simulated_metrics.get('cellData', {})
        for cell in cell_data:
            gid = cell.get('gid', None)
            if gid is None: continue # skip if no gid...but this should not happen
            conns = cell_data[gid].get('conns', None)
            if conns is not None:
                if len(conns) == 0:
                    print(f'Warning: Neuron {gid} has no connections. Deal broken.')
                    deal_broken = True
                    return deal_broken
            else:
                print(f'Warning: Neuron {gid} has no connections data. Deal broken.')
                deal_broken = True
                return deal_broken

        # all tests passed
        deal_broken = False
        return deal_broken
    
    # main logic =================================================================
    try:
        # print(simulated_data)
        # print(reference_data_path)
        time_start = time.time() # start timer
        experimental_metrics = np.load(reference_data_path, allow_pickle=True).item() # load experimental data for reference      
        kwargs['source'] = 'simulated' # set source to simulated
        kwargs = get_sim_data(kwargs) # fully define kwargs depending on run case
        simulated_metrics = get_simulated_metrics(kwargs) # load or compute simulated metrics
        if simulated_metrics is None:
            print('Error: simulated metrics is None.')
            return 1000
        
        kwargs = map_unit_locations(simulated_metrics, experimental_metrics, **kwargs) # map unit locations of experimental units to simulated units
        
        # calculate fitness
        fit_schema = kwargs.get('fit_schema', None)
        if fit_schema is None:
            raise ValueError('No fitness schema found in kwargs.')
        fitness_dict = calculate_fitness(simulated_metrics, experimental_metrics, fitness_dict={}, **kwargs)
        #fitness_dict = clean_fitness_dict(fitness_dict)    
        
        #avg_fitness, fitness_dict = get_avg_fitness(fitness_dict)
        avg_fitness, fitness_dict = get_avg_fitness_weighted_v2(fitness_dict)
        # print(f'Average fitness: {avg_fitness}')

        #TODO: implement deal breakers??
        # dealbreakers
        # broken = False
        # broken = break_deals(fitness_dict, simulated_metrics, kwargs)
        # if broken:
        #     print('Deal broken. Returning high fitness value.')
        #     fitness_dict['broken'] = True
        #     fitness_dict['unbroken_fit'] = avg_fitness
        #     avg_fitness = 1000 #redefine avg_fitness to be high if deal is broken
        #     fitness_dict['fit'] = avg_fitness 

        fitness_save_path = kwargs.get('fitness_save_path', None)
        if fitness_save_path is not None: # save fitness to .json file
            with open(fitness_save_path, 'w') as f:
                json.dump(fitness_dict, f, indent=4)
            print(f'Saved fitness to {fitness_save_path}')
        elapsed_time = time.time() - time_start
        print(f'Elapsed time: {elapsed_time} seconds.')        
        return avg_fitness        
    except Exception as e:
        # handle errors
        traceback.print_exc()
        print(f'Error calculating fitness: {e}')
        avg_fitness = 1000
        fitness_save_path = kwargs.get('fitness_save_path', None)
        if fitness_save_path is not None: # save fitness to .json file
            with open(fitness_save_path, 'w') as f:
                if fitness_dict is not None:
                    json.dump(fitness_dict, f, indent=4)
                else:
                    json.dump({
                        'error': str(e),
                        'fit': 1000,
                        }, f, indent=4)
            print(f'Saved fitness to {fitness_save_path}')
        else:
            print('No fitness save path found.')
        return avg_fitness

# Re-import libraries after state reset
import numpy as np
import matplotlib.pyplot as plt

def the_scoring_function_symmetric_sigmoid(val, target_val, maxFitness, weight,
                                           min_val=None, max_val=None, plot_fit_curve=False, fit_curve_path=None):
    if val is None:
        return maxFitness

    # Define sigmoid steepness
    inner_scale = weight / 10.0
    outer_scale = weight
    delta = abs(target_val)

    def smooth_sigmoid(x, scale):
        return maxFitness / (1 + np.exp(scale * abs(x - target_val))) - maxFitness / 2

    def compute_score(x):
        if min_val is not None and max_val is not None:
            if min_val <= x <= max_val:
                return smooth_sigmoid(x, inner_scale)
            elif x < min_val:
                shift = target_val - delta
                return maxFitness * (1 - 1 / (1 + np.exp(-outer_scale * (x - shift))))
            else:
                shift = target_val + delta
                return maxFitness * (1 - 1 / (1 + np.exp(outer_scale * (x - shift))))
        elif min_val is not None:
            if x >= min_val:
                return smooth_sigmoid(x, inner_scale)
            else:
                shift = target_val - delta
                return maxFitness * (1 - 1 / (1 + np.exp(-outer_scale * (x - shift))))
        elif max_val is not None:
            if x <= max_val:
                return smooth_sigmoid(x, inner_scale)
            else:
                shift = target_val + delta
                return maxFitness * (1 - 1 / (1 + np.exp(outer_scale * (x - shift))))
        else:
            return smooth_sigmoid(x, inner_scale)

    if plot_fit_curve:
        relevant_vals = [v for v in [min_val, max_val, val, target_val] if v is not None]
        min_x = min(relevant_vals)
        max_x = max(relevant_vals)
        span = max(0.1 * abs(min_x), 0.1 * abs(max_x), 1)
        x_start = min_x - span
        x_end = max_x + span

        x_range = np.linspace(x_start, x_end, 500)
        y_range = [compute_score(x) for x in x_range]

        plt.figure(figsize=(8, 4))
        plt.plot(x_range, y_range, label="Scoring Curve")
        plt.axvline(target_val, color='green', linestyle='--', label='Target Value')
        plt.axvline(val, color='red', linestyle=':', label='Actual Value')
        if min_val is not None:
            plt.axvline(min_val, color='blue', linestyle='-.', label='Min Bound')
        if max_val is not None:
            plt.axvline(max_val, color='purple', linestyle='-.', label='Max Bound')
        plt.xlabel("Input Value")
        plt.ylabel("Score")
        plt.title("Fitness Scoring Curve (Centered at 0)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if fit_curve_path is not None:
            plt.savefig(fit_curve_path, dpi=300)
            print(f'Saved fit curve to {fit_curve_path}')
            plt.savefig(fit_curve_path.replace('.png', '.svg'))
            print(f'Saved fit curve to {fit_curve_path.replace(".png", ".svg")}')
        plt.close()

    return compute_score(val)

# Re-define the updated function with blended sigmoid logic
def the_scoring_function_blended_sigmoid(val, target_val, maxFitness, weight,
                                         min_val=None, max_val=None, plot_fit_curve=False, fit_curve_path=None):
    if val is None:
        return maxFitness

    # Gentle slope for center sigmoid, steeper for outer penalty
    inner_scale = 0.02 * weight
    outer_scale = 0.1 * weight
    delta = abs(target_val)

    def smooth_centered_sigmoid(x):
        return maxFitness - (maxFitness - 1) / (1 + np.exp(-inner_scale * (x - target_val)))

    def shifted_sigmoid(x, direction):
        shift = direction * delta
        return maxFitness - (maxFitness - 1) / (1 + np.exp(direction * outer_scale * (x - (target_val + shift))))

    def compute_score(x):
        if min_val is not None and max_val is not None:
            if min_val <= x <= max_val:
                return smooth_centered_sigmoid(x)
            elif x < min_val:
                return shifted_sigmoid(x, -1)
            else:
                return shifted_sigmoid(x, 1)
        elif min_val is not None:
            if x >= min_val:
                return smooth_centered_sigmoid(x)
            else:
                return shifted_sigmoid(x, -1)
        elif max_val is not None:
            if x <= max_val:
                return smooth_centered_sigmoid(x)
            else:
                return shifted_sigmoid(x, 1)
        else:
            return smooth_centered_sigmoid(x)

    if plot_fit_curve:
        relevant_vals = [v for v in [min_val, max_val, val, target_val] if v is not None]
        min_x = min(relevant_vals)
        max_x = max(relevant_vals)
        span = max(0.1 * abs(min_x), 0.1 * abs(max_x), 1)
        x_start = min_x - span
        x_end = max_x + span

        x_range = np.linspace(x_start, x_end, 500)
        y_range = [compute_score(x) for x in x_range]

        plt.figure(figsize=(8, 4))
        plt.plot(x_range, y_range, label="Scoring Curve")
        plt.axvline(target_val, color='green', linestyle='--', label='Target Value')
        plt.axvline(val, color='red', linestyle=':', label='Actual Value')
        if min_val is not None:
            plt.axvline(min_val, color='blue', linestyle='-.', label='Min Bound')
        if max_val is not None:
            plt.axvline(max_val, color='purple', linestyle='-.', label='Max Bound')
        plt.xlabel("Input Value")
        plt.ylabel("Score")
        plt.title("Fitness Scoring Curve (Smooth)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if fit_curve_path is not None:
            plt.savefig(fit_curve_path, dpi=300)
            print(f'Saved fit curve to {fit_curve_path}')
            plt.savefig(fit_curve_path.replace('.png', '.svg'))
            print(f'Saved fit curve to {fit_curve_path.replace(".png", ".svg")}')
        plt.close()

    return compute_score(val)

def the_scoring_function_quadratic_smooth_sigmoid(val, target_val, maxFitness, weight, 
                                                  min_val=None, max_val=None, plot_fit_curve=False, fit_curve_path=None):
    if val is None:
        return maxFitness

    def shifted_sigmoid(x, direction):
        shift = direction * abs(target_val)
        scale = 0.1 / weight
        score = maxFitness - (maxFitness - 1) / (1 + np.exp(direction * scale * (x - (target_val + shift))))
        return score

    def compute_score(x):
        if min_val is not None and max_val is not None:
            a_left = (maxFitness - 1) / ((target_val - min_val) ** 2 / weight) if target_val != min_val else float('inf')
            a_right = (maxFitness - 1) / ((max_val - target_val) ** 2 / weight) if target_val != max_val else float('inf')
            if x <= target_val:
                return min(a_left * (x - target_val) ** 2 + 1, maxFitness)
            else:
                return min(a_right * (x - target_val) ** 2 + 1, maxFitness)
        if min_val is None and max_val is not None:
            if x >= target_val:
                return min((maxFitness - 1) / ((max_val - target_val) ** 2 / weight) * (x - target_val) ** 2 + 1, maxFitness)
            else:
                return shifted_sigmoid(x, -1)
        elif max_val is None and min_val is not None:
            if x <= target_val:
                return min((maxFitness - 1) / ((target_val - min_val) ** 2 / weight) * (x - target_val) ** 2 + 1, maxFitness)
            else:
                return shifted_sigmoid(x, 1)
        else:
            return shifted_sigmoid(x, -1) if x < target_val else shifted_sigmoid(x, 1)

    # if plot_fit_curve:
    #     if min_val is not None and max_val is not None:
    #         x_range = np.linspace(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val), 500)
    #     else:
    #         span = abs(target_val) * 2 if abs(target_val) > 1 else 2
    #         x_range = np.linspace(target_val - span, target_val + span, 500)

    #     y_range = [compute_score(x) for x in x_range]

    #     plt.figure(figsize=(8, 4))
    #     plt.plot(x_range, y_range, label="Scoring Curve")
    #     plt.axvline(target_val, color='green', linestyle='--', label='Target Value')
    #     plt.axvline(val, color='red', linestyle=':', label='Actual Value')
    #     plt.xlabel("Input Value")
    #     plt.ylabel("Score")
    #     plt.title("Fit Curve")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
        #plt.show()
    if plot_fit_curve:
        relevant_vals = [v for v in [min_val, max_val, val, target_val] if v is not None]
        min_x = min(relevant_vals)
        max_x = max(relevant_vals)
        span = max(0.1 * abs(min_x), 0.1 * abs(max_x), 1)
        x_start = min_x - span
        x_end = max_x + span

        x_range = np.linspace(x_start, x_end, 500)
        y_range = [compute_score(x) for x in x_range]

        plt.figure(figsize=(8, 4))
        plt.plot(x_range, y_range, label="Scoring Curve")
        plt.axvline(target_val, color='green', linestyle='--', label='Target Value')
        plt.axvline(val, color='red', linestyle=':', label='Actual Value')
        if min_val is not None:
            plt.axvline(min_val, color='blue', linestyle='-.', label='Min Bound')
        if max_val is not None:
            plt.axvline(max_val, color='purple', linestyle='-.', label='Max Bound')
        plt.xlabel("Input Value")
        plt.ylabel("Score")
        plt.title("Fitness Scoring Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        #save as png and svg
        if fit_curve_path is not None:
            plt.savefig(fit_curve_path, dpi=300)
            print(f'Saved fit curve to {fit_curve_path}')
            plt.savefig(fit_curve_path.replace('.png', '.svg'))
            print(f'Saved fit curve to {fit_curve_path.replace(".png", ".svg")}')
        plt.close()

    return compute_score(val)


def the_scoring_function_quadratic_smooth_sigmoid_dep(val, target_val, maxFitness, weight, min_val=None, max_val=None, plot_fit_curve=False):
    """
    Quadratic scoring function with a smoothly adjusted sigmoid penalty:
    - If both bounds exist, use a quadratic function with sharper slopes based on weight.
    - If one bound is missing, apply a flipped sigmoid penalty with an iteratively adjusted shift
      so that it smoothly approaches 1 near the target without engaging piecewise rules.
    - The weight parameter controls sharpness:
      - Higher weight makes the quadratic function steeper and the sigmoid approach 1 or 1000 faster.
    """
    if val is None:
        return maxFitness

    # Case 1: Both bounds are present (standard quadratic)
    if min_val is not None and max_val is not None:
        a_left = (maxFitness - 1) / ((target_val - min_val) ** 2 / weight) if target_val != min_val else float('inf')
        a_right = (maxFitness - 1) / ((max_val - target_val) ** 2 / weight) if target_val != max_val else float('inf')

        if val <= target_val:
            return min(a_left * (val - target_val) ** 2 + 1, maxFitness)
        else:
            return min(a_right * (val - target_val) ** 2 + 1, maxFitness)

    # Adaptive shifted sigmoid function
    def shifted_sigmoid(x, direction):
        """
        Flipped sigmoid function with adaptive shifting:
        - Ensures that values near the target smoothly approach 1.
        - Instead of scaling, we shift the sigmoid away from the target.
        - The weight parameter adjusts the steepness of the sigmoid.
        """
        shift = direction * abs(target_val)  # Shift proportional to the target value magnitude
        scale = 0.1 / weight  # Higher weight makes the transition steeper
        score = maxFitness - (maxFitness - 1) / (1 + np.exp(direction * scale * (x - (target_val + shift))))
        return score
        #return maxFitness - (maxFitness - 1) / (1 + np.exp(direction * scale * (x - (target_val + shift))))

    # Case 2: One bound is missing, apply the adaptive shifted sigmoid
    if min_val is None and max_val is not None:
        if max_val is not None and val >= target_val:
            return min((maxFitness - 1) / ((max_val - target_val) ** 2 / weight) * (val - target_val) ** 2 + 1, maxFitness)
        else:
            return shifted_sigmoid(val, -1)  # Shifted sigmoid for extreme low values

    elif max_val is None and min_val is not None:
        if min_val is not None and val <= target_val:
            return min((maxFitness - 1) / ((target_val - min_val) ** 2 / weight) * (val - target_val) ** 2 + 1, maxFitness)
        else:
            return shifted_sigmoid(val, 1)  # Shifted sigmoid for extreme high values

    # Case 3: No bounds at all (shifted sigmoid on both sides)
    elif min_val is None and max_val is None:
        if val < target_val:
            return shifted_sigmoid(val, -1)
        else:
            return shifted_sigmoid(val, 1)
        #return shifted_sigmoid(val, -1) if val < target_val else shifted_sigmoid(val, 1)

def fitnessFunc_v2(simulated_data, reference_data_path, **kwargs): #def fitnessFunc_v2(simulated_data, experimental_data, **kwargs):
    """
    Main logic of the calculate_fitness function.
    Ensures the function does not crash and always returns a fitness value.
    """
    experimental_metrics = np.load(reference_data_path, allow_pickle=True).item()
    
    # subfunctions =================================================================
    def calculate_fitness(simulated_data, experimental_data, fitness_dict={}, path=None, parent_key=None, **kwargs):
        """
        Calculate the fitness value based on the simulated and experimental data.
        """
        
        # subfunctions =========================================================
        
        def init_skip_keys():
            # skip keys of non-interest
            skip_keys = [
                
                #not sure about this, skipping for now, unidentifed units
                'u_frs',
                'u_isi',
                
                # dicts
                'convolved_data', # the fitness of these data are interrogated elsewhere
                
                # lists
                'gids',
                'unit_ids',
                'burst_id',
                'quiet_id',
                'burst_ids',
                
                # numpy arrays
                'data', # aw 2025-03-05 13:08:43 anything with a data key has summary statistics to be compared
                'timeVector', # aw 2025-03-05 10:15:05 - time will differ between simulated and experimental data
                
                #ints/floats
                'num_spikes', # aw 2025-03-05 10:15:05 - deprecated key
                
                # other
                'ax',
                
                # 'source', 
                # 'simulated_data', # aw 2025-03-05 10:15:05 - deprecated key, network_data outputs no longer distinguish between simulated and experimental data this way.
                # 'timeVector', # aw 2025-03-05 10:15:05 - time will differ between simulated and experimental data
                # 'spiking_summary_data', # aw 2025-03-05 10:15:05 - deprecated key
                # 'spiking_data_by_unit', # aw 2025-03-05 10:15:05 - deprecated key
                # 'wf_metrics', # # aw 2025-03-05 10:15:05 - not implmented yet for simulated data
                # 'data', # # aw 2025-03-05 13:08:43 anything with a data key has summary statistics to be compared
                'warnings', 
                # 'ax',
                # 'convolved_data', #ignore this, summary metrics are available 
                # 'burst_id', 
                # 'quiet_id',
                # #  'bursts',
                # #  'quiets',
                # 'burst_durations',
                # 'quiet_durations',
                # 'note',
                'num_bursts', #skip these raw data keys because they're recording duration dependent
                # 'num_spikes',
                # 'burst_start',
                # 'burst_end',
                # 'participating_units',
                # 'time_sequence',
                # 'classified_sequence',
                # 'cat_sequence',
                'spike_times',
                'spiking_times_by_unit',
                
                'unit_ids',
                'unit_locations',
                'unit_types',
                
                'wf_metrics',
                
                # 'classification_output',
                # 'classification_data',
                # 'unit_types',
                # 'sampling_rate',
                # 'gids',
                # 'unit_ids',
                # 'burst_ids',
                
                # #paths to skip
                # 'recording_path',
                # 'sorting_output',
                # 'waveform_output',
                # 'dtw_output',
                # 'mega_dtw_output',
                # 'sim_data_path',
                'source',
                'sampling_rate',

                        
                        ]
            return skip_keys
            
        def handle_list_comparison(simulated_data, experimental_data, key, fitness_dict):
            # compare lists
            #abs_path = f'{path}.{key}'
            list_list = []
            
            include_logic = [
                'participating_units' in key,
            ]
            exclude_logic = [
                'spike_times_by_unit' in key,
                
                # exclude unit sequence, time_sequence, relative_time_sequence, classified_sequence, cat_sequence. Sequence stack has all this info. Sequence stack would be 
                # best to compare using dtw - but I think I'll keep that turned off for now.
                'unit_sequence' in key,
                'unit_seqeunce' in key, # stupid typo leftover in some of the data
                'sequence_stack' in key,
                'time_sequence' in key,
                'relative_time_sequence' in key,
                'classified_sequence' in key,
                'cat_sequence' in key,
                
                # exclude waveform metrics and classification data - not comparable between simulated and experimental data
                'wf_metrics' in abs_path,
                'classified_units' in abs_path,
                'classification_output' in abs_path,
            ]
            if any(include_logic) and not any(exclude_logic):
                if 'participating_units' in key:
                    def compare_participating_units(simulated_data, experimental_data, key, fitness_dict):
                        # compare the set of both lists - get percent overlap
                        simulated_set = set(simulated_data[key])
                        experimental_set = set(experimental_data[key])
                        overlap = len(simulated_set.intersection(experimental_set))
                        percent_overlap = overlap / len(experimental_set)
                        return percent_overlap
                    
                    percent_overlap = compare_participating_units(simulated_data, experimental_data, key, fitness_dict)
                    
                    # score
                    target = 1 # best possible overlap is 1
                    weight = 1 # penalty for deviation from target
                    maxFitness = 1000 # maximum fitness score
                    min_val = 0 # minimum value allowed
                    max_val = 1
                    #score = the_scoring_function(percent_overlap, target, weight, maxFitness, min_val=min_val, max_val=max_val)
                    score = the_scoring_function_quadratic_smooth_sigmoid(percent_overlap, target, maxFitness, weight, min_val=min_val, max_val=max_val)
                    fitness_dict[key]['fit'] = score
                    
                    #return fitness_dict
                else:
                    raise ValueError(f'Unknown list for key {key}.')
            elif any(exclude_logic):
                pass
            else:
                raise ValueError(f'Unknown list for key {key}.')
            
            return fitness_dict
        
        def handle_numpy_array_comparison(simulated_data, experimental_data, key, fitness_dict):
            # # compare numpy arrays
            # print(f'Comparing numpy array for key {key}.')
            # #abs_path = f'{path}.{key}'
            # dtw_list = ['time', 'bursts', 'quiets', 'sequence_stack']
            
            #
            dtw_compare_keys = [
                'spike_times',
                'spiking_times_by_unit',
                'spiking_metrics_by_unit',
                'bursts',
                'quiets',
                
            ]
            
            if any(item in abs_path for item in dtw_compare_keys):
            #if 'times' in abs_path:
                #start = time.time()
                if len(simulated_data[key]) > 0 and len(experimental_data[key]) > 0:
                    # distance
                    distance = event_times_dtw_distance(simulated_data[key], experimental_data[key])
                    fitness_dict[key]['distance'] = distance
                    
                    #score
                    target = 0 # best possible distance is the same
                    weight = 1 # penalty for deviation from target
                    maxFitness = 1000 # maximum fitness score
                    min_val = 0 # minimum value allowed
                    max_val = 1000 # TODO: set max val based on experimental data. 
                                    #May need to work into regular network_metrics to make this easy
                    #score = the_scoring_function(distance, target, weight, maxFitness, min_val=min_val, max_val=max_val)
                    score = the_scoring_function_quadratic_smooth_sigmoid(distance, target, maxFitness, weight, min_val=min_val, max_val=max_val)
                    fitness_dict[key]['fit'] = score
                else:
                    #continue
                    raise ValueError(f'Empty numpy array for key {key}.')
                #time_elapsed = time.time() - start
                #print(f'DTW distance: {distance}')
                #print(f'DTW time elapsed: {time_elapsed} seconds.')
                
            else:
                raise ValueError(f'Unknown numpy array for key {key}.')
                
            return fitness_dict
        
        def handle_float_or_int_comparison(simulated_data, experimental_data, key, fitness_dict):
            # # compare floats or ints
            # #abs_path = f'{path}.{key}'
            # simple_comparison_list = ['fr', 'mean', 'median', 'std', 'cov',
            #                             'burst_part_rate', 'quiet_part_rate', 'burst_part_perc', 
            #                             'fano_factor.in_burst', 'fano_factor.out_burst', 
            #                             'burst_rate', 'spike_count',
            #                             'num_units_participating', 'duration', 'spike_rate',]
            
            compare_logic = [
                'fr' in abs_path, 
                'mean' in abs_path, 
                'median' in abs_path, 
                'std' in abs_path, 
                'cov' in abs_path,
                'fano_factor.in_burst' in abs_path, 
                'fano_factor.out_burst' in abs_path,
                'burst_part_rate' in abs_path,
                'quiet_part_rate' in abs_path,
                'burst_part_perc' in abs_path,
                'burst_rate' in abs_path,
                'max' in abs_path,
                'min' in abs_path,
                'num_units_participating' in abs_path,
                'burst_parts' in abs_path and 'duration' in abs_path,
                'burst_parts' in abs_path and 'spike_rate' in abs_path,
                
                #dtw keys - idk if it's practical to compare these - not sure if doing dtw analysis with all simulations
                'mean_dtw' in abs_path,
                'std_dtw' in abs_path,
                'variance_dtw' in abs_path,
                'cov_dtw' in abs_path,
                'median_dtw' in abs_path,
                'max_dtw' in abs_path,
                'min_dtw' in abs_path,
                
            ]
            
            exclude_logic = [
                'unit_metrics' in abs_path and 'num_spikes' in abs_path,
                'unit_metrics' in abs_path and 'burst_durations' in abs_path,
                'unit_metrics' in abs_path and 'quiet_durations' in abs_path,
                'burst_metrics' in abs_path and 'num_bursts' in abs_path,
                'burst_start' in abs_path,
                'burst_end' in abs_path,
                'spike_count' in abs_path,  # ignore spike count - it's duration dependent
                
                # ignore wf_metrics for now.. not implemented yet for simulated data
                'wf_metrics' in abs_path,
                'classified_units' in abs_path,
                'classification_output' in abs_path,
                
                # other keys specific to experimental data
                'sampling_rate' in abs_path,
                
                # exclude min diff burst... not sure how I'm using this yet.
                'min_dtw_diff_vector' in abs_path,
                
                #dtw
                'num_computations' in abs_path,
            ]
            
            if any(compare_logic) and not any(exclude_logic):
                
                #modulate min, max, and weights
                min_zero_list = ['fr', 'mean', 'median', 'std', 'cov', 'max', 'min', 
                                 'burst_rate', 'num_units_participating', 'duration',
                                 'spike_rate', ]
                zero_to_one_list = ['burst_part_rate', 'quiet_part_rate', 'burst_part_perc']
                
                if key in min_zero_list:
                    weight = 1
                    min_val = 0
                    max_val = None
                elif key in zero_to_one_list:
                    weight = 1
                    min_val = 0
                    max_val = 1
                elif 'fano_factor' in abs_path:
                    weight = 1
                    min_val = 0
                    max_val = None
                else:
                    print(f'warning: No min/max/weight defined for key {key}. Using default values.')
                
                
                # score
                target = experimental_data[key]
                maxFitness = 1000 # maximum fitness score
                #score = the_scoring_function(simulated_data[key], target, weight, maxFitness, min_val=min_val, max_val=max_val)
                score = the_scoring_function_quadratic_smooth_sigmoid(simulated_data[key], target, maxFitness, weight, min_val=min_val, max_val=max_val)
                fitness_dict[key]['fit'] = score
            elif any(exclude_logic):
                pass # ignore these keys
            else:
                raise ValueError(f'Unknown float or int for key {key}.')
            
            return fitness_dict
        
        def handle_string_comparison(simulated_data, experimental_data, key, fitness_dict):
            # compare strings
            str_list = []
            if any(item in abs_path for item in str_list):
                # simple comparison
                if simulated_data[key] == experimental_data[key]: 
                    fitness_dict[key]['fit'] = 1
                else: 
                    fitness_dict[key]['fit'] = 1000
            else:
                print(simulated_data[key])
                #raise ValueError(f'Unknown string for key {key}.')
                
            return fitness_dict
        
        # main logic =============================================================
        
        # skip keys of non-interest
        skip_keys = init_skip_keys()
        
        # update path
        if parent_key is not None and path is not None: 
            path += f'.{parent_key}'
            #fitness_dict[parent_key] = {}
        elif parent_key is not None and path is None: 
            path = parent_key
            #fitness_dict[parent_key] = {}
        elif parent_key is None and path is not None: pass
        else: path = None
        if path is not None: print(f'Path: {path}')
        
        # loop through all paths in both dictionaries and compare the values of the keys
        for key in simulated_data:
            #skips
            if key in skip_keys: continue # skip keys of non-interest
            #if path is not None and any(item in path for item in skip_paths): continue # skip paths of non-interest
            
            # print
            #print(f'Comparing key: {key}')
            
            # # update fit dict
            # if key not in fitness_dict:
            #     fitness_dict[key] = {}
            
            #debug - speed things up
            # try: 
            #     key = int(key)
            #     if key > 5: 
            #             print(f'Skipping key {key} to speed things up.')
            #             continue
            # except: pass            
            
            #if path is None: path = key
            if key in experimental_data:
                # #print('Comparing key:', key)
                # if path is not None: abs_path = f'{path}.{key}'
                # else: abs_path = key
                # # handle data types
                def handle_unit_mapping(path, key, kwargs):
                    if path is not None:
                        abs_path = f'{path}.{key}'
                        # sim_path = f'{path}.{key}'
                        # exp_path = f'{path}.{key}'
                    else:
                        abs_path = key
                        # sim_path = key
                        # exp_path = key
                    
                    try: 
                        if 'by_unit' in abs_path:
                            # split abs_path by .
                            parts = abs_path.split('.')
                            
                            #check if any part is a number
                            success_int = False
                            for i, part in enumerate(parts):
                                try:
                                    int(part)
                                    #parts[i] = int(part)
                                    exp_id = int(part)
                                    success_int = True
                                    break
                                except:
                                    pass
                            
                            #
                            if success_int: pass
                            else: raise ValueError('No unit ID found in path.')
                            
                            # reconstruct abs_path
                            # repalce int with mapped int.
                            # get int value, get mapped value, replace in abs_path
                            sim_id = kwargs.get('experimental_to_simulated_mapping', {}).get(exp_unit_id, None)
                            exp_path = abs_path
                            sim_path = abs_path.replace(str(exp_id), str(sim_id)) if sim_id is not None else abs_path
                            
                                
                            # exp_key = key
                            # #sim_key = kwargs.get('simulated_to_experimental_mapping', {}).get(key, {}).get('closest_exp_unit_id', key)
                            # sim_key = kwargs.get('experimental_to_simulated_mapping', {}).get(key, None)
                            # sim_path = f'{path}.{sim_key}' if path is not None else sim_key
                            # exp_path = f'{path}.{exp_key}' if path is not None else exp_key
                            # print()
                        else:
                            exp_id = None
                            sim_id = None
                            sim_path = abs_path
                            exp_path = abs_path
                    except: 
                        #exp_key = key
                        #sim_key = key
                        exp_id = None
                        sim_id = None
                        sim_path = abs_path
                        exp_path = abs_path
                    
                    # get the last item in sim and exp paths to get the keys
                    if sim_path is not None: 
                        try: sim_key = int(sim_path.split('.')[-1])
                        except: sim_key = sim_path.split('.')[-1]
                    if exp_path is not None: 
                        try: exp_key = int(exp_path.split('.')[-1])
                        except: exp_key = exp_path.split('.')[-1]
                    
                    return key, exp_key, sim_key, exp_id, sim_id, abs_path, exp_path, sim_path
                key, exp_key, sim_key, exp_id, sim_id, abs_path, exp_path, sim_path = handle_unit_mapping(path, key, kwargs)
                if exp_id is not None and sim_id is None: continue # dont score units if not mapped               
                
                # update fit dict
                if key not in fitness_dict:
                    fitness_dict[sim_key] = {}
                
                if isinstance(simulated_data[key], dict):
                    # recursive call
                    #fitness_dict[key] = calculate_fitness(simulated_data[key], experimental_data[key], fitness_dict[key], path=path, parent_key=key, **kwargs)
                    fitness_dict[key] = calculate_fitness(simulated_data[sim_key], experimental_data[exp_key], fitness_dict[sim_key], path=path, parent_key=sim_key, **kwargs)
                elif isinstance(simulated_data[key], list):
                    fitness_dict = handle_list_comparison(simulated_data, experimental_data, key, fitness_dict)
                elif isinstance(simulated_data[key], np.ndarray): 
                    continue
                    fitness_dict = handle_numpy_array_comparison(simulated_data, experimental_data, key, fitness_dict)
                elif isinstance(simulated_data[key], (float, int, np.int64, np.float64)):
                    fitness_dict = handle_float_or_int_comparison(simulated_data, experimental_data, key, fitness_dict)
                elif isinstance(simulated_data[key], str): 
                    continue
                    fitness_dict = handle_string_comparison(simulated_data, experimental_data, key, fitness_dict)
                elif isinstance(simulated_data[key], set):
                    continue # ignore sets for now
                    raise ValueError(f'Unknown data type for key {key}.')
                elif isinstance(simulated_data[key], tuple):
                    continue # ignore tuples for now
                    raise ValueError(f'Unknown data type for key {key}.')
                elif simulated_data[key] is None:
                    continue
                    raise ValueError(f'Key {key} is None.')
                else:
                    # other data types
                    try: 
                        print(type(simulated_data[key]))
                    except: 
                        print(f'Unknown data type for key {key}.')
                    raise ValueError(f'Unknown data type for key {key}.')
            else:
                #print(f'Key {key} not found in experimental data.')
                pass
            
        # # recurse through all keys in fitness_dict and delete empty dictionaries
        # for key in fitness_dict:
        #     if fitness_dict[key] == {}: del fitness_dict[key]
        
        # return fitness dictionary
        return fitness_dict
    
    def clean_fitness_dict(fitness_dict):
        # unscored keys
        unscored_keys = [
            # 'source',
            # 'sampling_rate',
            # 'unit_ids',
            # 'unit_locations',
            # 'warnings',
            # 'u_isi',
            # 'u_frs',
            # 'spike_times',
            # 'spiking_times_by_unit',
            # 'wf_metrics',
        ]
            
        
        # Create a list of keys to delete to avoid modifying the dictionary while iterating
        keys_to_delete = [key for key in fitness_dict if isinstance(fitness_dict[key], dict) and not fitness_dict[key]]

        # Delete empty dictionaries
        for key in keys_to_delete:
            if key in unscored_keys:
                del fitness_dict[key] # if key is unscored, delete it
            else:
                fitness_dict[key] = {'fit': 1000} # if key is scored, set it to 1000 if empty

        # Recurse through remaining dictionary values
        for key, value in list(fitness_dict.items()):  # Use list() to avoid RuntimeError due to dict size change
            if isinstance(value, dict):
                clean_fitness_dict(value)
                # If the nested dictionary becomes empty after recursion, delete it as well
                if not value:
                    if key in unscored_keys:
                        del fitness_dict[key] # if key is unscored, delete it
                    else:
                        fitness_dict[key] = {'fit': 1000}
                    
        # return cleaned fitness dictionary            
        return fitness_dict

    def get_avg_fitness(fitness_dict):
        """
        Recursively folds up the fitness values in a nested dictionary.
        It computes averages from the deepest level up and assigns an averaged 
        fitness value at each level.
        """
        
        #abs_path = ''

        def fold_up(fitness_dict, abs_path=''):
            # Check if current level contains 'fit' values
            if 'fit' in fitness_dict:
                return fitness_dict['fit'], 1  # Return fitness and count 1 comparison
            
            # Collect fitness values from child dictionaries
            fitness_values = []
            num_comparisons = 0  # Track number of fitness values aggregated

            for key, value in fitness_dict.items():
                if isinstance(value, dict):
                    prev_path = abs_path
                    abs_path = f'{abs_path}.{key}'
                    child_fitness, child_count = fold_up(value, abs_path)
                    abs_path = prev_path
                    fitness_values.append(child_fitness)
                    num_comparisons += child_count

            # If we collected fitness values, compute weighted average
            if fitness_values:
                avg_fitness = np.nanmean(fitness_values)
                fitness_dict['fit'] = avg_fitness  # Store computed fitness at this level
                return avg_fitness, num_comparisons

            #return 0, 0  # If no fitness values exist, return 0 (or another default)
            return 1000, 0 #fitness_dict  # If no fitness values exist, return 1000 (or another default)

        try:
            avg_fitness, _ = fold_up(fitness_dict)  # Start recursion
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error folding up fitness values: {e}")
            avg_fitness = 1000  # Default error value

        return avg_fitness, fitness_dict
    
    def get_sim_data_from_call_stack(kwargs):
        """
        Get the candidate and job path from the call stack.
        """
        #simData = kwargs.get('simData', None)
        #if simData is None: raise ValueError('No simulated data found in kwargs.')
        
        # get candidate and job path from call stack
        from .utils.extract_simulated_data import get_candidate_and_job_path_from_call_stack, retrieve_sim_data_from_call_stack, extract_data_of_interest_from_sim
        candidate_path, candidate_label = get_candidate_and_job_path_from_call_stack() #NOTE this part specifcally only works in batch processing, not in single candidate processing
        fitness_save_path = f'{candidate_path}_fitness.json'
        pkl_path = f'{candidate_path}_data.pkl'
        cfg_path = f'{candidate_path}_cfg.json'
        metrics_path = f'{candidate_path}_metrics.npy'
        
        # load simulated data
        sim.load(pkl_path)
        simData = sim.allSimData.todict().copy()
        popData = sim.net.allPops.copy()
        cellData = sim.net.allCells.copy()        
        simCfg = sim.cfg.todict().copy()
        netparams = sim.net.params.todict().copy()
        
        # update kwargs
        kwargs.update({
            'simData': simData,
            'cellData': cellData,
            'popData': popData,
            'simCfg': simCfg,
            'netParams': netparams,
            'simLabel': candidate_label,
            'data_file_path': pkl_path,
            'cfg_file_path': cfg_path,
            'fitness_save_path': fitness_save_path, #output
            'metrics_save_path': metrics_path, #output
        })
        
        # 
        print(f'data retrieved from {candidate_path}')
        return kwargs
    
    # main logic =================================================================
    try:
        # start timer
        time_start = time.time()
        kwargs['source'] = 'simulated'
        #kwargs['simData'] = simulated_data
        
        # check if batching - if so, kwargs need to be expanded to include all simulated data instead of just simData
        if kwargs.get('batching', False):
            kwargs = get_sim_data_from_call_stack(kwargs)
        
        # compute network metrics for simulated data
        simulated_metrics = compute_network_metrics(**kwargs)
        metrics_save_path = kwargs.get('metrics_save_path', None)
        if metrics_save_path is not None: # save metrics to .npy file
            np.save(metrics_save_path, simulated_metrics)
            print(f'Saved metrics to {metrics_save_path}')        
        
        # calculate fitness
        #fitness_dict = {}
        
        # TODO: MAP SIMULATED UNITS TO EXPERIMENTAL UNITS BY XY COORDINATES
        experimental_unit_locations = experimental_metrics['unit_locations']
        simulated_unit_locations = simulated_metrics['unit_locations']
        
        simulated_to_experimental_mapping = {}
        for sim_unit_id, sim_loc in simulated_unit_locations.items():
            closest_exp_unit_id = None
            closest_distance = float('inf')
            for exp_unit_id, exp_loc in experimental_unit_locations.items():
                distance = np.linalg.norm(np.array(sim_loc) - np.array(exp_loc))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_exp_unit_id = exp_unit_id
                if distance == 0:
                    closest_exp_unit_id = exp_unit_id
                    break
            if distance != 0: print(f'Warning: No exact match for unit {sim_unit_id}. Closest match is {closest_exp_unit_id} with distance {closest_distance}.')
            simulated_to_experimental_mapping[sim_unit_id] = {
                'closest_exp_unit_id': closest_exp_unit_id,
                'distance': closest_distance
            }
            
        # reorient
        experimental_to_simulated_mapping = {v['closest_exp_unit_id']: k for k, v in simulated_to_experimental_mapping.items()}
            
        kwargs['simulated_to_experimental_mapping'] = simulated_to_experimental_mapping
        kwargs['experimental_to_simulated_mapping'] = experimental_to_simulated_mapping
        
        fitness_dict = calculate_fitness(simulated_metrics, experimental_metrics, {}, **kwargs)
        fitness_dict = clean_fitness_dict(fitness_dict)
        
        # # add important keys back to fitness to avoid skewing results in data that dont have them
        # if 'spiking_data' not in fitness_dict:
        #     fitness_dict['spiking_data'] = {'fit':1000}
        # elif 'spiking_data' is in fitness_dict:
        #     if 'frs' not in fitness_dict['spiking_data']: fitness_dict['spiking_data']['frs'] = {'fit':1000}
        #     if 'i_frs' not in fitness_dict['spiking_data']: fitness_dict['spiking_data']['i_frs'] = {'fit':1000}
        #     if 'e_frs' not in fitness_dict['spiking_data']: fitness_dict['spiking_data']['e_frs'] = {'fit':1000}
        #     if 'isi' not in fitness_dict['spiking_data']: fitness_dict['spiking_data']['isi'] = {'fit':1000}
        #     if 'i_isi' not in fitness_dict['spiking_data']: fitness_dict['spiking_data']['i_isi'] = {'fit':1000}
        #     if 'e_isi' not in fitness_dict['spiking_data']: fitness_dict['spiking_data']['e_isi'] = {'fit':1000}
        
        
        
        
        avg_fitness, fitness_dict = get_avg_fitness(fitness_dict)
        fitness_save_path = kwargs.get('fitness_save_path', None)
        if fitness_save_path is not None: # save fitness to .json file
            with open(fitness_save_path, 'w') as f:
                json.dump(fitness_dict, f, indent=4)
            print(f'Saved fitness to {fitness_save_path}')
        elapsed_time = time.time() - time_start
        print(f'Elapsed time: {elapsed_time} seconds.')        
        return avg_fitness        
    except Exception as e:
        # handle errors
        traceback.print_exc()
        print(f'Error calculating fitness: {e}')
        avg_fitness = 1000
        return avg_fitness

''' Everything below this point is deprecated or not used in the current version of the code.'''

def fitnessFunc(simData=None, mode='optimizing', **kwargs):
    """
    Main logic of the calculate_fitness function.
    Ensures the function does not crash and always returns a fitness value.
    """

    def handle_optimizing_mode(simData, plot_sim, kwargs):
        
        """Handles the logic for 'optimizing' mode."""
        if simData is not None:
            from .utils.extract_simulated_data import get_candidate_and_job_path_from_call_stack, retrieve_sim_data_from_call_stack, extract_data_of_interest_from_sim
            kwargs['source'] = 'simulated'
            candidate_path = kwargs.get('candidate_path', None)
            if candidate_path is None:
                candidate_path, job_path = get_candidate_and_job_path_from_call_stack() #NOTE this part specifcally only works in batch processing, not in single candidate processing
                                                                                        # so, for testing fitness function, this part needs a workaround
            else: 
                candidate_path = kwargs['candidate_path']
            fitness_save_path = f'{candidate_path}_fitness.json'
            
            # Check if fitness results already exist
            skip_existing = kwargs.get('skip_existing', False)
            if skip_existing:
                existing_fitness = handle_existing_fitness(fitness_save_path)
                if existing_fitness is not None:
                    return existing_fitness, kwargs

            #from ....utils.extract_simulated_data import retrieve_sim_data_from_call_stack
            data_file_path = kwargs.get('data_file_path', None)
            if data_file_path is None:
                kwargs = retrieve_sim_data_from_call_stack(simData, **kwargs)
            else:
                # this is for debugging purposes
                data_file_path = data_file_path
                candidate_label = os.path.basename(candidate_path)
                cfg_file_path = f'{candidate_path}_cfg.json'
                extracted_data = extract_data_of_interest_from_sim(data_file_path, simData)
                kwargs.update({
                    'simData': extracted_data['simData'],
                    'cellData': extracted_data['cellData'],
                    'popData': extracted_data['popData'],
                    'simCfg': extracted_data['simCfg'],
                    'netParams': extracted_data['netParams'],
                    'simLabel': candidate_label,
                    'data_file_path': data_file_path,
                    'cfg_file_path': cfg_file_path,
                    'fitness_save_path': fitness_save_path, #output
                })
            error, kwargs = calculate_network_metrics(kwargs)
            if plot_sim:
                # data_file_path = kwargs.get('data_file_path', None)
                # sim_data_path = data_file_path
                # reference_data_path = kwargs.get('reference_data_path', None)
                # conv_params = kwargs.get('conv_params', None)
                # mega_params = kwargs.get('mega_params', None)
                # fitnessFuncArgs = kwargs.copy()
                # from RBS_network_models.sim_analysis import process_simulation
                # process_simulation(
                #     sim_data_path, 
                #     reference_data_path,
                #     DEBUG_MODE=False,
                #     conv_params = conv_params,
                #     mega_params = mega_params,
                #     fitnessFuncArgs = fitnessFuncArgs,
                #     )
                
                submit_plotting_job(candidate_path, kwargs)
            return error, kwargs
        return None, kwargs

    def handle_simulated_data_mode(simData, kwargs):
        """
        
        Handles the logic for 'simulated data' mode.
        
        """
        kwargs['source'] = 'simulated'
        #candidate_path = kwargs['simConfig']['filename']
        candidate_dir = kwargs['simConfig']['saveFolder']
        filename = kwargs['simConfig']['filename']
        fitness_save_path = os.path.join(candidate_dir, f'{filename}_fitness.json')
        kwargs.update({'fitness_save_path': fitness_save_path})
        assert simData is not None, 'Simulated data must be provided in "simulated data" mode.'
        kwargs.update({'simData': simData})
        
        #assert conv_params and mega_params in kwargs, 'Convolution parameters must be provided in "simulated data" mode.'
        assert 'conv_params' in kwargs, 'Convolution parameters must be provided in "simulated data" mode.'
        assert 'mega_params' in kwargs, 'Mega convolution parameters must be provided in "simulated data" mode.'
        
        error, kwargs = calculate_network_metrics(kwargs)
        return error, kwargs

    def handle_experimental_data_mode(
        #source='experimental',
        **kwargs):
        """
        
        Handles the logic for 'experimental data' mode.
        
        Important for initial calibration of the fitness targets against the experimental data prior 
        to the optimization process.
        
        """
        
        #assert source == 'experimental', 'Source must be "experimental" in "experimental data" mode.'
        #kwargs['source'] = 'experimental'
        # implemented = False
        # assert implemented, 'Experimental data source not yet implemented.'        
        #error, kwargs = calculate_network_metrics(kwargs)
        
        # aw 2025-01-26 17:46:29 - do nothing since we already have the network metrics?
        
        assert 'network_metrics' in kwargs, 'Network metrics must be provided in "experimental data" mode.'
        
        #source of the data
        kwargs['source'] = 'experimental'
        
        # define fitness save path
        sorter_dir = kwargs['network_metrics']['sorting_output']        
        fitness_dir = sorter_dir.replace('sorted', 'fitness_metrics').replace('sorter_output', '')        
        fitness_save_path = os.path.join(fitness_dir, 'calibrated_fitness.json')
        kwargs['fitness_save_path'] = fitness_save_path
        
        #break deals?
        kwargs['break_deals'] = False # TODO: need to rethink this anyway.
        
        # Max fitness
        kwargs['maxFitness'] = 1000        
        
        return None, kwargs

    def calculate_and_save_fitness(
        break_deals=False,
        #maxFitness=1000,
        fitness_save_path=None,
        source=None,
        #plot_sim = False,
        **kwargs
        ):
        
        """Calculates fitness and saves the results."""
        try:
            assert source is not None, 'Source must be provided.'
            average_fitness, fitnessResults = get_fitness(source, **kwargs)
            fitnessResults['average_fitness'] = average_fitness
            #fitnessResults['maxFitness'] = kwargs['maxFitness']
            #fitnessResults['maxFitness'] = maxFitness

            #break_deals = kwargs.get('break_deals', False)
            if break_deals:
                deal_broken = check_dealbreakers(fitnessResults, **kwargs)
                assert not deal_broken, 'Dealbreakers found in fitness results.'

            #save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
            assert fitness_save_path is not None, 'Fitness save path must be provided.'
            fitness_save_dir = os.path.dirname(fitness_save_path)
            if not os.path.exists(fitness_save_dir):
                os.makedirs(fitness_save_dir)
            save_fitness_results(fitness_save_path, fitnessResults)
            
            #assert average_fitness is an integer, 'Average fitness must be an integer.'
            try: assert isinstance(average_fitness, (int, float)), 'Average fitness must be an integer or float.'
            except:
                #print in all caps
                print('ERROR: AVERAGE FITNESS IS NOT AN INTEGER OR FLOAT')
                import sys
                sys.exit(1)
                
                average_fitness = 1000
                
                # print(f'Average fitness: {average_fitness}')
                # raise
            
            
            return average_fitness
        except Exception as e:
            maxFitness = kwargs.get('maxFitness', 1000)
            error_trace = str(e)
            fitnessResults = {
                #'average_fitness': kwargs['maxFitness'],
                'average_fitness': maxFitness,
                #'maxFitness': kwargs['maxFitness'],
                'maxFitness': maxFitness,
                'error': 'acceptable' if any(error in error_trace for error in []) else 'new',
                'error_trace': error_trace
            }
            print(f'Error calculating fitness: {e}')
            #save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
            save_fitness_results(fitness_save_path, fitnessResults)
            return 1000

    '''execute the main logic of the calculate_fitness function'''
    try:
        # Select mode and execute corresponding logic
        if mode == 'optimizing':
            plot_sim = kwargs.get('plot_sim', False)
            error, kwargs = handle_optimizing_mode(simData, plot_sim, kwargs)
        elif mode == 'simulated':
            error, kwargs = handle_simulated_data_mode(simData, kwargs)
        elif mode == 'experimental':
            error, kwargs = handle_experimental_data_mode(**kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        # Handle potential errors from network metrics calculation
        if error is not None:
            #return error
            raise ValueError(f'Error in network metrics: {error}')

        # Calculate fitness and return the result # TODO: this is broke rn, but will fix later
        average_fitness = calculate_and_save_fitness(**kwargs)
        return average_fitness
    except Exception as e:
        print(f'Error calculating fitness: {e}')
        fitnessResults = {
            'average_fitness': kwargs.get('maxFitness', 1000),
            'maxFitness': kwargs.get('maxFitness', 1000),
            'error': 'general',
            'error_trace': str(e)
        }
        save_fitness_results(kwargs.get('fitness_save_path', kwargs.get('fitness_save_path', 'unknown_path.json')), fitnessResults)
        #save_fitness_results(kwargs.get('fitness_save_path', 'unknown_path.json'), fitnessResults)
        return 1000        

def the_scoring_function(val, target_val, weight, maxFitness, min_val=None, max_val=None):
    """
    The function `the_scoring_function` calculates a fitness score for a given metric based on its proximity to a target value.
    The goal is to minimize this score, with lower values indicating better fitness. 
    
    The function employs an exponential penalty for deviations from the target value, where the sensitivity of this penalty 
    is controlled by the `weight` parameter. As the weight increases, the penalty for being far from the target decreases, approaching 
    a constant reward of 1 when the difference is small. 
    
    The function also checks if the input value falls within specified bounds (`min_val` and `max_val`); 
    if not specified, it defaults to allowing all values. If the input value is outside the specified bounds, it returns a 
    maximum fitness score (`maxFitness`), indicating poor fitness. 
    
    Overall, the function rewards values close to the target 
    more heavily while penalizing those further away, with the penalty capped at `maxFitness`.
    """
    if val is None:
        return maxFitness
    
    # Set default min and max values if not provided
    if min_val is None:
        min_val = float('-inf')  # Allow all values below positive infinity
    if max_val is None:
        max_val = float('inf')   # Allow all values above negative infinity

    # Calculate fitness score
    if min_val <= val <= max_val:
        #return min(np.exp(abs(target_val - val) / weight), maxFitness)
        return min(np.exp(abs(target_val - val) * weight), maxFitness)
    else:
        return maxFitness

def submit_plotting_job(candidate_path, kwargs):
    raise NotImplementedError("submit_plotting_job is not implemented yet.")

    # """
    # Submits the process_simulation step in handle_optimizing_mode as an MPI job.
    # """

    # # Paths and parameters
    # sim_data_path = kwargs.get('data_file_path', None)
    # reference_data_path = kwargs.get('reference_data_path', None)
    # conv_params = kwargs.get('conv_params', None)
    # mega_params = kwargs.get('mega_params', None)
    # fitnessFuncArgs = kwargs.copy()

    # # Define job folder and script path
    # job_folder = os.path.dirname(candidate_path)
    # job_name = os.path.basename(candidate_path) + "_fitness"
    # job_script = os.path.join(job_folder, job_name + ".sh")

    # # Define the command to run process_simulation
    # command = f"""
    # python -c "
    # from RBS_network_models.sim_analysis import process_simulation
    # process_simulation(
    #     sim_data_path='{sim_data_path}', 
    #     reference_data_path='{reference_data_path}',
    #     DEBUG_MODE=False,
    #     conv_params={conv_params},
    #     mega_params={mega_params},
    #     fitnessFuncArgs={fitnessFuncArgs}
    # )
    #     "
    #     """

    # # Create the job script for MPI submission
    # job_string = f"""#!/bin/bash
    # cd {job_folder}
    # {command}
    #     """

    # # Write the script to file
    # with open(job_script, "w") as f:
    #     f.write(job_string)

    # # Ensure the script is executable
    # os.chmod(job_script, 0o755)

    # # Submit the job using MPI direct
    # try:
    #     with open(f"{job_name}.run", "w") as outf, open(f"{job_name}.err", "w") as errf:
    #         subprocess.Popen(["/bin/bash", job_script], stdout=outf, stderr=errf, start_new_session=True)
    #     print(f"Submitted plotting job: {job_name}")
    # except Exception as e:
    #   print(f"Failed to submit job {job_name}: {e}")

def fit_firing_rates(data_source, mega_mode=False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False    
    
    #def fit_firing_rates(simulated=False, **kwargs):
    try:
        print('Calculating firing rate fitness...')
        MeanFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['MeanFireRate']
        
        # E_I_ratio = 5  # 1:5 ratio of E to I neurons
        # E_fr_target = MeanFireRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        # I_fr_target = MeanFireRate_target['target'] / (E_I_ratio + 1)
        
        # assumptions
        # max_E_assumption = MeanFireRate_target['max_E_assumption']
        # max_I_assumption = MeanFireRate_target['max_I_assumption']
        # min_E_assumption = MeanFireRate_target['min_E_assumption']
        # min_I_assumption = MeanFireRate_target['min_I_assumption']
        # max_E_assumption_inBurst = MeanFireRate_target['max_E_assumption_inBurst']
        # max_I_assumption_inBurst = MeanFireRate_target['max_I_assumption_inBurst']
        # min_E_assumption_inBurst = MeanFireRate_target['min_E_assumption_inBurst']
        # min_I_assumption_inBurst = MeanFireRate_target['min_I_assumption_inBurst']
        
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        
        # moduleate targets
        # typical case: 
        # target = MeanFireRate_target['target']
        # min_FR = MeanFireRate_target['min']
        # max_FR = MeanFireRate_target['max']
        # weight = MeanFireRate_target['weight']
        target = MeanFireRate_target.get('target', None)
        min_FR = MeanFireRate_target.get('min', None)
        max_FR = MeanFireRate_target.get('max', None)
        weight = MeanFireRate_target.get('weight', None)
        target_E = MeanFireRate_target.get('target_E', None)
        target_I = MeanFireRate_target.get('target_I', None)
        min_FR_E = MeanFireRate_target.get('min_E', None)
        min_FR_I = MeanFireRate_target.get('min_I', None)
        max_FR_E = MeanFireRate_target.get('max_E', None)
        max_FR_I = MeanFireRate_target.get('max_I', None)
        #excitatory_neurons = kwargs['network_metrics']['classification_data']['excitatory_neurons']
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        
        classified_targets = False
        if target_E is not None and target_I is not None:
            classified_targets = True
        
        # try:
        #     target_E = MeanFireRate_target['target_E']
        
        
        maxFitness = kwargs['maxFitness']
        
        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
       
        # choose bursting data
        if not mega_mode: 
            # typical case
            bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else:
            # if scoring with mega_bursting_data 
            mega_bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
            bursting_data_by_unit = mega_bursting_data_by_unit
            
            
        fitness_FRs = []
        val_FRs = []
        type_guesses = []
        classifications = []
        for unit, value in spiking_data_by_unit.items():
            try:
            
                unit_data = {
                    # 'spikes': spiking_data_by_unit[unit],
                    # 'bursts': bursting_data_by_unit[unit],
                    'spikes': spiking_data_by_unit.get(unit, {}),
                    'bursts': bursting_data_by_unit.get(unit, {}),
                }
                #val_FR = unit_data['spikes']['FireRate']
                val_FR = unit_data['spikes'].get('FireRate', None)
                val_FRs.append(val_FR)
                
                #''' if experimental '''
                if simulated is False: 
                    #target = MeanFireRate_target['target'] # experimental target
                    if not classified_targets:
                        fitness = the_scoring_function(val_FR, target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                    elif classified_targets:
                        if unit in excitatory_neurons:
                            E_fitness = the_scoring_function(val_FR, target_E, weight, maxFitness, min_val=min_FR_E, max_val=max_FR_E)
                            fitness = E_fitness
                        elif unit in inhibitory_neurons:
                            I_fitness = the_scoring_function(val_FR, target_I, weight, maxFitness, min_val=min_FR_I, max_val=max_FR_I)
                            fitness = I_fitness
                        else:
                            # unclassified
                            fitness = the_scoring_function(val_FR, target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                    # if fitness == maxFitness:
                    #     type_guesses.append('unclassified')
                #''' if simulated '''
                else:
                    E_gids = kwargs['network_metrics']['simulated_data']['E_Gids']
                    I_gids = kwargs['network_metrics']['simulated_data']['I_Gids']
                    if unit in E_gids:
                        # E_fitness = the_scoring_function(val_FR, E_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                        # fitness = E_fitness
                        E_fitness = the_scoring_function(val_FR, target_E, weight, maxFitness, min_val=min_FR_E, max_val=max_FR_E)
                        fitness = E_fitness
                    elif unit in I_gids:
                        # I_fitness = the_scoring_function(val_FR, I_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                        # fitness = I_fitness
                        I_fitness = the_scoring_function(val_FR, target_I, weight, maxFitness, min_val=min_FR_I, max_val=max_FR_I)
                        fitness = I_fitness
            except Exception as e:
                print(f'Error calculating firing rate fitness for unit {unit}.')
                print(e)
                fitness = 1000
            
            # #HACK
            # if np.isnan(val_FR):
            #     print(f'NaN found in firing rate for unit {unit}.')
            
            fitness_FRs.append(fitness)
        fitness_FR = np.mean(fitness_FRs)
    
        # Set to None if not defined
        E_fitness = E_fitness if 'E_fitness' in locals() else None
        I_fitness = I_fitness if 'I_fitness' in locals() else None
        
        # Create dictionary to store fitness values
        fitness_FR_dict = {
            'fit': fitness_FR,
            'value(s)': val_FRs,
            'fit_E': E_fitness,
            'fit_I': I_fitness,
            # 'target_E': E_fr_target,
            # 'target_I': I_fr_target,
            #
            'target_E': target_E,
            'target_I': target_I,
            'min_E': min_FR_E,
            'min_I': min_FR_I,
            'max_E': max_FR_E,
            'max_I': max_FR_I,
            #
            'target': target,
            'min': min_FR,
            'max': max_FR,
            'weight': weight,        
        }
        return fitness_FR_dict
    except Exception as e:
        print(f"Error in fit_firing_rates: {e}")
        return 1000

def fit_CoV_firing_rate(data_source, mega_mode = False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
            
    try:
        print('Calculating CoV firing rate fitness...')
        CoVFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['CoVFireRate']
        # E_I_ratio = 1.5 / 0.7  # ratio of E to I neurons
        # E_CoV_target = CoVFireRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        # I_CoV_target = CoVFireRate_target['target'] / (E_I_ratio + 1)
        # min_CoV = CoVFireRate_target['min']
        # max_CoV = CoVFireRate_target['max']
        # min_CoV = CoVFireRate_target.get('min', None)
        # max_CoV = CoVFireRate_target.get('max', None)
        weight = CoVFireRate_target['weight']
        maxFitness = kwargs['maxFitness']
        CoVFireRate_target_E = CoVFireRate_target.get('target_E', None)
        CoVFireRate_target_I = CoVFireRate_target.get('target_I', None)
        min_CoV = CoVFireRate_target.get('min', None)
        max_CoV = CoVFireRate_target.get('max', None)
        min_CoV_E = CoVFireRate_target.get('min_E', None)
        min_CoV_I = CoVFireRate_target.get('min_I', None)
        max_CoV_E = CoVFireRate_target.get('max_E', None)
        max_CoV_I = CoVFireRate_target.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)     
        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        
        fitness_CoVs = []
        val_CoVs = []
        for unit, value in spiking_data_by_unit.items():
            try: val_CoV = spiking_data_by_unit[unit]['CoV_fr']
            except: val_CoV = spiking_data_by_unit[unit]['fr_CoV']
            val_CoVs.append(val_CoV)
            if simulated is False:
                target = CoVFireRate_target['target']
                try: 
                    #fitness = the_scoring_function(val_CoV, target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
                    if unit in excitatory_neurons:
                        fitness = the_scoring_function(val_CoV, CoVFireRate_target_E, weight, maxFitness, min_val=min_CoV_E, max_val=max_CoV_E)
                    elif unit in inhibitory_neurons:
                        fitness = the_scoring_function(val_CoV, CoVFireRate_target_I, weight, maxFitness, min_val=min_CoV_I, max_val=max_CoV_I)
                        
                    # if fitness == maxFitness:
                    #     print(f"CoVFR fitness is maxFitness for unit {unit}.")
                except: 
                    fitness = 1000
            else:
                E_gids = kwargs['network_metrics']['simulated_data']['E_Gids']
                I_gids = kwargs['network_metrics']['simulated_data']['I_Gids']
                if unit in E_gids:
                    #fitness = the_scoring_function(val_CoV, E_CoV_target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
                    fitness = the_scoring_function(val_CoV, CoVFireRate_target_E, weight, maxFitness, min_val=min_CoV_E, max_val=max_CoV_E)
                elif unit in I_gids:
                    #fitness = the_scoring_function(val_CoV, I_CoV_target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
                    fitness = the_scoring_function(val_CoV, CoVFireRate_target_I, weight, maxFitness, min_val=min_CoV_I, max_val=max_CoV_I)
            
            fitness_CoVs.append(fitness)
        
        fitness_CoV = np.mean(fitness_CoVs)
        fitness_CoV_dict = {
            'fit': fitness_CoV,
            'value(s)': val_CoVs,
            # 'target_E': E_CoV_target,
            # 'target_I': I_CoV_target,
            'target_E': CoVFireRate_target_E,
            'target_I': CoVFireRate_target_I,
            # 'min': min_CoV,
            # 'max': max_CoV,
            'min_E': min_CoV_E,
            'min_I': min_CoV_I,
            'max_E': max_CoV_E,
            'max_I': max_CoV_I,
            'min': min_CoV,
            'max': max_CoV,
            'weight': weight,
        }
        return fitness_CoV_dict
    except Exception as e:
        print(f"Error in fit_CoV_firing_rate: {e}")
        return 1000

def fit_ISI(data_source, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating ISI fitness...')

        MeanISI_targets = kwargs['targets']['spiking_data']['spiking_summary_data']['MeanISI']
        #E_I_ratio_mean_ISI = 5.0 / 1.0  # Biologically founded E to I ratio for Mean ISI

        # Calculate targets for E and I populations
        #E_meanISI_target = MeanISI_target['target'] * (E_I_ratio_mean_ISI / (E_I_ratio_mean_ISI + 1))
        #I_meanISI_target = MeanISI_target['target'] / (E_I_ratio_mean_ISI + 1)

        # ISI bounds and weights
        min_ISI = MeanISI_targets['min']
        max_ISI = MeanISI_targets['max']
        weight = MeanISI_targets['weight']
        maxFitness = kwargs['maxFitness']
        MeanISI_target = MeanISI_targets.get('target', None)
        
        MeanISI_target_E = MeanISI_targets.get('target_E', None)
        MeanISI_target_I = MeanISI_targets.get('target_I', None)
        min_ISI_E = MeanISI_targets.get('min_E', None)
        min_ISI_I = MeanISI_targets.get('min_I', None)
        max_ISI_E = MeanISI_targets.get('max_E', None)
        max_ISI_I = MeanISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        

        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_ISIs = []
        val_ISIs = []

        for unit, value in spiking_data_by_unit.items():
            try:
                val_ISI = spiking_data_by_unit[unit]['meanISI']
                val_ISIs.append(val_ISI)

                if simulated:
                    # Use population-specific targets for simulated data
                    if unit in E_Gids:
                        #fitness = the_scoring_function(val_ISI, E_meanISI_target, weight, maxFitness, 
                                                    #    min_val=min_ISI, max_val=max_ISI)
                        fitness = the_scoring_function(val_ISI, MeanISI_target_E, weight, maxFitness, 
                                                       min_val=min_ISI_E, max_val=max_ISI_E)
                    elif unit in I_Gids:
                        #fitness = the_scoring_function(val_ISI, I_meanISI_target, weight, maxFitness, 
                                                       #min_val=min_ISI, max_val=max_ISI)
                        fitness = the_scoring_function(val_ISI, MeanISI_target_I, weight, maxFitness,
                                                         min_val=min_ISI_I, max_val=max_ISI_I)
                    else:
                        # Handle unclassified units
                        fitness = the_scoring_function(val_ISI, MeanISI_target, weight, maxFitness, 
                                                       min_val=min_ISI, max_val=max_ISI)
                else:
                    # # Use undifferentiated target for experimental data
                    # fitness = the_scoring_function(val_ISI, MeanISI_target['target'], weight, maxFitness, 
                    #                                min_val=min_ISI, max_val=max_ISI)
                    try:
                        if unit in excitatory_neurons:
                            fitness = the_scoring_function(val_ISI, MeanISI_target_E, weight, maxFitness, 
                                                           min_val=min_ISI_E, max_val=max_ISI_E)
                        elif unit in inhibitory_neurons:
                            fitness = the_scoring_function(val_ISI, MeanISI_target_I, weight, maxFitness, 
                                                           min_val=min_ISI_I, max_val=max_ISI_I)
                        else:
                            # Handle unclassified units
                            fitness = the_scoring_function(val_ISI, MeanISI_target, weight, maxFitness, 
                                                           min_val=min_ISI, max_val=max_ISI)
                    except:
                        # fitness = the_scoring_function(val_ISI, MeanISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_ISI, max_val=max_ISI)
                        fitness = maxFitness
                fitness_ISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_ISI = {
            'fit': np.mean(fitness_ISIs) if fitness_ISIs else None,
            'value(s)': val_ISIs,
            # 'target_E': E_meanISI_target,
            # 'target_I': I_meanISI_target,
            'target': MeanISI_target,
            'min': min_ISI,
            'max': max_ISI,
            
            'target_E': MeanISI_target_E,
            'target_I': MeanISI_target_I,
            'min_E': min_ISI_E,
            'min_I': min_ISI_I,
            'max_E': max_ISI_E,
            'max_I': max_ISI_I,
            
            'weight': weight,
        }

        return fitness_ISI

    except Exception as e:
        print(f"Error in fit_ISI: {e}")
        return 1000

def fit_CoV_ISI(data_source, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
        
    try:
        print('Calculating CoV ISI fitness...')

        CoV_ISI_targets = kwargs['targets']['spiking_data']['spiking_summary_data']['CoV_ISI']
        # E_I_ratio_CoV_ISI = 1.5 / 0.7  # Reflects variability differences

        # # Calculate targets for E and I populations
        # E_CoV_ISI_target = CoV_ISI_target['target'] * (E_I_ratio_CoV_ISI / (E_I_ratio_CoV_ISI + 1))
        # I_CoV_ISI_target = CoV_ISI_target['target'] / (E_I_ratio_CoV_ISI + 1)

        # CoV ISI bounds and weights
        # min_CoV_ISI = CoV_ISI_target['min']
        # max_CoV_ISI = CoV_ISI_target['max']
        CoV_ISI_target = CoV_ISI_targets.get('target', None)
        min_CoV_ISI = CoV_ISI_targets.get('min', None)
        max_CoV_ISI = CoV_ISI_targets.get('max', None)
        CoV_ISI_target_E = CoV_ISI_targets.get('target_E', None)
        CoV_ISI_target_I = CoV_ISI_targets.get('target_I', None)
        min_CoV_ISI_E = CoV_ISI_targets.get('min_E', None)
        min_CoV_ISI_I = CoV_ISI_targets.get('min_I', None)
        max_CoV_ISI_E = CoV_ISI_targets.get('max_E', None)
        max_CoV_ISI_I = CoV_ISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        
        weight = CoV_ISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CoV_ISIs = []
        val_CoV_ISIs = []

        for unit, value in spiking_data_by_unit.items():
            try:
                try: 
                    val_CoV_ISI = spiking_data_by_unit[unit]['CoV_ISI']
                except: 
                    val_CoV_ISI = spiking_data_by_unit[unit]['isi_CoV']
                val_CoV_ISIs.append(val_CoV_ISI)

                if simulated:
                    if unit in E_Gids:
                        # fitness = the_scoring_function(val_CoV_ISI, E_CoV_ISI_target, weight, maxFitness, 
                        #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                        fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_E, weight, maxFitness, 
                                                       min_val=min_CoV_ISI_E, max_val=max_CoV_ISI_E)
                    elif unit in I_Gids:
                        # fitness = the_scoring_function(val_CoV_ISI, I_CoV_ISI_target, weight, maxFitness, 
                        #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                        fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_I, weight, maxFitness, 
                                                       min_val=min_CoV_ISI_I, max_val=max_CoV_ISI_I)
                    else:
                        # Handle unclassified units
                        fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target, weight, maxFitness, 
                                                       min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                else:
                    # fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target['target'], weight, maxFitness, 
                    #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                    try:
                        if unit in excitatory_neurons:
                            fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_E, weight, maxFitness, 
                                                           min_val=min_CoV_ISI_E, max_val=max_CoV_ISI_E)
                        elif unit in inhibitory_neurons:
                            fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target_I, weight, maxFitness, 
                                                           min_val=min_CoV_ISI_I, max_val=max_CoV_ISI_I)
                        else:
                            # Handle unclassified units
                            fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target, weight, maxFitness, 
                                                           min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                    except:
                        # fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                        fitness = maxFitness
                    

                fitness_CoV_ISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_CoV_ISI = {
            'fit': np.mean(fitness_CoV_ISIs) if fitness_CoV_ISIs else None,
            'value(s)': val_CoV_ISIs,
            # 'target_E': E_CoV_ISI_target,
            # 'target_I': I_CoV_ISI_target,
            'target': CoV_ISI_target,
            'min': min_CoV_ISI,
            'max': max_CoV_ISI,
            'target_E': CoV_ISI_target_E,
            'target_I': CoV_ISI_target_I,
            'min_E': min_CoV_ISI_E,
            'min_I': min_CoV_ISI_I,
            'max_E': max_CoV_ISI_E,
            'max_I': max_CoV_ISI_I,
            
            'weight': weight,
        }

        return fitness_CoV_ISI

    except Exception as e:
        print(f"Error in fit_CoV_ISI: {e}")
        return 1000

def fit_mean_Burst_Rate(data_source, mega_mode = False, **kwargs):
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating mean burst rate fitness...')

        try:
            if not mega_mode: MeanBurstRate_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanBurstRate'] #TODO rerun analysis to get the correct target values
            else: MeanBurstRate_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['MeanBurstRate']
        except Exception as e: #if the program fails here print a big yellow warning to fix the knon issue.
            #activate_print()
            print(f"\033[93mWarning: {e}\033[0m")
            print(f"\033[93mWarning: The target values for mean_Burst_Rate are not available. Please rerun the analysis to get the correct target values.\033[0m")
            print(f"\033[93mWarning: Setting the fitness score to 1000.\033[0m")
            #suppress_print()
            return 1000
        #E_I_ratio = 2.0 / 1.0

        #E_meanBurstRate_target = MeanBurstRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        #I_meanBurstRate_target = MeanBurstRate_target['target'] / (E_I_ratio + 1)

        MeanBurstRate_target = MeanBurstRate_targets.get('target', None)
        # min_meanBurstRate = MeanBurstRate_targets['min']
        # max_meanBurstRate = MeanBurstRate_targets['max']
        # weight = MeanBurstRate_targets['weight']
        # maxFitness = kwargs['maxFitness']
        min_meanBurstRate = MeanBurstRate_targets.get('min', None)
        max_meanBurstRate = MeanBurstRate_targets.get('max', None)
        weight = MeanBurstRate_targets.get('weight', None)
        maxFitness = kwargs['maxFitness']

        if not mega_mode: mean_Burst_Rate = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        else: mean_Burst_Rate = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        
        mean_Burst_Rate_fitness = the_scoring_function(mean_Burst_Rate, MeanBurstRate_targets['target'], weight, maxFitness,)
        
        fitness_meanBurstRate = {
            'fit': mean_Burst_Rate_fitness,
            'value(s)': mean_Burst_Rate,
            'target' : MeanBurstRate_targets['target'],
            'min': min_meanBurstRate,
            'max': max_meanBurstRate,
            'weight': weight,
        }
        
        return fitness_meanBurstRate
    except Exception as e:
        print(f"Error in fit_mean_Burst_Rate: {e}")
        return 1000
    
def fit_baseline(data_source, mega_mode = False, **kwargs):
    try:
        print('Calculating baseline fitness...')

        if not mega_mode: baseline_target = kwargs['targets']['bursting_data']['bursting_summary_data']['baseline']
        else: baseline_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['baseline']
        
        target = baseline_target['target']
        min_baseline = baseline_target['min']
        max_baseline = baseline_target['max']
        weight = baseline_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: baseline = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['baseline']
        else: baseline = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['baseline']
        val_baseline = baseline

        fitness_baseline = the_scoring_function(val_baseline, target, weight, maxFitness, 
                                                min_val=min_baseline, max_val=max_baseline)

        fitness_baseline_dict = {
            'fit': fitness_baseline,
            'value': val_baseline,
            'target': target,
            'min': min_baseline,
            'max': max_baseline,
            'weight': weight,
        }

        return fitness_baseline_dict

    except Exception as e:
        print(f"Error in fit_baseline: {e}")
        return 1000

def fit_WithinBurstISI(data_source, mega_mode = False, **kwargs):
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating WithinBurstISI fitness...')

        if not mega_mode: WithinBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
        else: WithinBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
        #E_I_ratio = 2.0 / 1.0

        # E_WithinBurstISI_target = WithinBurstISI_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        # I_WithinBurstISI_target = WithinBurstISI_target['target'] / (E_I_ratio + 1)
        WithinBurstISI_target = WithinBurstISI_targets.get('target', None)
        # min_WithinBurstISI = WithinBurstISI_targets['min']
        # max_WithinBurstISI = WithinBurstISI_targets['max']
        
        min_WithinBurstISI = WithinBurstISI_targets.get('min', None)
        max_WithinBurstISI = WithinBurstISI_targets.get('max', None)
        
        E_WithinBurstISI_target = WithinBurstISI_targets.get('target_E', None)
        I_WithinBurstISI_target = WithinBurstISI_targets.get('target_I', None)
        E_min_WithinBurstISI = WithinBurstISI_targets.get('min_E', None)
        E_max_WithinBurstISI = WithinBurstISI_targets.get('max_E', None)
        I_min_WithinBurstISI = WithinBurstISI_targets.get('min_I', None)
        I_max_WithinBurstISI = WithinBurstISI_targets.get('max_I', None)
        
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = WithinBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_WithinBurstISIs = []
        val_WithinBurstISIs = []

        scored_by_individual = False
        if scored_by_individual:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_WithinBurstISI = bursting_data_by_unit[unit]['mean_isi_within']
                    val_WithinBurstISIs.append(val_WithinBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_WithinBurstISI, E_WithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                            fitness = the_scoring_function(val_WithinBurstISI, E_WithinBurstISI_target, weight, maxFitness, 
                                                        min_val=E_min_WithinBurstISI, max_val=E_max_WithinBurstISI)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_WithinBurstISI, I_WithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                            fitness = the_scoring_function(val_WithinBurstISI, I_WithinBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_WithinBurstISI, max_val=I_max_WithinBurstISI)
                        else:
                            # Handle unclassified units
                            fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target, weight, maxFitness, 
                                                        min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                    else:
                        # fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                        try:
                            if unit in excitatory_neurons:
                                fitness = the_scoring_function(val_WithinBurstISI, E_WithinBurstISI_target, weight, maxFitness, 
                                                            min_val=E_min_WithinBurstISI, max_val=E_max_WithinBurstISI)
                            elif unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_WithinBurstISI, I_WithinBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_WithinBurstISI, max_val=I_max_WithinBurstISI)
                            else:
                                # Handle unclassified units
                                fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target, weight, maxFitness, 
                                                            min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                        except:
                            # fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                            fitness = maxFitness

                    fitness_WithinBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val_WithinBurstISIs = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
            else: val_WithinBurstISIs = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
            
            try:
                fitness = the_scoring_function(val_WithinBurstISIs, WithinBurstISI_target, weight, maxFitness, min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
            except:
                fitness = maxFitness
            fitness_WithinBurstISIs.append(fitness)

        fitness_WithinBurstISI = {
            'fit': np.mean(fitness_WithinBurstISIs) if fitness_WithinBurstISIs else None,
            'value(s)': val_WithinBurstISIs,
            # 'target_E': E_WithinBurstISI_target,
            # 'target_I': I_WithinBurstISI_target,
            'target': WithinBurstISI_target,
            'min': min_WithinBurstISI,
            'max': max_WithinBurstISI,
            'target_E': E_WithinBurstISI_target,
            'target_I': I_WithinBurstISI_target,
            'min_E': E_min_WithinBurstISI,
            'min_I': I_min_WithinBurstISI,
            'max_E': E_max_WithinBurstISI,
            'max_I': I_max_WithinBurstISI,
            'weight': weight,
        }

        return fitness_WithinBurstISI

    except Exception as e:
        print(f"Error in fit_WithinBurstISI: {e}")
        return 1000

def fit_CovWithinBurstISI(data_source, mega_mode = False, **kwargs):
    # source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
        
    try:
        print('Calculating CovWithinBurstISI fitness...')

        if not mega_mode: CovWithinBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['CovWithinBurstISI']
        else: CovWithinBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['CovWithinBurstISI']
        # E_I_ratio_burst = 2.5 / 1.0

        # E_CovWithinBurstISI_target = CovWithinBurstISI_target['target'] * (E_I_ratio_burst / (E_I_ratio_burst + 1))
        # I_CovWithinBurstISI_target = CovWithinBurstISI_target['target'] / (E_I_ratio_burst + 1)

        CovWithinBurstISI_target = CovWithinBurstISI_targets.get('target', None)
        # min_CovWithinBurstISI = CovWithinBurstISI_targets['min']
        # max_CovWithinBurstISI = CovWithinBurstISI_targets['max']
        min_CovWithinBurstISI = CovWithinBurstISI_targets.get('min', None)
        max_CovWithinBurstISI = CovWithinBurstISI_targets.get('max', None)
        E_CovWithinBurstISI_target = CovWithinBurstISI_targets.get('target_E', None)
        I_CovWithinBurstISI_target = CovWithinBurstISI_targets.get('target_I', None)
        min_CovWithinBurstISI_I = CovWithinBurstISI_targets.get('min_I', None)
        min_CovWithinBurstISI_E = CovWithinBurstISI_targets.get('min_E', None)
        max_CovWithinBurstISI_I = CovWithinBurstISI_targets.get('max_I', None)
        max_CovWithinBurstISI_E = CovWithinBurstISI_targets.get('max_E', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = CovWithinBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovWithinBurstISIs = []
        val_CovWithinBurstISIs = []

        scored_by_individual_units = False
        if scored_by_individual_units:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_CovWithinBurstISI = bursting_data_by_unit[unit]['cov_isi_within']
                    val_CovWithinBurstISIs.append(val_CovWithinBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_CovWithinBurstISI, E_CovWithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                            fitness = the_scoring_function(val_CovWithinBurstISI, E_CovWithinBurstISI_target, weight, maxFitness, 
                                                        min_val=min_CovWithinBurstISI_E, max_val=max_CovWithinBurstISI_E)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_CovWithinBurstISI, I_CovWithinBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                            fitness = the_scoring_function(val_CovWithinBurstISI, I_CovWithinBurstISI_target, weight, maxFitness,
                                                            min_val=min_CovWithinBurstISI_I, max_val=max_CovWithinBurstISI_I)
                    else:
                        # fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                        try:
                            if unit in excitatory_neurons:
                                fitness = the_scoring_function(val_CovWithinBurstISI, E_CovWithinBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovWithinBurstISI_E, max_val=max_CovWithinBurstISI_E)
                            elif unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_CovWithinBurstISI, I_CovWithinBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovWithinBurstISI_I, max_val=max_CovWithinBurstISI_I)
                            else:
                                # Handle unclassified units
                                fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target['target'], weight, maxFitness, 
                                                            min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                        except:
                            # fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                            fitness = maxFitness

                    fitness_CovWithinBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val_CovWithinBurstISI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['CoVWithinBurstISI']
            else: val_CovWithinBurstISI = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['CoVWithinBurstISI']
            val_CovWithinBurstISIs.append(val_CovWithinBurstISI)
            try:
                fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target, weight, maxFitness, 
                                            min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
            except:
                fitness = maxFitness
            fitness_CovWithinBurstISIs.append(fitness)

        fitness_CovWithinBurstISI = {
            'fit': np.mean(fitness_CovWithinBurstISIs) if fitness_CovWithinBurstISIs else None,
            'value(s)': val_CovWithinBurstISIs,
            # 'target_E': E_CovWithinBurstISI_target,
            # 'target_I': I_CovWithinBurstISI_target,
            'target': CovWithinBurstISI_target,
            'min': min_CovWithinBurstISI,
            'max': max_CovWithinBurstISI,
            'target_E': E_CovWithinBurstISI_target,
            'target_I': I_CovWithinBurstISI_target,
            'min_E': min_CovWithinBurstISI_E,
            'min_I': min_CovWithinBurstISI_I,
            'max_E': max_CovWithinBurstISI_E,
            'max_I': max_CovWithinBurstISI_I,
            
            'weight': weight,
        }

        return fitness_CovWithinBurstISI

    except Exception as e:
        print(f"Error in fit_CovWithinBurstISI: {e}")
        return 1000

def fit_OutsideBurstISI(data_source, mega_mode = False, **kwargs):
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
        
    try:
        print('Calculating OutsideBurstISI fitness...')

        if not mega_mode: OutsideBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
        else: OutsideBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
        # E_I_ratio_outside_mean_ISI = 5.0 / 1.0

        # E_OutsideBurstISI_target = OutsideBurstISI_target['target'] * (E_I_ratio_outside_mean_ISI / (E_I_ratio_outside_mean_ISI + 1))
        # I_OutsideBurstISI_target = OutsideBurstISI_target['target'] / (E_I_ratio_outside_mean_ISI + 1)

        OutsideBurstISI_target = OutsideBurstISI_targets.get('target', None)                    
        # min_OutsideBurstISI = OutsideBurstISI_targets['min']
        # max_OutsideBurstISI = OutsideBurstISI_targets['max']
        min_OutsideBurstISI = OutsideBurstISI_targets.get('min', None)
        max_OutsideBurstISI = OutsideBurstISI_targets.get('max', None)
        E_OutsideBurstISI_target = OutsideBurstISI_targets.get('target_E', None)
        I_OutsideBurstISI_target = OutsideBurstISI_targets.get('target_I', None)
        E_min_OutsideBurstISI = OutsideBurstISI_targets.get('min_E', None)
        E_max_OutsideBurstISI = OutsideBurstISI_targets.get('max_E', None)
        I_min_OutsideBurstISI = OutsideBurstISI_targets.get('min_I', None)
        I_max_OutsideBurstISI = OutsideBurstISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = OutsideBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_OutsideBurstISIs = []
        val_OutsideBurstISIs = []
        
        score_by_inidividual = False
        if score_by_inidividual:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_OutsideBurstISI = bursting_data_by_unit[unit]['mean_isi_outside']
                    val_OutsideBurstISIs.append(val_OutsideBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_OutsideBurstISI, E_OutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                            fitness = the_scoring_function(val_OutsideBurstISI, E_OutsideBurstISI_target, weight, maxFitness, 
                                                        min_val=E_min_OutsideBurstISI, max_val=E_max_OutsideBurstISI)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_OutsideBurstISI, I_OutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                            fitness = the_scoring_function(val_OutsideBurstISI, I_OutsideBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_OutsideBurstISI, max_val=I_max_OutsideBurstISI)
                    else:
                        # fitness = the_scoring_function(val_OutsideBurstISI, OutsideBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                        try:
                            if unit in excitatory_neurons:
                                fitness = the_scoring_function(val_OutsideBurstISI, E_OutsideBurstISI_target, weight, maxFitness, 
                                                            min_val=E_min_OutsideBurstISI, max_val=E_max_OutsideBurstISI)
                            elif unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_OutsideBurstISI, I_OutsideBurstISI_target, weight, maxFitness,
                                                            min_val=I_min_OutsideBurstISI, max_val=I_max_OutsideBurstISI)
                            else:
                                # Handle unclassified units
                                fitness = the_scoring_function(val_OutsideBurstISI, OutsideBurstISI_target['target'], weight, maxFitness, 
                                                            min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                        except:
                            # fitness = the_scoring_function(val_OutsideBurstISI, OutsideBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                            fitness = maxFitness

                    
                    if fitness == maxFitness:
                        print(f"Warning: Unit {unit} has a fitness score of {maxFitness}. This may indicate an issue with the data or the scoring function. {val_OutsideBurstISI}")
                        print()
                    fitness_OutsideBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val = network_metrics['bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
            else: val = network_metrics['mega_bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
            val_OutsideBurstISIs.append(val)
            try:
                if simulated:
                    fitness = the_scoring_function(val, OutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                else:
                    fitness = the_scoring_function(val, OutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
            except:
                fitness = maxFitness
            fitness_OutsideBurstISIs.append(fitness)
            
        fitness_OutsideBurstISI = {
            'fit': np.mean(fitness_OutsideBurstISIs) if fitness_OutsideBurstISIs else None,
            'value(s)': val_OutsideBurstISIs,
            'fit_scores': fitness_OutsideBurstISIs,
            # 'target_E': E_OutsideBurstISI_target,
            # 'target_I': I_OutsideBurstISI_target,
            'target': OutsideBurstISI_target,
            'min': min_OutsideBurstISI,
            'max': max_OutsideBurstISI,
            'target_E': E_OutsideBurstISI_target,
            'target_I': I_OutsideBurstISI_target,
            'min_E': E_min_OutsideBurstISI,
            'min_I': I_min_OutsideBurstISI,
            'max_E': E_max_OutsideBurstISI,
            'max_I': I_max_OutsideBurstISI,
            
            'weight': weight,
        }

        return fitness_OutsideBurstISI

    except Exception as e:
        print(f"Error in fit_OutsideBurstISI: {e}")
        return 1000

def fit_CovOutsideBurstISI(data_source, mega_mode = False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    try:
        print('Calculating CovOutsideBurstISI fitness...')

        if not mega_mode: CovOutsideBurstISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
        else: CovOutsideBurstISI_targets = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
        # E_I_ratio_outside_CoV_ISI = 1.5 / 0.8

        # E_CovOutsideBurstISI_target = CovOutsideBurstISI_target['target'] * (E_I_ratio_outside_CoV_ISI / (E_I_ratio_outside_CoV_ISI + 1))
        # I_CovOutsideBurstISI_target = CovOutsideBurstISI_target['target'] / (E_I_ratio_outside_CoV_ISI + 1)

        # min_CovOutsideBurstISI = CovOutsideBurstISI_target['min']
        # max_CovOutsideBurstISI = CovOutsideBurstISI_target['max']
        CovOutsideBurstISI_min = CovOutsideBurstISI_targets.get('min', None)
        CovOutsideBurstISI_max = CovOutsideBurstISI_targets.get('max', None)
        CovOutsideBurstISI_target = CovOutsideBurstISI_targets.get('target', None)
        E_CovOutsideBurstISI_target = CovOutsideBurstISI_targets.get('target_E', None)
        I_CovOutsideBurstISI_target = CovOutsideBurstISI_targets.get('target_I', None)
        min_CovOutsideBurstISI_E = CovOutsideBurstISI_targets.get('min_E', None)
        min_CovOutsideBurstISI_I = CovOutsideBurstISI_targets.get('min_I', None)
        max_CovOutsideBurstISI_E = CovOutsideBurstISI_targets.get('max_E', None)
        max_CovOutsideBurstISI_I = CovOutsideBurstISI_targets.get('max_I', None)
        network_metrics = kwargs['network_metrics']
        classification_data = network_metrics.get('classification_data', None)
        inhibitory_neurons = classification_data.get('inhibitory_neurons', None)
        excitatory_neurons = classification_data.get('excitatory_neurons', None)
        
        weight = CovOutsideBurstISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        else: bursting_data_by_unit = kwargs['network_metrics']['mega_bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovOutsideBurstISIs = []
        val_CovOutsideBurstISIs = []
        
        scored_by_individual = False
        if scored_by_individual:
            for unit, value in bursting_data_by_unit.items():
                try:
                    val_CovOutsideBurstISI = bursting_data_by_unit[unit]['cov_isi_outside']
                    val_CovOutsideBurstISIs.append(val_CovOutsideBurstISI)

                    if simulated:
                        if unit in E_Gids:
                            # fitness = the_scoring_function(val_CovOutsideBurstISI, E_CovOutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                            fitness = the_scoring_function(val_CovOutsideBurstISI, E_CovOutsideBurstISI_target, weight, maxFitness, 
                                                        min_val=min_CovOutsideBurstISI_E, max_val=max_CovOutsideBurstISI_E)
                        elif unit in I_Gids:
                            # fitness = the_scoring_function(val_CovOutsideBurstISI, I_CovOutsideBurstISI_target, weight, maxFitness, 
                            #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                            fitness = the_scoring_function(val_CovOutsideBurstISI, I_CovOutsideBurstISI_target, weight, maxFitness,
                                                        min_val=min_CovOutsideBurstISI_I, max_val=max_CovOutsideBurstISI_I)
                    else:
                        # fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target['target'], weight, maxFitness, 
                        #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                        try:
                            if unit in inhibitory_neurons:
                                fitness = the_scoring_function(val_CovOutsideBurstISI, I_CovOutsideBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovOutsideBurstISI_I, max_val=max_CovOutsideBurstISI_I)
                            elif unit in excitatory_neurons:
                                fitness = the_scoring_function(val_CovOutsideBurstISI, E_CovOutsideBurstISI_target, weight, maxFitness, 
                                                            min_val=min_CovOutsideBurstISI_E, max_val=max_CovOutsideBurstISI_E)
                            else:
                                # # Handle unclassified units
                                # fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target['target'], weight, maxFitness, 
                                #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                                fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target, weight, maxFitness,
                                                                min_val=CovOutsideBurstISI_min, max_val=CovOutsideBurstISI_max)
                        except:
                            # fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target['target'], weight, maxFitness, 
                            #                                min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                            fitness = maxFitness

                    fitness_CovOutsideBurstISIs.append(fitness)
                except Exception as unit_error:
                    print(f"Error processing unit {unit}: {unit_error}")
                    continue  # Skip to the next unit if there's an issue
        else:
            if not mega_mode: val = network_metrics['bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
            else: val = network_metrics['mega_bursting_data']['bursting_summary_data']['CoVOutsideBurstISI']
            val_CovOutsideBurstISIs.append(val)
            try:
                if simulated:
                    fitness = the_scoring_function(val, CovOutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=CovOutsideBurstISI_min, max_val=CovOutsideBurstISI_max)
                else:
                    fitness = the_scoring_function(val, CovOutsideBurstISI_target, weight, maxFitness, 
                                                    min_val=CovOutsideBurstISI_min, max_val=CovOutsideBurstISI_max)
            except:
                fitness = maxFitness
            fitness_CovOutsideBurstISIs.append(fitness)
            

        fitness_CovOutsideBurstISI = {
            'fit': np.mean(fitness_CovOutsideBurstISIs) if fitness_CovOutsideBurstISIs else None,
            'value(s)': val_CovOutsideBurstISIs,
            # 'target_E': E_CovOutsideBurstISI_target,
            # 'target_I': I_CovOutsideBurstISI_target,
            # 'min': min_CovOutsideBurstISI,
            # 'max': max_CovOutsideBurstISI,
            'target': CovOutsideBurstISI_target,
            'min': CovOutsideBurstISI_min,
            'max': CovOutsideBurstISI_max,
            'target_E': E_CovOutsideBurstISI_target,
            'target_I': I_CovOutsideBurstISI_target,
            'min_E': min_CovOutsideBurstISI_E,
            'min_I': min_CovOutsideBurstISI_I,
            'max_E': max_CovOutsideBurstISI_E,
            'max_I': max_CovOutsideBurstISI_I,
            
            'weight': weight,
        }

        return fitness_CovOutsideBurstISI

    except Exception as e:
        print(f"Error in fit_CovOutsideBurstISI: {e}")
        return 1000

def fit_NetworkISI(data_source, mega_mode = False, **kwargs):
    
    #source
    if data_source == 'simulated':
        simulated = True
    elif data_source == 'experimental':
        simulated = False
    
    
    try:
        print('Calculating NetworkISI fitness...')

        NetworkISI_targets = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanNetworkISI']
        # E_I_ratio_network_mean_ISI = 4.0 / 1.0

        # E_NetworkISI_target = NetworkISI_target['target'] * (E_I_ratio_network_mean_ISI / (E_I_ratio_network_mean_ISI + 1))
        # I_NetworkISI_target = NetworkISI_target['target'] / (E_I_ratio_network_mean_ISI + 1)

        NetworkISI_target = NetworkISI_targets.get('target', None)
        min_NetworkISI = NetworkISI_targets['min']
        max_NetworkISI = NetworkISI_targets['max']
        weight = NetworkISI_targets['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_NetworkISIs = []
        val_NetworkISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_NetworkISI = bursting_data_by_unit[unit]['mean_isi_all']
                val_NetworkISIs.append(val_NetworkISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_NetworkISI, E_NetworkISI_target, weight, maxFitness, 
                                                       min_val=min_NetworkISI, max_val=max_NetworkISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_NetworkISI, I_NetworkISI_target, weight, maxFitness, 
                                                       min_val=min_NetworkISI, max_val=max_NetworkISI)
                else:
                    fitness = the_scoring_function(val_NetworkISI, NetworkISI_target['target'], weight, maxFitness, 
                                                   min_val=min_NetworkISI, max_val=max_NetworkISI)

                fitness_NetworkISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_NetworkISI = {
            'fit': np.mean(fitness_NetworkISIs) if fitness_NetworkISIs else None,
            'value(s)': val_NetworkISIs,
            'target_E': E_NetworkISI_target,
            'target_I': I_NetworkISI_target,
            'min': min_NetworkISI,
            'max': max_NetworkISI,
            'weight': weight,
        }

        return fitness_NetworkISI

    except Exception as e:
        print(f"Error in fit_NetworkISI: {e}")
        return 1000

def fit_CovNetworkISI(simulated=False, **kwargs):
    try:
        print('Calculating CovNetworkISI fitness...')

        CovNetworkISI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['CoVNetworkISI']
        E_I_ratio_network_CoV_ISI = 2.0 / 1.0

        E_CovNetworkISI_target = CovNetworkISI_target['target'] * (E_I_ratio_network_CoV_ISI / (E_I_ratio_network_CoV_ISI + 1))
        I_CovNetworkISI_target = CovNetworkISI_target['target'] / (E_I_ratio_network_CoV_ISI + 1)

        min_CovNetworkISI = CovNetworkISI_target['min']
        max_CovNetworkISI = CovNetworkISI_target['max']
        weight = CovNetworkISI_target['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovNetworkISIs = []
        val_CovNetworkISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_CovNetworkISI = bursting_data_by_unit[unit]['cov_isi_all']
                val_CovNetworkISIs.append(val_CovNetworkISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_CovNetworkISI, E_CovNetworkISI_target, weight, maxFitness, 
                                                       min_val=min_CovNetworkISI, max_val=max_CovNetworkISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_CovNetworkISI, I_CovNetworkISI_target, weight, maxFitness, 
                                                       min_val=min_CovNetworkISI, max_val=max_CovNetworkISI)
                else:
                    fitness = the_scoring_function(val_CovNetworkISI, CovNetworkISI_target['target'], weight, maxFitness, 
                                                   min_val=min_CovNetworkISI, max_val=max_CovNetworkISI)

                fitness_CovNetworkISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_CovNetworkISI = {
            'fit': np.mean(fitness_CovNetworkISIs) if fitness_CovNetworkISIs else None,
            'value(s)': val_CovNetworkISIs,
            'target_E': E_CovNetworkISI_target,
            'target_I': I_CovNetworkISI_target,
            'min': min_CovNetworkISI,
            'max': max_CovNetworkISI,
            'weight': weight,
        }

        return fitness_CovNetworkISI

    except Exception as e:
        print(f"Error in fit_CovNetworkISI: {e}")
        return 1000

def fit_Number_Bursts(mega_mode = False, **kwargs):
    try:
        print('Calculating NumBursts fitness...')

        if not mega_mode: NumBursts_target = kwargs['targets']['bursting_data']['bursting_summary_data']['Number_Bursts']
        else: NumBursts_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['Number_Bursts']
        target = NumBursts_target['target']
        min_NumBursts = NumBursts_target['min']
        max_NumBursts = NumBursts_target['max']
        weight = NumBursts_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: Number_Bursts = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['Number_Bursts']
        else: Number_Bursts = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['Number_Bursts']
        
        val_NumBursts = Number_Bursts

        fitness_NumBurst = the_scoring_function(val_NumBursts, target, weight, maxFitness, 
                                                min_val=min_NumBursts, max_val=max_NumBursts)

        fitness_NumBurst_dict = {
            'fit': fitness_NumBurst,
            'value': val_NumBursts,
            'target': target,
            'min': min_NumBursts,
            'max': max_NumBursts,
            'weight': weight,
        }

        return fitness_NumBurst_dict

    except Exception as e:
        print(f"Error in fit_Number_Bursts: {e}")
        return 1000

def fit_mean_IBI(mega_mode = False, **kwargs):
    try:
        print('Calculating mean_IBI fitness...')

        if not mega_mode: mean_IBI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['mean_IBI']
        else: mean_IBI_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['mean_IBI']
        target = mean_IBI_target['target']
        min_mean_IBI = mean_IBI_target['min']
        max_mean_IBI = mean_IBI_target['max']
        weight = mean_IBI_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: mean_IBI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_IBI']
        else: mean_IBI = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['mean_IBI']
        val_mean_IBI = mean_IBI

        fitness_mean_IBI = the_scoring_function(val_mean_IBI, target, weight, maxFitness, 
                                                min_val=min_mean_IBI, max_val=max_mean_IBI)

        fitness_mean_IBI_dict = {
            'fit': fitness_mean_IBI,
            'value': val_mean_IBI,
            'target': target,
            'min': min_mean_IBI,
            'max': max_mean_IBI,
            'weight': weight,
        }

        return fitness_mean_IBI_dict

    except Exception as e:
        print(f"Error in fit_mean_IBI: {e}")
        return 1000

def fit_cov_IBI(mega_mode=False, **kwargs):
    try:
        print('Calculating cov_IBI fitness...')

        if not mega_mode: cov_IBI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['cov_IBI']
        else: cov_IBI_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['cov_IBI']
        target = cov_IBI_target['target']
        min_cov_IBI = cov_IBI_target['min']
        max_cov_IBI = cov_IBI_target['max']
        weight = cov_IBI_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: cov_IBI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['cov_IBI']
        else: cov_IBI = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['cov_IBI']
        val_cov_IBI = cov_IBI

        fitness_cov_IBI = the_scoring_function(val_cov_IBI, target, weight, maxFitness, 
                                               min_val=min_cov_IBI, max_val=max_cov_IBI)

        fitness_cov_IBI_dict = {
            'fit': fitness_cov_IBI,
            'value': val_cov_IBI,
            'target': target,
            'min': min_cov_IBI,
            'max': max_cov_IBI,
            'weight': weight,
        }

        return fitness_cov_IBI_dict

    except Exception as e:
        print(f"Error in fit_cov_IBI: {e}")
        return 1000

def fit_mean_Burst_Peak(mega_mode=False, **kwargs):
    try:
        print('Calculating mean_Burst_Peak fitness...')

        if not mega_mode: mean_Burst_Peak_target = kwargs['targets']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        else: mean_Burst_Peak_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        target = mean_Burst_Peak_target['target']
        # min_mean_Burst_Peak = mean_Burst_Peak_target['min']
        # max_mean_Burst_Peak = mean_Burst_Peak_target['max']
        min_mean_Burst_Peak = mean_Burst_Peak_target.get('min', None)
        max_mean_Burst_Peak = mean_Burst_Peak_target.get('max', None)
        weight = mean_Burst_Peak_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: mean_Burst_Peak = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        else: mean_Burst_Peak = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        val_mean_Burst_Peak = mean_Burst_Peak

        fitness_mean_Burst_Peak = the_scoring_function(val_mean_Burst_Peak, target, weight, maxFitness, 
                                                       min_val=min_mean_Burst_Peak, max_val=max_mean_Burst_Peak)

        fitness_mean_Burst_Peak_dict = {
            'fit': fitness_mean_Burst_Peak,
            'value': val_mean_Burst_Peak,
            'target': target,
            'min': min_mean_Burst_Peak,
            'max': max_mean_Burst_Peak,
            'weight': weight,
        }

        return fitness_mean_Burst_Peak_dict

    except Exception as e:
        print(f"Error in fit_mean_Burst_Peak: {e}")
        return 1000

def fit_cov_Burst_Peak(mega_mode = False, **kwargs):
    try:
        print('Calculating cov_Burst_Peak fitness...')

        if not mega_mode: cov_Burst_Peak_target = kwargs['targets']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        else: cov_Burst_Peak_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        target = cov_Burst_Peak_target['target']
        min_cov_Burst_Peak = cov_Burst_Peak_target['min']
        max_cov_Burst_Peak = cov_Burst_Peak_target['max']
        weight = cov_Burst_Peak_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: cov_Burst_Peak = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        else: cov_Burst_Peak = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        val_cov_Burst_Peak = cov_Burst_Peak

        fitness_cov_Burst_Peak = the_scoring_function(val_cov_Burst_Peak, target, weight, maxFitness, 
                                                      min_val=min_cov_Burst_Peak, max_val=max_cov_Burst_Peak)

        fitness_cov_Burst_Peak_dict = {
            'fit': fitness_cov_Burst_Peak,
            'value': val_cov_Burst_Peak,
            'target': target,
            'min': min_cov_Burst_Peak,
            'max': max_cov_Burst_Peak,
            'weight': weight,
        }

        return fitness_cov_Burst_Peak_dict

    except Exception as e:
        print(f"Error in fit_cov_Burst_Peak: {e}")
        return 1000

def fit_fano_factor(mega_mode=False, **kwargs):
    try:
        print('Calculating fano_factor fitness...')

        if not mega_mode: fano_factor_target = kwargs['targets']['bursting_data']['bursting_summary_data']['fano_factor']
        else: fano_factor_target = kwargs['targets']['mega_bursting_data']['bursting_summary_data']['fano_factor']
        target = fano_factor_target['target']
        min_fano_factor = fano_factor_target['min']
        max_fano_factor = fano_factor_target['max']
        weight = fano_factor_target['weight']
        maxFitness = kwargs['maxFitness']

        if not mega_mode: fano_factor = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
        else: fano_factor = kwargs['network_metrics']['mega_bursting_data']['bursting_summary_data']['fano_factor']
        val_fano_factor = fano_factor

        fitness_fano_factor = the_scoring_function(val_fano_factor, target, weight, maxFitness, 
                                                   min_val=min_fano_factor, max_val=max_fano_factor)

        fitness_fano_factor_dict = {
            'fit': fitness_fano_factor,
            'value': val_fano_factor,
            'target': target,
            'min': min_fano_factor,
            'max': max_fano_factor,
            'weight': weight,
        }

        return fitness_fano_factor_dict

    except Exception as e:
        print(f"Error in fit_fano_factor: {e}")
        return 1000

def I_frs_must_be_greater_than_E_frs(fitnessVals, network_metrics):
    '''I firing rates must be greater than E firing rates'''
    dealbroken = False
    try:
        simulated_data = network_metrics['simulated_data']
        spiking_data_by_unit = network_metrics['simulated_data']['spiking_data_by_unit']
        
        # Get population data for each population
        E_rates = []
        I_rates = []
        for unit, value in spiking_data_by_unit.items():
            if unit in simulated_data['E_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                E_rates.append(firing_rate)
            elif unit in simulated_data['I_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                I_rates.append(firing_rate)
                
        # Check if I firing rates are greater than E firing rates
        if np.mean(I_rates) < np.mean(E_rates):
            dealbroken = True
            print(f'I firing rates are not greater than E firing rates. I: {np.mean(I_rates)}, E: {np.mean(E_rates)}')
            
        return dealbroken
    except Exception as e:
        print(f"Error in I_frs_must_be_greater_than_E_frs: {e}")
        return False

def E_and_I_neurons_must_fire(fitnessVals, network_metrics):
    '''E and I neurons must fire'''
    dealbroken = False
    try:
        simulated_data = network_metrics['simulated_data']
        spiking_data_by_unit = network_metrics['simulated_data']['spiking_data_by_unit']
        
        #get population data for each population
        E_rates = []
        I_rates = []
        for unit, value in spiking_data_by_unit.items():
            if unit in simulated_data['E_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                E_rates.append(firing_rate)
            elif unit in simulated_data['I_Gids']:
                firing_rate = spiking_data_by_unit[unit]['FireRate']
                I_rates.append(firing_rate)
                
        #check if E and I neurons are firing
        if np.mean(E_rates) == 0 or np.mean(I_rates) == 0:
            dealbroken = True
            print(f'E and I neurons are not firing. E: {np.mean(E_rates)}, I: {np.mean(I_rates)}')
    except Exception as e:
        print(f"Error in E_and_I_neurons_must_fire: {e}")
        dealbroken = False
        
    return dealbroken

def check_dealbreakers(fitnessVals, **kwargs):
    '''Check if any fitness values are dealbreakers'''
    network_metrics = kwargs['network_metrics']
    dealbreakers = []
    # dealbroken = I_frs_must_be_greater_than_E_frs(fitnessVals, network_metrics)
    # dealbroken = E_and_I_neurons_must_fire(fitnessVals, network_metrics)  
    dealbreakers.append(I_frs_must_be_greater_than_E_frs(fitnessVals, network_metrics))
    dealbreakers.append(E_and_I_neurons_must_fire(fitnessVals, network_metrics))
    if any(dealbreakers):
        return True
    else:
        return False  
    #return dealbroken

def save_fitness_results(output_path, fitnessResults):
    with open(output_path, 'w') as f:
        json.dump(fitnessResults, f, indent=4)
    print(f'fitness results saved to {output_path}')

def get_fitness(data_source, **kwargs):
    '''Main fitness calculation function'''
    try:
        fitnessVals = {}
        
        # Spiking Data
        fitnessVals['rate_fit'] = fit_firing_rates(data_source, **kwargs)
        fitnessVals['CoV_rate_fit'] = fit_CoV_firing_rate(data_source, **kwargs)
        # fitnessVals['ISI_fit'] = fit_ISI(simulated=True, **kwargs)
        # fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(simulated=True, **kwargs)
        fitnessVals['ISI_fit'] = fit_ISI(data_source, **kwargs)
        fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(data_source, **kwargs)
        
        
        # aw 2025-02-03 08:08:13 - left off here. NEed to update these bursting functions
        # also need to add the mega bursting functions
        # Bursting Data
        #fitnessVals['mean_Burst_Rate_fit'] = fit_mean_Burst_Rate(simulated=True, **kwargs)
        fitnessVals['mean_Burst_Rate_fit'] = fit_mean_Burst_Rate(data_source, **kwargs)
        fitnessVals['baseline_fit'] = fit_baseline(data_source, **kwargs)
        fitnessVals['WithinBurstISI_fit'] = fit_WithinBurstISI(data_source, **kwargs)
        fitnessVals['CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(data_source, **kwargs)
        fitnessVals['OutsideBurstISI_fit'] = fit_OutsideBurstISI(data_source, **kwargs)
        fitnessVals['CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(data_source, **kwargs)
        # fitnessVals['NetworkISI_fit'] = fit_NetworkISI(data_source, **kwargs)
        # fitnessVals['CoVNetworkISI_fit'] = fit_CovNetworkISI(simulated=True, **kwargs)
        # fitnessVals['Number_Bursts_fit'] = fit_Number_Bursts(**kwargs)
        fitnessVals['mean_IBI_fit'] = fit_mean_IBI(**kwargs)
        fitnessVals['cov_IBI_fit'] = fit_cov_IBI(**kwargs)
        fitnessVals['mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(**kwargs)
        fitnessVals['cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(**kwargs)
        fitnessVals['fano_factor_fit'] = fit_fano_factor(**kwargs)
        
        # Mega Bursting Data
        fitnessVals['mega_mean_Burst_Rate_fit'] = fit_mean_Burst_Rate(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_baseline_fit'] = fit_baseline(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_WithinBurstISI_fit'] = fit_WithinBurstISI(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_OutsideBurstISI_fit'] = fit_OutsideBurstISI(data_source, mega_mode=True, **kwargs)
        fitnessVals['mega_CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(data_source, mega_mode=True, **kwargs)
        # fitnessVals['mega_NetworkISI_fit'] = fit_NetworkISI(data_source, mega_mode=True, **kwargs)
        # fitnessVals['mega_CoVNetworkISI_fit'] = fit_CovNetworkISI(simulated=True, mega_mode=True, **kwargs)
        # fitnessVals['mega_Number_Bursts_fit'] = fit_Number_Bursts(mega_mode=True, **kwargs)
        fitnessVals['mega_mean_IBI_fit'] = fit_mean_IBI(mega_mode=True, **kwargs)
        fitnessVals['mega_cov_IBI_fit'] = fit_cov_IBI(mega_mode=True, **kwargs)
        fitnessVals['mega_mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(mega_mode=True, **kwargs)
        fitnessVals['mega_cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(mega_mode=True, **kwargs)
        #fitnessVals['mega_fano_factor_fit'] = fit_fano_factor(mega_mode=True, **kwargs)
        
        # for debugging, uncomment as needed:
        # # print warning if mega and non mega versions of the same metric have the same  'fit' and 'value'
        
        # for key in fitnessVals:
        #     if 'mega' in key:
        #         non_mega_key = key.replace('mega_', '')
        #         try:
        #             if fitnessVals[key]['fit'] == fitnessVals[non_mega_key]['fit'] and fitnessVals[key]['fit'] != 1:
        #                 print(f'Warning: {key} and {non_mega_key} have the same fit. Consider removing one of them.')
                        
        #                 value_key = fitnessVals[key].get('value', fitnessVals[key].get('value(s)'))
        #                 value_non_mega_key = fitnessVals[non_mega_key].get('value', fitnessVals[non_mega_key].get('value(s)'))
                        
        #                 if isinstance(value_key, list) and isinstance(value_non_mega_key, list):
        #                     if all(v1 == v2 for v1, v2 in zip(value_key, value_non_mega_key)):
        #                         print(f'Warning: {key} and {non_mega_key} have the same fit and value. Consider removing one of them.')
        #                 elif value_key == value_non_mega_key:
        #                     print(f'Warning: {key} and {non_mega_key} have the same fit and value. Consider removing one of them.')
        #                 # aw 2025-02-04 11:46:05 - fanofactor between mega and non mega bursts is the same.
        #                 # this is because the fano factor is calculated using the same data for both mega and non mega bursts
        #                 # it may be more appropriate to calculate the fano factor using individual spiking data for each neuron
        #         except Exception as e:
        #             print(f'Error comparing {key} and {non_mega_key}: {e}')
        #             continue

        # if any fit values are equal to 1000, change to {'fit': 1000}
        for key in fitnessVals:
            if fitnessVals[key] == 1000:
                fitnessVals[key] = {'fit': 1000}
        average_fitness = np.mean([fitnessVals[key]['fit'] for key in fitnessVals])
        
        # assert that average_fitness is an integer or float
        assert isinstance(average_fitness, (int, float)), 'average_fitness is not an integer or float'
        
        return average_fitness, fitnessVals
    except Exception as e:
        print(f"Error in get_fitness: {e}")
        return 1000, {}

def handle_existing_fitness(fitness_save_path):
    if os.path.exists(fitness_save_path):
        with open(fitness_save_path, 'r') as f:
            fitnessResults = json.load(f)
        average_fitness = fitnessResults['average_fitness']
        print(f'Fitness results already exist: {average_fitness}')
        return average_fitness
    return None

def handle_simulated_data_mode(simData, kwargs):
    """Handles the logic for 'simulated data' mode."""
    kwargs['source'] = 'simulated'
    candidate_path = kwargs['simConfig']['filename']
    fitness_save_path = f'{candidate_path}_fitness.json'
    assert simData is not None, 'Simulated data must be provided in "simulated data" mode.'
    kwargs.update({'simData': simData})
    error, kwargs = calculate_network_metrics(kwargs)
    return error, kwargs

def handle_experimental_data_mode(kwargs):
    """Handles the logic for 'experimental data' mode."""
    kwargs['source'] = 'experimental'
    implemented = False
    assert implemented, 'Experimental data source not yet implemented.'
    return None, kwargs

def calculate_and_save_fitness(kwargs):
    """Calculates fitness and saves the results."""
    try:
        average_fitness, fitnessResults = get_fitness(kwargs['source'], **kwargs)
        fitnessResults['average_fitness'] = average_fitness
        fitnessResults['maxFitness'] = kwargs['maxFitness']

        # deal_broken = check_dealbreakers(fitnessResults, **kwargs)
        # assert not deal_broken, 'Dealbreakers found in fitness results.'

        save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
        return average_fitness
    except Exception as e:
        error_trace = str(e)
        fitnessResults = {
            'average_fitness': kwargs['maxFitness'],
            'maxFitness': kwargs['maxFitness'],
            'error': 'acceptable' if any(error in error_trace for error in []) else 'new',
            'error_trace': error_trace
        }
        print(f'Error calculating fitness: {e}')
        save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
        return 1000
    
def calculate_network_metrics(kwargs):
    print('Calculating network activity metrics...')
    #from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import get_simulated_network_activity_metrics
    from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import compute_network_metrics
    #network_metrics = get_simulated_network_activity_metrics(**kwargs)
    network_metrics = compute_network_metrics(**kwargs)
        
    # Save the network metrics to a file
    # if networks_metrics is None, save an error message
    if network_metrics is None:
        print('Network activity metrics could not be calculated.')
        fitnessResults = {
            'average_fitness': kwargs['maxFitness'],
            'maxFitness': kwargs['maxFitness'],
            'error': 'Network activity metrics could not be calculated. Fitness set to maxFitness.'
        }
        print(f'Fitness set to maxFitness: {kwargs["maxFitness"]}')
        save_fitness_results(kwargs['fitness_save_path'], fitnessResults)
        kwargs['network_metrics'] = None
        error = fitnessResults['error']
        return error, kwargs
    #else, return the network_metrics
    else:
        kwargs['network_metrics'] = network_metrics
        #kwargs['mega_network_metrics'] = mega_metrics
        return None, kwargs
    
def submit_fitness_job(simData=None, mode='optimizing', **kwargs):
    ''' Copy the initial logic of calulate_fitness() and modify it to submit a job to the cluster '''
    
    try:
        # Select mode and execute corresponding logic
        if mode == 'optimizing':
            kwargs = retrieve_sim_data_from_call_stack(simData, **kwargs)
            kwargs.pop('simData', None) # Remove simData from kwargs, it will be added back in calculate_fitness()
            #error, kwargs = handle_optimizing_mode(simData, kwargs)
            
        # elif mode == 'simulated data':
        #     error, kwargs = handle_simulated_data_mode(simData, kwargs)
        # elif mode == 'experimental data':
        #     error, kwargs = handle_experimental_data_mode(kwargs)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        # Save kwargs to a file
        sim_data_path = kwargs.get('data_file_path', None)
        assert sim_data_path is not None, 'Data file path not provided.'
        kwargs_path = sim_data_path.replace('_data.json', '_kwargs.json')
        kwargs['kwargs_path'] = kwargs_path
        
    except Exception as e:
        print(f'Error calculating fitness: {e}')
        fitnessResults = {
            'average_fitness': kwargs.get('maxFitness', 1000),
            'maxFitness': kwargs.get('maxFitness', 1000),
            'error': 'general',
            'error_trace': str(e)
        }
        save_fitness_results(kwargs.get('fitness_save_path', kwargs.get('fitness_save_path', 'unknown_path.json')), fitnessResults)
        #save_fitness_results(kwargs.get('fitness_save_path', 'unknown_path.json'), fitnessResults)
        return 1000

    # Prepare sbatch or srun command
    fitness_script = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/batch_scripts/fitness_helper.py'
    import sys
    from pprint import pprint
    
    #global_kwargs_path = temp_user_args.global_kwargs_path
    #global_kwargs_path = '/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/optimization_projects/CDKL5_DIV21/scripts/temp_global_kwargs.pkl'
    
    #load pickle file from global_kwargs_path
    #import pickle
        
    # # Ensure the file exists
    # if os.path.exists(global_kwargs_path):
    #     try:
    #         print(f'Loading global kwargs from {global_kwargs_path}')
    #         with open(global_kwargs_path, 'rb') as f:
    #             global_kwargs = pickle.load(f)
    #         print("Global kwargs loaded successfully.")
    #         print(global_kwargs)  # Print the loaded data for debugging purposes
    #     except Exception as e:
    #         print(f"Error loading pickle file: {e}")
    # else:
    #    print(f"Pickle file does not exist at: {global_kwargs_path}")
    
    from __main__ import global_kwargs
    nodes_per_sim = global_kwargs.get('nodes_per_sim', 1)
    cores_per_sim = global_kwargs.get('cores_per_sim', 1)
    cores_per_task = global_kwargs.get('cores_per_task', 1)
    tasks_per_sim = global_kwargs.get('tasks_per_sim', 1)
    #available_cores = tasks_per_sim * cores_per_task
    
    nodes_per_sim = str(global_kwargs.get('nodes_per_sim', 1))  # Convert to string
    cores_per_task = str(global_kwargs.get('cores_per_task', 1))  # Convert to string
    tasks_per_sim = str(global_kwargs.get('tasks_per_sim', 1))  # Convert to string
    
    slurm_command = [
        "srun", "--exclusive", 
        "-N", nodes_per_sim,
        "-n", '1',
        #"--cpus-per-task", f'{cores_per_sim}',
        "--cpus-per-task", f'{cores_per_task}',
        "python3", fitness_script, sim_data_path, kwargs_path
    ]
    
    pprint(f"Submitting fitness job with command: {slurm_command}")

    # Submit the job
    subprocess.run(slurm_command)


    # def submit_fitness_job(kwargs):
    #     try:
    #         sim_data_path = kwargs.get('data_file_path', None)
    #         assert sim_data_path is not None, 'Data file path not provided.'
    #         kwargs_path = kwargs.get('kwargs_path', sim_data_path.replace('_data.json', '_kwargs.json'))

    #         # Prepare sbatch or srun command
    #         fitness_script = "/path/to/fitness_calculation.py"
    #         slurm_command = [
    #             "srun", "--exclusive", "--cpus-per-task=1",
    #             "python3", fitness_script, sim_data_path, kwargs_path
    #         ]

    #         # Submit the job
    #         subprocess.run(slurm_command)
    #     except Exception as e:
    #         print(f'Error submitting fitness job: {e}')
    #         return 1000 