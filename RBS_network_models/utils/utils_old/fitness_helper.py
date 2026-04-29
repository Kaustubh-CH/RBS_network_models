import os
import json
import sys
from pprint import pprint
import numpy as np

# def add_repo_root_to_sys_path():
#     #path_for_external_repos = "/pscratch/sd/a/adammwea/workspace/RBS_neuronal_network_models/"
#     repo_path = get_git_root()
#     sys.path.insert(0, repo_path)
#     #sys.path.append(path_for_external_repos)    

# def get_git_root():
#     try:
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         while current_dir != os.path.dirname(current_dir):  # Root directory check
#             if os.path.isdir(os.path.join(current_dir, '.git')):
#                 return current_dir
#             current_dir = os.path.dirname(current_dir)
#         return None  # No .git directory found
#     except Exception as e:
#         print(f"An error occurred while determining the Git root: {e}")
#         return None
    
# add_repo_root_to_sys_path()
# #from _external.RBS_network_simulation_optimization_tools.modules.analysis.analyze_network_activity import get_simulated_network_activity_metrics
# #from _external.RBS_network_simulation_optimization_tools.workspace.optimization_projects.CDKL5_DIV21_dep._2_batchrun_optimization.extract_simulated_data import retrieve_sim_data_from_call_stack
# import numpy as np
# #from optimization_projects.CDKL5_DIV21.scripts.helper import *
# #from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
# from DIV21.utils.sim_helper import *
# import pickle
# #from analysis_helper import *
# from DIV21.utils.analysis_helper import *


'''The Scoring Functions'''        
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
	
	# Set default min and max values if not provided
	if min_val is None:
		min_val = float('-inf')  # Allow all values below positive infinity
	if max_val is None:
		max_val = float('inf')   # Allow all values above negative infinity

	# Calculate fitness score
	if min_val <= val <= max_val:
		return min(np.exp(abs(target_val - val) / weight), maxFitness)
	else:
		return maxFitness

'''fitness functions for the network activity metrics'''
def fit_firing_rates(simulated=False, **kwargs):
    try:
        print('Calculating firing rate fitness...')
        MeanFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['MeanFireRate']
        E_I_ratio = 5  # 1:5 ratio of E to I neurons
        E_fr_target = MeanFireRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        I_fr_target = MeanFireRate_target['target'] / (E_I_ratio + 1)
        min_FR = MeanFireRate_target['min']
        max_FR = MeanFireRate_target['max']
        weight = MeanFireRate_target['weight']
        maxFitness = kwargs['maxFitness']
        
        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        fitness_FRs = []
        val_FRs = []
        for unit, value in spiking_data_by_unit.items():
            val_FR = spiking_data_by_unit[unit]['FireRate']
            val_FRs.append(val_FR)
            if simulated is False: 
                target = MeanFireRate_target['target'] # experimental target
                fitness = the_scoring_function(val_FR, target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
            else:
                E_gids = kwargs['network_metrics']['simulated_data']['E_Gids']
                I_gids = kwargs['network_metrics']['simulated_data']['I_Gids']
                if unit in E_gids:
                    E_fitness = the_scoring_function(val_FR, E_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                    fitness = E_fitness
                elif unit in I_gids:
                    I_fitness = the_scoring_function(val_FR, I_fr_target, weight, maxFitness, min_val=min_FR, max_val=max_FR)
                    fitness = I_fitness
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
            'target_E': E_fr_target,
            'target_I': I_fr_target,
            'min': min_FR,
            'max': max_FR,
            'weight': weight,        
        }
        return fitness_FR_dict
    except Exception as e:
        print(f"Error in fit_firing_rates: {e}")
        return 1000

def fit_CoV_firing_rate(simulated=False, **kwargs):
    try:
        print('Calculating CoV firing rate fitness...')
        CoVFireRate_target = kwargs['targets']['spiking_data']['spiking_summary_data']['CoVFireRate']
        E_I_ratio = 1.5 / 0.7  # ratio of E to I neurons
        E_CoV_target = CoVFireRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        I_CoV_target = CoVFireRate_target['target'] / (E_I_ratio + 1)
        min_CoV = CoVFireRate_target['min']
        max_CoV = CoVFireRate_target['max']
        weight = CoVFireRate_target['weight']
        maxFitness = kwargs['maxFitness']
        
        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        
        fitness_CoVs = []
        val_CoVs = []
        for unit, value in spiking_data_by_unit.items():
            val_CoV = spiking_data_by_unit[unit]['CoV_fr']
            val_CoVs.append(val_CoV)
            if simulated is False:
                target = CoVFireRate_target['target']
                fitness = the_scoring_function(val_CoV, target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
            else:
                E_gids = kwargs['network_metrics']['simulated_data']['E_Gids']
                I_gids = kwargs['network_metrics']['simulated_data']['I_Gids']
                if unit in E_gids:
                    fitness = the_scoring_function(val_CoV, E_CoV_target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
                elif unit in I_gids:
                    fitness = the_scoring_function(val_CoV, I_CoV_target, weight, maxFitness, min_val=min_CoV, max_val=max_CoV)
            
            fitness_CoVs.append(fitness)
        
        fitness_CoV = np.mean(fitness_CoVs)
        fitness_CoV_dict = {
            'fit': fitness_CoV,
            'value(s)': val_CoVs,
            'target_E': E_CoV_target,
            'target_I': I_CoV_target,
            'min': min_CoV,
            'max': max_CoV,
            'weight': weight,
        }
        return fitness_CoV_dict
    except Exception as e:
        print(f"Error in fit_CoV_firing_rate: {e}")
        return 1000

def fit_ISI(simulated=False, **kwargs):
    try:
        print('Calculating ISI fitness...')

        MeanISI_target = kwargs['targets']['spiking_data']['spiking_summary_data']['MeanISI']
        E_I_ratio_mean_ISI = 5.0 / 1.0  # Biologically founded E to I ratio for Mean ISI

        # Calculate targets for E and I populations
        E_meanISI_target = MeanISI_target['target'] * (E_I_ratio_mean_ISI / (E_I_ratio_mean_ISI + 1))
        I_meanISI_target = MeanISI_target['target'] / (E_I_ratio_mean_ISI + 1)

        # ISI bounds and weights
        min_ISI = MeanISI_target['min']
        max_ISI = MeanISI_target['max']
        weight = MeanISI_target['weight']
        maxFitness = kwargs['maxFitness']

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
                        fitness = the_scoring_function(val_ISI, E_meanISI_target, weight, maxFitness, 
                                                       min_val=min_ISI, max_val=max_ISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_ISI, I_meanISI_target, weight, maxFitness, 
                                                       min_val=min_ISI, max_val=max_ISI)
                else:
                    # Use undifferentiated target for experimental data
                    fitness = the_scoring_function(val_ISI, MeanISI_target['target'], weight, maxFitness, 
                                                   min_val=min_ISI, max_val=max_ISI)

                fitness_ISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_ISI = {
            'fit': np.mean(fitness_ISIs) if fitness_ISIs else None,
            'value(s)': val_ISIs,
            'target_E': E_meanISI_target,
            'target_I': I_meanISI_target,
            'min': min_ISI,
            'max': max_ISI,
            'weight': weight,
        }

        return fitness_ISI

    except Exception as e:
        print(f"Error in fit_ISI: {e}")
        return 1000

def fit_CoV_ISI(simulated=False, **kwargs):
    try:
        print('Calculating CoV ISI fitness...')

        CoV_ISI_target = kwargs['targets']['spiking_data']['spiking_summary_data']['CoV_ISI']
        E_I_ratio_CoV_ISI = 1.5 / 0.7  # Reflects variability differences

        # Calculate targets for E and I populations
        E_CoV_ISI_target = CoV_ISI_target['target'] * (E_I_ratio_CoV_ISI / (E_I_ratio_CoV_ISI + 1))
        I_CoV_ISI_target = CoV_ISI_target['target'] / (E_I_ratio_CoV_ISI + 1)

        # CoV ISI bounds and weights
        min_CoV_ISI = CoV_ISI_target['min']
        max_CoV_ISI = CoV_ISI_target['max']
        weight = CoV_ISI_target['weight']
        maxFitness = kwargs['maxFitness']

        spiking_data_by_unit = kwargs['network_metrics']['spiking_data']['spiking_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CoV_ISIs = []
        val_CoV_ISIs = []

        for unit, value in spiking_data_by_unit.items():
            try:
                val_CoV_ISI = spiking_data_by_unit[unit]['CoV_ISI']
                val_CoV_ISIs.append(val_CoV_ISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_CoV_ISI, E_CoV_ISI_target, weight, maxFitness, 
                                                       min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_CoV_ISI, I_CoV_ISI_target, weight, maxFitness, 
                                                       min_val=min_CoV_ISI, max_val=max_CoV_ISI)
                else:
                    fitness = the_scoring_function(val_CoV_ISI, CoV_ISI_target['target'], weight, maxFitness, 
                                                   min_val=min_CoV_ISI, max_val=max_CoV_ISI)

                fitness_CoV_ISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_CoV_ISI = {
            'fit': np.mean(fitness_CoV_ISIs) if fitness_CoV_ISIs else None,
            'value(s)': val_CoV_ISIs,
            'target_E': E_CoV_ISI_target,
            'target_I': I_CoV_ISI_target,
            'min': min_CoV_ISI,
            'max': max_CoV_ISI,
            'weight': weight,
        }

        return fitness_CoV_ISI

    except Exception as e:
        print(f"Error in fit_CoV_ISI: {e}")
        return 1000

def fit_mean_Burst_Rate(simulated=False, **kwargs):
    try:
        print('Calculating mean burst rate fitness...')

        try:
            MeanBurstRate_target = kwargs['targets']['bursting_data']['bursting_summary_data']['mean_Burst_Rate'] #TODO rerun analysis to get the correct target values
        except Exception as e: #if the program fails here print a big yellow warning to fix the knon issue.
            activate_print()
            print(f"\033[93mWarning: {e}\033[0m")
            print(f"\033[93mWarning: The target values for mean_Burst_Rate are not available. Please rerun the analysis to get the correct target values.\033[0m")
            print(f"\033[93mWarning: Setting the fitness score to 1000.\033[0m")
            suppress_print()
            return 1000
        #E_I_ratio = 2.0 / 1.0

        #E_meanBurstRate_target = MeanBurstRate_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        #I_meanBurstRate_target = MeanBurstRate_target['target'] / (E_I_ratio + 1)

        min_meanBurstRate = MeanBurstRate_target['min']
        max_meanBurstRate = MeanBurstRate_target['max']
        weight = MeanBurstRate_target['weight']
        maxFitness = kwargs['maxFitness']

        mean_Burst_Rate = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Rate']
        
        mean_Burst_Rate_fitness = the_scoring_function(mean_Burst_Rate, MeanBurstRate_target['target'], weight, maxFitness,)
        
        fitness_meanBurstRate = {
            'fit': mean_Burst_Rate_fitness,
            'value(s)': mean_Burst_Rate,
            'target' : MeanBurstRate_target['target'],
            'min': min_meanBurstRate,
            'max': max_meanBurstRate,
            'weight': weight,
        }
    except Exception as e:
        print(f"Error in fit_mean_Burst_Rate: {e}")
        return 1000
    
def fit_baseline(**kwargs):
    try:
        print('Calculating baseline fitness...')

        baseline_target = kwargs['targets']['bursting_data']['bursting_summary_data']['baseline']
        target = baseline_target['target']
        min_baseline = baseline_target['min']
        max_baseline = baseline_target['max']
        weight = baseline_target['weight']
        maxFitness = kwargs['maxFitness']

        baseline = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['baseline']
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

def fit_WithinBurstISI(simulated=False, **kwargs):
    try:
        print('Calculating WithinBurstISI fitness...')

        WithinBurstISI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanWithinBurstISI']
        E_I_ratio = 2.0 / 1.0

        E_WithinBurstISI_target = WithinBurstISI_target['target'] * (E_I_ratio / (E_I_ratio + 1))
        I_WithinBurstISI_target = WithinBurstISI_target['target'] / (E_I_ratio + 1)

        min_WithinBurstISI = WithinBurstISI_target['min']
        max_WithinBurstISI = WithinBurstISI_target['max']
        weight = WithinBurstISI_target['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_WithinBurstISIs = []
        val_WithinBurstISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_WithinBurstISI = bursting_data_by_unit[unit]['mean_isi_within']
                val_WithinBurstISIs.append(val_WithinBurstISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_WithinBurstISI, E_WithinBurstISI_target, weight, maxFitness, 
                                                       min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_WithinBurstISI, I_WithinBurstISI_target, weight, maxFitness, 
                                                       min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)
                else:
                    fitness = the_scoring_function(val_WithinBurstISI, WithinBurstISI_target['target'], weight, maxFitness, 
                                                   min_val=min_WithinBurstISI, max_val=max_WithinBurstISI)

                fitness_WithinBurstISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_WithinBurstISI = {
            'fit': np.mean(fitness_WithinBurstISIs) if fitness_WithinBurstISIs else None,
            'value(s)': val_WithinBurstISIs,
            'target_E': E_WithinBurstISI_target,
            'target_I': I_WithinBurstISI_target,
            'min': min_WithinBurstISI,
            'max': max_WithinBurstISI,
            'weight': weight,
        }

        return fitness_WithinBurstISI

    except Exception as e:
        print(f"Error in fit_WithinBurstISI: {e}")
        return 1000

def fit_CovWithinBurstISI(simulated=False, **kwargs):
    try:
        print('Calculating CovWithinBurstISI fitness...')

        CovWithinBurstISI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['CovWithinBurstISI']
        E_I_ratio_burst = 2.5 / 1.0

        E_CovWithinBurstISI_target = CovWithinBurstISI_target['target'] * (E_I_ratio_burst / (E_I_ratio_burst + 1))
        I_CovWithinBurstISI_target = CovWithinBurstISI_target['target'] / (E_I_ratio_burst + 1)

        min_CovWithinBurstISI = CovWithinBurstISI_target['min']
        max_CovWithinBurstISI = CovWithinBurstISI_target['max']
        weight = CovWithinBurstISI_target['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovWithinBurstISIs = []
        val_CovWithinBurstISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_CovWithinBurstISI = bursting_data_by_unit[unit]['cov_isi_within']
                val_CovWithinBurstISIs.append(val_CovWithinBurstISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_CovWithinBurstISI, E_CovWithinBurstISI_target, weight, maxFitness, 
                                                       min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_CovWithinBurstISI, I_CovWithinBurstISI_target, weight, maxFitness, 
                                                       min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)
                else:
                    fitness = the_scoring_function(val_CovWithinBurstISI, CovWithinBurstISI_target['target'], weight, maxFitness, 
                                                   min_val=min_CovWithinBurstISI, max_val=max_CovWithinBurstISI)

                fitness_CovWithinBurstISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_CovWithinBurstISI = {
            'fit': np.mean(fitness_CovWithinBurstISIs) if fitness_CovWithinBurstISIs else None,
            'value(s)': val_CovWithinBurstISIs,
            'target_E': E_CovWithinBurstISI_target,
            'target_I': I_CovWithinBurstISI_target,
            'min': min_CovWithinBurstISI,
            'max': max_CovWithinBurstISI,
            'weight': weight,
        }

        return fitness_CovWithinBurstISI

    except Exception as e:
        print(f"Error in fit_CovWithinBurstISI: {e}")
        return 1000

def fit_OutsideBurstISI(simulated=False, **kwargs):
    try:
        print('Calculating OutsideBurstISI fitness...')

        OutsideBurstISI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanOutsideBurstISI']
        E_I_ratio_outside_mean_ISI = 5.0 / 1.0

        E_OutsideBurstISI_target = OutsideBurstISI_target['target'] * (E_I_ratio_outside_mean_ISI / (E_I_ratio_outside_mean_ISI + 1))
        I_OutsideBurstISI_target = OutsideBurstISI_target['target'] / (E_I_ratio_outside_mean_ISI + 1)

        min_OutsideBurstISI = OutsideBurstISI_target['min']
        max_OutsideBurstISI = OutsideBurstISI_target['max']
        weight = OutsideBurstISI_target['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_OutsideBurstISIs = []
        val_OutsideBurstISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_OutsideBurstISI = bursting_data_by_unit[unit]['mean_isi_outside']
                val_OutsideBurstISIs.append(val_OutsideBurstISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_OutsideBurstISI, E_OutsideBurstISI_target, weight, maxFitness, 
                                                       min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_OutsideBurstISI, I_OutsideBurstISI_target, weight, maxFitness, 
                                                       min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)
                else:
                    fitness = the_scoring_function(val_OutsideBurstISI, OutsideBurstISI_target['target'], weight, maxFitness, 
                                                   min_val=min_OutsideBurstISI, max_val=max_OutsideBurstISI)

                fitness_OutsideBurstISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_OutsideBurstISI = {
            'fit': np.mean(fitness_OutsideBurstISIs) if fitness_OutsideBurstISIs else None,
            'value(s)': val_OutsideBurstISIs,
            'target_E': E_OutsideBurstISI_target,
            'target_I': I_OutsideBurstISI_target,
            'min': min_OutsideBurstISI,
            'max': max_OutsideBurstISI,
            'weight': weight,
        }

        return fitness_OutsideBurstISI

    except Exception as e:
        print(f"Error in fit_OutsideBurstISI: {e}")
        return 1000

def fit_CovOutsideBurstISI(simulated=False, **kwargs):
    try:
        print('Calculating CovOutsideBurstISI fitness...')

        CovOutsideBurstISI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['CovOutsideBurstISI']
        E_I_ratio_outside_CoV_ISI = 1.5 / 0.8

        E_CovOutsideBurstISI_target = CovOutsideBurstISI_target['target'] * (E_I_ratio_outside_CoV_ISI / (E_I_ratio_outside_CoV_ISI + 1))
        I_CovOutsideBurstISI_target = CovOutsideBurstISI_target['target'] / (E_I_ratio_outside_CoV_ISI + 1)

        min_CovOutsideBurstISI = CovOutsideBurstISI_target['min']
        max_CovOutsideBurstISI = CovOutsideBurstISI_target['max']
        weight = CovOutsideBurstISI_target['weight']
        maxFitness = kwargs['maxFitness']

        bursting_data_by_unit = kwargs['network_metrics']['bursting_data']['bursting_data_by_unit']
        E_Gids = kwargs['network_metrics']['simulated_data']['E_Gids']
        I_Gids = kwargs['network_metrics']['simulated_data']['I_Gids']

        fitness_CovOutsideBurstISIs = []
        val_CovOutsideBurstISIs = []

        for unit, value in bursting_data_by_unit.items():
            try:
                val_CovOutsideBurstISI = bursting_data_by_unit[unit]['cov_isi_outside']
                val_CovOutsideBurstISIs.append(val_CovOutsideBurstISI)

                if simulated:
                    if unit in E_Gids:
                        fitness = the_scoring_function(val_CovOutsideBurstISI, E_CovOutsideBurstISI_target, weight, maxFitness, 
                                                       min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                    elif unit in I_Gids:
                        fitness = the_scoring_function(val_CovOutsideBurstISI, I_CovOutsideBurstISI_target, weight, maxFitness, 
                                                       min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)
                else:
                    fitness = the_scoring_function(val_CovOutsideBurstISI, CovOutsideBurstISI_target['target'], weight, maxFitness, 
                                                   min_val=min_CovOutsideBurstISI, max_val=max_CovOutsideBurstISI)

                fitness_CovOutsideBurstISIs.append(fitness)
            except Exception as unit_error:
                print(f"Error processing unit {unit}: {unit_error}")
                continue  # Skip to the next unit if there's an issue

        fitness_CovOutsideBurstISI = {
            'fit': np.mean(fitness_CovOutsideBurstISIs) if fitness_CovOutsideBurstISIs else None,
            'value(s)': val_CovOutsideBurstISIs,
            'target_E': E_CovOutsideBurstISI_target,
            'target_I': I_CovOutsideBurstISI_target,
            'min': min_CovOutsideBurstISI,
            'max': max_CovOutsideBurstISI,
            'weight': weight,
        }

        return fitness_CovOutsideBurstISI

    except Exception as e:
        print(f"Error in fit_CovOutsideBurstISI: {e}")
        return 1000

def fit_NetworkISI(simulated=False, **kwargs):
    try:
        print('Calculating NetworkISI fitness...')

        NetworkISI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['MeanNetworkISI']
        E_I_ratio_network_mean_ISI = 4.0 / 1.0

        E_NetworkISI_target = NetworkISI_target['target'] * (E_I_ratio_network_mean_ISI / (E_I_ratio_network_mean_ISI + 1))
        I_NetworkISI_target = NetworkISI_target['target'] / (E_I_ratio_network_mean_ISI + 1)

        min_NetworkISI = NetworkISI_target['min']
        max_NetworkISI = NetworkISI_target['max']
        weight = NetworkISI_target['weight']
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

def fit_Number_Bursts(**kwargs):
    try:
        print('Calculating NumBursts fitness...')

        NumBursts_target = kwargs['targets']['bursting_data']['bursting_summary_data']['Number_Bursts']
        target = NumBursts_target['target']
        min_NumBursts = NumBursts_target['min']
        max_NumBursts = NumBursts_target['max']
        weight = NumBursts_target['weight']
        maxFitness = kwargs['maxFitness']

        Number_Bursts = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['Number_Bursts']
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

def fit_mean_IBI(**kwargs):
    try:
        print('Calculating mean_IBI fitness...')

        mean_IBI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['mean_IBI']
        target = mean_IBI_target['target']
        min_mean_IBI = mean_IBI_target['min']
        max_mean_IBI = mean_IBI_target['max']
        weight = mean_IBI_target['weight']
        maxFitness = kwargs['maxFitness']

        mean_IBI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_IBI']
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

def fit_cov_IBI(**kwargs):
    try:
        print('Calculating cov_IBI fitness...')

        cov_IBI_target = kwargs['targets']['bursting_data']['bursting_summary_data']['cov_IBI']
        target = cov_IBI_target['target']
        min_cov_IBI = cov_IBI_target['min']
        max_cov_IBI = cov_IBI_target['max']
        weight = cov_IBI_target['weight']
        maxFitness = kwargs['maxFitness']

        cov_IBI = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['cov_IBI']
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

def fit_mean_Burst_Peak(**kwargs):
    try:
        print('Calculating mean_Burst_Peak fitness...')

        mean_Burst_Peak_target = kwargs['targets']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
        target = mean_Burst_Peak_target['target']
        min_mean_Burst_Peak = mean_Burst_Peak_target['min']
        max_mean_Burst_Peak = mean_Burst_Peak_target['max']
        weight = mean_Burst_Peak_target['weight']
        maxFitness = kwargs['maxFitness']

        mean_Burst_Peak = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['mean_Burst_Peak']
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

def fit_cov_Burst_Peak(**kwargs):
    try:
        print('Calculating cov_Burst_Peak fitness...')

        cov_Burst_Peak_target = kwargs['targets']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
        target = cov_Burst_Peak_target['target']
        min_cov_Burst_Peak = cov_Burst_Peak_target['min']
        max_cov_Burst_Peak = cov_Burst_Peak_target['max']
        weight = cov_Burst_Peak_target['weight']
        maxFitness = kwargs['maxFitness']

        cov_Burst_Peak = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['cov_Burst_Peak']
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

def fit_fano_factor(**kwargs):
    try:
        print('Calculating fano_factor fitness...')

        fano_factor_target = kwargs['targets']['bursting_data']['bursting_summary_data']['fano_factor']
        target = fano_factor_target['target']
        min_fano_factor = fano_factor_target['min']
        max_fano_factor = fano_factor_target['max']
        weight = fano_factor_target['weight']
        maxFitness = kwargs['maxFitness']

        fano_factor = kwargs['network_metrics']['bursting_data']['bursting_summary_data']['fano_factor']
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

'''dealbreakers'''
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

'''helper functions'''
def save_fitness_results(output_path, fitnessResults):
    with open(output_path, 'w') as f:
        json.dump(fitnessResults, f, indent=4)
    print(f'fitness results saved to {output_path}')

def get_fitness(data_source, **kwargs):
    '''Main fitness calculation function'''
    fitnessVals = {}
    if data_source == 'experimental':
        # Priority 1: Spiking Data
        fitnessVals['rate_fit'] = fit_firing_rates(**kwargs)
        fitnessVals['CoV_rate_fit'] = fit_CoV_firing_rate(**kwargs)
        fitnessVals['ISI_fit'] = fit_ISI(**kwargs)
        fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(**kwargs)
        # Priority 2: Bursting Data
        fitnessVals['baseline_fit'] = fit_baseline(**kwargs)
        fitnessVals['WithinBurstISI_fit'] = fit_WithinBurstISI(**kwargs)
        fitnessVals['CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(**kwargs)
        fitnessVals['OutsideBurstISI_fit'] = fit_OutsideBurstISI(**kwargs)
        fitnessVals['CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(**kwargs)
        fitnessVals['NetworkISI_fit'] = fit_NetworkISI(**kwargs)
        fitnessVals['CoVNetworkISI_fit'] = fit_CovNetworkISI(**kwargs)
        fitnessVals['Number_Bursts_fit'] = fit_Number_Bursts(**kwargs)
        fitnessVals['mean_IBI_fit'] = fit_mean_IBI(**kwargs)
        fitnessVals['cov_IBI_fit'] = fit_cov_IBI(**kwargs)
        fitnessVals['mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(**kwargs)
        fitnessVals['cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(**kwargs)
        fitnessVals['fano_factor_fit'] = fit_fano_factor(**kwargs)
    elif data_source == 'simulated':
        # Priority 1: Spiking Data
        fitnessVals['rate_fit'] = fit_firing_rates(simulated=True, **kwargs)
        fitnessVals['CoV_rate_fit'] = fit_CoV_firing_rate(simulated=True, **kwargs)
        fitnessVals['ISI_fit'] = fit_ISI(simulated=True, **kwargs)
        fitnessVals['CoV_ISI_fit'] = fit_CoV_ISI(simulated=True, **kwargs)
        # Priority 2: Bursting Data
        fitnessVals['mean_Burst_Rate_fit'] = fit_mean_Burst_Rate(simulated=True, **kwargs)
        fitnessVals['baseline_fit'] = fit_baseline(**kwargs)
        fitnessVals['WithinBurstISI_fit'] = fit_WithinBurstISI(simulated=True, **kwargs)
        fitnessVals['CoVWithinBurstISI_fit'] = fit_CovWithinBurstISI(simulated=True, **kwargs)
        fitnessVals['OutsideBurstISI_fit'] = fit_OutsideBurstISI(simulated=True, **kwargs)
        fitnessVals['CoVOutsideBurstISI_fit'] = fit_CovOutsideBurstISI(simulated=True, **kwargs)
        fitnessVals['NetworkISI_fit'] = fit_NetworkISI(simulated=True, **kwargs)
        fitnessVals['CoVNetworkISI_fit'] = fit_CovNetworkISI(simulated=True, **kwargs)
        fitnessVals['Number_Bursts_fit'] = fit_Number_Bursts(**kwargs)
        fitnessVals['mean_IBI_fit'] = fit_mean_IBI(**kwargs)
        fitnessVals['cov_IBI_fit'] = fit_cov_IBI(**kwargs)
        fitnessVals['mean_Burst_Peak_fit'] = fit_mean_Burst_Peak(**kwargs)
        fitnessVals['cov_Burst_Peak_fit'] = fit_cov_Burst_Peak(**kwargs)
        fitnessVals['fano_factor_fit'] = fit_fano_factor(**kwargs)

    # if any fit values are equal to 1000, change to {'fit': 1000}
    for key in fitnessVals:
        if fitnessVals[key] == 1000:
            fitnessVals[key] = {'fit': 1000}
    average_fitness = np.mean([fitnessVals[key]['fit'] for key in fitnessVals])
    return average_fitness, fitnessVals

def handle_existing_fitness(fitness_save_path):
    if os.path.exists(fitness_save_path):
        with open(fitness_save_path, 'r') as f:
            fitnessResults = json.load(f)
        average_fitness = fitnessResults['average_fitness']
        print(f'Fitness results already exist: {average_fitness}')
        return average_fitness
    return None

# def handle_optimizing_mode(simData, kwargs):
#     """Handles the logic for 'optimizing' mode."""
#     if simData is not None:
#         kwargs['source'] = 'simulated'
#         import batch_scripts.temp_user_args as temp_user_args
#         recalculate_fitness = temp_user_args.USER_recalculate_fitness

#         recalculate_fitness = True  #TODO: remove this later
#         if not recalculate_fitness:
#             from batch_scripts.extract_simulated_data import get_candidate_and_job_path_from_call_stack
#             candidate_path, job_path = get_candidate_and_job_path_from_call_stack()
#             fitness_save_path = f'{candidate_path}_fitness.json'
#             existing_fitness = handle_existing_fitness(fitness_save_path)
#             if existing_fitness is not None:
#                 return existing_fitness, kwargs

#         kwargs = retrieve_sim_data_from_call_stack(simData, **kwargs)
#         error, kwargs = calculate_network_metrics(kwargs)
#         return error, kwargs
#     return None, kwargs

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
    
'''main functions'''
def calculate_network_metrics(kwargs):
    print('Calculating network activity metrics...')
    from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import get_simulated_network_activity_metrics
    #from RBS_network_models.utils.analysis_helper import get_simulated_network_activity_metrics
    #from RBS_network_models.developing.utils.analysis_helper import get_simulated_network_activity_metrics
    #from modules.analysis.analyze_network_activity import get_simulated_network_activity_metrics
    #from _external.RBS_network_simulation_optimization_tools.modules.analysis.analyze_network_activity import get_simulated_network_activity_metrics
    # if 'conv_params' not in kwargs or kwargs['conv_params'] is None: #TODO: remove this later
    #     convolution_params = import_module_from_path('/pscratch/sd/a/adammwea/workspace'
    #                                                 '/RBS_neuronal_network_models/optimization_projects'
    #                                                 '/CDKL5_DIV21/_config/convolution_params/241202_convolution_params.py')
    #     conv_params = convolution_params.conv_params
    #     kwargs['conv_params'] = conv_params
    network_metrics = get_simulated_network_activity_metrics(**kwargs)
    # HACK
    #mega_conv_params = kwargs['conv_params'].copy()
    #mega_conv_params['binSize'] *= 5
    #mega_conv_params['gaussianSigma'] *= 15
    #kwargs['conv_params'] = mega_conv_params
    #mega_metrics = get_simulated_network_activity_metrics(**kwargs)
    # HACK
    
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
    
'''parallelized fitness calculation'''
import subprocess
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