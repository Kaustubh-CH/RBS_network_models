'''Setup Python environment for running the script'''
from pprint import pprint
# import setup_environment
# setup_environment.set_pythonpath()

'''Import necessary modules'''
#from modules.analysis.calculate_fitness import fitnessFunc
from netpyne import sim
import dill
import os
import time

def extract_data_of_interest_from_sim(data_path, simData=None, load_extract=True):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            sim.load(data_path)
            simData = sim.allSimData.copy()
            cellData = sim.net.allCells.copy()
            popData = sim.net.allPops.copy()
            simCfg = sim.cfg.__dict__.copy()
            netParams = sim.net.params.__dict__.copy()
            
            sim.clearAll()
            
            extracted_data = {
                'simData': simData,
                'cellData': cellData,
                'popData': popData,
                'simCfg': simCfg,
                'netParams': netParams,
            }
            
            return extracted_data
        #except EOFError as e:
        except Exception as e:
            print(f"Error loading data (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait for a second before retrying
            else:
                raise e

def find_simulation_files(target_dir):
    simulation_files = {}
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('_data.json'):
                data_path = os.path.join(root, file)
                cfg_path = os.path.join(root, file.replace('_data.json', '_cfg.json'))
                fitness_path = os.path.join(root, file.replace('_data.json', '_Fitness.json'))
                if not os.path.exists(fitness_path):
                    fitness_path = os.path.join(root, file.replace('_data.json', '_fitness.json'))
                
                if os.path.exists(cfg_path) and os.path.exists(data_path):
                    simulation_files[file] = {
                        'data_path': os.path.abspath(data_path),
                        'cfg_path': os.path.abspath(cfg_path),
                        'fitness_path': os.path.abspath(fitness_path) if os.path.exists(fitness_path) else None,
                    }
                else:
                    print(f'File {cfg_path} or {data_path} not found.')
    return simulation_files

# def process_simulation_file(file_name, file_paths, selection=None):
#     import re
#     if selection and selection not in file_paths['data_path']:
#         print(f'Skipping {file_name} because it does not match the selection criteria.')
#         return None, None, None
    
#     #generate output path. Replace 'Fitness' with 'fitness' in the file name. Also modify path to save in the same location as this script
#     fitness_save_name = os.path.basename(file_paths['fitness_path'])
#     fitness_save_name = re.sub(r'Fitness', 'fitness', fitness_save_name)
#     fitness_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fitness_save_name)
    
#     extracted_data = extract_data_of_interest_from_sim(file_paths['data_path'])
#     kwargs = {
#         'simData': extracted_data['simData'],
#         'cellData': extracted_data['cellData'],
#         'popData': extracted_data['popData'],
#         'simCfg': extracted_data['simCfg'],
#         'netParams': extracted_data['netParams'],
#         'simLabel': extracted_data['simCfg']['simLabel'],
#         'data_file_path': file_paths['data_path'],
#         'cfg_file_path': file_paths['cfg_path'],
#         'fitness_file_path': file_paths['fitness_path'], #input
#         #replace 'Fitness' with 'fitness' in the file name
#         'fitness_save_path': fitness_save_path, #output
#         #'fitnessFuncArgs': fitnessFuncArgs_dep,
#     }
    
#     return kwargs

import inspect

def find_variable_in_call_stack(var_name):
    """
    Traverses the call stack to find a specified variable in the local variables.
    Returns the variable's value if found, or `None` if not found.
    """
    stack = inspect.stack()
    for frame_info in stack:
        local_vars = frame_info.frame.f_locals
        if var_name in local_vars:
            return local_vars[var_name]
    # If the variable is not found in any frame
    return None

def find_job_path_in_call_stack():
    """
    Traverses the call stack to find `jobPath` in the local variables.
    Returns the `jobPath` if found, or `None` if not found.
    """
    return find_variable_in_call_stack('jobPath')

def get_candidate_and_job_path_from_call_stack():
    """
    Traverses the call stack to find `jobPath`, `candidate_index`, and `candidate_label`.
    Returns the `jobPath`, `candidate_label` if all are found, or raises an error.
    """
    # Find variables in the stack
    jobPath = find_job_path_in_call_stack()
    candidate_index = find_variable_in_call_stack('candidate_index')
    candidate_label = find_variable_in_call_stack('_')  # Assuming `_` is the candidate label

    # Raise an error if any of the required variables are not found
    if jobPath is None:
        raise ValueError("`jobPath` not found in the call stack.")
    if candidate_index is None:
        raise ValueError("`candidate_index` not found in the call stack.")
    if candidate_label is None:
        raise ValueError("`candidate_label` (variable '_') not found in the call stack.")

    return jobPath, candidate_label

def get_candidate_and_job_path_from_call_stack_dep(**kwargs):
    import inspect
    stack = inspect.stack()
    two_levels_up = stack[2].frame
    two_levels_up_locals = two_levels_up.f_locals
    jobPath = two_levels_up_locals['jobPath']
    candidate_index = two_levels_up_locals['candidate_index']
    candidate_label = two_levels_up_locals['_']
    return jobPath, candidate_label
    
def retrieve_sim_data_from_call_stack(simData, **kwargs):
    ## retreive data from 2 call stack levels up
    import inspect
    
    stack = inspect.stack()
    jobPath = None
    candidate_index = None
    candidate_label = None
    
    for frame_info in stack:
        frame_locals = frame_info.frame.f_locals
        if 'jobPath' in frame_locals and '_' in frame_locals:
            jobPath = frame_locals['jobPath']
            candidate_index = frame_locals['candidate_index']
            candidate_label = frame_locals['_']
            break
    
    # Continue with the rest of the code using jobPath, candidate_index, and candidate_label
    
    #data_file_path = f'{jobPath}_data.json'
    data_file_path = f'{jobPath}_data.pkl'
    #check if the file exists
    assert os.path.exists(data_file_path), f'File {data_file_path} not found.'
    
    cfg_file_path = f'{jobPath}_cfg.json'
    #check if the file exists
    assert os.path.exists(cfg_file_path), f'File {cfg_file_path} not found.'
    
    fitness_save_path = f'{jobPath}_fitness.json' #this file should not exist yet
    
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
    
    return kwargs

#def prepare_simulation_data(simData, **kwargs):
    
    
    #kwargs = retrieve_data_from_call_stack(simData, **kwargs)
    # Set the target directory to the directory containing the simulation output
    # target_dir = os.path.abspath('simulation_output/240808_Run1_testing_data/gen_0')
    # simulation_files = find_simulation_files(target_dir)
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # bad_files_path = os.path.join(script_dir, 'bad_files.txt')
    
    # bad_files = []
    # selection = None  # Set to a specific selection if needed
    
    # for file_name, file_paths in simulation_files.items():
    #     try:
    #         average_fitness, avg_scaled_fitness, fitnessVals = process_simulation_file(file_name, file_paths, selection)
    #         if average_fitness is not None:
    #             print(f'Average Fitness for {file_name}: {average_fitness}')
    #             print(f'Average Scaled Fitness for {file_name}: {avg_scaled_fitness}')
    #             print(f'Fitness Values for {file_name}: {fitnessVals}')
    #     except Exception as e:
    #         print(f'Error in processing {file_name}: {e}')
    #         bad_files.append(file_name)
    

    # with open(bad_files_path, 'w') as f:
    #     for file in bad_files:
    #         f.write(f'{file}\n')

# if __name__ == "__main__":
#     main()
