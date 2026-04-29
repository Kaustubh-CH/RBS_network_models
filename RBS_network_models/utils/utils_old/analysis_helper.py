#from modules.analysis_functions.network_activity_analysis import measure_network_activity
import numpy as np
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
import matplotlib.pyplot as plt
#import workspace.RBS_network_simulation_optimization_tools.external.MEA_Analysis.IPNAnalysis.helper_functions as helper
#from MEAProcessingLibrary import helper_functions as helper
#from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import *
#from DIV21.utils.sim_helper import *
#from RBS_network_models.developing.utils.sim_helper import *
from .sim_helper import *


'''deprecated fuctions'''
def convolve_network_activity(rasterData, binSize=None, gaussianSigma=None, thresholdBurst=None, min_peak_distance=None):

    conv_params = init_convolution_params(binSize, gaussianSigma, thresholdBurst, min_peak_distance)
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    thresholdBurst = conv_params['thresholdBurst']
    min_peak_distance = conv_params['min_peak_distance']
    
    relativeSpikeTimes = np.array(rasterData['spkTimes'])
    relativeSpikeTimes -= relativeSpikeTimes[0]
    
    timeVector = np.arange(0, max(relativeSpikeTimes) + binSize, binSize)
    binnedTimes, _ = np.histogram(relativeSpikeTimes, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    spike_counts_at_time = convolve(binnedTimes, kernel, mode='same') / binSize

    rmsSpikeCounts = np.sqrt(np.mean(spike_counts_at_time**2))
    peaks, properties = find_peaks(spike_counts_at_time, height=thresholdBurst * rmsSpikeCounts, distance=min_peak_distance)

    burstPeakTimes = timeVector[peaks]
    burstPeakValues = properties['peak_heights']

    return burstPeakTimes, burstPeakValues, spike_counts_at_time, timeVector, rmsSpikeCounts

def _analyze_network_activity(spike_times):
    #try: network_data = get_network_data(spike_times)
    try: get_network_data(spike_times)
    except Exception as e:
        print(f'Error calculating network activity metrics: {e}')
        pass
    
    print('Network Activity Metrics Calculated')
    #return network_data

'''Initialization Functions'''
def init_convolution_params(binSize=None, gaussianSigma=None, thresholdBurst=None, min_peak_distance=None):
    try:
        #from simulate._config_files.convolution_params import conv_params
        conv_params = import_module_from_path('/pscratch/sd/a/adammwea/workspace/'
                                              'RBS_neuronal_network_models/optimization_projects'
                                              '/CDKL5_DIV21/_config/convolution_params/241202_convolution_params.py')
        conv_params = conv_params.conv_params
        #from optimization_projects.CDKL5_DIV21._config.convolution_params import conv_params
        conv_params_temp = conv_params
        conv_params_temp['binSize'] = binSize if binSize is not None else conv_params['binSize']
        conv_params_temp['gaussianSigma'] = gaussianSigma if gaussianSigma is not None else conv_params['gaussianSigma']
        conv_params_temp['thresholdBurst'] = thresholdBurst if thresholdBurst is not None else conv_params['thresholdBurst']
        conv_params_temp['min_peak_distance'] = min_peak_distance if min_peak_distance is not None else conv_params['min_peak_distance']
    except:
        conv_params_temp = {
            'binSize': 0.1 if binSize is None else binSize,
            'gaussianSigma': 0.15 if gaussianSigma is None else gaussianSigma,
            'thresholdBurst': 1.0 if thresholdBurst is None else thresholdBurst,
            'min_peak_distance':1 if min_peak_distance is None else min_peak_distance,
        }
    
    conv_params = conv_params_temp
    return conv_params


'''Spiking Analysis Functions'''
def get_spiking_stats_by_unit(spike_times_by_unit):
    spiking_data_by_unit = {}
    # if 'E_Gids' in network_data['simulated_data']:
    #     E_Gids = network_data['simulated_data']['E_Gids']
    for unit, spike_times in spike_times_by_unit.items():
        isi = np.diff(spike_times)
        meanISI = np.mean(isi)
        #CoV_ISI = np.std(isi) / meanISI
        CoV_ISI = np.cov(isi) 
        spiking_data_by_unit[unit] = {
            #'pop': 'E' if unit in E_Gids else 'I',
            'meanISI': meanISI,
            'CoV_ISI': CoV_ISI,
            'spike_times': spike_times,
        }
    return spiking_data_by_unit

def get_spiking_summary_data(spiking_data_by_unit):
    meanISI_E = np.mean([unit_stats['meanISI'] for unit_stats in spiking_data_by_unit.values() if unit_stats['meanISI'] is not None])
    CoV_ISI_E = np.mean([unit_stats['CoV_ISI'] for unit_stats in spiking_data_by_unit.values() if unit_stats['CoV_ISI'] is not None])
    return meanISI_E, CoV_ISI_E

'''Bursting Analysis Functions'''
def convolve_signal_get_baseline(spike_times, binSize=None, gaussianSigma=None):  
        
    #init convolution parameters
    conv_params = init_convolution_params(binSize, gaussianSigma)
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    
    #convolve signal and get baseline
    timeVector = np.arange(0, max(spike_times) + binSize, binSize)
    binnedTimes, _ = np.histogram(spike_times, bins=timeVector)
    binnedTimes = np.append(binnedTimes, 0)

    kernelRange = np.arange(-3*gaussianSigma, 3*gaussianSigma + binSize, binSize)
    kernel = norm.pdf(kernelRange, 0, gaussianSigma)
    kernel *= binSize
    spike_counts_at_time = convolve(binnedTimes, kernel, mode='same') / binSize
    
    baseline = np.mean(spike_counts_at_time)

    return baseline


''' Use Functions from MEA Analysis Repository to do Bursting Analysis'''
#import norm

import numpy as np

# #from MEAProcessingLibrary.detect_burst_statistics import detect_bursts_statistics
# from MEA_Analysis.MEAProcessingLibrary.detect_burst_statistics import detect_bursts_statistics
# #import numpy as np

'''Main Functions to be used - Warpper Functions for Experimental and Simulated Cases'''

'''Simulated Data Functions'''


#nvm i dont actually need a seperate function for this
def plot_simulated_network_activity(simData=None, popData=None, **kwargs):
    import time
    
    #candidate_label = kwargs.get('candidate_label', None)
    print('') #for formatting
    print('Calculating Network Activity Metrics for Simulated Data...')
    start_time = time.time()
    #this part should be useful for fitness during simulation
    if simData is None:
        try: 
            simData = sim.simData
            print('Using simData from netpyne.sim.simData')
        except:
            print('No simData provided or found in netpyne.sim.simData')
            return None

    #initialize network_data
    #network_data = init_network_data_dict()
    network_data = init_network_data_dict()
    rasterData = simData.copy()
    
    #check if rasterData['spkt'] is empty
    if len(rasterData['spkt']) == 0:
        print('No spike times found in rasterData')
        return None
    
    #convert time to seconds - get initially available data
    spike_times = np.array(rasterData['spkt']) / 1000
    timeVector = np.array(rasterData['t']) / 1000
    spike_times_by_unit = {int(i): spike_times[rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])} #mea_analysis_pipeline.py expects spike_times as dictionary    
    
    #extract spiking metrics from simulated data
    try: 
        extract_metrics_from_simulated_data(spike_times, timeVector, spike_times_by_unit, rasterData, popData, **kwargs)
    except Exception as e:
        print(f'Error extracting metrics from simulated data: {e}')
        pass
    
    #extract bursting metrics from simulated data (but this one works for both simulated and experimental data)
    try: 
        extract_bursting_activity_data(spike_times, spike_times_by_unit)
    except Exception as e:
        print(f'Error calculating bursting activity: {e}')
        pass
    
    #return network_data
    
    def convert_single_element_arrays(data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = convert_single_element_arrays(value)
        elif isinstance(data, np.ndarray) and data.size == 1:
            return data.item()
        return data

    network_data = convert_single_element_arrays(network_data)    
    print('Network Activity Metrics Calculated and Extracted!')
    print(f'Elapsed time: {time.time() - start_time} seconds')
    print('') #for formatting    
    return network_data



def get_min_fr(results):
    min_fr = float('inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        recording_object = result['recording_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 1:
                    well_id = f'well{str(0).zfill(2)}{stream_select}'
                    recording_segment = recording_object[well_id]['recording_segments'][0]
                    duration = recording_segment.get_total_duration()  # seconds
                    fr = len(spike_train) / duration  # spikes per second
                    if not np.isnan(fr) and not np.isinf(fr) and fr < min_fr:
                        min_fr = fr
    return min_fr if min_fr != float('inf') else None

def get_max_fr(results):
    max_fr = float('-inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        recording_object = result['recording_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 1:
                    well_id = f'well{str(0).zfill(2)}{stream_select}'
                    recording_segment = recording_object[well_id]['recording_segments'][0]
                    duration = recording_segment.get_total_duration()  # seconds
                    fr = len(spike_train) / duration  # spikes per second
                    if not np.isnan(fr) and not np.isinf(fr) and fr > max_fr:
                        max_fr = fr
    return max_fr if max_fr != float('-inf') else None

def get_min_CoV_fr(results):
    min_CoV_fr = float('inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        recording_object = result['recording_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 2:
                    well_id = f'well{str(0).zfill(2)}{stream_select}'
                    recording_segment = recording_object[well_id]['recording_segments'][0]
                    duration = recording_segment.get_total_duration()  # seconds
                    fr = len(spike_train) / duration  # spikes per second
                    isi = np.diff(spike_train)
                    #assert isi length is greater than 1
                    if len(isi) <= 1: continue
                    CoV = np.std(isi) / np.mean(isi)
                    # if CoV == 0:
                    #     print('CoV is 0')
                    if not np.isnan(CoV) and not np.isinf(CoV) and CoV < min_CoV_fr:
                        min_CoV_fr = CoV
    return min_CoV_fr if min_CoV_fr != float('inf') else None

def get_max_CoV_fr(results):
    max_CoV_fr = float('-inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        recording_object = result['recording_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 1:
                    well_id = f'well{str(0).zfill(2)}{stream_select}'
                    recording_segment = recording_object[well_id]['recording_segments'][0]
                    duration = recording_segment.get_total_duration()  # seconds
                    fr = len(spike_train) / duration  # spikes per second
                    isi = np.diff(spike_train)
                    CoV = np.std(isi) / np.mean(isi)
                    if not np.isnan(CoV) and not np.isinf(CoV) and CoV > max_CoV_fr:
                        max_CoV_fr = CoV
    return max_CoV_fr if max_CoV_fr != float('-inf') else None

def get_min_mean_isi(results):
    min_mean_isi = float('inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 2:
                    isi = np.diff(spike_train)
                    mean_isi = np.mean(isi)
                    if not np.isnan(mean_isi) and not np.isinf(mean_isi) and mean_isi < min_mean_isi:
                        min_mean_isi = mean_isi
    return min_mean_isi if min_mean_isi != float('inf') else None

def get_max_mean_isi(results): 
    max_mean_isi = float('-inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 1:
                    isi = np.diff(spike_train)
                    mean_isi = np.mean(isi)
                    if not np.isnan(mean_isi) and not np.isinf(mean_isi) and mean_isi > max_mean_isi:
                        max_mean_isi = mean_isi
    return max_mean_isi if max_mean_isi != float('-inf') else None

def get_min_CoV_isi(results):
    min_CoV_isi = float('inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 2:
                    isi = np.diff(spike_train)
                    
                    CoV = np.std(isi) / np.mean(isi)
                    if not np.isnan(CoV) and not np.isinf(CoV) and CoV < min_CoV_isi:
                        min_CoV_isi = CoV
    return min_CoV_isi if min_CoV_isi != float('inf') else None

def get_max_CoV_isi(results):
    max_CoV_isi = float('-inf')
    for recording_path, result in results.items():
        sorting_object_list = result['sorting_objects']
        for sorting_object in sorting_object_list:
            units = sorting_object.get_unit_ids()
            for unit in units:
                spike_train = sorting_object.get_unit_spike_train(unit)
                if len(spike_train) > 1:
                    isi = np.diff(spike_train)
                    CoV = np.std(isi) / np.mean(isi)
                    if not np.isnan(CoV) and not np.isinf(CoV) and CoV > max_CoV_isi:
                        max_CoV_isi = CoV
    return max_CoV_isi if max_CoV_isi != float('-inf') else None