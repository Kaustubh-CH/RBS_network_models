#from modules.analysis_functions.network_activity_analysis import measure_network_activity
import numpy as np
from scipy.signal import convolve, find_peaks
from scipy.stats import norm
import matplotlib.pyplot as plt
#import external.MEA_Analysis.IPNAnalysis.helper_functions as helper

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
        from simulate._config_files.convolution_params import conv_params
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

def init_burst_analysis_params(isi_threshold=None):
    try:
        from simulate._config_files.burst_analysis_params import burst_analysis_params
        burst_analysis_params_temp = burst_analysis_params
        burst_analysis_params_temp['isi_threshold'] = isi_threshold if isi_threshold is not None else burst_analysis_params['isi_threshold']
    except:
        burst_analysis_params_temp = {
            'isi_threshold': 0.1 if isi_threshold is None else isi_threshold,
        }
    
    burst_analysis_params = burst_analysis_params_temp
    return burst_analysis_params

def init_network_data_dict():
    empty_network_data = {
        #General Data
        'source': None, # 'simulated' or 'experimental'
        'timeVector': None,
        
        #Simulated Data
        'simulated_data': {
            'soma_voltage': None,
            'E_Gids': None,
            'I_Gids': None,
            'MeanFireRate_E': None,
            'CoVFireRate_E': None,
            'MeanFireRate_I': None,
            'CoVFireRate_I': None,
            'MeanISI_E': None,
            'MeanISI_I': None,
            'CoV_ISI_E': None,
            'CoV_ISI_I': None,
            'spiking_data_by_unit': None, 
        },
        
        #Spiking Data
        'spiking_data': {
            'spike_times': None,
            'spiking_summary_data': {
                #'spike_times': None,
                'MeanFireRate': None,
                'CoVFireRate': None,
                'MeanISI': None,
                'CoV_ISI': None,         
            },
            'spiking_times_by_unit': None,
            'spiking_data_by_unit': None,
        },
        
        #Bursting Data
        'bursting_data': {
            'bursting_summary_data': {
                'baseline': None,
                'MeanWithinBurstISI': None,
                'CovWithinBurstISI': None,
                'MeanOutsideBurstISI': None,
                'CoVOutsideBurstISI': None,
                'MeanNetworkISI': None,
                'CoVNetworkISI': None,
                'NumUnits': None,
                'Number_Bursts': None,
                'mean_IBI': None,
                'cov_IBI': None,
                'mean_Burst_Peak': None,
                'cov_Burst_Peak': None,
                'fano_factor': None,
            },
            'bursting_data_by_unit': None,
        }
    }
    global network_data
    network_data = empty_network_data
    return empty_network_data
    
    ## Keep for reference
    
            # return {
        #     # 'burstPeakValues': None,
        #     # 'IBIs': None,
        #     # 'baseline': None,
        #     # 'peakFreq': None,
        #     # 'firingRate': None,
        #     # 'burstPeakTimes': None,
        #     # 'timeVector': None,
        #     # 'threshold': None,
            
        #     'Number_Bursts': None,
        #     'mean_IBI': None,
        #     'cov_IBI': None,
        #     'mean_Burst_Peak': None,
        #     'cov_Burst_Peak': None,
        #     'fano_factor': None,
        #     'MeanWithinBurstISI': None,
        #     'CoVWithinBurstISI': None,
        #     'MeanOutsideBurstISI': None,
        #     'CoVOutsideBurstISI': None,
        #     'MeanNetworkISI': None,
        #     'CoVNetworkISI': None,
        #     'NumUnits': None,
        #     #'fileName': None
        # }

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
def analyze_convolved_spiking_signal(spike_times, spike_times_by_unit, min_peak_distance, binSize, gaussianSigma, thresholdBurst):
    fig, ax = plt.subplots()
    #spike_times = {i: rasterData['spkt'][rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])} #plot_network_activity expects spike_times as dictionary
    #spike_times_dict = {int(i): spike_times[spike_times['spkid'] == i] for i in np.unique(spike_times['spkid'])}
    try:
        ax, convolved_signal_metrics = helper.plot_network_activity(ax, spike_times_by_unit, min_peak_distance=min_peak_distance, binSize=binSize, gaussianSigma=gaussianSigma, thresholdBurst=thresholdBurst)
        # network_data['bursting_data']['bursting_summary_data']['mean_Burst_Peak'] = convolved_signal_metrics['mean_Burst_Peak']
        # network_data['bursting_data']['bursting_summary_data']['cov_Burst_Peak'] = convolved_signal_metrics['cov_Burst_Peak']
        # network_data['bursting_data']['bursting_summary_data']['fano_factor'] = convolved_signal_metrics['fano_factor']
        # network_data['bursting_data']['bursting_summary_data']['mean_IBI'] = convolved_signal_metrics['mean_IBI']
        # network_data['bursting_data']['bursting_summary_data']['cov_IBI'] = convolved_signal_metrics['cov_IBI']
        # network_data['bursting_data']['bursting_summary_data']['Number_Bursts'] = convolved_signal_metrics['Number_Bursts']    
        # network_data['bursting_data']['bursting_summary_data']['baseline'] = convolve_signal_get_baseline(spike_times)
        
        #pull y data out of ax
        y_data = ax.lines[0].get_ydata()
        signal = y_data
        baseline = np.mean(signal)
        
        
        convolved_data = {
            'mean_Burst_Peak': convolved_signal_metrics['mean_Burst_Peak'],
            'cov_Burst_Peak': convolved_signal_metrics['cov_Burst_Peak'],
            'fano_factor': convolved_signal_metrics['fano_factor'],
            'mean_IBI': convolved_signal_metrics['mean_IBI'],
            'cov_IBI': convolved_signal_metrics['cov_IBI'],
            'Number_Bursts': convolved_signal_metrics['Number_Bursts'],
            #'baseline': convolve_signal_get_baseline(spike_times, binSize=binSize, gaussianSigma=gaussianSigma),
            'baseline': baseline,
            #'fig': fig,
            'ax': ax,
        }
        return convolved_data
    except Exception as e:
        #set all convolved data to nan
        print(f'Error analyzing convolved spiking signal: {e}')
        print(f'might be cause by all spike times being in the same bin')
        convolved_data = {
            'mean_Burst_Peak': np.nan,
            'cov_Burst_Peak': np.nan,
            'fano_factor': np.nan,
            'mean_IBI': np.nan,
            'cov_IBI': np.nan,
            'Number_Bursts': np.nan,
            'baseline': np.nan,
        }
        return convolved_data

def analyze_bursting_activity(spike_times, spike_times_by_unit, isi_threshold):
    '''Slightly modified version of the code from mea_analysis_pipeline.py'''
    #spike_times = {int(i): rasterData['spkt'][rasterData['spkid'] == i] for i in np.unique(rasterData['spkid'])}
    
    # convert spike_times to dictionary
    #spike_times_dict = {int(i): spike_times[spike_times['spkid'] == i] for i in np.unique(spike_times['spkid'])}
    #spike_times = spike_times_dict
    spike_times = spike_times_by_unit
    
    burst_statistics = helper.detect_bursts_statistics(spike_times, isi_threshold=isi_threshold)
    bursts_by_unit = [unit_stats['bursts'] for unit_stats in burst_statistics.values()]
    
    all_isis_within_bursts = np.concatenate([stats['isis_within_bursts'] for stats in burst_statistics.values() if stats['isis_within_bursts'].size > 0])
    all_isis_outside_bursts = np.concatenate([stats['isis_outside_bursts'] for stats in burst_statistics.values() if stats['isis_outside_bursts'].size > 0])
    all_isis = np.concatenate([stats['isis_all'] for stats in burst_statistics.values() if stats['isis_all'].size > 0])

    mean_isi_within_combined = np.mean(all_isis_within_bursts) if all_isis_within_bursts.size > 0 else np.nan
    cov_isi_within_combined = np.cov(all_isis_within_bursts) if all_isis_within_bursts.size > 0 else np.nan

    mean_isi_outside_combined = np.mean(all_isis_outside_bursts) if all_isis_outside_bursts.size > 0 else np.nan
    cov_isi_outside_combined = np.cov(all_isis_outside_bursts) if all_isis_outside_bursts.size > 0 else np.nan

    mean_isi_all_combined = np.mean(all_isis) if all_isis.size > 0 else np.nan
    cov_isi_all_combined = np.cov(all_isis) if all_isis.size > 0 else np.nan

    bursting_summary_data = {}
    bursting_summary_data['MeanWithinBurstISI'] = mean_isi_within_combined
    bursting_summary_data['CoVWithinBurstISI'] = cov_isi_within_combined
    bursting_summary_data['MeanOutsideBurstISI'] = mean_isi_outside_combined   
    bursting_summary_data['CoVOutsideBurstISI'] = cov_isi_outside_combined
    bursting_summary_data['MeanNetworkISI'] = mean_isi_all_combined
    bursting_summary_data['CoVNetworkISI'] = cov_isi_all_combined
    bursting_summary_data['NumUnits'] = len(spike_times)
    bursting_summary_data['Number_Bursts'] = sum(len(unit_stats['bursts']) for unit_stats in burst_statistics.values())
    bursting_summary_data['mean_IBI'] = np.mean(all_isis) if all_isis.size > 0 else np.nan
    bursting_summary_data['cov_IBI'] = np.cov(all_isis) if all_isis.size > 0 else np.nan
    
    bursting_data = {
        'bursting_summary_data': bursting_summary_data,
        'bursting_data_by_unit': burst_statistics,
    }

    return bursting_data

def extract_bursting_activity_data(spike_times, spike_times_by_unit, **kwargs):

    #conv_params = init_convolution_params()
    conv_params = kwargs.get('conv_params', None)
    assert conv_params is not None, 'No convolution parameters provided'
    
    #extract convolution parameters
    binSize = conv_params['binSize']
    gaussianSigma = conv_params['gaussianSigma']
    thresholdBurst = conv_params['thresholdBurst']
    min_peak_distance = conv_params['min_peak_distance']
    
    #unit-wise burst analysis
    burst_analysis_params = init_burst_analysis_params()
    isi_threshold = burst_analysis_params['isi_threshold']
    
    convolved_data = analyze_convolved_spiking_signal(spike_times, spike_times_by_unit, min_peak_distance, binSize, gaussianSigma, thresholdBurst)
    bursting_data = analyze_bursting_activity(spike_times, spike_times_by_unit, isi_threshold)
    
    #add convolved data to the bursting summary data
    bursting_summary_data = bursting_data['bursting_summary_data']
    for key in convolved_data.keys():
        bursting_summary_data[key] = convolved_data[key]
    bursting_data['bursting_summary_data'] = bursting_summary_data
    
    # Verify that any single-element array is converted to a scalar
    for key in bursting_data['bursting_summary_data'].keys():
        value = bursting_data['bursting_summary_data'][key]
        if isinstance(value, np.ndarray) and value.size == 1:
            bursting_data['bursting_summary_data'][key] = value.item()  # .item() fetches the scalar value directly
    
    network_data['bursting_data'] = bursting_data
    # network_data = {
    #     'bursting_data': bursting_data,
    # }

    #return network_data

'''Main Functions to be used - Warpper Functions for Experimental and Simulated Cases'''

'''Simulated Data Functions'''

def get_CoV_fr_simulated(spike_times, total_duration, window_size=1.0): 
    """
    Calculate the Coefficient of Variation of Firing Rate (CoV FR) over time windows.

    Parameters:
    - spike_times: List or array of spike times for a single unit (in seconds).
    - total_duration: Total duration of the simulation (in seconds).
    - window_size: Size of the time window for calculating firing rates (default: 1.0 second).

    Returns:
    - CoV_fr: Coefficient of Variation of Firing Rate (float).
    """
    #TODO import window from convolution_params?
    if len(spike_times) == 0:
        # If no spikes, CoV FR is undefined (return NaN)
        return np.nan

    # Divide the total duration into non-overlapping time windows
    #window_size = 0.1
    total_duration = total_duration[-1]
    num_windows = int(total_duration / window_size)
    window_edges = np.linspace(0, total_duration, num_windows + 1)

    # Count spikes in each window
    spike_counts = np.histogram(spike_times, bins=window_edges)[0]

    # Convert spike counts to firing rates (spikes per second)
    firing_rates = spike_counts / window_size

    # Compute CoV FR: standard deviation divided by mean
    mean_fr = np.mean(firing_rates)
    std_fr = np.std(firing_rates)

    # Avoid division by zero if mean_fr is 0
    if mean_fr == 0:
        return np.nan

    CoV_fr = std_fr / mean_fr
    return CoV_fr

def calculate_simulated_network_activity_metrics(spike_times_by_unit):
    # Initialize spiking data dictionary
    network_data['simulated_data']['spiking_data_by_unit'] = {}
    
    # Iterate over each unit to calculate individual metrics
    for unit, spike_times in spike_times_by_unit.items():
        if len(spike_times) < 2:
            meanISI = np.nan
            CoV_ISI = np.nan
        else:
            isi = np.diff(spike_times)
            meanISI = np.mean(isi)
            CoV_ISI = np.std(isi) / meanISI  # Correct CoV calculation for ISI
        
        fr = len(spike_times) / network_data['timeVector'][-1]  # Firing rate (spikes per second)
        
        # Calculate CoV FR using the get_CoV_fr function with a 1-second time window
        CoV_fr = get_CoV_fr_simulated(spike_times, network_data['timeVector'])

        # Store calculated metrics for each unit
        network_data['simulated_data']['spiking_data_by_unit'][unit] = {
            'FireRate': fr,
            'CoV_fr': CoV_fr,
            'meanISI': meanISI,
            'CoV_ISI': CoV_ISI,
            'spike_times': spike_times,
        }
        
        # Determine population type
        if unit in network_data['simulated_data']['E_Gids']:
            network_data['simulated_data']['spiking_data_by_unit'][unit]['pop'] = 'E'
        elif unit in network_data['simulated_data']['I_Gids']:
            network_data['simulated_data']['spiking_data_by_unit'][unit]['pop'] = 'I'
        else:
            raise ValueError(f'Unit {unit} not found in E_Gids or I_Gids')

    # Extract E and I gids
    E_Gids = network_data['simulated_data']['E_Gids']
    I_Gids = network_data['simulated_data']['I_Gids']

    # Calculate mean and CoV metrics for excitatory and inhibitory populations
    E_CoV_ISI = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['CoV_ISI'] 
                            for unit in E_Gids if unit in network_data['simulated_data']['spiking_data_by_unit']])
    I_CoV_ISI = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['CoV_ISI'] 
                            for unit in I_Gids if unit in network_data['simulated_data']['spiking_data_by_unit']])
    MeanISI_E = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['meanISI'] 
                            for unit in E_Gids if unit in network_data['simulated_data']['spiking_data_by_unit']])
    MeanISI_I = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['meanISI'] 
                            for unit in I_Gids if unit in network_data['simulated_data']['spiking_data_by_unit']])
    CoVFiringRate_E = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['CoV_fr'] 
                                  for unit in E_Gids if unit in network_data['simulated_data']['spiking_data_by_unit']])
    CoVFiringRate_I = np.nanmean([network_data['simulated_data']['spiking_data_by_unit'][unit]['CoV_fr'] 
                                  for unit in I_Gids if unit in network_data['simulated_data']['spiking_data_by_unit']])
    
    # Add population-level metrics to the network data
    network_data['simulated_data']['CoV_ISI_E'] = E_CoV_ISI
    network_data['simulated_data']['CoV_ISI_I'] = I_CoV_ISI
    network_data['simulated_data']['MeanISI_E'] = MeanISI_E
    network_data['simulated_data']['MeanISI_I'] = MeanISI_I
    network_data['simulated_data']['CoVFireRate_E'] = CoVFiringRate_E
    network_data['simulated_data']['CoVFireRate_I'] = CoVFiringRate_I
            
def extract_metrics_from_simulated_data(spike_times, timeVector, spike_times_by_unit, rasterData, popData, **kwargs):
    
    #add immediately available data to network_data
    network_data['source'] = 'simulated'
    network_data['timeVector'] = timeVector #seconds
    network_data['spiking_data']['spike_times'] = spike_times
    network_data['simulated_data']['soma_voltage'] = rasterData['soma_voltage']
    network_data['spiking_data']['spiking_times_by_unit'] = spike_times_by_unit

    #add uniquely simulated data to network_data
    network_data['simulated_data']['E_Gids'] = popData['E']['cellGids']
    network_data['simulated_data']['I_Gids'] = popData['I']['cellGids']
    network_data['simulated_data']['MeanFireRate_E'] = rasterData['popRates']['E']
    network_data['simulated_data']['MeanFireRate_I'] = rasterData['popRates']['I']
    
    #get simulated network activity metrics
    calculate_simulated_network_activity_metrics(spike_times_by_unit)
    
    #derive remaining spiking data from simulated data
    num_E = len(network_data['simulated_data']['E_Gids'])
    num_I = len(network_data['simulated_data']['I_Gids'])
    
    #overall mean firing rate
    E_popRates = rasterData['popRates']['E']
    I_popRates = rasterData['popRates']['I']
    network_data['spiking_data']['spiking_summary_data']['MeanFireRate'] = (
        (E_popRates * num_E + I_popRates * num_I) / (num_E + num_I)
        )
    
    #overall CoV firing rate
    E_CoV = network_data['simulated_data']['CoVFireRate_E']
    I_CoV = network_data['simulated_data']['CoVFireRate_I']
    network_data['spiking_data']['spiking_summary_data']['CoVFireRate'] = (
        (E_CoV * num_E + I_CoV * num_I) / (num_E + num_I)
        )
    
    #overall mean ISI
    E_meanISI = network_data['simulated_data']['MeanISI_E']
    I_meanISI = network_data['simulated_data']['MeanISI_I']
    network_data['spiking_data']['spiking_summary_data']['MeanISI'] = (
        (E_meanISI * num_E + I_meanISI * num_I) / (num_E + num_I)
        )
    
    #overall CoV ISI
    E_CoV_ISI = network_data['simulated_data']['CoV_ISI_E']
    I_CoV_ISI = network_data['simulated_data']['CoV_ISI_I']
    network_data['spiking_data']['spiking_summary_data']['CoV_ISI'] = (
        (E_CoV_ISI * num_E + I_CoV_ISI * num_I) / (num_E + num_I)
        )
    
    #spiking data by unit - just go through the simulated version of the data but de-identify pop
    simulated_spiking_data_by_unit = network_data['simulated_data']['spiking_data_by_unit']
    spiking_data_by_unit = {}
    for unit, unit_data in simulated_spiking_data_by_unit.items():
        spiking_data_by_unit[unit] = {
            'FireRate': unit_data['FireRate'],
            'CoV_fr': unit_data['CoV_fr'],
            'meanISI': unit_data['meanISI'],
            'CoV_ISI': unit_data['CoV_ISI'],
            'spike_times': unit_data['spike_times'],
        }
    network_data['spiking_data']['spiking_data_by_unit'] = spiking_data_by_unit

def get_simulated_network_activity_metrics(simData=None, popData=None, **kwargs):
    import time
    
    #candidate_label = kwargs.get('candidate_label', None)
    print('') #for formatting
    print('Calculating Network Activity Metrics for Simulated Data...')
    start_time = time.time()
    assert simData is not None, 'No simData provided'
    #this part should be useful for fitness during simulation
    # if simData is None:
    #     try: 
    #         simData = sim.simData
    #         print('Using simData from netpyne.sim.simData')
    #     except:
    #         print('No simData provided or found in netpyne.sim.simData')
    #         return None

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
        extract_bursting_activity_data(spike_times, spike_times_by_unit, **kwargs)
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

'''Experimental Data Functions'''
def get_spike_times(recording_object, sorting_object, sampling_rate=10000):
    spike_times = []
    units = sorting_object.get_unit_ids()
    total_duration = recording_object.get_total_duration() #seconds
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) #in samples
        spike_times.extend(spike_train)
    spike_times.sort()
    spike_times = np.array(spike_times) / sampling_rate #convert to seconds
    print(f'Max Spike Time: {max(spike_times)}')
    assert max(spike_times) <= total_duration, 'Spike times are not in seconds'
    assert all(spike_time >= 0 for spike_time in spike_times), 'Spike times contain negative values'
    return spike_times

def get_time_vector(recording_object, sampling_rate=10000):
    duration = recording_object.get_total_duration() #seconds
    time_vector = np.linspace(0, duration, int(duration * sampling_rate))
    assert len(time_vector) == int(duration * sampling_rate), 'Time vector length mismatch'
    return time_vector

def get_spike_times_by_unit(sorting_object, sampling_rate=10000):
    spike_times_by_unit = {}
    units = sorting_object.get_unit_ids()
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
        assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
        spike_times_by_unit[unit] = spike_train
    return spike_times_by_unit

def get_mean_fr(recording_object, sorting_object, sampling_rate=10000):
    total_fr = 0
    unit_count = 0
    units = sorting_object.get_unit_ids()
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
        assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
        if len(spike_train) > 1:
            duration = recording_object.get_total_duration()  # seconds
            fr = len(spike_train) / duration  # spikes per second
            if not np.isnan(fr) and not np.isinf(fr):
                total_fr += fr
                unit_count += 1
    return total_fr / unit_count if unit_count > 0 else None

def get_CoV_fr_experimental(recording_object, sorting_object, sampling_rate=10000, window_size=1.0):
    """
    Calculate the Coefficient of Variation of Firing Rate (CoV FR) for experimental data over time windows.

    Parameters:
    - recording_object: A recording extractor object containing the experiment's total duration.
    - sorting_object: A sorting extractor object with spike times for each unit.
    - sampling_rate: Sampling rate of the recording (in Hz, default: 10000).
    - window_size: Size of the time window for calculating firing rates (default: 1.0 second).

    Returns:
    - CoV_fr: Coefficient of Variation of Firing Rate (float).
    """
    # Get total duration of the recording (in seconds)
    total_duration = recording_object.get_total_duration()

    # Initialize list to hold firing rates across all windows for all units
    firing_rates = []

    # Get unit IDs from the sorting object
    units = sorting_object.get_unit_ids()

    # Process each unit's spike train
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate  # Convert spike times to seconds
        assert all(spike_time >= 0 for spike_time in spike_train), f"Spike times for unit {unit} contain negative values"

        if len(spike_train) > 1:
            # Divide the total duration into non-overlapping time windows
            num_windows = int(np.ceil(total_duration / window_size))
            window_edges = np.linspace(0, total_duration, num_windows + 1)

            # Count spikes in each window for the current unit
            spike_counts = np.histogram(spike_train, bins=window_edges)[0]

            # Convert spike counts to firing rates (spikes per second) for this unit
            unit_firing_rates = spike_counts / window_size

            # Append non-NaN and non-inf firing rates to the global list
            firing_rates.extend([fr for fr in unit_firing_rates if not np.isnan(fr) and not np.isinf(fr)])

    # Calculate CoV FR if there are valid firing rates
    if len(firing_rates) > 1:
        mean_fr = np.mean(firing_rates)
        std_fr = np.std(firing_rates)
        CoV_fr = std_fr / mean_fr
        return CoV_fr
    else:
        # Return None if there are no valid firing rates
        return None


def get_mean_isi(recording_object, sorting_object, sampling_rate=10000):
    all_means = []
    units = sorting_object.get_unit_ids()
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
        assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
        if len(spike_train) > 1:
            isi = np.diff(spike_train)
            mean_isi = np.mean(isi)
            if not np.isnan(mean_isi) and not np.isinf(mean_isi):
                all_means.append(mean_isi)
    
    overall_mean = np.mean(all_means) if all_means else None
    return overall_mean if overall_mean is not None else None  # already in seconds

def get_CoV_isi(recording_object, sorting_object, sampling_rate=10000):
    all_CoVs = []
    units = sorting_object.get_unit_ids()
    for unit in units:
        spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
        assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
        if len(spike_train) > 1:
            isi = np.diff(spike_train)  # already in seconds
            if len(isi) > 1:
                CoV = np.std(isi) / np.mean(isi)
                if not np.isnan(CoV) and not np.isinf(CoV):
                    all_CoVs.append(CoV)
    
    overall_mean_CoV = np.mean(all_CoVs) if all_CoVs else None
    return overall_mean_CoV

def get_unit_fr(recording_object, sorting_object, unit, sampling_rate=10000):
    spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
    assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
    if len(spike_train) > 1:
        duration = recording_object.get_total_duration()  # seconds
        fr = len(spike_train) / duration  # spikes per second
        return fr
    else:
        return 0

def get_unit_fr_CoV(recording_object, sorting_object, unit, sampling_rate=10000):
    spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
    assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
    if len(spike_train) > 1:
        isi = np.diff(spike_train)  # inter-spike intervals
        if len(isi) > 1:
            CoV = np.std(isi) / np.mean(isi)
            return CoV
        else:
            return None
    else:
        return None
    
def get_unit_mean_isi(recording_object, sorting_object, unit, sampling_rate=10000):
    spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
    assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
    if len(spike_train) > 1:
        isi = np.diff(spike_train)
        mean_isi = np.mean(isi)
        return mean_isi  # already in seconds
    else:
        return None
    
def get_unit_isi_CoV(recording_object, sorting_object, unit, sampling_rate=10000):
    spike_train = sorting_object.get_unit_spike_train(unit) / sampling_rate # convert to seconds
    assert all(spike_time >= 0 for spike_time in spike_train), f'Spike times for unit {unit} contain negative values'
    if len(spike_train) > 2:
        isi = np.diff(spike_train)  # already in seconds
        if len(isi) > 1:
            CoV = np.std(isi) / np.mean(isi)
            return CoV
        else:
            return None
    else:
        return None

def extract_metrics_from_experimental_data(spike_times, timeVector, spike_times_by_unit, **kwargs):
    
    #extract spiking metrics from experimental data
    recording_object = kwargs['recording_object']
    sorting_object = kwargs['sorting_object']
    sampling_rate = recording_object.get_sampling_frequency()    
    
    #add immediately available data to network_data
    network_data['source'] = 'experimental'
    network_data['timeVector'] = timeVector
    network_data['spiking_data']['spike_times'] = spike_times
    network_data['spiking_data']['spiking_times_by_unit'] = spike_times_by_unit
    
    #overall mean firing rate
    network_data['spiking_data']['spiking_summary_data']['MeanFireRate'] = get_mean_fr(
        recording_object, sorting_object, sampling_rate=sampling_rate
        )
    
    #overall CoV firing rate
    network_data['spiking_data']['spiking_summary_data']['CoVFireRate'] = get_CoV_fr_experimental(
        recording_object, sorting_object, sampling_rate=sampling_rate
        )
    
    #overall mean ISI
    network_data['spiking_data']['spiking_summary_data']['MeanISI'] = get_mean_isi(
        recording_object, sorting_object, sampling_rate=sampling_rate
    )
    
    #overall CoV ISI
    network_data['spiking_data']['spiking_summary_data']['CoV_ISI'] = get_CoV_isi(
        recording_object, sorting_object, sampling_rate=sampling_rate
    )
    
    #spiking data by unit - just go through the simulated version of the data but de-identify pop
    #simulated_spiking_data_by_unit = network_data['simulated_data']['spiking_data_by_unit']
    spiking_data_by_unit = {}
    units = sorting_object.get_unit_ids()
    for unit in units:
        spiking_data_by_unit[unit] = {
            #'unitProperty': sorting_object.get_unit_property(unit),
            #'SpikeTrain': sorting_object.get_unit_spike_train(unit),
            'FireRate': get_unit_fr(recording_object, sorting_object, unit),
            'fr_CoV': get_unit_fr_CoV(recording_object, sorting_object, unit),
            'meanISI': get_unit_mean_isi(recording_object, sorting_object, unit),
            'isi_CoV': get_unit_isi_CoV(recording_object, sorting_object, unit),
            'spike_times': spike_times_by_unit[unit],
        }
    network_data['spiking_data']['spiking_data_by_unit'] = spiking_data_by_unit

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

def get_experimental_network_activity_metrics(**kwargs):
    
    #initialize network_data
    network_data = init_network_data_dict()
    
    #get data
    sorting_object = kwargs['sorting_object']
    recording_object = kwargs['recording_object']
    sampling_rate = recording_object.get_sampling_frequency() 
    
    #convert time to seconds - get initially available data
    spike_times = get_spike_times(recording_object, sorting_object, sampling_rate=sampling_rate) #seconds
    timeVector = get_time_vector(recording_object, sampling_rate=sampling_rate) #seconds
    spike_times_by_unit = get_spike_times_by_unit(sorting_object, sampling_rate=sampling_rate) 
    
    #extract spiking metrics from simulated data
    try: 
        extract_metrics_from_experimental_data(spike_times, timeVector, spike_times_by_unit, **kwargs)
    except Exception as e:
        print(f'Error extracting metrics from simulated data: {e}')
        pass
    
    #extract bursting metrics from simulated data (but this one works for both simulated and experimental data)
    try: 
        extract_bursting_activity_data(spike_times, spike_times_by_unit, **kwargs)
    except Exception as e:
        print(f'Error calculating bursting activity: {e}')
        pass
    
    return network_data