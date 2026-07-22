# use this script to process experimental data and exract features to be used developing network models.
# primarily we aim to get experimental data into a form that is similar to the simulated data 
# - such that the same analysis can be applied to both.
# Notes: ========================================

# Imports ========================================
import os
import glob
import MEA_Analysis.MEAProcessingLibrary.mea_processing_library as mea
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
import spikeinterface.postprocessing as spost
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import get_experimental_network_metrics_v3
import traceback
# from .utils.helper import indent_mode_on, indent_mode_off
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import compute_network_metrics
import json

# Functions ======================================
''' Newer/Updated Functions'''
def run_partial_analysis(
        recording_object, 
        sorting_object,
        #wf_extractor, 
        sorting_analyzer,
        stream_num, 
        conv_params,
        mega_params,
        #convolution_params_path, 
        output_path,
        save_path=None,
        bursting_plot_path=None, bursting_fig_path=None, 
        plot=False, 
        limit_seconds = None,
        debug_mode = False,
        **kwargs):
    
    # import
    from MEA_Analysis.NetworkAnalysis_aw.compute_network_metrics import compute_network_metrics
    
    # Subfunctions ======================================
    def get_metrics(sorting_object, recording_object, sorting_analyzer, conv_params, mega_params, debug_mode=False, **kwargs):
        # get network metrics
        well_id = f'well{str(0).zfill(2)}{stream_num}'
        well_recording_segment = recording_object 
        
        # define paths based on wf_extractor
        analyzer_folder = sorting_analyzer.folder._str
        #wf_extractor_info = wf_extractor._save_data()
        #wf_folder = wf_extractor.folder._str
        #print(f"wf_folder: {wf_folder}")
        
        # replace 'waveforms' with 'dtw'
        dtw_folder = analyzer_folder.replace('analyzer', 'dtw')
        #dtw_folder = wf_folder.replace('waveforms', 'dtw')
        #dtw_output = os.path.join(dtw_folder, 'dtw_output')
        dtw_temp = os.path.join(dtw_folder, 'dtw_temp')
        #mega_dtw_output = os.path.join(dtw_folder, 'mega_dtw_output')
        # print(f"dtw_output: {dtw_output}")
        # print(f"dtw_temp: {dtw_temp}")
        # print(f"mega_dtw: {mega_dtw_output}")
        
        try:
            source = 'experimental'
            netkwargs = {
                'debug_mode': debug_mode,
                'well_id': well_id,
                'stream_num': stream_num,
                'recording_object': recording_object,
                'sorting_object': sorting_object,
                #'wf_extractor': wf_extractor,
                'sorting_analyzer': sorting_analyzer,
                'run_parallel': True,
                'max_workers': kwargs['max_workers'],
                #'max_workers': 32,
                #'max_workers': 16,
                #'max_workers' : 256,
                #'plot_wfs': True,
                
                #'plot_wfs': False,
                'plot_wfs': kwargs['plot_wfs'],
                'burst_sequencing': True,
                #'burst_sequencing': False,
                
                # debug - move to run script later #HACK
                # fitness_save_path = kwargs['fitness_save_path']
                # basename = os.path.basename(fitness_save_path)
                # sa_dir = os.path.dirname(fitness_save_path)
                # # remove .json from basename
                # basename = basename.replace('.json', '')
                # dtw_dir = os.path.join(sa_dir, basename, 'dtw_temp')    
                # kwargs['dtw_temp'] = dtw_dir
                
                'dtw_temp': dtw_temp,
                # 'dtw_output': dtw_output,
                # 'mega_dtw': mega_dtw_output,
                
                #analysis options
                'compute_spike_metrics': True, # compute spike metrics
                'compute_burst_metrics': True, # compute burst metrics
                'classify_units': False, # classify units
                #'locate_units': False, # locate units
                'compute_dynamic_time_warping': False, # compute dynamic time warping
                'compute_summary_metrics': False,                
                
            }
            network_metrics = compute_network_metrics(conv_params, mega_params, source, **netkwargs)
            #network_metrics = get_experimental_network_metrics_v3(sorting_object, well_recording_segment, wf_extractor, conv_params, mega_params, debug_mode=debug_mode, **kwargs)
            return network_metrics
        except Exception as e:
            print(f'Error: Could not get network metrics for {well_id}')
            traceback.print_exc()
            return (e, traceback.format_exc())
        
    # Main ======================================
    # assertions
    assert sorting_object is not None, f"Error: sorting_object is None"
        
    # get network metrics
    network_metrics = get_metrics(sorting_object, recording_object, sorting_analyzer, conv_params, mega_params, debug_mode=debug_mode, **kwargs)
        
    # get recording details
    print("Saving network metrics as numpy...")
    recording_details = kwargs['details']
    projectName = recording_details['projectName']
    date = recording_details['date']
    chipID = recording_details['chipID']
    scanType = recording_details['scanType']
    runID = recording_details['runID']
    
    # get sorting output dir from network metrics
    #sorting_output_dir = network_metrics['sorting_output']
    # remove /sorter_output from sorting_output_dir
    #sorting_output_dir = sorting_output_dir.replace('sorter_output', '')
    # replace sorted with network_metrics
    #output_dir = sorting_output_dir.replace('sorted', 'network_metrics')
    output_dir = output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"metrics.npy")
    print(f"Saving network metrics to {save_path}")
    # save_path = os.path.join(output_path, f"network_metrics_well00{stream_num}.npy")
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, network_metrics)
    
    #plot network activity
    if plot:
        
        #plot neuron locations and class. Inhibitory neurons are red, excitatory neurons are blue.
        #save_path = save_path.replace('.npy', '_neuron_locations.pdf')
        #save_path = save_path.replace('.npy', '_neuron_locations.png')
        #neuron_loc_plot_path = save_path.replace('network_metrics', 'neuron_loc_plots')
        neuron_loc_plot_path = save_path.replace('network_analysis', 'location_plots')
        neuron_loc_parent_dir = os.path.dirname(neuron_loc_plot_path)
        if not os.path.exists(neuron_loc_parent_dir):
            os.makedirs(neuron_loc_parent_dir)
        neuron_loc_plot_path = os.path.join(neuron_loc_parent_dir, f"neuron_locations.png")
        #plot_neuron_locations_and_class(sorting_object, wf_extractor, network_metrics, save_path=neuron_loc_plot_path)
        plot_neuron_locations_and_class(sorting_object, sorting_analyzer, network_metrics, save_path=neuron_loc_plot_path)
        
        
        try: 
            print("Generating network summary plot...")
            # aw 2025-01-20 17:25:21 - I guess I'll just plot both for now. I like how mine looks, but additional context is nice for Roy.
            #network_plot_path = save_path.replace('network_metrics', 'network_plots')
            network_plot_path = save_path.replace('network_analysis', 'network_plots')
            network_plot_parent_dir = os.path.dirname(network_plot_path)
            if not os.path.exists(network_plot_parent_dir):
                os.makedirs(network_plot_parent_dir)
            #network_plot_path = os.path.join(network_plot_parent_dir, f"network_summary_plot.pdf")
            network_plot_path_3p = os.path.join(network_plot_parent_dir, f"network_summary_3pannels.pdf")
            network_plot_path_2p = os.path.join(network_plot_parent_dir, f"network_summary_2pannels.pdf")
            network_plot_path_3p_classed = os.path.join(network_plot_parent_dir, f"network_summary_3pannels_classed.pdf")
            network_plot_path_2p_classed = os.path.join(network_plot_parent_dir, f"network_summary_2pannels_classed.pdf")
            
            #
            # unit_types = network_metrics['unit_types']
            # print(f"unit_types: {unit_types}")
            # import sys
            # sys.exit()
            #
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_3p,
                #mode = '2p',
                mode = '3p',
                limit_seconds = limit_seconds,
                plot_class = False,
                )
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_2p,
                mode = '2p',
                limit_seconds = limit_seconds,
                plot_class = False,
                )  
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_3p_classed,
                #mode = '2p',
                mode = '3p',
                limit_seconds = limit_seconds,
                plot_class = True,
                )
            
            # #debug
            # # print keys in network_metrics
            # print(f"Keys in network_metrics:")
            # for key in network_metrics.keys():
            #     print(f"{key}")
            
            # import sys
            # sys.exit() 
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_2p_classed,
                mode = '2p',
                limit_seconds = limit_seconds,
                plot_class = True,
                )  
        except Exception as e:
            print(e)
            #print(f"Error: Could not plot network activity for {well_id}") 
            traceback.print_exc()
            print(f"Error: Could not plot network activity")

    # return network metrics and save path
    print(f"Saved network metrics to {save_path}")
    return network_metrics, save_path

def tune_conv_params(
    #raw_data_paths, 
    sorted_data_dirs = None, 
    output_dirs = None, 
    stream_select=None, 
    plot=True, 
    conv_params=None,
    mega_params=None,
    limit_seconds=None,
    plot_wfs=False,
    max_workers = 4, # safe for all computers - if not specified, will use all available cores
    debug_mode = False,
    **kwargs,
    ):
    
    ## subfunctions =================================================================
    def initialize_output_dir(output_dirs):
        assert output_dirs is not None, f"Error: output_dirs is None"
        for output_dir in output_dirs:
            assert output_dir is not None, f"Error: output_dir is None"
            output_dir = os.path.abspath(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        return output_dirs
    
    def initialize_h5_paths(sorted_data_dirs):
        assert sorted_data_dirs is not None, f"Error: sorted_data_dirs is None"
        h5_paths = []
        for sorted_data_dir in sorted_data_dirs:
            # look for a file called spikeinterface_recording.json in the directory
            if not os.path.exists(sorted_data_dir):
                print(f"Error: sorted_data_dir does not exist. {sorted_data_dir}")
                raise FileNotFoundError(f"Error: sorted_data_dir does not exist. {sorted_data_dir}")
                # continue
            json_path = os.path.join(sorted_data_dir, 'spikeinterface_recording.json')
            if not os.path.exists(json_path):
                print(f"Error: sorted_data_dir does not contain spikeinterface_recording.json. {sorted_data_dir}")
                raise FileNotFoundError(f"Error: sorted_data_dir does not contain spikeinterface_recording.json. {sorted_data_dir}")
                #continue
            # load json file
            with open(json_path, 'r') as f:
                recording_details = json.load(f)
            h5_path = recording_details['kwargs']['file_path']
            
            #HACK stupid patch to fix old sorting path
            if 'zinputs' in h5_path:
                h5_path = h5_path.replace('zinputs', 'z_raw_data')
                
            h5_paths.append(h5_path)
        return h5_paths
    
    def initialize_sorted_output_dirs(sorted_data_dirs):
        assert sorted_data_dirs is not None, f"Error: sorted_data_dir is None"  
        #sorted_data_dir = os.path.abspath(sorted_data_dir)
        
        # iterate through sorter_output_dir and get all well output directories
        sorted_output_folders = []
        for sorted_data_dir in sorted_data_dirs:
            for root, dirs, files in os.walk(sorted_data_dir):
                if root.endswith('sorter_output'):
                    if not os.path.exists(root):
                        print(f"Error: sorted_output_folder does not exist. {root}")
                        #continue
                        raise FileNotFoundError(f"Error: sorted_output_folder does not exist. {root}")
                    sorted_output_folders.append(os.path.join(root))
        return sorted_output_folders
    
    def validate_three_objects(well_data):
        #init skip flag
        skip = False
        
        recording_segments, sort_obj, wf_extractor = well_data
        
        objs = [recording_segments, sort_obj, wf_extractor]
        for obj in objs:
            try:
                for i, item in enumerate(obj):
                    if isinstance(item, Exception):
                        skip = True
                        return objs, skip
            except:
                if isinstance(obj, Exception):
                    skip = True
                    return objs, skip
        return objs, skip
    
    ## main function =================================================================
    # assertions - assert conv_params and mega_params are defined, this is required for network analysis
    assert conv_params is not None, f"Error: conv_params is None - must be provided for network analysis"
    assert mega_params is not None, f"Error: mega_params is None - must be provided for network analysis"
    
    #init paths
    output_dirs = initialize_output_dir(output_dirs)
    h5_paths = initialize_h5_paths(sorted_data_dirs)
    sorted_output_folders = initialize_sorted_output_dirs(sorted_data_dirs)
     
    # get paired data objects - network analysis requires both recording and sorting objects
    well_data_list, recording_details, path_pairs = get_data_obj_groups_v3(h5_paths, sorted_output_folders)
    
    # iterate through data_obj_list and get network metrics for each pair
    for i, well_data in enumerate(well_data_list):
        
        # choose to skip or not by validating objects
        objs, skip = validate_three_objects(well_data)
        if skip: continue
        #recording_segments, sort_obj, wf_extractor = objs
        recording_segments, sort_obj, sorting_analyzer = objs
        recording_segment = recording_segments[0] # HACK: this function is really only going to be used for network scans... but if I try to use it for multiple segments, I'll need to update this.
        
        # init kwargs         
        stream_id = recording_segment.stream_id
        stream_num = int(stream_id.split('well')[1][:3])
        kwargs = recording_details.copy()
        kwargs['plot_wfs'] = plot_wfs
        kwargs['max_workers'] = max_workers
        
        # init print statements
        print(f"Analyzing network data collected in well{str(0).zfill(2)}{stream_num}...")
        # indent_mode_on(level=1) # indent all print statements in this block
        print(f"Initializing...")
        
        output_dir = output_dirs[i]
        
        # run analysis
        kwargs['recording_object'] = recording_segment
        kwargs['sorting_object'] = sort_obj
        kwargs['sorting_analyzer'] = sorting_analyzer
        kwargs['stream_num'] = stream_num
        kwargs['conv_params'] = conv_params
        kwargs['mega_params'] = mega_params
        kwargs['output_path'] = output_dir
        kwargs['plot'] = plot
        kwargs['details'] = recording_details
        kwargs['limit_seconds'] = limit_seconds
        kwargs['debug_mode'] = debug_mode
        run_partial_analysis(**kwargs)
        # run_analysis(
        #     recording_segment,
        #     sort_obj,
        #     sorting_analyzer, 
        #     stream_num,
        #     conv_params,
        #     mega_params,
        #     output_dir,             
        #     plot=plot, 
        #     details=recording_details, 
        #     limit_seconds = limit_seconds,
        #     debug_mode = debug_mode, # limit number of units and bursts to analyze to get through functions quickly
        #     **kwargs)    
    print('done')
    indent_mode_off() # turn off indenting for all print statements
    return

def run_analysis(
        recording_object, 
        sorting_object,
        #wf_extractor, 
        sorting_analyzer,
        stream_num, 
        conv_params,
        mega_params,
        #convolution_params_path, 
        output_path,
        save_path=None,
        bursting_plot_path=None, bursting_fig_path=None, 
        plot=False, 
        limit_seconds = None,
        debug_mode = False,
        **kwargs):
    
    # Subfunctions ======================================
    def get_metrics(sorting_object, recording_object, sorting_analyzer, conv_params, mega_params, max_duration= None,debug_mode=False, **kwargs):
        # get network metrics
        well_id = f'well{str(0).zfill(2)}{stream_num}'
        well_recording_segment = recording_object 
        
        # define paths based on wf_extractor
        analyzer_folder = sorting_analyzer.folder._str
        #wf_extractor_info = wf_extractor._save_data()
        #wf_folder = wf_extractor.folder._str
        #print(f"wf_folder: {wf_folder}")
        
        # replace 'waveforms' with 'dtw'
        dtw_folder = analyzer_folder.replace('analyzer', 'dtw')
        #dtw_folder = wf_folder.replace('waveforms', 'dtw')
        #dtw_output = os.path.join(dtw_folder, 'dtw_output')
        dtw_temp = os.path.join(dtw_folder, 'dtw_temp')
        #mega_dtw_output = os.path.join(dtw_folder, 'mega_dtw_output')
        # print(f"dtw_output: {dtw_output}")
        # print(f"dtw_temp: {dtw_temp}")
        # print(f"mega_dtw: {mega_dtw_output}")
        
        try:
            source = 'experimental'
            netkwargs = {
                'debug_mode': debug_mode,
                'well_id': well_id,
                'stream_num': stream_num,
                'recording_object': recording_object,
                'sorting_object': sorting_object,
                #'wf_extractor': wf_extractor,
                'sorting_analyzer': sorting_analyzer,
                'run_parallel': True,
                'max_workers': kwargs['max_workers'],
                'max_duration': max_duration,
                #'max_workers': 32,
                #'max_workers': 16,
                #'max_workers' : 256,
                #'plot_wfs': True,
                
                #'plot_wfs': False,
                'plot_wfs': kwargs['plot_wfs'],
                'burst_sequencing': True,
                #'burst_sequencing': False,
                
                # debug - move to run script later #HACK
                # fitness_save_path = kwargs['fitness_save_path']
                # basename = os.path.basename(fitness_save_path)
                # sa_dir = os.path.dirname(fitness_save_path)
                # # remove .json from basename
                # basename = basename.replace('.json', '')
                # dtw_dir = os.path.join(sa_dir, basename, 'dtw_temp')    
                # kwargs['dtw_temp'] = dtw_dir
                
                'dtw_temp': dtw_temp,
                'max_duration': kwargs.get('max_duration', 20),
                # 'dtw_output': dtw_output,
                # 'mega_dtw': mega_dtw_output,
                
            }
            network_metrics = compute_network_metrics(conv_params, mega_params, source, **netkwargs)
            #network_metrics = get_experimental_network_metrics_v3(sorting_object, well_recording_segment, wf_extractor, conv_params, mega_params, debug_mode=debug_mode, **kwargs)
            return network_metrics
        except Exception as e:
            print(f'Error: Could not get network metrics for {well_id}')
            traceback.print_exc()
            return (e, traceback.format_exc())
        
    # Main ======================================
    # assertions
    assert sorting_object is not None, f"Error: sorting_object is None"
        
    # get network metrics
    network_metrics = get_metrics(sorting_object, recording_object, sorting_analyzer, conv_params, mega_params, debug_mode=debug_mode, **kwargs)
        
    # get recording details
    print("Saving network metrics as numpy...")
    recording_details = kwargs['details']
    projectName = recording_details['projectName']
    date = recording_details['date']
    chipID = recording_details['chipID']
    scanType = recording_details['scanType']
    runID = recording_details['runID']
    
    # get sorting output dir from network metrics
    #sorting_output_dir = network_metrics['sorting_output']
    # remove /sorter_output from sorting_output_dir
    #sorting_output_dir = sorting_output_dir.replace('sorter_output', '')
    # replace sorted with network_metrics
    #output_dir = sorting_output_dir.replace('sorted', 'network_metrics')
    output_dir = output_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"metrics.npy")
    print(f"Saving network metrics to {save_path}")
    # save_path = os.path.join(output_path, f"network_metrics_well00{stream_num}.npy")
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, network_metrics)
    
    #plot network activity
    if plot:
        
        #plot neuron locations and class. Inhibitory neurons are red, excitatory neurons are blue.
        #save_path = save_path.replace('.npy', '_neuron_locations.pdf')
        #save_path = save_path.replace('.npy', '_neuron_locations.png')
        #neuron_loc_plot_path = save_path.replace('network_metrics', 'neuron_loc_plots')
        neuron_loc_plot_path = save_path.replace('network_analysis', 'location_plots')
        neuron_loc_parent_dir = os.path.dirname(neuron_loc_plot_path)
        if not os.path.exists(neuron_loc_parent_dir):
            os.makedirs(neuron_loc_parent_dir)
        neuron_loc_plot_path = os.path.join(neuron_loc_parent_dir, f"neuron_locations.png")
        #plot_neuron_locations_and_class(sorting_object, wf_extractor, network_metrics, save_path=neuron_loc_plot_path)
        plot_neuron_locations_and_class(sorting_object, sorting_analyzer, network_metrics, save_path=neuron_loc_plot_path)
        
        
        try: 
            print("Generating network summary plot...")
            # aw 2025-01-20 17:25:21 - I guess I'll just plot both for now. I like how mine looks, but additional context is nice for Roy.
            #network_plot_path = save_path.replace('network_metrics', 'network_plots')
            network_plot_path = save_path.replace('network_analysis', 'network_plots')
            network_plot_parent_dir = os.path.dirname(network_plot_path)
            if not os.path.exists(network_plot_parent_dir):
                os.makedirs(network_plot_parent_dir)
            #network_plot_path = os.path.join(network_plot_parent_dir, f"network_summary_plot.pdf")
            network_plot_path_3p = os.path.join(network_plot_parent_dir, f"network_summary_3pannels.pdf")
            network_plot_path_2p = os.path.join(network_plot_parent_dir, f"network_summary_2pannels.pdf")
            network_plot_path_3p_classed = os.path.join(network_plot_parent_dir, f"network_summary_3pannels_classed.pdf")
            network_plot_path_2p_classed = os.path.join(network_plot_parent_dir, f"network_summary_2pannels_classed.pdf")
            
            #
            # unit_types = network_metrics['unit_types']
            # print(f"unit_types: {unit_types}")
            # import sys
            # sys.exit()
            #
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_3p,
                #mode = '2p',
                mode = '3p',
                limit_seconds = limit_seconds,
                plot_class = False,
                )
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_2p,
                mode = '2p',
                limit_seconds = limit_seconds,
                plot_class = False,
                )  
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_3p_classed,
                #mode = '2p',
                mode = '3p',
                limit_seconds = limit_seconds,
                plot_class = True,
                )
            
            # #debug
            # # print keys in network_metrics
            # print(f"Keys in network_metrics:")
            # for key in network_metrics.keys():
            #     print(f"{key}")
            
            # import sys
            # sys.exit() 
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_2p_classed,
                mode = '2p',
                limit_seconds = limit_seconds,
                plot_class = True,
                )  
        except Exception as e:
            print(e)
            #print(f"Error: Could not plot network activity for {well_id}") 
            traceback.print_exc()
            print(f"Error: Could not plot network activity")

    # return network metrics and save path
    print(f"Saved network metrics to {save_path}")
    return network_metrics, save_path

def get_data_obj_groups_v3(h5_paths, sorted_output_folders):
    # aw 2025-02-11
    # get recording details from each h5 file, check for match where all details match in sorted_output_folders
    # this way, we pair up recordings with their corresponding sorting output
    # get paired data objects - network analysis requires both recording and sorting objects
    
    # Subfunctions ======================================
    def load_three_objects(h5_path, sorted_output_folder, recording_details):
        # this function expects to load one sorting_obj, one recording_obj, and waveform data for the related well.
        # there may be any number of rec_segments in the recording_obj, but only one sorting_obj
        
        # get stream_select from sorted_output_folder path
        # look for the word 'well' in the string. It will beb followed by three digits.
        # get the int value of those digits. Should be 0-5
        stream_select = int(sorted_output_folder.split('well')[1][:3])
        wellid = f'well{str(0).zfill(2)}{stream_select}'
        
        # load recording object
        try:
            _, well_recs, _, _ = mea.load_recordings(h5_path, stream_select=stream_select)
            rec_segments = well_recs[wellid]
        except Exception as e: rec_segments = (e, traceback.format_exc()) # put error in rec_segments for debugging
        
        # load sorting object
        try: sort_obj = mea.load_kilosort2_results(sorted_output_folder)
        except Exception as e: sort_obj = (e, traceback.format_exc()) # put error in sort_obj for debugging
        
        # # load waveform data
        # try:
        #     waveform_output_dir = sorted_output_folder.replace('sorter_output', '')
        #     waveform_output_dir = waveform_output_dir.replace('sorted', 'waveforms')
        #     sort_obj.register_recording(rec_segments[0])
        #     waveform_extractor = mea.load_waveforms(waveform_output_dir, sorting=sort_obj)
        # except Exception as e: waveform_extractor = (e, traceback.format_exc()) # put error in waveform_extractor for debugging
        
        #import traceback
        #from spikeinterface.analysis import load_sorting_analyzer

        # load waveform data
        try:
            #waveform_output_dir = sorted_output_folder.replace('sorter_output', '')
            #waveform_output_dir = waveform_output_dir.replace('sorted', 'waveforms')
            sorting_analyzer_output_dir = sorted_output_folder.replace('sorter_output', '')
            sorting_analyzer_output_dir = sorting_analyzer_output_dir.replace('sorted', 'analyzer')
            
            # register rec
            sort_obj.register_recording(rec_segments[0])

            # load SortingAnalyzer from folder
            sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_output_dir)

            # get waveforms extension
            # waveform_extractor = sorting_analyzer.get_extension("waveforms")

        except Exception as e:
            #waveform_extractor = (e, traceback.format_exc())  # put error in waveform_extractor for debugging
            sorting_analyzer = (e, traceback.format_exc())  # put error in waveform_extractor for debugging
        
        # return paired objects
        # return (rec_segments, sort_obj, waveform_extractor), recording_details
        return (rec_segments, sort_obj, sorting_analyzer), recording_details
    
    # Main ======================================   
    well_data_list = []
    path_pairs = []
    for h5_path in h5_paths:
        
        # get recording details
        recording_details = mea.extract_recording_details(h5_path)[0] # NOTE: this works for a list of dirs or a single dir - but treats single dir as a list of a single dir
        
        #remove h5_file_path from recording_details - this wont match, this is the old path
        h5_path = recording_details.pop('h5_file_path')
        
        for sorted_output_folder in sorted_output_folders:         
            
            # shortform
            found = all([f'/{recording_details[key]}/' in sorted_output_folder for key in recording_details.keys()])
            
            if found:
                path_pairs.append((h5_path, sorted_output_folder))                       
                well_data, recording_details = load_three_objects(h5_path, sorted_output_folder, recording_details) # NOTE: on error, well_data will be exception information
                well_data_list.append(well_data)
                
    return well_data_list, recording_details, path_pairs

def get_data_obj_groups_v2(h5_paths, raw_data_path, sorted_output_folders):
    # aw 2025-02-11
    # get recording details from each h5 file, check for match where all details match in sorted_output_folders
    # this way, we pair up recordings with their corresponding sorting output
    # get paired data objects - network analysis requires both recording and sorting objects
    
    # Subfunctions ======================================
    def load_three_objects(h5_path, sorted_output_folder, recording_details):
        # this function expects to load one sorting_obj, one recording_obj, and waveform data for the related well.
        # there may be any number of rec_segments in the recording_obj, but only one sorting_obj
        
        # get stream_select from sorted_output_folder path
        # look for the word 'well' in the string. It will beb followed by three digits.
        # get the int value of those digits. Should be 0-5
        stream_select = int(sorted_output_folder.split('well')[1][:3])
        wellid = f'well{str(0).zfill(2)}{stream_select}'
        
        # load recording object
        try:
            _, well_recs, _, _ = mea.load_recordings(h5_path, stream_select=stream_select)
            rec_segments = well_recs[wellid]
        except Exception as e: rec_segments = (e, traceback.format_exc()) # put error in rec_segments for debugging
        
        # load sorting object
        try: sort_obj = mea.load_kilosort2_results(sorted_output_folder)
        except Exception as e: sort_obj = (e, traceback.format_exc()) # put error in sort_obj for debugging
        
        # load waveform data
        try:
            waveform_output_dir = sorted_output_folder.replace('sorter_output', '')
            waveform_output_dir = waveform_output_dir.replace('sorted', 'waveforms')
            sort_obj.register_recording(rec_segments[0])
            waveform_extractor = mea.load_waveforms(waveform_output_dir, sorting=sort_obj)
        except Exception as e: waveform_extractor = (e, traceback.format_exc()) # put error in waveform_extractor for debugging
        
        # return paired objects
        return (rec_segments, sort_obj, waveform_extractor), recording_details
    
    # Main ======================================   
    well_data_list = []
    path_pairs = []
    for h5_path in h5_paths:
        
        # get recording details
        recording_details = mea.extract_recording_details(raw_data_path)[0] # NOTE: this works for a list of dirs or a single dir - but treats single dir as a list of a single dir
        
        #remove h5_file_path from recording_details - this wont match, this is the old path
        h5_path = recording_details.pop('h5_file_path')
        
        for sorted_output_folder in sorted_output_folders:         
            
            # shortform
            found = all([f'/{recording_details[key]}/' in sorted_output_folder for key in recording_details.keys()])
            
            if found:
                path_pairs.append((h5_path, sorted_output_folder))                       
                well_data, recording_details = load_three_objects(h5_path, sorted_output_folder, recording_details) # NOTE: on error, well_data will be exception information
                well_data_list.append(well_data)
                
    return well_data_list, recording_details, path_pairs

def analyze_network_data(
    #raw_data_paths, 
    sorted_data_dirs = None, 
    output_dirs = None, 
    stream_select=None, 
    plot=True, 
    conv_params=None,
    mega_params=None,
    limit_seconds=None,
    plot_wfs=False,
    max_workers = 4, # safe for all computers - if not specified, will use all available cores
    debug_mode = False,
    **kwargs,
    ):
    
    ## subfunctions =================================================================
    def initialize_output_dir(output_dirs):
        assert output_dirs is not None, f"Error: output_dirs is None"
        for output_dir in output_dirs:
            assert output_dir is not None, f"Error: output_dir is None"
            output_dir = os.path.abspath(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        return output_dirs
    
    def initialize_h5_paths(sorted_data_dirs):
        assert sorted_data_dirs is not None, f"Error: sorted_data_dirs is None"
        h5_paths = []
        for sorted_data_dir in sorted_data_dirs:
            # look for a file called spikeinterface_recording.json in the directory
            if not os.path.exists(sorted_data_dir):
                print(f"Error: sorted_data_dir does not exist. {sorted_data_dir}")
                raise FileNotFoundError(f"Error: sorted_data_dir does not exist. {sorted_data_dir}")
                # continue
            json_path = os.path.join(sorted_data_dir, 'spikeinterface_recording.json')
            if not os.path.exists(json_path):
                print(f"Error: sorted_data_dir does not contain spikeinterface_recording.json. {sorted_data_dir}")
                raise FileNotFoundError(f"Error: sorted_data_dir does not contain spikeinterface_recording.json. {sorted_data_dir}")
                #continue
            # load json file
            with open(json_path, 'r') as f:
                recording_details = json.load(f)
            h5_path = recording_details['kwargs']['parent_recording']['kwargs']['file_path']
            
            #HACK stupid patch to fix old sorting path
            if 'zinputs' in h5_path:
                h5_path = h5_path.replace('zinputs', 'z_raw_data')
                
            h5_paths.append(h5_path)
        return h5_paths
    
    def initialize_sorted_output_dirs(sorted_data_dirs):
        assert sorted_data_dirs is not None, f"Error: sorted_data_dir is None"  
        #sorted_data_dir = os.path.abspath(sorted_data_dir)
        
        # iterate through sorter_output_dir and get all well output directories
        sorted_output_folders = []
        for sorted_data_dir in sorted_data_dirs:
            for root, dirs, files in os.walk(sorted_data_dir):
                if root.endswith('sorter_output'):
                    if not os.path.exists(root):
                        print(f"Error: sorted_output_folder does not exist. {root}")
                        #continue
                        raise FileNotFoundError(f"Error: sorted_output_folder does not exist. {root}")
                    sorted_output_folders.append(os.path.join(root))
        return sorted_output_folders
    
    def validate_three_objects(well_data):
        #init skip flag
        skip = False
        
        recording_segments, sort_obj, wf_extractor = well_data
        
        objs = [recording_segments, sort_obj, wf_extractor]
        for obj in objs:
            try:
                for i, item in enumerate(obj):
                    if isinstance(item, Exception):
                        skip = True
                        return objs, skip
            except:
                if isinstance(obj, Exception):
                    skip = True
                    return objs, skip
        return objs, skip
    
    ## main function =================================================================
    # assertions - assert conv_params and mega_params are defined, this is required for network analysis
    assert conv_params is not None, f"Error: conv_params is None - must be provided for network analysis"
    assert mega_params is not None, f"Error: mega_params is None - must be provided for network analysis"
    
    #init paths
    output_dirs = initialize_output_dir(output_dirs)
    h5_paths = initialize_h5_paths(sorted_data_dirs)
    sorted_output_folders = initialize_sorted_output_dirs(sorted_data_dirs)
     
    # get paired data objects - network analysis requires both recording and sorting objects
    well_data_list, recording_details, path_pairs = get_data_obj_groups_v3(h5_paths, sorted_output_folders)
    
    # iterate through data_obj_list and get network metrics for each pair
    for i, well_data in enumerate(well_data_list):
        
        # choose to skip or not by validating objects
        objs, skip = validate_three_objects(well_data)
        if skip: continue
        #recording_segments, sort_obj, wf_extractor = objs
        recording_segments, sort_obj, sorting_analyzer = objs
        recording_segment = recording_segments[0] # HACK: this function is really only going to be used for network scans... but if I try to use it for multiple segments, I'll need to update this.
        
        # init kwargs         
        stream_id = recording_segment.stream_id
        stream_num = int(stream_id.split('well')[1][:3])
        kwargs = recording_details.copy()
        kwargs['plot_wfs'] = plot_wfs
        kwargs['max_workers'] = max_workers
        
        # init print statements
        print(f"Analyzing network data collected in well{str(0).zfill(2)}{stream_num}...")
        # indent_mode_on(level=1) # indent all print statements in this block
        print(f"Initializing...")
        
        output_dir = output_dirs[i]
        
        # run analysis
        kwargs['recording_object'] = recording_segment
        kwargs['sorting_object'] = sort_obj
        kwargs['sorting_analyzer'] = sorting_analyzer
        kwargs['stream_num'] = stream_num
        kwargs['conv_params'] = conv_params
        kwargs['mega_params'] = mega_params
        kwargs['output_path'] = output_dir
        kwargs['plot'] = plot
        kwargs['details'] = recording_details
        kwargs['limit_seconds'] = limit_seconds
        kwargs['debug_mode'] = debug_mode
        run_analysis(**kwargs)
        # run_analysis(
        #     recording_segment,
        #     sort_obj,
        #     sorting_analyzer, 
        #     stream_num,
        #     conv_params,
        #     mega_params,
        #     output_dir,             
        #     plot=plot, 
        #     details=recording_details, 
        #     limit_seconds = limit_seconds,
        #     debug_mode = debug_mode, # limit number of units and bursts to analyze to get through functions quickly
        #     **kwargs)    
    print('done')
    # indent_mode_off() # turn off indenting for all print statements
    return

# aw 2025-04-09 10:30:26 - deprecated so that inputs are more minimal
def analyze_network_data_dep(
    raw_data_paths, 
    sorted_data_dir = None, 
    output_dir = None, 
    stream_select=None, 
    plot=True, 
    conv_params=None,
    mega_params=None,
    limit_seconds=None,
    plot_wfs=False,
    max_workers = 4, # safe for all computers - if not specified, will use all available cores
    debug_mode = False,
    ):
    
    ## subfunctions =================================================================
    def initialize_output_dir(output_dir):
        assert output_dir is not None, f"Error: output_dir is None"
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
    
    def initialize_h5_paths(raw_data_paths):
        abs_raw_paths = [os.path.abspath(raw_data_path) for raw_data_path in raw_data_paths]
        h5_paths = []
        for raw_data_path in abs_raw_paths:
            if os.path.isdir(raw_data_path):
                h5_paths.extend(glob.glob(os.path.join(raw_data_path, '**/*.h5'), recursive=True))
            else:
                h5_paths.append(raw_data_path)
        return h5_paths
    
    def initialize_sorted_output_dirs(sorted_data_dir):
        assert sorted_data_dir is not None, f"Error: sorted_data_dir is None"  
        sorted_data_dir = os.path.abspath(sorted_data_dir)
        
        # iterate through sorter_output_dir and get all well output directories
        sorted_output_folders = []
        for root, dirs, files in os.walk(sorted_data_dir):
            if root.endswith('sorter_output'):
                if not os.path.exists(root):
                    print(f"Error: sorted_output_folder does not exist. {root}")
                    continue
                sorted_output_folders.append(os.path.join(root))
        return sorted_output_folders
    
    def validate_three_objects(well_data):
        #init skip flag
        skip = False
        
        recording_segments, sort_obj, wf_extractor = well_data
        
        objs = [recording_segments, sort_obj, wf_extractor]
        for obj in objs:
            try:
                for i, item in enumerate(obj):
                    if isinstance(item, Exception):
                        skip = True
                        return objs, skip
            except:
                if isinstance(obj, Exception):
                    skip = True
                    return objs, skip
        return objs, skip
    
    ## main function =================================================================
    # assertions - assert conv_params and mega_params are defined, this is required for network analysis
    assert conv_params is not None, f"Error: conv_params is None - must be provided for network analysis"
    assert mega_params is not None, f"Error: mega_params is None - must be provided for network analysis"
    
    #init paths
    output_dir = initialize_output_dir(output_dir)
    h5_paths = initialize_h5_paths(raw_data_paths)
    sorted_output_folders = initialize_sorted_output_dirs(sorted_data_dir)
     
    # get paired data objects - network analysis requires both recording and sorting objects
    well_data_list, recording_details, path_pairs = get_data_obj_groups_v2(h5_paths, raw_data_paths, sorted_output_folders)
    
    # iterate through data_obj_list and get network metrics for each pair
    for well_data in well_data_list:
        
        # choose to skip or not by validating objects
        objs, skip = validate_three_objects(well_data)
        if skip: continue
        recording_segments, sort_obj, wf_extractor = objs
        recording_segment = recording_segments[0] # HACK: this function is really only going to be used for network scans... but if I try to use it for multiple segments, I'll need to update this.
        
        # init kwargs         
        stream_id = recording_segment.stream_id
        stream_num = int(stream_id.split('well')[1][:3])
        kwargs = recording_details.copy()
        kwargs['plot_wfs'] = plot_wfs
        kwargs['max_workers'] = max_workers
        
        # init print statements
        print(f"Analyzing network data collected in well{str(0).zfill(2)}{stream_num}...")
        # indent_mode_on(level=1) # indent all print statements in this block
        print(f"Initializing...")
        
        # run analysis
        run_analysis(
            recording_segment,
            sort_obj,
            wf_extractor, 
            stream_num,
            conv_params,
            mega_params,
            output_dir,             
            plot=plot, 
            details=recording_details, 
            limit_seconds = limit_seconds,
            debug_mode = debug_mode, # limit number of units and bursts to analyze to get through functions quickly
            **kwargs)    
    print('done')
    indent_mode_off() # turn off indenting for all print statements
    return

# aw 2025-02-24 16:07:47
def plot_neuron_locations_and_class(sorting_object, sa, network_metrics, save_path=None):
        
        # # aw 2025-02-24 16:11:37
        # TODO: Finish this function. It should plot the locations of neurons on the MEA, color coded by class.        
        # we = waveform_extractor
        # unit_ids = sorting_object.get_unit_ids()
        
        # unit_locations = spost.compute_unit_locations(we)
        # unit_locations_dict = {unit_id: unit_locations[i] for i, unit_id in enumerate(unit_ids)}
        # unit_locations = unit_locations_dict
        # sampling_frequency = we.sampling_frequency
        
        # classification_dict = {}
        # for i, unit_id in enumerate(unit_id_list):
        #     classification_dict[unit_id] = f"Cluster {cluster_labels[i] + 1}"
        
        # # plot locations of neurons on probe 2000 x 4000 um rectangle. ignore z coordinate in unit_locations
        # #unit_locations_2d = np.array([[loc[0], loc[1]] for loc in unit_locations])
        # unit_locations_2d = {unit_id: np.array([loc[0], loc[1]]) for unit_id,loc in unit_locations.items()}
        # #unit_locations_2d = unit_locations_2d.T # transpose to get x and y coordinates
        # # label inhibitory and excitatory neurons
        # # inhib_neuron_locs = unit_locations_2d[[i for i, unit_id in enumerate(unit_id_list) if classification_dict[unit_id] == f"Cluster {inhibitory_cluster + 1}"]]
        # # excit_neuron_locs = unit_locations_2d[[i for i, unit_id in enumerate(unit_id_list) if classification_dict[unit_id] == f"Cluster {1 - inhibitory_cluster + 1}"]]
        # inhib_neuron_locs = np.array([unit_locations_2d[unit_id] for unit_id in unit_id_list if classification_dict[unit_id] == f"Cluster {inhibitory_cluster + 1}"])
        # excit_neuron_locs = np.array([unit_locations_2d[unit_id] for unit_id in unit_id_list if classification_dict[unit_id] == f"Cluster {1 - inhibitory_cluster + 1}"])
        
        #
        classification_output = network_metrics['classification_output']
        include_unit_ids = classification_output['include_units']
        classified_units = classification_output['classified_units']
        
        #
        unit_locations = spost.compute_unit_locations(sa)
        unit_locations_dict = {unit_id: unit_locations[i] for i, unit_id in enumerate(include_unit_ids)}
        inhib_neuron_locs = np.array([unit_locations_dict[i] for i in include_unit_ids if classified_units[i]['desc'] == 'inhib'])
        excit_neuron_locs = np.array([unit_locations_dict[i] for i in include_unit_ids if classified_units[i]['desc'] == 'excit'])
        
        #
        
        
        min_x = min(np.min(inhib_neuron_locs[:, 0]), np.min(excit_neuron_locs[:, 0]))
        min_y = min(np.min(inhib_neuron_locs[:, 1]), np.min(excit_neuron_locs[:, 1]))
        max_x = max(np.max(inhib_neuron_locs[:, 0]), np.max(excit_neuron_locs[:, 0]))
        max_y = max(np.max(inhib_neuron_locs[:, 1]), np.max(excit_neuron_locs[:, 1]))
        
        if min_x > 0: min_x = 0
        if min_y > 0: min_y = 0
        if max_x < 4000: max_x = 4000
        if max_y < 2000: max_y = 2000
        
        
        plt.figure(figsize=(10, 6))
        plt.scatter(excit_neuron_locs[:, 0], excit_neuron_locs[:, 1], c='blue', label='Excitatory Neurons', s=75)
        #plt.scatter(inhib_neuron_locs[:, 0], inhib_neuron_locs[:, 1], c='blue', label='Inhibitory Neurons', s=10)
        #plot inhib with open circles in case of overlap
        plt.scatter(inhib_neuron_locs[:, 0], inhib_neuron_locs[:, 1], facecolors='none', edgecolors='red', label='Inhibitory Neurons', s=75)
        
        
        # make the dots smaller
        plt.title('Neuron Locations on MEA')
        plt.xlabel('X Coordinate (um)')
        plt.ylabel('Y Coordinate (um)')
        # plt.xlim(0, 2000)
        # plt.xlim(0, 4000)
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.legend()
        
        # plt.savefig(os.path.join(os.path.dirname(wfs_output_path), 'Neuron_Locations_on_MEA.png'))
        # print(f"Saved neuron locations plot to {os.path.join(os.path.dirname(wfs_output_path), 'Neuron_Locations_on_MEA.png')}")
        png_path = save_path
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        print(f"Saved neuron locations plot to {png_path}")
        print(f"Saved neuron locations plot to {pdf_path}")
        # import sys
        # sys.exit()

def get_data_obj_groups(h5_paths, raw_data_path, sorted_output_folders):
    # aw 2025-02-11
    # get recording details from each h5 file, check for match where all details match in sorted_output_folders
    # this way, we pair up recordings with their corresponding sorting output
    # get paired data objects - network analysis requires both recording and sorting objects
    
    well_data_list = []
    path_pairs = []
    for h5_path in h5_paths:
        
        # get recording details
        recording_details = mea.extract_recording_details(raw_data_path)[0] # NOTE: this works for a list of dirs or a single dir - but treats single dir as a list of a single dir
        
        #remove h5_file_path from recording_details - this wont match, this is the old path
        h5_path = recording_details.pop('h5_file_path')
        
        for sorted_output_folder in sorted_output_folders:         
            
            # shortform
            found = all([f'/{recording_details[key]}/' in sorted_output_folder for key in recording_details.keys()])
            
            if found:
                path_pairs.append((h5_path, sorted_output_folder))
                
                def load_three_objects():
                    # this function expects to load one sorting_obj, one recording_obj, and waveform data for the related well.
                    # there may be any number of rec_segments in the recording_obj, but only one sorting_obj
                    
                    # get stream_select from sorted_output_folder path
                    # look for the word 'well' in the string. It will beb followed by three digits.
                    # get the int value of those digits. Should be 0-5
                    stream_select = int(sorted_output_folder.split('well')[1][:3])
                    wellid = f'well{str(0).zfill(2)}{stream_select}'
                    
                    # try to load sorting object
                    try: sort_obj = mea.load_kilosort2_results(sorted_output_folder)
                    except Exception as e:
                        print(f"Error: Could not load sorting object for {sorted_output_folder}")
                        print(e)
                        sort_obj = e # put error in sort_obj for debugging
                    
                    #try to load recording object
                    try: 
                        _, well_recs, _, _ = mea.load_recordings(h5_path, stream_select=stream_select)
                        rec_segments = well_recs[wellid]
                    except Exception as e:
                        print(f"Error: Could not load recording object for {h5_path}")
                        print(e)
                        #well_recs = e
                        rec_segments = e
                        
                    #try to load waveform data
                    try:
                        #waveform_output_dir = os.path.join(sorted_output_folder, 'waveform_output')
                        waveform_output_dir = sorted_output_folder.replace('sorter_output', '')
                        #waveform_output_dir = os.path.join(waveform_output_dir)
                        waveform_output_dir = waveform_output_dir.replace('sorted', 'waveforms')
                        # register recordings to the sorting object before loading waveforms
                        # this is necessary for the waveform data to be loaded correctly
                        sort_obj.register_recording(rec_segments[0])
                        waveform_extractor = mea.load_waveforms(waveform_output_dir, sorting = sort_obj)
                    except Exception as e:
                        print(f"Error: Could not load waveform data for {h5_path}")
                        print(e)
                        waveform_extractor= e
                    
                    # return paired objects
                    return (rec_segments, sort_obj, waveform_extractor), recording_details                        
                well_data, recording_details = load_three_objects()
                well_data_list.append(well_data)
                
    return well_data_list, recording_details, path_pairs

def extract_network_features_v2(
    raw_data_paths, 
    sorted_data_dir = None, 
    output_dir = None, 
    stream_select=None, 
    plot=True, 
    conv_params=None,
    mega_params=None,
    limit_seconds=None,
    plot_wfs=False,
    max_workers = 4, # safe for all computers - if not specified, will use all available cores
    debug_mode = False,
    ):
    
    #init
    assert output_dir is not None, f"Error: output_dir is None"
    output_dir = os.path.abspath(output_dir)
    #assert os.path.exists(output_dir), f"Error: output_dir does not exist. {output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
                                                                                   
    #assert sorted_data_dir is not None, f"Error: sorted_data_dir is None"  
    raw_data_paths = [os.path.abspath(raw_data_path) for raw_data_path in raw_data_paths]
    sorted_data_dir = os.path.abspath(sorted_data_dir)
    
    #assert conv_params is defined, this is required for network analysis
    assert conv_params is not None, f"Error: conv_params is None - must be provided for network analysis"
    assert mega_params is not None, f"Error: mega_params is None - must be provided for network analysis"
    
    #iterate through raw data paths and get all h5 files
    h5_paths = []
    for raw_data_path in raw_data_paths:
        if os.path.isdir(raw_data_path):
            h5_paths.extend(glob.glob(os.path.join(raw_data_path, '**/*.h5'), recursive=True))
        else:
            h5_paths.append(raw_data_path)
                
    # iterate through sorter_output_dir and get all well output directories
    sorted_output_folders = []
    for root, dirs, files in os.walk(sorted_data_dir):
        if root.endswith('sorter_output'):
            if not os.path.exists(root):
                print(f"Error: sorted_output_folder does not exist. {root}")
                continue
            sorted_output_folders.append(os.path.join(root))
     
    well_data_list, recording_details, path_pairs = get_data_obj_groups(h5_paths, raw_data_paths, sorted_output_folders)
    
    # iterate through data_obj_list and get network metrics for each pair
    for well_data in well_data_list:
        try:
            # get recording, sorting, and waveform extractors
            recording_segments, sort_obj, wf_extractor = well_data      # NOTE: I think in general, for this whole process, we shouldnt need more than one
                        
            #if sort_obj is an error, skip
            if isinstance(sort_obj, Exception):
                #print(f"Error: Could not load sorting object for {sort_obj}")
                print(f"Error: Could not load sorting object.")
                print(f'Error details: {sort_obj}')
                print(f"Skipping well...")
                continue
            stream_id = recording_segments[0].stream_id
            stream_num = int(stream_id.split('well')[1][:3])
            kwargs = recording_details.copy()
            kwargs['plot_wfs'] = plot_wfs
            kwargs['max_workers'] = max_workers
            #wf_extractor = well_data[2]
            extract_network_metrics(
                recording_segments[0], # TODO: testing, only sending one seg.
                sort_obj,
                wf_extractor, 
                stream_num,
                conv_params,
                mega_params,                
                # mega_params,
                # conv_params, 
                output_dir,             
                plot=plot, 
                details=recording_details, 
                limit_seconds = limit_seconds,
                debug_mode = debug_mode, # limit number of units and bursts to analyze to get through functions quickly
                **kwargs)    
        except Exception as e:
            print(f"Error: Could not get network metrics for:")
            from pprint import pprint
            #pprint('recording_segments:', recording_segments)
            #pprint(recording_segments)
            pprint(recording_details)
            #pprint(sort_obj)
            print(e)
            continue         
    print('done')

def extract_network_metrics(
        recording_object, 
        sorting_object,
        wf_extractor, 
        stream_num, 
        conv_params,
        mega_params,
        #convolution_params_path, 
        output_path,
        save_path=None,
        bursting_plot_path=None, bursting_fig_path=None, 
        plot=False, 
        limit_seconds = None,
        debug_mode = False,
        **kwargs):
    
    # 
    assert sorting_object is not None, f"Error: sorting_object is None"
        
    # get network metrics
    get_activity = True
    if get_activity:
        well_id = f'well{str(0).zfill(2)}{stream_num}'
        well_recording_segment = recording_object 
        try: 
            network_metrics = get_experimental_network_activity_metrics(sorting_object, well_recording_segment, wf_extractor, conv_params, mega_params, debug_mode=debug_mode, **kwargs)
        except Exception as e:
            print(e)
            print(f"Error: Could not get network metrics for {well_id}")
        
    # get recording details
    print("Saving network metrics as numpy...")
    recording_details = kwargs['details']
    projectName = recording_details['projectName']
    date = recording_details['date']
    chipID = recording_details['chipID']
    scanType = recording_details['scanType']
    runID = recording_details['runID']
    
    # get sorting output dir from network metrics
    sorting_output_dir = network_metrics['sorting_output']
    # remove /sorter_output from sorting_output_dir
    sorting_output_dir = sorting_output_dir.replace('sorter_output', '')
    # replace sorted with network_metrics
    output_dir = sorting_output_dir.replace('sorted', 'network_metrics')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"network_metrics.npy")
    print(f"Saving network metrics to {save_path}")
    # save_path = os.path.join(output_path, f"network_metrics_well00{stream_num}.npy")
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, network_metrics)
    
    #plot network activity
    if plot:
        
        #plot neuron locations and class. Inhibitory neurons are red, excitatory neurons are blue.
        #save_path = save_path.replace('.npy', '_neuron_locations.pdf')
        #save_path = save_path.replace('.npy', '_neuron_locations.png')
        neuron_loc_plot_path = save_path.replace('network_metrics', 'neuron_loc_plots')
        neuron_loc_parent_dir = os.path.dirname(neuron_loc_plot_path)
        if not os.path.exists(neuron_loc_parent_dir):
            os.makedirs(neuron_loc_parent_dir)
        neuron_loc_plot_path = os.path.join(neuron_loc_parent_dir, f"neuron_locations.png")
        plot_neuron_locations_and_class(sorting_object, wf_extractor, network_metrics, save_path=neuron_loc_plot_path)
        
        
        try: 
            print("Generating network summary plot...")
            # aw 2025-01-20 17:25:21 - I guess I'll just plot both for now. I like how mine looks, but additional context is nice for Roy.
            network_plot_path = save_path.replace('network_metrics', 'network_plots')
            network_plot_parent_dir = os.path.dirname(network_plot_path)
            if not os.path.exists(network_plot_parent_dir):
                os.makedirs(network_plot_parent_dir)
            #network_plot_path = os.path.join(network_plot_parent_dir, f"network_summary_plot.pdf")
            network_plot_path_3p = os.path.join(network_plot_parent_dir, f"network_summary_3pannels.pdf")
            network_plot_path_2p = os.path.join(network_plot_parent_dir, f"network_summary_2pannels.pdf")
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_3p,
                #mode = '2p',
                mode = '3p',
                limit_seconds = limit_seconds,
                )
            
            plot_network_metrics(
                network_metrics, 
                bursting_plot_path, 
                bursting_fig_path,
                save_path=network_plot_path_2p,
                mode = '2p',
                limit_seconds = limit_seconds,
                )  
        except Exception as e:
            print(e)
            print(f"Error: Could not plot network activity for {well_id}")  

    # return network metrics and save path
    print(f"Saved network metrics to {save_path}")
    return network_metrics, save_path

def plot_network_metrics(
    network_metrics,
    bursting_plot_path,
    bursting_fig_path,
    save_path=None,
    mode='2p',
    limit_seconds = None,
    plot_class = False,
    ):
    
    # TODO: blend with plot comparison plot? I think.
    # aw 2025-02-21 00:57:16 - pretty sure this is done.
    
    #from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_summary
    from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.network_analysis import plot_network_summary_v3 as plot_network_summary
    
    # plot network activity
    # plot_network_summary(network_metrics, 
    #                      bursting_plot_path, 
    #                      bursting_fig_path, 
    #                      save_path=save_path, 
    #                      mode=mode,
    #                      limit_seconds=limit_seconds, 
    #                      plot_class=plot_class,
    #                      )  
    
    kwargs = {
        'bursting_plot_path': bursting_plot_path,
        'bursting_fig_path': bursting_fig_path,
        #'save_path': save_path,
        'mode': mode,
        'y_lim': (0, 16),
        'x_lim': (0, limit_seconds),
        'plot_class': plot_class,
        }
    plot_network_summary(network_metrics, **kwargs)
    
    # plot shorter plots for better view of bursting identification
    # save_path_35s = save_path.replace('.pdf', '_35s.pdf')
    # plot_network_summary(network_metrics, 
    #                      bursting_plot_path, 
    #                      bursting_fig_path, 
    #                      save_path=save_path_35s, 
    #                      mode=mode,
    #                      limit_seconds=35, 
    #                      plot_class=plot_class,
    #                     ) 
    
    #save_path_35s = save_path.replace('.pdf', '_35s.pdf')
    kwargs={
        'bursting_plot_path': bursting_plot_path,
        'bursting_fig_path': bursting_fig_path,
        #'save_path': save_path_35s,
        'mode': mode,
        'x_lim': (0, 35),
        'y_lim': (0, 16),
        'plot_class': plot_class,
    }
    plot_network_summary(network_metrics, **kwargs)
    
    # plot shorter plots for better view of bursting identification
    # save_path_60s = save_path.replace('.pdf', '_60s.pdf')
    # plot_network_summary(network_metrics, 
    #                      bursting_plot_path, 
    #                      bursting_fig_path, 
    #                      save_path=save_path_60s, 
    #                      mode=mode,
    #                      limit_seconds=60,
    #                      plot_class=plot_class,
    #                     )
    #save_path_60s = save_path.replace('.pdf', '_60s.pdf')
    # kwargs = {
    #     'bursting_plot_path': bursting_plot_path,
    #     'bursting_fig_path': bursting_fig_path,
    #     #'save_path': save_path_60s,
    #     'mode': mode,
    #     'x_lim': (0, 60),
    #     'y_lim': (0, 16),
    #     'plot_class': plot_class,
    # }
    # plot_network_summary(network_metrics, **kwargs)

    # 140s plot 
    kwargs = {
        'bursting_plot_path': bursting_plot_path,
        'bursting_fig_path': bursting_fig_path,
        #'save_path': save_path,
        'mode': mode,
        'x_lim': (0, 140),
        'y_lim': (0, 16),
        'plot_class': plot_class,
    }
    plot_network_summary(network_metrics, **kwargs)

''' Functions below this point predate 2025-02-11 21:11:58'''
# =============================================================================
# retired - # aw 2025-02-26 10:54:16
    # def extract_network_features(
    #     raw_data_paths, 
    #     sorted_data_dir = None, 
    #     output_dir = None, 
    #     stream_select=None, 
    #     plot=True, 
    #     conv_params=None,
    #     mega_params=None,
    #     limit_seconds=None,
    #     plot_wfs=False,
    #     ):
        
    #     #init
    #     assert output_dir is not None, f"Error: output_dir is None"
    #     output_dir = os.path.abspath(output_dir)
    #     #assert os.path.exists(output_dir), f"Error: output_dir does not exist. {output_dir}"
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
                                                                                    
    #     #assert sorted_data_dir is not None, f"Error: sorted_data_dir is None"  
    #     raw_data_paths = [os.path.abspath(raw_data_path) for raw_data_path in raw_data_paths]
    #     sorted_data_dir = os.path.abspath(sorted_data_dir)
        
    #     #assert conv_params is defined, this is required for network analysis
    #     assert conv_params is not None, f"Error: conv_params is None - must be provided for network analysis"
    #     assert mega_params is not None, f"Error: mega_params is None - must be provided for network analysis"
    #     # mega_params = conv_params.mega_params
    #     # conv_params = conv_params.conv_params
        
        
    #     #iterate through raw data paths and get all h5 files
    #     h5_paths = []
    #     for raw_data_path in raw_data_paths:
    #         if os.path.isdir(raw_data_path):
    #             h5_paths.extend(glob.glob(os.path.join(raw_data_path, '**/*.h5'), recursive=True))
    #         else:
    #             h5_paths.append(raw_data_path)
                    
    #     # iterate through sorter_output_dir and get all well output directories
    #     sorted_output_folders = []
    #     for root, dirs, files in os.walk(sorted_data_dir):
    #         # for dir in dirs:
    #         #     sorted_data_paths.append(os.path.join(root, dir))
    #         if root.endswith('sorter_output'):
    #             sorted_output_folders.append(os.path.join(root))
        
    #     # get recording details from each h5 file, check for match where all details match in sorted_output_folders
    #     # this way, we pair up recordings with their corresponding sorting output
    #     # get paired data objects - network analysis requires both recording and sorting objects
    #     def get_data_obj_pairs():
    #         well_data_list = []
    #         path_pairs = []
    #         for h5_path in h5_paths:
    #             # get recording details
    #             recording_details = mea.extract_recording_details(raw_data_path)[0]     # NOTE: this works for a list of dirs or a single dir - but treats single dir 
    #                                                                                         # as a list of a single dir
    #             # h5_file_path = recording_details['h5_file_path']
    #             # runID = recording_details['runID']
    #             # scanType = recording_details['scanType']
    #             # chipID = recording_details['chipID']
    #             # date = recording_details['date']
    #             # projectName = recording_details['projectName'] 
                
    #             #remove h5_file_path from recording_details - this wont match, this is the old path
    #             h5_path = recording_details.pop('h5_file_path')
                
    #             for sorted_output_folder in sorted_output_folders:         
                    
    #                 # check if all details match, long form for debugging
    #                 # elements = [recording_details[key] for key in recording_details.keys()]
    #                 # elements_in_sorted_output_folder = [element in sorted_output_folder for element in elements]
    #                 # found = all(elements_in_sorted_output_folder)
                    
    #                 # shortform
    #                 found = all([f'/{recording_details[key]}/' in sorted_output_folder for key in recording_details.keys()])
                    
    #                 if found:
    #                     path_pairs.append((h5_path, sorted_output_folder))
    #                     # get recording and sorting objects
    #                     #print(f"Found matching recording and sorting objects for {h5_path}")
                        
    #                     def load_recording_and_sorting_objects():
    #                         # this function expects to load one sorting_obj and one recording_obj for the related well.
    #                         # there may be any number of rec_segments in the recording_obj, but only one sorting_obj
                            
    #                         # get stream_select from sorted_output_folder path
    #                         # look for the word 'well' in the string. It will beb followed by three digits.
    #                         # get the int value of those digits. Should be 0-5
    #                         stream_select = int(sorted_output_folder.split('well')[1][:3])
    #                         wellid = f'well{str(0).zfill(2)}{stream_select}'
                            
    #                         # try to load sorting object
    #                         try: sort_obj = mea.load_kilosort2_results(sorted_output_folder)
    #                         except Exception as e:
    #                             print(f"Error: Could not load sorting object for {sorted_output_folder}")
    #                             print(e)
    #                             sort_obj = e # put error in sort_obj for debugging
                            
    #                         #try to load recording object
    #                         try: 
    #                             _, well_recs, _, _ = mea.load_recordings(h5_path, stream_select=stream_select)
    #                             rec_segments = well_recs[wellid]
    #                         except Exception as e:
    #                             print(f"Error: Could not load recording object for {h5_path}")
    #                             print(e)
    #                             #well_recs = e
    #                             rec_segments = e
                            
    #                         # return paired objects
    #                         return (rec_segments, sort_obj), recording_details                        
    #                         #print(f"Loaded recording and sorting objects for {h5_path}")
    #                     well_data, recording_details = load_recording_and_sorting_objects()
    #                     well_data_list.append(well_data)
                        
    #         return well_data_list, recording_details, path_pairs
    #     well_data_list, recording_details, path_pairs = get_data_obj_pairs()
        
    #     # iterate through data_obj_list and get network metrics for each pair
    #     for well_data in well_data_list:
    #         try:
    #             recording_segments, sort_obj = well_data    # NOTE: I think in general, for this whole process, we shouldnt need more than one
    #                                                         # recording segment per well - we just need to get attributes from the recording object
    #                                                         # TODO: if so, reduce the amount of data we're passing around to just the attributes we need.
                
    #             #if sort_obj is an error, skip
    #             if isinstance(sort_obj, Exception):
    #                 #print(f"Error: Could not load sorting object for {sort_obj}")
    #                 print(f"Error: Could not load sorting object.")
    #                 print(f'Error details: {sort_obj}')
    #                 print(f"Skipping well...")
    #                 continue
    #             stream_id = recording_segments[0].stream_id
    #             stream_num = int(stream_id.split('well')[1][:3])
    #             kwargs = recording_details.copy()
    #             kwargs['plot_wfs'] = plot_wfs
    #             extract_network_metrics(
    #                 recording_segments[0], # TODO: testing, only sending one seg.
    #                 sort_obj, 
    #                 stream_num,
    #                 conv_params,
    #                 mega_params,                
    #                 # mega_params,
    #                 # conv_params, 
    #                 output_dir,             
    #                 plot=plot, details=recording_details, 
    #                 limit_seconds = limit_seconds,
    #                 **kwargs)    
    #         except Exception as e:
    #             print(f"Error: Could not get network metrics for:")
    #             from pprint import pprint
    #             #pprint('recording_segments:', recording_segments)
    #             #pprint(recording_segments)
    #             pprint(recording_details)
    #             pprint(sort_obj)
    #             print(e)
    #             continue         
    #     print('done')
    # =============================================================================
    # reference
    # def save_network_metrics(recording_object, sorting_object, stream_num, 
    #                          convolution_params_path, output_path,
    #                          bursting_plot_path=None, bursting_fig_path=None, 
    #                          plot=False, **kwargs):
    #     add_repo_root_to_sys_path()
    #     #import _external.RBS_network_simulation_optimization_tools.modules.analysis.analyze_network_activity as ana
    #     import RBS_network_models.developing.utils.analyze_network_activity as ana 
    #     #load recordings and sorting data                    
    #     #get recording segment - note: network scans should only have one recording segment
    #     well_id = f'well{str(0).zfill(2)}{stream_num}'
    #     #recording_object = kwargs['recording_object']
    #     well_recording_segment = recording_object[well_id]['recording_segments'][0]  #note: network scans should only have one recording segment
        
    #     #get network activity metrics
    #     #import conv_params from convolution_params_path
    #     conv_params = import_module_from_path(convolution_params_path)
    #     conv_params = conv_params.conv_params
    #     #sorting_object = kwargs['sorting_object']
    #     network_metrics = ana.get_experimental_network_activity_metrics(sorting_object, well_recording_segment, conv_params)
        
    #     # save network metrics
    #     print("Saving network metrics as numpy...")
    #     recording_details = kwargs['details']
    #     projectName = recording_details['projectName']
    #     date = recording_details['date']
    #     chipID = recording_details['chipID']
    #     scanType = recording_details['scanType']
    #     runID = recording_details['runID']
    #     save_path = os.path.join(output_path, f"{projectName}_{date}_{chipID}_{scanType}_{runID}_network_metrics_well00{stream_num}.npy")
    #     #save_path = os.path.join(output_path, f"network_metrics_well00{stream_num}.npy")
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     np.save(save_path, network_metrics)
        
    #     #plot network activity
    #     print("Plotting network activity...")
    #     if plot:
    #         fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    #         if bursting_plot_path is None and bursting_fig_path is None:
    #             #modify save path
    #             raster_plot_path = save_path.replace('.npy', '_raster_plot.pdf')
    #             raster_fig_path = save_path.replace('.npy', '_raster_fig.pkl')
    #             raster_plot_path_png = save_path.replace('.npy', '_raster_plot.png')
    #             bursting_plot_path = save_path.replace('.npy', '_bursting_plot.pdf')
    #             bursting_fig_path = save_path.replace('.npy', '_bursting_fig.pkl')
    #             bursting_plot_path_png = save_path.replace('.npy', '_bursting_plot.png')
    #         # assert bursting_plot_path is not None, f"Error: bursting_plot_path is None"
    #         # assert bursting_fig_path is not None, f"Error: bursting_fig_path is None"
    #         # bursting_plot_path_png = save_path.replace('.npy', '_bursting_plot.png')

    #         #generate_network_bursting_plot(network_metrics, bursting_plot_path, bursting_fig_path)
    #         print("Generating raster plot...")
    #         spiking_data_by_unit = network_metrics['spiking_data']['spiking_data_by_unit']
    #         ax[0] = plot_raster_plot_experimental(ax[0], spiking_data_by_unit)       
    #         print("Generating network bursting plot...")
    #         bursting_ax = network_metrics['bursting_data']['bursting_summary_data']['ax']
    #         ax[1] = plot_network_bursting_experimental(ax[1], bursting_ax)
            
    #         #save plots
    #         plt.tight_layout()
    #         fig.savefig(bursting_plot_path) #save as pdf
    #         print(f"Network summary plot saved to {bursting_plot_path}")
    #         fig.savefig(bursting_plot_path_png, dpi=600) #save as png
    #         print(f"Network summary plot saved to {bursting_plot_path_png}")
        
    #     print(f"Saved network metrics to {save_path}")

    # def load_recording_and_sorting_object_tuples(recording_paths, sorting_output_parent_path, stream_select=None):
    #     paired_objects = []
    #     for recording_path in recording_paths:
        
    #         #prep sorting objects
    #         sorting_output_paths = check_for_sorting_objects_associated_to_recordings(
    #             recording_paths, 
    #             sorting_output_parent_path
    #             )
            
    #         recording_details = mea_lib.extract_recording_details(recording_path)[0]
            
    #         for sorting_output_path in sorting_output_paths:
    #             stream_sorting_object = load_spike_sorted_data(sorting_output_path)

    #             #load recordings and sorting data
    #             MaxID, recording_object, expected_well_count, rec_counts = mea_lib.load_recordings(recording_path, stream_select=stream_select)
                
    #             # Append the paired objects as a tuple
    #             paired_objects.append((recording_object, stream_sorting_object, recording_details))
                
    #     return paired_objects

    # def load_recording_and_sorting_object_tuples_dep(recording_paths, sorting_output_parent_path, stream_select=None):
    #     paired_objects = []
    #     for recording_path in recording_paths:
        
    #         #prep sorting objects
    #         sorting_output_paths = check_for_sorting_objects_associated_to_recordings(
    #             recording_paths, 
    #             sorting_output_parent_path
    #             )
            
    #         recording_details = mea_lib.extract_recording_details(recording_path)[0]
            
    #         for sorting_output_path in sorting_output_paths:
    #             stream_sorting_object = load_spike_sorted_data(sorting_output_path)

    #             #load recordings and sorting data
    #             MaxID, recording_object, expected_well_count, rec_counts = mea_lib.load_recordings(recording_path, stream_select=stream_select)
                
    #             # Append the paired objects as a tuple
    #             paired_objects.append((recording_object, stream_sorting_object, recording_details))
                
    #     return paired_objects

