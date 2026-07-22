

## aw 2025-05-16 10:00:19 lets try only including mega burst rate
fit_schema = {
    'source': False,
    'trimmed': False,
    
    #'spiking_data':False,
    # 2025-05-19 17:46:57 - including spiking data now after optimizing burst rate.
    'spiking_data':{
        'spike_times': True,
        'spiking_times_by_unit': False,
        'spiking_metrics_by_unit': False,

        # firing rates
        'frs': { # aw 2025-05-23 13:45:35 current mean target aroun 2.5 spikes/s, max around 21 spikes/s min around 1 spikes/s
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 0.001, # value int/float submitted to fitness function
            'max_val': 22.0, # value int/float submitted to fitness function
            #'max_val': 40.0, # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values 
                                # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
        },
        'i_frs': {
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 0.25, # value int/float submitted to fitness function
            'max_val': 22,  # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values
        }, # aw 2025-05-27 12:56:59
        'e_frs': {
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 0.15, # value int/float submitted to fitness function
            'max_val': 3.5,  # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values
        }, # aw 2025-05-27 12:57:01
        'u_frs': False,

        # number of neurons actually firing
        'num_e_firing': { # aw 2025-05-20 15:11:08 current target=203
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 99, # value int/float submitted to fitness function
            'max_val': 203,  # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, 
            }, # aw 2025-05-27 12:57:00
            # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values
                                # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
        'num_i_firing': { # aw 2025-05-20 15:11:08 current target=98
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 40, # arbitrary, a little less than half of target e_firing
            'max_val': 98,  # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values
                                # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
        }, # aw 2025-05-27 12:57:02
        'num_total_firing': False, # aw 2025-05-20 15:11:08 current target=301

        # spike counts
        'E_spikes': True, # aw 2025-05-20 15:11:08 false - time dependent, so only comparable to experimental value if simulation time is the same
        'I_spikes': True, # aw 2025-05-20 15:11:08 false - time dependent, so only comparable to experimental value if simulation time is the same
        'U_spikes': True, # aw 2025-05-20 15:11:08 false - time dependent, so only comparable to experimental value if simulation time is the same
        
        # fr ratios
        'EI_fr_ratios': { # # aw 2025-05-20 15:11:08 current target=0.16742355633713643
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 0.001, # value int/float submitted to fitness function
            'max_val': 0.5,  # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values 
                                # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
        },

        #spiking ratios
        'EI_spike_ratios': { # aw 2025-05-20 15:11:08
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 0.001, # value int/float submitted to fitness function
            'max_val': 2,  # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values
                                # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
        }, # aw 2025-05-27 13:04:43

        # interspike intervals
        'isi': True,
        'i_isi': True,
        'e_isi': True,
        'u_isi': True,
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
        'baseline': { # aw 2025-05-23 12:00:31 current target is around 2.5
            'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
            'min_val': 2.0, # value int/float submitted to fitness function
            #'max_val': 10.0, # value int/float submitted to fitness function
            'max_val': 4.0, # value int/float submitted to fitness function
            'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
            'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values 
                                # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
        },  
        'burst_metrics': {
            'num_bursts': True,
            'burst_rate': { # aw 2025-05-20 16:16:30 current target=0.19333333333333333
                'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
                'min_val': 0.02, # value int/float submitted to fitness function
                'max_val': 1.0, # value int/float submitted to fitness function
                'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
                'normalize': True, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values 
                                    # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
            },
            'burst_ids': False,
            'ibi': True,

            # aw 2025-05-25 22:56:22 - tightening up burst amplitude
            #  mean: 4.636081156813823
            #  min:  1.7842541797570484
            #  max: 9.519041012337574
            'burst_amp': {
                'include': True, # boolean - if True, this will be included in fitness scoring. This is absolutely required if wanting to pass min, max, and normalize keys to handle_inclusion in fitness function
                'min_val': 0.25, # value int/float submitted to fitness function
                'max_val': 11.0, # value int/float submitted to fitness function
                #'max_val': 20.0, # value int/float submitted to fitness function
                'weight': 1.0, # value int/float submitted to fitness function - bigger values widen the trough of fitcurve
                'normalize': False, # boolean - if True, after fitness scoring, score will be adjusted - normalized to the range of seed values 
                                    # NOTE: ignored if seeds are not provided. This should be used to pin down already optimized values
            },
            'burst_duration': True,
            'burst_parts': True,
            'num_units_per_burst': True,
            'in_burst_fr': True,                        
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


# aw 2025-05-20 15:01:33
# old format

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
