'''
netParams.py
Specifications for CDKL5_DIV21 network model using NetPyNE
'''

#version control
#version = 0.0 # prior to 28Dec2024
#version = 1.0 # major updates on 28Dec2024
#version = 2.0 # aw 2025-02-04 14:00:39 - added support for specifying cell positions and types
version = 3.0 # aw 2025-03-12 10:28:16 - adding support for specific cell positions and gids aligning with reference data
if version == 3.0:
    from netpyne import specs
    try:
        from __main__ import cfg  # import SimConfig object with params from parent module
    except:
        from .cfg import cfg  # if no simConfig in parent module, import directly from tut8_cfg module
    import numpy as np
    import os
    
    ###############################################################################
    #
    # NETWORK PARAMETERS
    #
    ###############################################################################

    netParams = specs.NetParams()   # object of class NetParams to store the network parameters
    
    ###############################################################################
    #
    # Helper functions
    #
    ###############################################################################
    
    def dynamic_positive_normal(mean, std, size):    
        if std is None or std < 1e-12:
            return np.full(size, max(mean, 0.0))
        from scipy.stats import truncnorm
        a, b = (0 - mean) / std, float("inf")
        initial_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=size * 2)
        left_truncated_count = np.sum(initial_values < 0)
        upper_truncation_value = np.percentile(initial_values, 100 - (left_truncated_count / len(initial_values)) * 100)
        b_dynamic = (upper_truncation_value - mean) / std
        values = truncnorm.rvs(a, b_dynamic, loc=mean, scale=std, size=size)
        return values
    
    def init_normally_distributed_properties(cfg, num_excite, num_inhib):

        excitatory_props = {
            'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, num_excite),
            'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, num_excite),
            'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, num_excite),
            'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, num_excite),
            'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, num_excite),
        }
        inhibitory_props = {
            'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, num_inhib),
            'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, num_inhib),
            'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, num_inhib),
            'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, num_inhib),
            'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, num_inhib),
        }
        return excitatory_props, inhibitory_props
    
    def generate_clustered_neuronal_positions(y_range, x_range, z_range, num_cells=500, n_clusters=5):
        """
        Generate realistically clustered positions of neurons in a rectangular culture.
        
        Parameters:
            y_range (list): Range of Y coordinates [min, max].
            x_range (list): Range of X coordinates [min, max].
            z_range (list): Range of Z coordinates [min, max].
            num_cells (int): Total number of neurons to generate.
            n_clusters (int): Number of clusters to simulate.
        
        Returns:
            np.ndarray: Array of clustered neuron positions (x, y, z).
        """
        # Generate cluster centers
        cluster_centers = np.column_stack((
            np.random.uniform(x_range[0], x_range[1], n_clusters),  # X cluster centers
            np.random.uniform(y_range[0], y_range[1], n_clusters),  # Y cluster centers
            np.random.uniform(z_range[0], z_range[1], n_clusters)   # Z cluster centers
        ))
        
        # Distribute cells among clusters
        cells_per_cluster = [num_cells // n_clusters] * n_clusters
        remainder = num_cells % n_clusters
        for i in range(remainder):
            cells_per_cluster[i] += 1  # Distribute remainder randomly
        
        # Generate points for each cluster
        positions = []
        for i, center in enumerate(cluster_centers):
            cluster_points = np.random.multivariate_normal(
                mean=center, 
                cov=np.diag([100, 100, 1]),  # Covariance matrix for spread
                size=cells_per_cluster[i]
            )
            positions.append(cluster_points)
        
        # Combine all cluster points
        positions = np.vstack(positions)
        
        return positions
    
    def config_cell_params(cell_type, props, num_cells, positions):
        # Default 3D space ranges for random placement  
            
        #build individual cell rules for each cell
        if hasattr(netParams, 'cellsLists') is False:
            netParams.cellsLists = {}
        netParams.cellsLists[cell_type] = []
        secsList = []
        cellLabelList = []
        #for i in range(num_cells):
        i = -1 # props are indexed, not dict with gids
        for gid, pos in positions.items():
            #pos = positions[i]
            i += 1
            x, y, z = pos
            cellsLabel = f'{cell_type}{gid}'
            #i = gid # #HACK
            cellsLabel = str(cellsLabel) #convert to string to avoid serialization issues
            cellRule = {
                #'pop': cell_type,
                #'cellLabel': cellsLabel,
                #'cellModel': 'HH',                   
                # 'cellType': cellsLabel,
                'conds': {                          # I finally understgand how conds works. This sets the conditions required for population
                                                    # attributes to be applied to the cell.
                                                    # In this case, the cellType, cellModel, cellLabel, and pop must be equal to the values
                                                    # specified later in the popParams dictionary.
                                                    # I do cell params before pop params so I can pass cellsList 
                                                    # to popParams - but now I'm not sure if I need to do that at all.
                                                    # everything is based on cellTags I guess.
                    #'cellType': cell_type,
                    #'cellType': cellsLabel, #This helps with loading the simulation data later
                    'cellModel': 'HH',
                    'pop': cell_type,
                    'cellLabel': cellsLabel,
                    #'x': x, 'y': y, 'z': z,
                },
                'secs': {
                    'soma': {
                        'geom': {
                            'L': props['L'][i],
                            'diam': props['diam'][i],
                            'Ra': props['Ra'][i],
                            ## apprently I can also pass position this way
                            #'pt3d': [x, y, z, diam], # x, y, z, diameter #nvm idk how to get this to validate
                            
                            #REVISED COMMENT:
                            # I could do this if I wwanted. what this is useful for is building morphologies
                            # of specific compartments.
                            #the correct way to implement:
                            # 'pt3d': [
                            #    (0, 0, 0, 1),
                            #    (0, 0, 1, 1),
                            #    (0, 0, 2, 1),
                            # etc... a list of tuples of x, y, z, diameter
                            # ],
                            # TODO: I will need to do this eventually.
                            # TODO: extract morphologies from CDKL5_DIV21 and get them into this format
                        },
                        'mechs': {
                            'hh': {
                                'gnabar': props['gnabar'][i],
                                'gkbar': props['gkbar'][i],
                                'gl': 0.003, 'el': -70,
                            }
                        },
                    }
                },
                #'x': x, 'y': y, 'z': z,
            },
            cellRule = cellRule[0]
            netParams.cellsLists[cell_type].append(
                {
                    'cellType': cellsLabel,
                    'cellLabel': cellsLabel,
                    'x': x, 'y': y, 'z': z,     #yup, the only thing cellsList is good for is to pass exact positions to the popParams
                    #'cellModel': 'HH',         #Let me ammend that, cellsList is meant to pass positions and popParams to specific cells
                    #                           #cellLabel lets me apply specific cell rules to specifc cells
                    #                           #any other conds that can be applied globably to all cells in a population
                    #                           #should be applied in popParams config
                    #maybe over here?
                    #'pt3d': (x, y, z, props['diam'][i]), # x, y, z, diameter NOPE SEE ABOVE FOR REVISED COMMENT
                }
            )
            netParams.cellParams[cellsLabel] = cellRule         
        #print(f'Created {num_cells} cells of type {cell_type}')
        #print(f'netParams.cellParams configured for {cell_type}')
    
    def config_pop_params(cell_type, cfg, props, num_cells):
        #assert cellsList[cell_type] is a list
        assert isinstance(netParams.cellsLists[cell_type], list), f'netParams.cellsLists[{cell_type}] is not a list'
        netParams.popParams[cell_type] = {
            #'cellType': cell_type,
            'pop': cell_type,
            'numCells': num_cells,
            'cellModel': 'HH',
            'cellsList': netParams.cellsLists[cell_type],
            #'cellModel': 'HH',
            #'diversity': True,
        }
        #print(f'Created {num_cells} populations of type {cell_type}')
        #print(f'netParams.popParams configured for {cell_type}')
    
    def select_cells_from_positions(positions, num_excite, num_inhib):
            # Shuffle indices to ensure random selection
            indices = np.arange(len(positions))
            np.random.shuffle(indices)
            
            # Select cells for excitatory and inhibitory populations
            E_cells = indices[:num_excite]
            I_cells = indices[num_excite:num_excite + num_inhib]
            
            posE = positions[E_cells]
            posI = positions[I_cells]
            
            return posE, posI
    
    def config_connectivity(cfg):
        from time import time
        start = time()
        #print('Static normal applied')
        #print('Generating static normal values for connectivity properties...')
        conn_specs = {
            'E->I': {'pre': 'E', 'post': 'I', 'synMech': 'exc', 
                    'prob': cfg.probEI, 
                    'weight': cfg.weightEI},
            'I->E': {'pre': 'I', 'post': 'E', 'synMech': 'inh', 
                    'prob': cfg.probIE, 
                    'weight': cfg.weightIE},
            'E->E': {'pre': 'E', 'post': 'E', 'synMech': 'exc', 
                    'prob': cfg.probEE, 
                    'weight': cfg.weightEE},
            'I->I': {'pre': 'I', 'post': 'I', 'synMech': 'inh', 
                    'prob': cfg.probII, 
                    'weight': cfg.weightII},
        }
        for key, spec in conn_specs.items():
            netParams.connParams[key] = {
                'preConds': {
                    #'cellType': spec['pre']
                    'pop': spec['pre']
                    },
                'postConds': {
                    #'cellType': spec['post']
                    'pop': spec['post']
                    },
                #'probability': spec['prob'],   # NOTE: apparently I've been forgetting to apply problengthconst 
                                                # and decay to the probability...good thing I'm doing this
                                                # sensitivity analysis thing...
                'probability': 'exp(-dist_3D / {})*{}'.format(cfg.probLengthConst, spec['prob']),  
                'weight': spec['weight'],
                'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                'synMech': spec['synMech']
            }

        netParams.synMechParams['exc'] = {
            'mod': 'Exp2Syn', 
            'tau1': cfg.tau1_exc,
            'tau2': cfg.tau2_exc, 
            'e': 0
            }
        netParams.synMechParams['inh'] = {
            'mod': 'Exp2Syn', 
            'tau1': cfg.tau1_inh, 
            'tau2': cfg.tau2_inh, 
            'e': -75
            }
        
        #print('Connectivity properties generated')
        time_elapsed = time() - start
        #print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
    
    def save_network(cfg, filename = None):
        #assert that cfg.saveFolder and cfg.simLabel are not '' or None
        try:
            assert cfg.saveFolder, 'cfg.saveFolder not found'
            assert cfg.simLabel, 'cfg.simLabel not found'
        except:
            import warnings
            warnings.simplefilter('always')
            
            # warning, if running a batch process but unable to find cfg.saveFolder and cfg.simLabel,
            # something is wrong.
            # however if running a single simulation, this is normal
            warning_message = ('\n'
                                'WARNING: cfg.saveFolder and cfg.simLabel not found.\n'
                                'This is normal if running a single simulation.\n'
                                'Otherwise, something is wrong.\n'
                                'cfg.saveFolder and cfg.simLabel have been set to "../tests/outputs" and "test" \n'
                                'respectively.'
                                )
            warnings.warn(warning_message)
            
            #print('generating temporary saveFolder and simLabel...')
            cfg.saveFolder = '../tests/outputs'
            cfg.saveFolder = os.path.abspath(cfg.saveFolder)
            if not os.path.exists(cfg.saveFolder):
                os.makedirs(cfg.saveFolder)
            cfg.simLabel = 'test'
        output_path = os.path.join(cfg.saveFolder, f"{cfg.simLabel}_netParams.json")
        filename = output_path # if filename is None else filename maybe implement this later
        netParams.save(filename)
        #print(f'Network parameters saved to {filename}')
    ###############################################################################
    #
    # Initialize normally distributed properties and cell positions
    #
    ###############################################################################
    
    #extract num_excite and num_inhib from cfg
    num_excite = cfg.num_excite
    num_inhib = cfg.num_inhib
    
    # initialize normally distributed properties
    excitatory_props, inhibitory_props = init_normally_distributed_properties(cfg, num_excite, num_inhib)
    
    # cfg locations
    if cfg.locations_known is False:
        DEFAULT_Y_RANGE = [6.72, 2065.09]
        DEFAULT_X_RANGE = [173.66, 3549.91]
        DEFAULT_Z_RANGE = [1, 11.61]
        netParams.sizeX = 4000
        netParams.sizeY = 2100
        netParams.sizeZ = 12
        
        #Generate clustered positions of neurons in a rectangular culture
        num_cells = num_excite + num_inhib
        positions = generate_clustered_neuronal_positions(DEFAULT_Y_RANGE, DEFAULT_X_RANGE, DEFAULT_Z_RANGE, 
                                                        num_cells=num_cells, 
                                                        n_clusters=5
                                                        )
        
        #of the generated positions, randomly select subsets for excitatory and inhibitory cells
        posE, posI = select_cells_from_positions(positions, num_excite, num_inhib)
        #print(type(posE))
        print('positions generated') 
    elif cfg.locations_known is True:
        features_path = cfg.features_path
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features path {features_path} does not exist.")
        
        # Load unit locations from HDF5 experimental features file
        import h5py
        unit_locations = {}
        with h5py.File(features_path, 'r') as _f:
            for _key in _f.keys():
                if _key.startswith('unit_'):
                    _uid_str = _key[len('unit_'):]
                    try:
                        _uid = int(_uid_str)
                    except ValueError:
                        _uid = _uid_str
                    _attrs = _f[_key].attrs
                    _x = float(_attrs['loc_x'])
                    _y = float(_attrs['loc_y'])
                    _z = float(_attrs['loc_z']) if 'loc_z' in _attrs else 0.0
                    unit_locations[_uid] = (_x, _y, _z)

        inhib_units = cfg.inhib_units
        excit_units = cfg.excit_units

        # derive posE and posI similar to the above method
        posE = {gid: pos for gid, pos in unit_locations.items() if gid in excit_units}
        posI = {gid: pos for gid, pos in unit_locations.items() if gid in inhib_units}
        print('positions loaded from experimental HDF5')
    ###############################################################################
    #
    # Config cell, pop, and connectivity params
    #
    ###############################################################################
    
    #config cell params
    config_cell_params('E', excitatory_props, num_excite, posE)
    config_cell_params('I', inhibitory_props, num_inhib, posI)
    
    #config pop params
    config_pop_params('E', cfg, excitatory_props, num_excite)
    config_pop_params('I', cfg, inhibitory_props, num_inhib)
    
    #config connectivity params
    config_connectivity(cfg)
    
    ###############################################################################
    #
    # Save network
    #
    ###############################################################################
    
    #save network
    #save_network(cfg)  
elif version == 2.0:
    from netpyne import specs
    try:
        from __main__ import cfg  # import SimConfig object with params from parent module
    except:
        from .cfg import cfg  # if no simConfig in parent module, import directly from tut8_cfg module
    import numpy as np
    import os
    
    ###############################################################################
    #
    # NETWORK PARAMETERS
    #
    ###############################################################################

    netParams = specs.NetParams()   # object of class NetParams to store the network parameters
    
    ###############################################################################
    #
    # Helper functions
    #
    ###############################################################################
    
    def dynamic_positive_normal(mean, std, size):    
        if std is None or std < 1e-12:
            return np.full(size, max(mean, 0.0))
        from scipy.stats import truncnorm
        a, b = (0 - mean) / std, float("inf")
        initial_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=size * 2)
        left_truncated_count = np.sum(initial_values < 0)
        upper_truncation_value = np.percentile(initial_values, 100 - (left_truncated_count / len(initial_values)) * 100)
        b_dynamic = (upper_truncation_value - mean) / std
        values = truncnorm.rvs(a, b_dynamic, loc=mean, scale=std, size=size)
        return values
        
    ###############################################################################
    #
    # Initialize normally distributed properties and cell positions
    #
    ###############################################################################
    
    #extract num_excite and num_inhib from cfg
    num_excite = cfg.num_excite
    num_inhib = cfg.num_inhib
    
    # initialize normally distributed properties
    def init_normally_distributed_properties(cfg, num_excite, num_inhib):

        excitatory_props = {
            'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, num_excite),
            'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, num_excite),
            'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, num_excite),
            'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, num_excite),
            'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, num_excite),
        }
        inhibitory_props = {
            'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, num_inhib),
            'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, num_inhib),
            'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, num_inhib),
            'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, num_inhib),
            'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, num_inhib),
        }
        return excitatory_props, inhibitory_props
    excitatory_props, inhibitory_props = init_normally_distributed_properties(cfg, num_excite, num_inhib)
    
    #set size of 3D space
    #for new method
    #these are position params taken from old KCNT1 data I think...these are a place holder.
    #TODO: get position data from CDKL5_DIV21 data
        # aw 2025-02-04 14:13:25 - adding support for specifying cell positions and types now
    #cfg.locations_known = False
    if cfg.locations_known is False:
        DEFAULT_Y_RANGE = [6.72, 2065.09]
        DEFAULT_X_RANGE = [173.66, 3549.91]
        DEFAULT_Z_RANGE = [1, 11.61]
        netParams.sizeX = 4000
        netParams.sizeY = 2100
        netParams.sizeZ = 12
        
        #Generate clustered positions of neurons in a rectangular culture
        num_cells = num_excite + num_inhib
        def generate_clustered_neuronal_positions(y_range, x_range, z_range, num_cells=500, n_clusters=5):
            """
            Generate realistically clustered positions of neurons in a rectangular culture.
            
            Parameters:
                y_range (list): Range of Y coordinates [min, max].
                x_range (list): Range of X coordinates [min, max].
                z_range (list): Range of Z coordinates [min, max].
                num_cells (int): Total number of neurons to generate.
                n_clusters (int): Number of clusters to simulate.
            
            Returns:
                np.ndarray: Array of clustered neuron positions (x, y, z).
            """
            # Generate cluster centers
            cluster_centers = np.column_stack((
                np.random.uniform(x_range[0], x_range[1], n_clusters),  # X cluster centers
                np.random.uniform(y_range[0], y_range[1], n_clusters),  # Y cluster centers
                np.random.uniform(z_range[0], z_range[1], n_clusters)   # Z cluster centers
            ))
            
            # Distribute cells among clusters
            cells_per_cluster = [num_cells // n_clusters] * n_clusters
            remainder = num_cells % n_clusters
            for i in range(remainder):
                cells_per_cluster[i] += 1  # Distribute remainder randomly
            
            # Generate points for each cluster
            positions = []
            for i, center in enumerate(cluster_centers):
                cluster_points = np.random.multivariate_normal(
                    mean=center, 
                    cov=np.diag([100, 100, 1]),  # Covariance matrix for spread
                    size=cells_per_cluster[i]
                )
                positions.append(cluster_points)
            
            # Combine all cluster points
            positions = np.vstack(positions)
            
            return positions
        positions = generate_clustered_neuronal_positions(DEFAULT_Y_RANGE, DEFAULT_X_RANGE, DEFAULT_Z_RANGE, 
                                                        num_cells=num_cells, 
                                                        n_clusters=5
                                                        )
        
        #of the generated positions, randomly select subsets for excitatory and inhibitory cells
        def select_cells_from_positions(positions, num_excite, num_inhib):
            # Shuffle indices to ensure random selection
            indices = np.arange(len(positions))
            np.random.shuffle(indices)
            
            # Select cells for excitatory and inhibitory populations
            E_cells = indices[:num_excite]
            I_cells = indices[num_excite:num_excite + num_inhib]
            
            posE = positions[E_cells]
            posI = positions[I_cells]
            
            return posE, posI
        posE, posI = select_cells_from_positions(positions, num_excite, num_inhib)
        #print(type(posE))
        print('positions generated') 
    elif cfg.locations_known is True:
        unit_locations = cfg.unit_locations
        inhib_units = cfg.inhib_units
        excit_units = cfg.excit_units
        
        # derive posE and posI similar to the above method
        # posE = unit_locations[excit_units]
        # posI = unit_locations[inhib_units]
        posE = [unit_locations[str(i)] for i in excit_units]
        posI = [unit_locations[str(i)] for i in inhib_units]
        # converst from list of tuples to numpy array or numpy arrays to align with the above method
        posE = np.array(posE)
        posI = np.array(posI)
        
        # print(type(posE))
        # print(type(posE[0])) # verify that this is a numpy array
        print('positions loaded from cfg')
    ###############################################################################
    #
    # Config cell, pop, and connectivity params
    #
    ###############################################################################
    
    #config cell params
    def config_cell_params(cell_type, props, num_cells, positions):
        # Default 3D space ranges for random placement  
            
        #build individual cell rules for each cell
        if hasattr(netParams, 'cellsLists') is False:
            netParams.cellsLists = {}
        netParams.cellsLists[cell_type] = []
        secsList = []
        cellLabelList = []
        for i in range(num_cells):
            pos = positions[i]
            x, y, z = pos
            #make sure all values are floats, not np.float64
            #diam = float(props['diam'][i])
            #x, y, z = float(x), float(y), float(z)
            cellsLabel = f'{cell_type}{i}' # TODO: this doesnt save as a pickle correctly? maybe? I'm not sure.
            cellsLabel = str(cellsLabel) #convert to string to avoid serialization issues
            cellRule = {
                #'pop': cell_type,
                #'cellLabel': cellsLabel,
                #'cellModel': 'HH',                   
                'cellType': cellsLabel,
                'conds': {                          # I finally understgand how conds works. This sets the conditions required for population
                                                    # attributes to be applied to the cell.
                                                    # In this case, the cellType, cellModel, cellLabel, and pop must be equal to the values
                                                    # specified later in the popParams dictionary.
                                                    # I do cell params before pop params so I can pass cellsList 
                                                    # to popParams - but now I'm not sure if I need to do that at all.
                                                    # everything is based on cellTags I guess.
                    #'cellType': cell_type,
                    #'cellType': cellsLabel, #This helps with loading the simulation data later
                    'cellModel': 'HH',
                    'pop': cell_type,
                    'cellLabel': cellsLabel,
                    #'x': x, 'y': y, 'z': z,
                },
                'secs': {
                    'soma': {
                        'geom': {
                            'L': props['L'][i],
                            'diam': props['diam'][i],
                            'Ra': props['Ra'][i],
                            ## apprently I can also pass position this way
                            #'pt3d': [x, y, z, diam], # x, y, z, diameter #nvm idk how to get this to validate
                            
                            #REVISED COMMENT:
                            # I could do this if I wwanted. what this is useful for is building morphologies
                            # of specific compartments.
                            #the correct way to implement:
                            # 'pt3d': [
                            #    (0, 0, 0, 1),
                            #    (0, 0, 1, 1),
                            #    (0, 0, 2, 1),
                            # etc... a list of tuples of x, y, z, diameter
                            # ],
                            # TODO: I will need to do this eventually.
                            # TODO: extract morphologies from CDKL5_DIV21 and get them into this format
                        },
                        'mechs': {
                            'hh': {
                                'gnabar': props['gnabar'][i],
                                'gkbar': props['gkbar'][i],
                                'gl': 0.003, 'el': -70,
                            }
                        },
                    }
                },
                #'x': x, 'y': y, 'z': z,
            },
            cellRule = cellRule[0]
            netParams.cellsLists[cell_type].append(
                {
                    'cellType': cellsLabel,
                    'cellLabel': cellsLabel,
                    'x': x, 'y': y, 'z': z,     #yup, the only thing cellsList is good for is to pass exact positions to the popParams
                    #'cellModel': 'HH',         #Let me ammend that, cellsList is meant to pass positions and popParams to specific cells
                    #                           #cellLabel lets me apply specific cell rules to specifc cells
                    #                           #any other conds that can be applied globably to all cells in a population
                    #                           #should be applied in popParams config
                    #maybe over here?
                    #'pt3d': (x, y, z, props['diam'][i]), # x, y, z, diameter NOPE SEE ABOVE FOR REVISED COMMENT
                }
            )
            netParams.cellParams[cellsLabel] = cellRule         
        #print(f'Created {num_cells} cells of type {cell_type}')
        #print(f'netParams.cellParams configured for {cell_type}')
    config_cell_params('E', excitatory_props, num_excite, posE)
    config_cell_params('I', inhibitory_props, num_inhib, posI)
    
    #config pop params
    def config_pop_params(cell_type, cfg, props, num_cells):
        #assert cellsList[cell_type] is a list
        assert isinstance(netParams.cellsLists[cell_type], list), f'netParams.cellsLists[{cell_type}] is not a list'
        netParams.popParams[cell_type] = {
            #'cellType': cell_type,
            'pop': cell_type,
            'numCells': num_cells,
            'cellModel': 'HH',
            'cellsList': netParams.cellsLists[cell_type],
            #'cellModel': 'HH',
            #'diversity': True,
        }
        #print(f'Created {num_cells} populations of type {cell_type}')
        #print(f'netParams.popParams configured for {cell_type}')
    config_pop_params('E', cfg, excitatory_props, num_excite)
    config_pop_params('I', cfg, inhibitory_props, num_inhib)
    
    #config connectivity params
    def config_connectivity(cfg):
        from time import time
        start = time()
        #print('Static normal applied')
        #print('Generating static normal values for connectivity properties...')
        conn_specs = {
            'E->I': {'pre': 'E', 'post': 'I', 'synMech': 'exc', 
                    'prob': cfg.probEI, 
                    'weight': cfg.weightEI},
            'I->E': {'pre': 'I', 'post': 'E', 'synMech': 'inh', 
                    'prob': cfg.probIE, 
                    'weight': cfg.weightIE},
            'E->E': {'pre': 'E', 'post': 'E', 'synMech': 'exc', 
                    'prob': cfg.probEE, 
                    'weight': cfg.weightEE},
            'I->I': {'pre': 'I', 'post': 'I', 'synMech': 'inh', 
                    'prob': cfg.probII, 
                    'weight': cfg.weightII},
        }
        for key, spec in conn_specs.items():
            netParams.connParams[key] = {
                'preConds': {
                    #'cellType': spec['pre']
                    'pop': spec['pre']
                    },
                'postConds': {
                    #'cellType': spec['post']
                    'pop': spec['post']
                    },
                #'probability': spec['prob'],   # NOTE: apparently I've been forgetting to apply problengthconst 
                                                # and decay to the probability...good thing I'm doing this
                                                # sensitivity analysis thing...
                'probability': 'exp(-dist_3D / {})*{}'.format(cfg.probLengthConst, spec['prob']),  
                'weight': spec['weight'],
                'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                'synMech': spec['synMech']
            }

        netParams.synMechParams['exc'] = {
            'mod': 'Exp2Syn', 
            'tau1': cfg.tau1_exc,
            'tau2': cfg.tau2_exc, 
            'e': 0
            }
        netParams.synMechParams['inh'] = {
            'mod': 'Exp2Syn', 
            'tau1': cfg.tau1_inh, 
            'tau2': cfg.tau2_inh, 
            'e': -75
            }
        
        #print('Connectivity properties generated')
        time_elapsed = time() - start
        #print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
    config_connectivity(cfg)
    
    ###############################################################################
    #
    # Save network
    #
    ###############################################################################
    
    #save network
    def save_network(cfg, filename = None):
        #assert that cfg.saveFolder and cfg.simLabel are not '' or None
        try:
            assert cfg.saveFolder, 'cfg.saveFolder not found'
            assert cfg.simLabel, 'cfg.simLabel not found'
        except:
            import warnings
            warnings.simplefilter('always')
            
            # warning, if running a batch process but unable to find cfg.saveFolder and cfg.simLabel,
            # something is wrong.
            # however if running a single simulation, this is normal
            warning_message = ('\n'
                                'WARNING: cfg.saveFolder and cfg.simLabel not found.\n'
                                'This is normal if running a single simulation.\n'
                                'Otherwise, something is wrong.\n'
                                'cfg.saveFolder and cfg.simLabel have been set to "../tests/outputs" and "test" \n'
                                'respectively.'
                                )
            warnings.warn(warning_message)
            
            #print('generating temporary saveFolder and simLabel...')
            cfg.saveFolder = '../tests/outputs'
            cfg.saveFolder = os.path.abspath(cfg.saveFolder)
            if not os.path.exists(cfg.saveFolder):
                os.makedirs(cfg.saveFolder)
            cfg.simLabel = 'test'
        output_path = os.path.join(cfg.saveFolder, f"{cfg.simLabel}_netParams.json")
        filename = output_path # if filename is None else filename maybe implement this later
        netParams.save(filename)
        #print(f'Network parameters saved to {filename}')
    #save_network(cfg)   
elif version == 1.0:
    from netpyne import specs
    try:
        from __main__ import cfg  # import SimConfig object with params from parent module
    except:
        from .cfg import cfg  # if no simConfig in parent module, import directly from tut8_cfg module
    import numpy as np
    import os
    
    ###############################################################################
    #
    # NETWORK PARAMETERS
    #
    ###############################################################################

    netParams = specs.NetParams()   # object of class NetParams to store the network parameters
    
    ###############################################################################
    #
    # Helper functions
    #
    ###############################################################################
    
    def dynamic_positive_normal(mean, std, size):    
        if std is None or std < 1e-12:
            return np.full(size, max(mean, 0.0))
        from scipy.stats import truncnorm
        a, b = (0 - mean) / std, float("inf")
        initial_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=size * 2)
        left_truncated_count = np.sum(initial_values < 0)
        upper_truncation_value = np.percentile(initial_values, 100 - (left_truncated_count / len(initial_values)) * 100)
        b_dynamic = (upper_truncation_value - mean) / std
        values = truncnorm.rvs(a, b_dynamic, loc=mean, scale=std, size=size)
        return values
        
    ###############################################################################
    #
    # Initialize normally distributed properties and cell positions
    #
    ###############################################################################
    
    #extract num_excite and num_inhib from cfg
    num_excite = cfg.num_excite
    num_inhib = cfg.num_inhib
    
    # initialize normally distributed properties
    def init_normally_distributed_properties(cfg, num_excite, num_inhib):

        excitatory_props = {
            'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, num_excite),
            'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, num_excite),
            'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, num_excite),
            'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, num_excite),
            'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, num_excite),
        }
        inhibitory_props = {
            'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, num_inhib),
            'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, num_inhib),
            'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, num_inhib),
            'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, num_inhib),
            'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, num_inhib),
        }
        return excitatory_props, inhibitory_props
    excitatory_props, inhibitory_props = init_normally_distributed_properties(cfg, num_excite, num_inhib)
    
    #set size of 3D space
    #for new method
    #these are position params taken from old KCNT1 data I think...these are a place holder.
    #TODO: get position data from CDKL5_DIV21 data
    DEFAULT_Y_RANGE = [6.72, 2065.09]
    DEFAULT_X_RANGE = [173.66, 3549.91]
    DEFAULT_Z_RANGE = [1, 11.61]
    netParams.sizeX = 4000
    netParams.sizeY = 2100
    netParams.sizeZ = 12
    
    #Generate clustered positions of neurons in a rectangular culture
    num_cells = num_excite + num_inhib
    def generate_clustered_neuronal_positions(y_range, x_range, z_range, num_cells=500, n_clusters=5):
        """
        Generate realistically clustered positions of neurons in a rectangular culture.
        
        Parameters:
            y_range (list): Range of Y coordinates [min, max].
            x_range (list): Range of X coordinates [min, max].
            z_range (list): Range of Z coordinates [min, max].
            num_cells (int): Total number of neurons to generate.
            n_clusters (int): Number of clusters to simulate.
        
        Returns:
            np.ndarray: Array of clustered neuron positions (x, y, z).
        """
        # Generate cluster centers
        cluster_centers = np.column_stack((
            np.random.uniform(x_range[0], x_range[1], n_clusters),  # X cluster centers
            np.random.uniform(y_range[0], y_range[1], n_clusters),  # Y cluster centers
            np.random.uniform(z_range[0], z_range[1], n_clusters)   # Z cluster centers
        ))
        
        # Distribute cells among clusters
        cells_per_cluster = [num_cells // n_clusters] * n_clusters
        remainder = num_cells % n_clusters
        for i in range(remainder):
            cells_per_cluster[i] += 1  # Distribute remainder randomly
        
        # Generate points for each cluster
        positions = []
        for i, center in enumerate(cluster_centers):
            cluster_points = np.random.multivariate_normal(
                mean=center, 
                cov=np.diag([100, 100, 1]),  # Covariance matrix for spread
                size=cells_per_cluster[i]
            )
            positions.append(cluster_points)
        
        # Combine all cluster points
        positions = np.vstack(positions)
        
        return positions
    positions = generate_clustered_neuronal_positions(DEFAULT_Y_RANGE, DEFAULT_X_RANGE, DEFAULT_Z_RANGE, 
                                                    num_cells=num_cells, 
                                                    n_clusters=5
                                                    )
    
    #of the generated positions, randomly select subsets for excitatory and inhibitory cells
    def select_cells_from_positions(positions, num_excite, num_inhib):
        # Shuffle indices to ensure random selection
        indices = np.arange(len(positions))
        np.random.shuffle(indices)
        
        # Select cells for excitatory and inhibitory populations
        E_cells = indices[:num_excite]
        I_cells = indices[num_excite:num_excite + num_inhib]
        
        posE = positions[E_cells]
        posI = positions[I_cells]
        
        return posE, posI
    posE, posI = select_cells_from_positions(positions, num_excite, num_inhib) 
    
    ###############################################################################
    #
    # Config cell, pop, and connectivity params
    #
    ###############################################################################
    
    #config cell params
    def config_cell_params(cell_type, props, num_cells, positions):
        # Default 3D space ranges for random placement  
            
        #build individual cell rules for each cell
        if hasattr(netParams, 'cellsLists') is False:
            netParams.cellsLists = {}
        netParams.cellsLists[cell_type] = []
        secsList = []
        cellLabelList = []
        for i in range(num_cells):
            pos = positions[i]
            x, y, z = pos
            #make sure all values are floats, not np.float64
            #diam = float(props['diam'][i])
            #x, y, z = float(x), float(y), float(z)
            cellsLabel = f'{cell_type}{i}' # TODO: this doesnt save as a pickle correctly? maybe? I'm not sure.
            cellsLabel = str(cellsLabel) #convert to string to avoid serialization issues
            cellRule = {
                #'pop': cell_type,
                #'cellLabel': cellsLabel,
                #'cellModel': 'HH',                   
                'cellType': cellsLabel,
                'conds': {                          # I finally understgand how conds works. This sets the conditions required for population
                                                    # attributes to be applied to the cell.
                                                    # In this case, the cellType, cellModel, cellLabel, and pop must be equal to the values
                                                    # specified later in the popParams dictionary.
                                                    # I do cell params before pop params so I can pass cellsList 
                                                    # to popParams - but now I'm not sure if I need to do that at all.
                                                    # everything is based on cellTags I guess.
                    #'cellType': cell_type,
                    #'cellType': cellsLabel, #This helps with loading the simulation data later
                    'cellModel': 'HH',
                    'pop': cell_type,
                    'cellLabel': cellsLabel,
                    #'x': x, 'y': y, 'z': z,
                },
                'secs': {
                    'soma': {
                        'geom': {
                            'L': props['L'][i],
                            'diam': props['diam'][i],
                            'Ra': props['Ra'][i],
                            ## apprently I can also pass position this way
                            #'pt3d': [x, y, z, diam], # x, y, z, diameter #nvm idk how to get this to validate
                            
                            #REVISED COMMENT:
                            # I could do this if I wwanted. what this is useful for is building morphologies
                            # of specific compartments.
                            #the correct way to implement:
                            # 'pt3d': [
                            #    (0, 0, 0, 1),
                            #    (0, 0, 1, 1),
                            #    (0, 0, 2, 1),
                            # etc... a list of tuples of x, y, z, diameter
                            # ],
                            # TODO: I will need to do this eventually.
                            # TODO: extract morphologies from CDKL5_DIV21 and get them into this format
                        },
                        'mechs': {
                            'hh': {
                                'gnabar': props['gnabar'][i],
                                'gkbar': props['gkbar'][i],
                                'gl': 0.003, 'el': -70,
                            }
                        },
                    }
                },
                #'x': x, 'y': y, 'z': z,
            },
            cellRule = cellRule[0]
            netParams.cellsLists[cell_type].append(
                {
                    'cellType': cellsLabel,
                    'cellLabel': cellsLabel,
                    'x': x, 'y': y, 'z': z,     #yup, the only thing cellsList is good for is to pass exact positions to the popParams
                    #'cellModel': 'HH',         #Let me ammend that, cellsList is meant to pass positions and popParams to specific cells
                    #                           #cellLabel lets me apply specific cell rules to specifc cells
                    #                           #any other conds that can be applied globably to all cells in a population
                    #                           #should be applied in popParams config
                    #maybe over here?
                    #'pt3d': (x, y, z, props['diam'][i]), # x, y, z, diameter NOPE SEE ABOVE FOR REVISED COMMENT
                }
            )
            netParams.cellParams[cellsLabel] = cellRule         
        #print(f'Created {num_cells} cells of type {cell_type}')
        #print(f'netParams.cellParams configured for {cell_type}')
    config_cell_params('E', excitatory_props, num_excite, posE)
    config_cell_params('I', inhibitory_props, num_inhib, posI)
    
    #config pop params
    def config_pop_params(cell_type, cfg, props, num_cells):
        #assert cellsList[cell_type] is a list
        assert isinstance(netParams.cellsLists[cell_type], list), f'netParams.cellsLists[{cell_type}] is not a list'
        netParams.popParams[cell_type] = {
            #'cellType': cell_type,
            'pop': cell_type,
            'numCells': num_cells,
            'cellModel': 'HH',
            'cellsList': netParams.cellsLists[cell_type],
            #'cellModel': 'HH',
            #'diversity': True,
        }
        #print(f'Created {num_cells} populations of type {cell_type}')
        #print(f'netParams.popParams configured for {cell_type}')
    config_pop_params('E', cfg, excitatory_props, num_excite)
    config_pop_params('I', cfg, inhibitory_props, num_inhib)
    
    #config connectivity params
    def config_connectivity(cfg):
        from time import time
        start = time()
        #print('Static normal applied')
        #print('Generating static normal values for connectivity properties...')
        conn_specs = {
            'E->I': {'pre': 'E', 'post': 'I', 'synMech': 'exc', 
                    'prob': cfg.probEI, 
                    'weight': cfg.weightEI},
            'I->E': {'pre': 'I', 'post': 'E', 'synMech': 'inh', 
                    'prob': cfg.probIE, 
                    'weight': cfg.weightIE},
            'E->E': {'pre': 'E', 'post': 'E', 'synMech': 'exc', 
                    'prob': cfg.probEE, 
                    'weight': cfg.weightEE},
            'I->I': {'pre': 'I', 'post': 'I', 'synMech': 'inh', 
                    'prob': cfg.probII, 
                    'weight': cfg.weightII},
        }
        for key, spec in conn_specs.items():
            netParams.connParams[key] = {
                'preConds': {
                    #'cellType': spec['pre']
                    'pop': spec['pre']
                    },
                'postConds': {
                    #'cellType': spec['post']
                    'pop': spec['post']
                    },
                #'probability': spec['prob'],   # NOTE: apparently I've been forgetting to apply problengthconst 
                                                # and decay to the probability...good thing I'm doing this
                                                # sensitivity analysis thing...
                'probability': 'exp(-dist_3D / {})*{}'.format(cfg.probLengthConst, spec['prob']),  
                'weight': spec['weight'],
                'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                'synMech': spec['synMech']
            }

        netParams.synMechParams['exc'] = {
            'mod': 'Exp2Syn', 
            'tau1': cfg.tau1_exc,
            'tau2': cfg.tau2_exc, 
            'e': 0
            }
        netParams.synMechParams['inh'] = {
            'mod': 'Exp2Syn', 
            'tau1': cfg.tau1_inh, 
            'tau2': cfg.tau2_inh, 
            'e': -75
            }
        
        #print('Connectivity properties generated')
        time_elapsed = time() - start
        #print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
    config_connectivity(cfg)
    
    ###############################################################################
    #
    # Save network
    #
    ###############################################################################
    
    #save network
    def save_network(cfg, filename = None):
        #assert that cfg.saveFolder and cfg.simLabel are not '' or None
        try:
            assert cfg.saveFolder, 'cfg.saveFolder not found'
            assert cfg.simLabel, 'cfg.simLabel not found'
        except:
            import warnings
            warnings.simplefilter('always')
            
            # warning, if running a batch process but unable to find cfg.saveFolder and cfg.simLabel,
            # something is wrong.
            # however if running a single simulation, this is normal
            warning_message = ('\n'
                                'WARNING: cfg.saveFolder and cfg.simLabel not found.\n'
                                'This is normal if running a single simulation.\n'
                                'Otherwise, something is wrong.\n'
                                'cfg.saveFolder and cfg.simLabel have been set to "../tests/outputs" and "test" \n'
                                'respectively.'
                                )
            warnings.warn(warning_message)
            
            #print('generating temporary saveFolder and simLabel...')
            cfg.saveFolder = '../tests/outputs'
            cfg.saveFolder = os.path.abspath(cfg.saveFolder)
            if not os.path.exists(cfg.saveFolder):
                os.makedirs(cfg.saveFolder)
            cfg.simLabel = 'test'
        output_path = os.path.join(cfg.saveFolder, f"{cfg.simLabel}_netParams.json")
        filename = output_path # if filename is None else filename maybe implement this later
        netParams.save(filename)
        #print(f'Network parameters saved to {filename}')
    #save_network(cfg)
elif version == 0.0:
    import os
    import numpy as np
    from netpyne import specs
    from scipy.stats import truncnorm
    #from CDKL5_DIV21.utils.sim_helper import import_module_from_path
    from workspace.RBS_neuronal_network_models.optimizing.CDKL5_DIV21.scripts_dep.sim_helper import import_module_from_path
    import numpy as np
    #from netpyne.batchtools import specs
            
    def dynamic_positive_normal(mean, std, size):    
        if std is None or std < 1e-12:
            return np.full(size, max(mean, 0.0))
        a, b = (0 - mean) / std, float("inf")
        initial_values = truncnorm.rvs(a, b, loc=mean, scale=std, size=size * 2)
        left_truncated_count = np.sum(initial_values < 0)
        upper_truncation_value = np.percentile(initial_values, 100 - (left_truncated_count / len(initial_values)) * 100)
        b_dynamic = (upper_truncation_value - mean) / std
        values = truncnorm.rvs(a, b_dynamic, loc=mean, scale=std, size=size)
        return values

    def main(cfg):
        
        #extract num_excite and num_inhib from cfg
        num_excite = cfg.num_excite
        num_inhib = cfg.num_inhib
        
        # initialize normally distributed properties
        def init_normally_distributed_properties(cfg, num_excite, num_inhib):

            excitatory_props = {
                'L': dynamic_positive_normal(cfg.E_L_mean, cfg.E_L_stdev, num_excite),
                'diam': dynamic_positive_normal(cfg.E_diam_mean, cfg.E_diam_stdev, num_excite),
                'Ra': dynamic_positive_normal(cfg.E_Ra_mean, cfg.E_Ra_stdev, num_excite),
                'gnabar': dynamic_positive_normal(cfg.gnabar_E, cfg.gnabar_E_std, num_excite),
                'gkbar': dynamic_positive_normal(cfg.gkbar_E, cfg.gkbar_E_std, num_excite),
            }
            inhibitory_props = {
                'L': dynamic_positive_normal(cfg.I_L_mean, cfg.I_L_stdev, num_inhib),
                'diam': dynamic_positive_normal(cfg.I_diam_mean, cfg.I_diam_stdev, num_inhib),
                'Ra': dynamic_positive_normal(cfg.I_Ra_mean, cfg.I_Ra_stdev, num_inhib),
                'gnabar': dynamic_positive_normal(cfg.gnabar_I, cfg.gnabar_I_std, num_inhib),
                'gkbar': dynamic_positive_normal(cfg.gkbar_I, cfg.gkbar_I_std, num_inhib),
            }
            return excitatory_props, inhibitory_props
        excitatory_props, inhibitory_props = init_normally_distributed_properties(cfg, num_excite, num_inhib)
        
        #set size of 3D space
        #for new method
        #these are position params taken from old KCNT1 data I think...these are a place holder.
        #TODO: get position data from CDKL5_DIV21 data
        DEFAULT_Y_RANGE = [6.72, 2065.09]
        DEFAULT_X_RANGE = [173.66, 3549.91]
        DEFAULT_Z_RANGE = [1, 11.61]
        netParams.sizeX = 4000
        netParams.sizeY = 2100
        netParams.sizeZ = 12
        
        #Generate clustered positions of neurons in a rectangular culture
        num_cells = num_excite + num_inhib
        def generate_clustered_neuronal_positions(y_range, x_range, z_range, num_cells=500, n_clusters=5):
            """
            Generate realistically clustered positions of neurons in a rectangular culture.
            
            Parameters:
                y_range (list): Range of Y coordinates [min, max].
                x_range (list): Range of X coordinates [min, max].
                z_range (list): Range of Z coordinates [min, max].
                num_cells (int): Total number of neurons to generate.
                n_clusters (int): Number of clusters to simulate.
            
            Returns:
                np.ndarray: Array of clustered neuron positions (x, y, z).
            """
            # Generate cluster centers
            cluster_centers = np.column_stack((
                np.random.uniform(x_range[0], x_range[1], n_clusters),  # X cluster centers
                np.random.uniform(y_range[0], y_range[1], n_clusters),  # Y cluster centers
                np.random.uniform(z_range[0], z_range[1], n_clusters)   # Z cluster centers
            ))
            
            # Distribute cells among clusters
            cells_per_cluster = [num_cells // n_clusters] * n_clusters
            remainder = num_cells % n_clusters
            for i in range(remainder):
                cells_per_cluster[i] += 1  # Distribute remainder randomly
            
            # Generate points for each cluster
            positions = []
            for i, center in enumerate(cluster_centers):
                cluster_points = np.random.multivariate_normal(
                    mean=center, 
                    cov=np.diag([100, 100, 1]),  # Covariance matrix for spread
                    size=cells_per_cluster[i]
                )
                positions.append(cluster_points)
            
            # Combine all cluster points
            positions = np.vstack(positions)
            
            return positions
        positions = generate_clustered_neuronal_positions(DEFAULT_Y_RANGE, DEFAULT_X_RANGE, DEFAULT_Z_RANGE, 
                                                        num_cells=num_cells, 
                                                        n_clusters=5
                                                        )
        
        #of the generated positions, randomly select subsets for excitatory and inhibitory cells
        def select_cells_from_positions(positions, num_excite, num_inhib):
            # Shuffle indices to ensure random selection
            indices = np.arange(len(positions))
            np.random.shuffle(indices)
            
            # Select cells for excitatory and inhibitory populations
            E_cells = indices[:num_excite]
            I_cells = indices[num_excite:num_excite + num_inhib]
            
            posE = positions[E_cells]
            posI = positions[I_cells]
            
            return posE, posI
        posE, posI = select_cells_from_positions(positions, num_excite, num_inhib) 
        
        #config cell params
        def config_cell_params(cell_type, props, num_cells, positions):
            # Default 3D space ranges for random placement  
                
            #build individual cell rules for each cell
            if hasattr(netParams, 'cellsLists') is False:
                netParams.cellsLists = {}
            netParams.cellsLists[cell_type] = []
            secsList = []
            cellLabelList = []
            for i in range(num_cells):
                pos = positions[i]
                x, y, z = pos
                #make sure all values are floats, not np.float64
                #diam = float(props['diam'][i])
                #x, y, z = float(x), float(y), float(z)
                cellsLabel = f'{cell_type}{i}' # TODO: this doesnt save as a pickle correctly? maybe? I'm not sure.
                cellsLabel = str(cellsLabel) #convert to string to avoid serialization issues
                cellRule = {
                    #'pop': cell_type,
                    #'cellLabel': cellsLabel,
                    #'cellModel': 'HH',                   
                    'cellType': cellsLabel,
                    'conds': {                          # I finally understgand how conds works. This sets the conditions required for population
                                                        # attributes to be applied to the cell.
                                                        # In this case, the cellType, cellModel, cellLabel, and pop must be equal to the values
                                                        # specified later in the popParams dictionary.
                                                        # I do cell params before pop params so I can pass cellsList 
                                                        # to popParams - but now I'm not sure if I need to do that at all.
                                                        # everything is based on cellTags I guess.
                        #'cellType': cell_type,
                        #'cellType': cellsLabel, #This helps with loading the simulation data later
                        'cellModel': 'HH',
                        'pop': cell_type,
                        'cellLabel': cellsLabel,
                        #'x': x, 'y': y, 'z': z,
                    },
                    'secs': {
                        'soma': {
                            'geom': {
                                'L': props['L'][i],
                                'diam': props['diam'][i],
                                'Ra': props['Ra'][i],
                                ## apprently I can also pass position this way
                                #'pt3d': [x, y, z, diam], # x, y, z, diameter #nvm idk how to get this to validate
                                
                                #REVISED COMMENT:
                                # I could do this if I wwanted. what this is useful for is building morphologies
                                # of specific compartments.
                                #the correct way to implement:
                                # 'pt3d': [
                                #    (0, 0, 0, 1),
                                #    (0, 0, 1, 1),
                                #    (0, 0, 2, 1),
                                # etc... a list of tuples of x, y, z, diameter
                                # ],
                                # TODO: I will need to do this eventually.
                                # TODO: extract morphologies from CDKL5_DIV21 and get them into this format
                            },
                            'mechs': {
                                'hh': {
                                    'gnabar': props['gnabar'][i],
                                    'gkbar': props['gkbar'][i],
                                    'gl': 0.003, 'el': -70,
                                }
                            },
                        }
                    },
                    #'x': x, 'y': y, 'z': z,
                },
                cellRule = cellRule[0]
                netParams.cellsLists[cell_type].append(
                    {
                        'cellType': cellsLabel,
                        'cellLabel': cellsLabel,
                        'x': x, 'y': y, 'z': z,     #yup, the only thing cellsList is good for is to pass exact positions to the popParams
                        #'cellModel': 'HH',         #Let me ammend that, cellsList is meant to pass positions and popParams to specific cells
                        #                           #cellLabel lets me apply specific cell rules to specifc cells
                        #                           #any other conds that can be applied globably to all cells in a population
                        #                           #should be applied in popParams config
                        #maybe over here?
                        #'pt3d': (x, y, z, props['diam'][i]), # x, y, z, diameter NOPE SEE ABOVE FOR REVISED COMMENT
                    }
                )
                netParams.cellParams[cellsLabel] = cellRule         
            #print(f'Created {num_cells} cells of type {cell_type}')
            #print(f'netParams.cellParams configured for {cell_type}')
        config_cell_params('E', excitatory_props, num_excite, posE)
        config_cell_params('I', inhibitory_props, num_inhib, posI)
        
        #config pop params
        def config_pop_params(cell_type, cfg, props, num_cells):
            #assert cellsList[cell_type] is a list
            assert isinstance(netParams.cellsLists[cell_type], list), f'netParams.cellsLists[{cell_type}] is not a list'
            netParams.popParams[cell_type] = {
                #'cellType': cell_type,
                'pop': cell_type,
                'numCells': num_cells,
                'cellModel': 'HH',
                'cellsList': netParams.cellsLists[cell_type],
                #'cellModel': 'HH',
                #'diversity': True,
            }
            #print(f'Created {num_cells} populations of type {cell_type}')
            #print(f'netParams.popParams configured for {cell_type}')
        config_pop_params('E', cfg, excitatory_props, num_excite)
        config_pop_params('I', cfg, inhibitory_props, num_inhib)
        
        #config connectivity params
        def config_connectivity(cfg):
            from time import time
            start = time()
            #print('Static normal applied')
            #print('Generating static normal values for connectivity properties...')
            conn_specs = {
                'E->I': {'pre': 'E', 'post': 'I', 'synMech': 'exc', 
                        'prob': cfg.probEI, 
                        'weight': cfg.weightEI},
                'I->E': {'pre': 'I', 'post': 'E', 'synMech': 'inh', 
                        'prob': cfg.probIE, 
                        'weight': cfg.weightIE},
                'E->E': {'pre': 'E', 'post': 'E', 'synMech': 'exc', 
                        'prob': cfg.probEE, 
                        'weight': cfg.weightEE},
                'I->I': {'pre': 'I', 'post': 'I', 'synMech': 'inh', 
                        'prob': cfg.probII, 
                        'weight': cfg.weightII},
            }
            for key, spec in conn_specs.items():
                netParams.connParams[key] = {
                    'preConds': {
                        #'cellType': spec['pre']
                        'pop': spec['pre']
                        },
                    'postConds': {
                        #'cellType': spec['post']
                        'pop': spec['post']
                        },
                    #'probability': spec['prob'],   # NOTE: apparently I've been forgetting to apply problengthconst 
                                                    # and decay to the probability...good thing I'm doing this
                                                    # sensitivity analysis thing...
                    'probability': 'exp(-dist_3D / {})*{}'.format(cfg.probLengthConst, spec['prob']),  
                    'weight': spec['weight'],
                    'delay': 'dist_3D / {}'.format(cfg.propVelocity),
                    'synMech': spec['synMech']
                }

            netParams.synMechParams['exc'] = {
                'mod': 'Exp2Syn', 
                'tau1': cfg.tau1_exc,
                'tau2': cfg.tau2_exc, 
                'e': 0
                }
            netParams.synMechParams['inh'] = {
                'mod': 'Exp2Syn', 
                'tau1': cfg.tau1_inh, 
                'tau2': cfg.tau2_inh, 
                'e': -75
                }
            
            #print('Connectivity properties generated')
            time_elapsed = time() - start
            #print(f'Connectivity properties generated in {time_elapsed:.2f} seconds')
        config_connectivity(cfg)
        
        #save network
        def save_network(cfg, filename = None):
            #assert that cfg.saveFolder and cfg.simLabel are not '' or None
            try:
                assert cfg.saveFolder, 'cfg.saveFolder not found'
                assert cfg.simLabel, 'cfg.simLabel not found'
            except:
                import warnings
                warnings.simplefilter('always')
                
                # warning, if running a batch process but unable to find cfg.saveFolder and cfg.simLabel,
                # something is wrong.
                # however if running a single simulation, this is normal
                warning_message = ('\n'
                                    'WARNING: cfg.saveFolder and cfg.simLabel not found.\n'
                                    'This is normal if running a single simulation.\n'
                                    'Otherwise, something is wrong.\n')
                warnings.warn(warning_message)
                
                #print('generating temporary saveFolder and simLabel...')
                cfg.saveFolder = './testing'
                cfg.saveFolder = os.path.abspath(cfg.saveFolder)
                if not os.path.exists(cfg.saveFolder):
                    os.makedirs(cfg.saveFolder)
                cfg.simLabel = 'test'
            output_path = os.path.join(cfg.saveFolder, f"{cfg.simLabel}_netParams.json")
            filename = output_path # if filename is None else filename maybe implement this later
            netParams.save(filename)
            #print(f'Network parameters saved to {filename}')
        save_network(cfg)

    # Import cfg and run
    try:
        from __main__ import cfg
        
        # Initialize NetParams object
        netParams = specs.NetParams()
        
        # Run main function
        main(cfg)
    except ImportError:
        raise Exception("cfg not found")
