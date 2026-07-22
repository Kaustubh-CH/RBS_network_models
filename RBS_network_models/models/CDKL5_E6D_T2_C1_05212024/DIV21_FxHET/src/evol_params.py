'''
evolutionary parameter space for CDKL5_DIV21 project
'''
from netpyne import specs
#version = 1.0
version = 2.0

# aw 2025-02-07 16:47:42 - maximizing plausible ranges for each param
if version == 2.0:
    # Evolutionary Parameters
    params = specs.ODict()

    # Propagation Parameters
    params['propVelocity'] = [0.1, 0.3]  # Conduction velocities in mammalian neurons range from ~0.1 m/s in unmyelinated fibers to over 100 m/s in large myelinated axons. [Source: Wikipedia - Axon](https://en.wikipedia.org/wiki/Axon)
    # NOTE:
        # Propagation Parameters
        # # For neuronal cultures from dissociated mouse pup cortices (unmyelinated)
        # params['propVelocity_mouse_cultures'] = [0.1, 0.3]  # Conduction velocities in unmyelinated neurons range from ~0.1 to 0.3 m/s. [Source: Engineering a 3D functional human peripheral nerve in vitro using the Nerve-on-a-Chip platform](https://www.nature.com/articles/s41598-019-45407-5)

        # # For human iPSC-derived brain organoids (partially myelinated)
        # params['propVelocity_human_organoids'] = [0.1, 1.0]  # Partially myelinated fibers in organoids can have conduction velocities up to ~1.0 m/s. [Source: A Human Brain Microphysiological System Derived from Induced Pluripotent Stem Cells](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6047513/)


    # Morphology Parameters (Excitatory and Inhibitory Cells)
    params.update({

        'E_diam_stdev': [0, 12],  # Dendritic diameters in excitatory neurons can vary significantly, with standard deviations up to 10 µm observed in cortical pyramidal cells. [Source: Nature - Morphological diversity of single neurons](https://www.nature.com/articles/s41586-021-03941-1)
        'E_L_stdev': [0, 150],  # Dendritic lengths in excitatory neurons show variability, with standard deviations reaching up to 100 µm, reflecting diverse branching patterns. [Source: Nature - Morphological diversity of single neurons](https://www.nature.com/articles/s41586-021-03941-1)
        'E_Ra_stdev': [0, 50],  # Axial resistance (Ra) variability in excitatory neurons can have standard deviations up to 50 Ω·cm, influencing signal propagation. [Source: PMC - Neuronal Morphology Enhances Robustness](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9974411/)

        'I_diam_stdev': [0, 6],  # Inhibitory interneurons exhibit dendritic diameter variability with standard deviations up to 5 µm. [Source: Nature - Morphological diversity of single neurons](https://www.nature.com/articles/s41586-021-03941-1)
        'I_L_stdev': [0, 100],  # Dendritic length variability in inhibitory neurons can have standard deviations up to 50 µm. [Source: Nature - Morphological diversity of single neurons](https://www.nature.com/articles/s41586-021-03941-1)
        'I_Ra_stdev': [0, 40],  # Axial resistance variability in inhibitory neurons can have standard deviations up to 40 Ω·cm. [Source: PMC - Neuronal Morphology Enhances Robustness](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9974411/)
        # NOTE:# Morphology Parameters (Excitatory and Inhibitory Cells) Parsed by Source
            # params.update({

            #     ## Standard Deviations (Accounting for Biological Variability)
            #     'E_diam_stdev': [0, 12],  # Increased variability in excitatory neuron diameters.
            #     'E_L_stdev': [0, 150],  # Higher variance in dendritic lengths.
            #     'E_Ra_stdev': [0, 50],  # Variability in axial resistance due to neuron structure.

            #     'I_diam_stdev': [0, 6],  # Variability in inhibitory neuron diameters.
            #     'I_L_stdev': [0, 100],  # Variability in inhibitory neuron dendritic lengths.
            #     'I_Ra_stdev': [0, 40],  # Variability in inhibitory neuron Ra.
            # })
        
        
        # Mean Morphological Constants with Maximal Plausible Ranges
        'E_diam_mean': [5, 30],  # Mean diameter of excitatory neurons (µm). Pyramidal cells average around 20 µm, with variability depending on species and brain region. [Source: Pyramidal cell](https://en.wikipedia.org/wiki/Pyramidal_cell)
        'E_L_mean': [50, 1000],  # Mean length of excitatory neurons (µm). Individual dendrites are usually several hundred micrometers long. [Source: Pyramidal cell](https://en.wikipedia.org/wiki/Pyramidal_cell)
        'E_Ra_mean': [70, 200],  # Mean axial resistance of excitatory neurons (Ω·cm). Axial resistance is inversely related to the cross-sectional area of the neuron. [Source: Length constant](https://en.wikipedia.org/wiki/Length_constant)

        'I_diam_mean': [4, 15],  # Mean diameter of inhibitory neurons (µm). Inhibitory interneurons have soma diameters averaging around 10 µm, with some variability. [Source: Golgi cell](https://en.wikipedia.org/wiki/Golgi_cell)
        'I_L_mean': [50, 500],  # Mean length of inhibitory neurons (µm). Inhibitory neurons typically have shorter dendrites compared to excitatory neurons.
        'I_Ra_mean': [80, 200],  # Mean axial resistance of inhibitory neurons (Ω·cm). Due to their generally smaller diameters, inhibitory neurons exhibit higher axial resistance. [Source: Length constant](https://en.wikipedia.org/wiki/Length_constant)

        # NOTE:# Morphology Parameters (Excitatory and Inhibitory Cells) Parsed by Source
            # params.update({

            #     ## Mouse Neuronal Cultures (Unmyelinated, Smaller Morphology)
            #     'E_diam_mouse': [5, 20],  # Mean excitatory neuron diameter (µm) in mouse cortical cultures.
            #     'E_L_mouse': [50, 800],  # Mean excitatory neuron dendritic length (µm).
            #     'E_Ra_mouse': [100, 200],  # Higher axial resistance (Ω·cm) due to smaller neuron diameters.

            #     'I_diam_mouse': [4, 12],  # Mean inhibitory neuron diameter (µm) in mouse cortical cultures.
            #     'I_L_mouse': [30, 400],  # Mean inhibitory neuron dendritic length (µm).
            #     'I_Ra_mouse': [150, 250],  # Even higher Ra due to small inhibitory neuron size.

            #     ## Human iPSC-Derived Brain Organoids (Larger, Partial Myelination)
            #     'E_diam_human': [10, 30],  # Mean excitatory neuron diameter (µm). Human pyramidal cells are generally larger.
            #     'E_L_human': [100, 1200],  # Mean excitatory neuron dendritic length (µm).
            #     'E_Ra_human': [50, 130],  # Lower axial resistance (Ω·cm) due to larger neurons and partial myelination.

            #     'I_diam_human': [6, 18],  # Mean inhibitory neuron diameter (µm) in human organoids.
            #     'I_L_human': [50, 600],  # Mean inhibitory neuron dendritic length (µm).
            #     'I_Ra_human': [80, 180],  # Lower Ra in human inhibitory neurons compared to mouse.
            # })
    })

    # Connection Probability Length Constant
    params['probLengthConst'] = [1, 5000]  # Connection probability length constants can range from short-range (1 µm) to long-range (several mm) depending on neural circuitry. [Source: Neuronal Dynamics Online Book](https://neuronaldynamics.epfl.ch/online/Ch3.S1.html)

    # Connectivity Parameters
    params.update({
        'probIE': [0, 1],  # Inhibitory to Excitatory connection probability ranges from 0 to 1, representing no connection to full connectivity.
        'probEE': [0, 1],  # Excitatory to Excitatory connection probability ranges from 0 to 1.
        'probII': [0, 1],  # Inhibitory to Inhibitory connection probability ranges from 0 to 1.
        'probEI': [0, 1],  # Excitatory to Inhibitory connection probability ranges from 0 to 1.

        # Excitatory weights split by receptor type: AMPA (fast) and NMDA (slow)
        # AMPA-mediated weights (blocked by NBQX)
        'weightEI_AMPA': [0, 1000],  # AMPA weight for E->I connections. [Source: PMC - Determination of effective synaptic conductances](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6420044/)
        'weightEE_AMPA': [0, 1000],  # AMPA weight for E->E connections.

        # NMDA-mediated weights (blocked by AP5)
        'weightEI_NMDA': [0, 1000],  # NMDA weight for E->I connections.
        'weightEE_NMDA': [0, 1000],  # NMDA weight for E->E connections.

        # GABA-mediated weights (blocked by Bicuculline / Gabazine)
        'weightIE_GABA': [0, 1000],  # GABA weight for I->E connections.
        'weightII_GABA': [0, 1000],  # GABA weight for I->I connections.
    })

    # Sodium (gnabar) and Potassium (gkbar) Conductances
    params.update({
        # 'gnabar_E': [0, 0.12],  # Sodium conductance in excitatory neurons can range up to 0.12 S/cm². [Source: Hodgkin–Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)
        'gnabar_E': [0, 12],  # Widening the range significantly to account for variability and unknowns in sodium conductance for excitatory neurons.

        # 'gnabar_E_std': [0, 0.04],  # Standard deviation of sodium conductance in excitatory neurons can be up to 0.04 S/cm², reflecting variability in channel expression.
        'gnabar_E_std': [0, 4],  # Widening the range significantly to account for variability and unknowns in standard deviation of sodium conductance.

        # 'gkbar_E': [0, 0.036],  # Potassium conductance in excitatory neurons can reach up to 0.036 S/cm². [Source: Hodgkin–Huxley model](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)
        'gkbar_E': [0, 4],  # Widening the range significantly to account for variability and unknowns in potassium conductance for excitatory neurons.

        # 'gkbar_E_std': [0, 0.01],  # Standard deviation of potassium conductance in excitatory neurons can be up to 0.01 S/cm², accounting for differences in channel distribution.
        'gkbar_E_std': [0, 1],  # Widening the range significantly to account for variability and unknowns in standard deviation of potassium conductance.

        # 'gnabar_I': [0, 0.1],  # Sodium conductance in inhibitory neurons can range up to 0.1 S/cm², higher than in excitatory neurons due to different channel compositions. [Source: The influence of sodium and potassium dynamics on excitability](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2951284/)
        'gnabar_I': [0, 10],  # Widening the range significantly to account for variability and unknowns in sodium conductance for inhibitory neurons.

        # 'gnabar_I_std': [0, 0.03],  # Standard deviation of sodium conductance in inhibitory neurons can be up to 0.03 S/cm², indicating variability in sodium channel expression.
        'gnabar_I_std': [0, 3],  # Widening the range significantly to account for variability and unknowns in standard deviation of sodium conductance.

        # 'gkbar_I': [0, 0.05],  # Potassium conductance in inhibitory neurons can reach up to 0.05 S/cm², slightly higher than in excitatory neurons. [Source: The influence of sodium and potassium dynamics on excitability](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2951284/)
        'gkbar_I': [0, 5],  # Widening the range significantly to account for variability and unknowns in potassium conductance for inhibitory neurons.

        # 'gkbar_I_std': [0, 0.015],  # Standard deviation of potassium conductance in inhibitory neurons can be up to 0.015 S/cm², reflecting variability in potassium channel expression.
        'gkbar_I_std': [0, 2],  # Widening the range significantly to account for variability and unknowns in standard deviation of potassium conductance.
    })

    # Synaptic Time Constants — split by receptor type
    params.update({
        # AMPA time constants (fast excitatory)
        'tau1_AMPA': [0.1, 5],    # AMPA rise time is fast, typically < 1 ms. Wide range kept for fitting flexibility.
        'tau2_AMPA': [0.1, 50],   # AMPA decay time is fast, typically 1-10 ms.

        # NMDA time constants (slow excitatory)
        'tau1_NMDA': [0.1, 100],  # NMDA rise time is slower than AMPA. [Source: Estimating the Time Course of the Excitatory Synaptic Conductance](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6793890/)
        'tau2_NMDA': [1, 500],    # NMDA decay time is very slow, typically 50-150 ms but can be longer.

        # GABA time constants (inhibitory, GABA_A)
        'tau1_GABA': [0.1, 100],  # GABA rise time. [Source: Neurotransmitter Time Constants (PSCs)](https://compneuro.uwaterloo.ca/research/constants-constraints/neurotransmitter-time-constants-pscs.html)
        'tau2_GABA': [0.1, 1000], # GABA decay time, wide range to account for GABA_A and GABA_B contributions.
    })



# at least, before 28Dec2024
if version == 1.0:

    # Evolutionary Parameters
    params = specs.ODict()

    # Propagation Parameters
    params['propVelocity'] = 1  # Propagation velocity (arbitrary scaling examples in comments)

    # Morphology Parameters (Excitatory and Inhibitory Cells)
    params.update({
        
        #'E_diam_stdev': [0, 18.8 / 3],     # Standard deviation of excitatory diameter
        'E_diam_stdev': [0, 18.8],          # no obvious trend in the data, so I'm significantly increasing possible E_diam_stdev value
                                            # TODO: i hope increasing standard deviation of values like this doesn't break parameter assignment...
        
        #'E_L_stdev': [0, 18.8 / 2],         # Standard deviation of excitatory length
        'E_L_stdev': [0, 18.8],              # modest trend in the positive direction for better sims. I'm increasing the range to 0-18.8. Which is the max value for E_L_mean.
        
        #'E_Ra_stdev': [0, 128 / 2],      # Standard deviation of axial resistance
        'E_Ra_stdev': [0, 128],           # no obvious trend in the data, so I'm significantly increasing possible E_Ra_stdev value
        
        #'I_diam_stdev': [0, 10.0 / 3],   # Standard deviation of inhibitory diameter
        'I_diam_stdev': [0, 10.0],        # no obvious trend in the data, so I'm significantly increasing possible I_diam_stdev value
        
        #'I_L_stdev': [0, 9.0 / 3],         # Standard deviation of inhibitory length
        'I_L_stdev': [0, 9.0],              # no obvious trend in the data, so I'm significantly increasing possible I_L_stdev value
                                            # TODO: learn about the implications of this parameter for the model and neurophysiology
        
        #'I_Ra_stdev': [0, 110 / 2]       # Standard deviation of axial resistance
        'I_Ra_stdev': [0, 110],            # no obvious trend in the data, so I'm significantly increasing possible I_Ra_stdev value
        
        ## constants
        'E_diam_mean': 18.8,              # Mean diameter of excitatory cells (um)
        'E_L_mean': 18.8,                # Mean length of excitatory cells (um)
        'E_Ra_mean': 128,                # Mean axial resistance of excitatory cells (ohm/cm)
        'I_diam_mean': 10.0,             # Mean diameter of inhibitory cells (um)
        'I_L_mean': 9.0,                 # Mean length of inhibitory cells (um)
        'I_Ra_mean': 110,                # Mean axial resistance of inhibitory cells (ohm/cm)
        
        
    })

    # Connection Probability Length Constant
    #params['probLengthConst'] = [50, 3000]  # Length constant for connection probability (um)
    params['probLengthConst'] = [1, 5000]  # no obvious trend in the data, so I'm significantly increasing possible probLengthConst value

    # Connectivity Parameters
    params.update({
        ## probabiilties
        #'probIE': [0, 0.75],            # Inhibitory to Excitatory probability
        'probIE': [0, 1],                # No obvious trend. I'm increasing the range to 0-1. Which is the max value for a probability. obv.
        
        #'probEE': [0, 0.75],            # Excitatory to Excitatory probability
        'probEE': [0, 1],                # clear trend in the positive direction for better sims. I'm increasing the range to 0-1. Which is the max value for a probability. obv.    
        
        #'probII': [0, 0.75],            # Inhibitory to Inhibitory probability
        'probII': [0, 1],                # clear trend in the positive direction for better sims. I'm increasing the range to 0-1. Which is the max value for a probability. obv.
        
        #'probEI': [0.25, 1],            # Excitatory to Inhibitory probability
        'probEI': [0, 1],                # no obvious trend in the data, so I'm significantly increasing possible probEI value
        
        ## weights
        #'weightEI': [0, 2],                # Weight for Excitatory to Inhibitory connections
        'weightEI': [0, 10],                # no obvious trend in the data, so I'm significantly increasing possible weightEI value
                                            # Weight parameters are where I can exert the most arbitrary control over the model...I think.
                                            # TODO: Justify my intuition about the weight parameters.
        
        #'weightIE': [0, 1],             # Weight for Inhibitory to Excitatory connections
        'weightIE': [0, 10],             # no obvious trend in the data, so I'm significantly increasing possible weightIE value
        
        #'weightEE': [0, 2],             # Weight for Excitatory to Excitatory connections
        'weightEE': [0, 10],             # no obvious trend in the data, so I'm significantly increasing possible weightEE value
        
        #'weightII': [0, 2]              # Weight for Inhibitory to Inhibitory connections    
        'weightII': [0, 10]              # no obvious trend in the data, so I'm significantly increasing possible weightII value
    })

    # Sodium (gnabar) and Potassium (gkbar) Conductances
    params.update({
        
        #'gnabar_E': [0.5, 7],           # Sodium conductance for excitatory cells
        'gnabar_E': [0, 15],             # no clear trend in the data, so I'm increasing the range to 0-15
            
        #'gnabar_E_std': [0, 5 / 3], # Standard deviation of sodium conductance for excitatory cells
        'gnabar_E_std': [0, 15],          # no clear trend in the data, so I'm increasing the range to 0-5
        
        'gkbar_E': [0, 1],              # Potassium conductance for excitatory cells
                                        # pretty clear negative trend. holding constant at 0-1 for now.
                                        
        
        #'gkbar_E_std': [0, 1 / 2],         # Standard deviation of potassium conductance for excitatory cells
        'gkbar_E_std': [0, 1],              # actually, looks like a negative trend in the data. not sure if I should tighten the range or not.
                                            # there are values at the positive end of the range still. I have limited data to work with. Just going to widen the range anyway.
        
        # 'gnabar_I': [0, 5],               # Sodium conductance for inhibitory cells
        'gnabar_I': [0, 15],                # clear trend in the positive direction for better sims. I'm increasing the range to 0-10
        
        #'gnabar_I_std': [0, 3 / 3], # Standard deviation of sodium conductance for inhibitory cells
        'gnabar_I_std': [0, 15],          # no clear trend in the data, so I'm increasing the range to 0-3
        
        #'gkbar_I': [0, 10],             # Potassium conductance for inhibitory cells
        'gkbar_I': [0, 15/2],              # clear negative trend. very clear. reducing the range to 0-7.5
        
        #'gkbar_I_std': [0, 7.5 / 3] # Standard deviation of potassium conductance for inhibitory cells
        'gkbar_I_std': [0, 10]          # no clear trend in the data, so I'm increasing the range to 0-10
    })

    # Synaptic Time Constants
    params.update({
        #'tau1_exc': [0, 7.5],              # Rise time of excitatory synaptic conductance
        'tau1_exc': [0, 15],                # no clear trend in the data, so I'm increasing the range to 0-15
        
        #'tau2_exc': [0, 30.0],          # Decay time of excitatory synaptic conductance
        'tau2_exc': [0, 60],            # no clear trend in the data, so I'm increasing the range to 0-60
        
        # 'tau1_inh': [0, 10],            # Rise time of inhibitory synaptic conductance
        'tau1_inh': [0, 20],            # no clear trend in the data, so I'm increasing the range to 0-20
        
        #'tau2_inh': [0, 20.0]           # Decay time of inhibitory synaptic conductance
        'tau2_inh': [0, 40]             # no clear trend in the data, so I'm increasing the range to 0-40
    })

    # Stimulation Parameters (default values commented out for now)
    # params['Erhythmic_stimWeight'] = [0, 0.02]
    # params['Irhythmic_stimWeight'] = [0, 0.02]
    # params['rythmic_stiminterval'] = [0, 5000]  # Interval between spikes (ms)
    # params['rythmic_stimnoise'] = [0, 0.6]
