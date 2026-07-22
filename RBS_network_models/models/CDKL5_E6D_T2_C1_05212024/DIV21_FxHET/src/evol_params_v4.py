'''
evolutionary parameter space for CDKL5_DIV21 project

Seeded from trial_226_cfg.json
v4:
  - Parameters with ±50% variation based on the seed cfg values.
  - Standard deviations fixed to 0.
'''
from netpyne import specs

version = 4.0

if version == 4.0:
    # Evolutionary Parameters
    params = specs.ODict()

    # Propagation Parameters
    params['propVelocity'] = [1.3185967058719756, 3.955790117615927]

    # Morphology Parameters (Excitatory and Inhibitory Cells)
    params.update({

        # --- Standard Deviations: fixed to 0 (homogeneous populations) ---
        'E_diam_stdev': [0.0, 0.0],
        'E_L_stdev':    [0.0, 0.0],
        'E_Ra_stdev':   [0.0, 0.0],

        'I_diam_stdev': [0.0, 0.0],
        'I_L_stdev':    [0.0, 0.0],
        'I_Ra_stdev':   [0.0, 0.0],

        # --- Mean Morphology: ±50% variation ---
        'E_diam_mean': [15.679580993975359, 47.03874298192608],
        'E_L_mean': [299.8675873024155, 899.6027619072464],
        'E_Ra_mean': [71.42654069257571, 214.27962207772714],

        'I_diam_mean': [24.99627878503219, 74.98883635509657],
        'I_L_mean': [89.6957872747861, 269.0873618243583],
        'I_Ra_mean': [123.38032138429931, 370.14096415289794],
    })

    # Connection Probability Length Constant
    params['probLengthConst'] = [22458.65530916739, 67375.96592750217]

    # Connectivity Parameters
    params.update({
        'probIE': [0.3182332970714261, 0.9546998912142783],
        'probEE': [0.2345331408361066, 0.7035994225083197],
        'probII': [0.29237995262320093, 0.8771398578696028],
        'probEI': [0.37300182540313864, 1.0],

        # Excitatory synaptic weights split by receptor type
        # AMPA-mediated weights (blocked by NBQX)
        'weightEI_AMPA': [0, 2000],
        'weightEE_AMPA': [0, 20000.8920047738575],

        # NMDA-mediated weights (blocked by AP5)
        'weightEI_NMDA': [0, 2000],
        'weightEE_NMDA': [0, 20000.850358398238],

        # GABA-mediated weights (blocked by Bicuculline / Gabazine)
        'weightIE_GABA': [0, 2000],
        'weightII_GABA': [0, 20000],
    })

    # Sodium (gnabar) and Potassium (gkbar) Conductances
    params.update({
        'gnabar_E':     [129.73179249339006, 389.1953774801702],
        'gnabar_E_std': [0.0, 0.0],

        'gkbar_E':      [3.0202916525655947, 9.060874957696784],
        'gkbar_E_std':  [0.0, 0.0],

        'gnabar_I':     [83.40593146871005, 250.21779440613017],
        'gnabar_I_std': [0.0, 0.0],

        'gkbar_I':      [4.559856178823562, 13.679568536470686],
        'gkbar_I_std':  [0.0, 0.0],
    })

    # Synaptic Time Constants — split by receptor type
    params.update({
        # AMPA time constants (fast excitatory)
        'tau1_AMPA': [2.219199869599101, 6.657599608797303],
        'tau2_AMPA': [1.0095314501236747, 3.028594350371024],

        # NMDA time constants (slow excitatory)
        'tau1_NMDA': [26.333236813271416, 78.99971043981425],
        'tau2_NMDA': [1.6825408834593738, 5.047622650378122],

        # GABA time constants (inhibitory)
        'tau1_GABA': [67.16712376877902, 201.50137130633706],
        'tau2_GABA': [110.06424593279861, 330.19273779839585],
    })
