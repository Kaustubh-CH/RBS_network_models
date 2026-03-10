'''
evolutionary parameter space for CDKL5_DIV21 project

Seeded from trial_243_cfg.json
v3 (from seed):
  - Parameters with ±30% variation based on the seed cfg values.
  - Standard deviations fixed to 0.
'''
from netpyne import specs

version = 3.0

if version == 3.0:
    # Evolutionary Parameters
    params = specs.ODict()

    # Propagation Parameters
    params['propVelocity'] = [1.423701682906636, 2.644017411112324]

    # Morphology Parameters (Excitatory and Inhibitory Cells)
    params.update({

        # --- Standard Deviations: fixed to 0 (homogeneous populations) ---
        'E_diam_stdev': [0.0, 0.0],
        'E_L_stdev':    [0.0, 0.0],
        'E_Ra_stdev':   [0.0, 0.0],

        'I_diam_stdev': [0.0, 0.0],
        'I_L_stdev':    [0.0, 0.0],
        'I_Ra_stdev':   [0.0, 0.0],

        # --- Mean Morphology: ±30% variation ---
        'E_diam_mean': [15.036185489912864, 27.924344481266746],
        'E_L_mean': [671.9720837320332, 1247.9481555023473],
        'E_Ra_mean': [77.55386926949771, 144.0286143576386],

        'I_diam_mean': [15.524511614647295, 28.83123585577355],
        'I_L_mean': [125.73653007379612, 233.51069870847848],
        'I_Ra_mean': [252.4174511475531, 468.7752664168843],
    })

    # Connection Probability Length Constant
    params['probLengthConst'] = [32856.84857328382, 61019.86163609852]

    # Connectivity Parameters
    params.update({
        'probIE': [0.16532978848234348, 0.30704103575292356],
        'probEE': [0.1633402402335605, 0.30334616043375525],
        'probII': [0.6216144629053935, 1.0], # capped at 1.0 (0.888 * 1.3 is > 1.0)
        'probEI': [0.37803864985926724, 0.7020717783100678],

        # Synaptic weights
        'weightEI': [607.431725321466, 1128.0874898827226],
        'weightIE': [1171.7629052476008, 2176.131109745544],
        'weightEE': [6087.890562150114, 11306.082472564497],
        'weightII': [6383.559767219216, 11855.182424835687],
    })

    # Sodium (gnabar) and Potassium (gkbar) Conductances
    params.update({
        'gnabar_E':     [80.89048760463843, 150.22519126575708],
        'gnabar_E_std': [0.0, 0.0],

        'gkbar_E':      [2.4005189734458364, 4.458106664970839],
        'gkbar_E_std':  [0.0, 0.0],

        'gnabar_I':     [36.06686195654405, 66.98131506215324],
        'gnabar_I_std': [0.0, 0.0],

        'gkbar_I':      [3.056645488019885, 5.676627334894072],
        'gkbar_I_std':  [0.0, 0.0],
    })

    # Synaptic Time Constants
    params.update({
        'tau1_exc': [27.637369240806697, 51.326542875783864],
        'tau2_exc': [1.8829246461771447, 3.496860057186126],
        'tau1_inh': [201.12175124716758, 373.51182374473983],
        'tau2_inh': [182.46626642973243, 338.8659233695031],
    })
