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
        'E_diam_mean': [05.036185489912864, 37.924344481266746],
        'E_L_mean': [400.9720837320332, 1647.9481555023473],
        'E_Ra_mean': [37.55386926949771, 444.0286143576386],

        'I_diam_mean': [5.524511614647295, 50.83123585577355],
        'I_L_mean': [105.73653007379612, 533.51069870847848],
        'I_Ra_mean': [102.4174511475531, 868.7752664168843],
    })

    # Connection Probability Length Constant
    params['probLengthConst'] = [12856.84857328382, 91019.86163609852]

    # Connectivity Parameters
    params.update({
        'probIE': [0.16532978848234348, 0.30704103575292356],
        'probEE': [0.1633402402335605, 0.30334616043375525],
        'probII': [0.6216144629053935, 1.0], # capped at 1.0 (0.888 * 1.3 is > 1.0)
        'probEI': [0.37803864985926724, 0.7020717783100678],

        # Synaptic weights
        'weightEI': [407.431725321466, 1428.0874898827226],
        'weightIE': [671.7629052476008, 2776.131109745544],
        'weightEE': [4087.890562150114, 18306.082472564497],
        'weightII': [4383.559767219216, 18855.182424835687],
    })

    # Sodium (gnabar) and Potassium (gkbar) Conductances
    params.update({
        'gnabar_E':     [30.89048760463843, 350.22519126575708],
        'gnabar_E_std': [0.0, 0.0],

        'gkbar_E':      [0.4005189734458364, 14.458106664970839],
        'gkbar_E_std':  [0.0, 0.0],

        'gnabar_I':     [16.06686195654405, 166.98131506215324],
        'gnabar_I_std': [0.0, 0.0],

        'gkbar_I':      [3.056645488019885, 15.676627334894072],
        'gkbar_I_std':  [0.0, 0.0],
    })

    # Synaptic Time Constants
    params.update({
        'tau1_exc': [17.637369240806697, 81.326542875783864],
        'tau2_exc': [1.8829246461771447, 13.496860057186126],
        'tau1_inh': [101.12175124716758, 673.51182374473983],
        'tau2_inh': [102.46626642973243, 638.8659233695031],
    })
