'''
evolutionary parameter space for CDKL5_DIV21 project

Seeded from trial_389_cfg.json
v389 (from seed):/pscratch/sd/k/ktub1999/networkSimulations/z_simulated_data/KCNT_Test_Mar_03_seed_params_nostd_v4/batch_runs/batch_2026-03-03_spiking_only/gen_433/trial_433_cfg.json

  - Parameters with ±30% variation based on trial_389 simConfig values.
  - Standard deviations fixed to 0.
  - Probabilities clipped to [0, 1].
'''
from netpyne import specs

version = 389.0

if version == 389.0:
    # Evolutionary Parameters
    params = specs.ODict()

    # Propagation Parameters
    params['propVelocity'] = [26.1430645022351, 48.551405504151]

    # Morphology Parameters (Excitatory and Inhibitory Cells)
    params.update({

        # --- Standard Deviations: fixed to 0 (homogeneous populations) ---
        'E_diam_stdev': [0.0, 0.0],
        'E_L_stdev':    [0.0, 0.0],
        'E_Ra_stdev':   [0.0, 0.0],

        'I_diam_stdev': [0.0, 0.0],
        'I_L_stdev':    [0.0, 0.0],
        'I_Ra_stdev':   [0.0, 0.0],

        # --- Mean Morphology: ±30% variation around trial_389 ---
        'E_diam_mean': [37.5421379113718, 69.7211132639762],
        'E_L_mean': [951.280758362652, 1766.66426553064],
        'E_Ra_mean': [90.0699446492753, 167.272754348654],

        'I_diam_mean': [33.6428640981287, 62.4796047536677],
        'I_L_mean': [1367.50002234731, 2539.642898645],
        'I_Ra_mean': [234.403298112477, 435.320410780314],
    })

    # Connection Probability Length Constant
    # NOTE: Keeping this from prior seeded spaces because trial_389 cfg does not include it.
    params['probLengthConst'] = [32856.84857328382, 61019.86163609852]

    # Connectivity Parameters
    params.update({
        'probIE': [0.673688674864798, 1.0],
        'probEE': [0.656606815112939, 1.0],
        'probII': [0.51856231138211, 0.963044292566775],
        'probEI': [0.301348853511865, 0.559647870807749],

        # Synaptic weights
        'weightEI': [0, 7816.05654454867],
        'weightIE': [0, 95.5643203477992],
        'weightEE': [0, 7061.76664831422],
        'weightII': [0, 965.066443256747],
    })

    # Sodium (gnabar) and Potassium (gkbar) Conductances
    params.update({
        'gnabar_E':     [59.4388358919386, 110.3864095136],
        'gnabar_E_std': [0.0, 0.0],

        'gkbar_E':      [4.15446591644711, 7.7154367019732],
        'gkbar_E_std':  [0.0, 0.0],

        'gnabar_I':     [16.3989475328098, 30.4551882752181],
        'gnabar_I_std': [0.0, 0.0],

        'gkbar_I':      [0.780452505490025, 1.44941179591005],
        'gkbar_I_std':  [0.0, 0.0],
    })

    # Synaptic Time Constants
    params.update({
        'tau1_exc': [0, 1127.70397746002],
        'tau2_exc': [0, 7.60327919020474],
        'tau1_inh': [0, 29.3541116495538],
        'tau2_inh': [0, 209.835664842618],
    })
