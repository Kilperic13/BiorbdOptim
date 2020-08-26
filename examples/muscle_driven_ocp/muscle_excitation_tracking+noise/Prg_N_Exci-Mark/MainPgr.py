import numpy as np

### Define the problem
model = "arm26_modifie.bioMod"
final_time = 1.5
n_shooting_points = 29
use_residual_torque = True

### Difine where are your originals datas - Name of the groupe of essaies - name of which data are safe
pathRef = '/home/lim/Devel_V3/BiorbdOptim/examples/muscle_driven_ocp/muscle_excitation_tracking+noise/Data'
DataEssaieName = ['D28-1', 'D10-2', 'D14-3']
DataRef = ['excitations_ref.npy', 'markers_ref.npy', 'x_ref.npy', 'activations_ref.npy', 't.npy']

RefData = np.ones((len(DataEssaieName), len(DataRef)))
RefData2 = RefData.tolist()

#### Define Path of creation of data
pathRamd = '/home/lim/Devel_V3/BiorbdOptim/examples/muscle_driven_ocp/muscle_excitation_tracking+noise/Prg_N_Exci-Mark/Data_NEM'
# How many % do you want ?
PourcentBase = 6
PorcentMany = 3
Ramd_E = [f'DR_E {i * PourcentBase}%' for i in range(PorcentMany)]
Ramd_E2 = ['DR_E 0%', 'DR_E 6%', 'DR_E 12%']
    # Here, for exemple, Ramd_E == Ramd_E2
#How many essaie by % do you want ?
PorcentEssaie = 3
c = [f'.{i+1}' for i in range(PorcentEssaie)]

