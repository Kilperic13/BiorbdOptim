from scipy.integrate import solve_ivp
import numpy as np
import random
import biorbd
from casadi import MX, Function
from matplotlib import pyplot as plt
import MainPgr as MP
import os

from biorbd_optim import (
    OptimalControlProgram,
    BidirectionalMapping,
    Mapping,
    DynamicsTypeList,
    DynamicsType,
    DynamicsFunctions,
    Data,
    ObjectiveList,
    Objective,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialConditionsList,
)

def Ramdomisation(nb_shooting, nivR_E = 0, nivR_M = 0, Data_E = True, Data_M = True):
    # Aliases
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()
    dt = final_time / nb_shooting
    Natur_DataE = type(Data_E)
    Natur_DataM = type(Data_M)

    # Data exist ?
    if Natur_DataE == bool:
        Data_E = np.zeros((nb_shooting, nb_mus))
    if Natur_DataM == bool:
        Data_M = np.zeros((3, nb_markers, nb_shooting))

    # Generate the level of randomisation
    ND_M = (nivR_M / 100) * np.random.rand(3, nb_markers, nb_shooting + 1)
    ND_E = (nivR_E / 100) * np.random.rand(nb_shooting, nb_mus)
    ND_M = LevelRand(Data_M, nivR_M, ND_M, Info = 'M')
    ND_E = LevelRand(Data_E, nivR_E, ND_E, Info = 'E')

    # Creation of the new data
    NewData_M = np.ones((3, nb_markers, nb_shooting + 1))
    NewData_E = np.ones((nb_shooting, nb_mus))
    NewData_M = GeneratND(NewData_M, ND_M, Data_M, Info = 'M')
    NewData_E = GeneratND(NewData_E, ND_E, Data_E, Info = 'E')

    if Natur_DataE != bool and Natur_DataM != bool:
        return NewData_M, NewData_E
    elif Natur_DataE != bool and Natur_DataM == bool:
        return NewData_E
    elif Natur_DataM != bool and Natur_DataE == bool:
        return NewData_M
    else:
        return None

def LevelRand(Data, nivR, ND, Info = None):
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()

    if Info == None:
        return 0

    # Generate the level of randomisation
    if Info == 'M':
        for j in range(3):
            for i in range(len(Data[j])):
                Dmax = max(Data[j][i])
                Dmin = min(Data[j][i])
                if (Dmax - Dmin) == 0:
                    Dmax = Dmin*(1 + nivR/1000)
                    # Dmin = 0
                ND[j][i] = ND[j][i] * (Dmax - Dmin)
    elif Info == 'E':
        for i in range(len(Data.T)):
            Dmax = max(Data[i])
            Dmin = min(Data[i])
            if (Dmax - Dmin) == 0:
                Dmax = Dmin * (1 + nivR / 1000)
            ND.T[i] = ND.T[i] * (Dmax - Dmin)

    return ND

def GeneratND(NewData, ND, Data, Info = None):

    if (type(ND) == int and ND == 0) or Info == None:
        print('Problem creation New Data')
        return Data

    if Info == 'M':
        for i in range(len(Data)):
            for j in range(len(Data[0])):
                for k in range(len(Data[0][0])):
                    NewData[i][j][k] = Data[i][j][k] + ((-1) ** random.randint(1, 2)) * ND[i][j][k]
    elif Info == 'E':
        for i in range(len(Data)):
            for j in range(len(Data.T)):
                NewData[i][j] = Data[i][j] + ((-1) ** random.randint(1, 2)) * ND[i][j]
                if NewData[i][j] > 1:
                    NewData[i][j] = 1
                elif NewData[i][j] < 0:
                    NewData[i][j] = 0

    return NewData

def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    excitations_ref,
    q_ref,
    use_residual_torque,
    kin_data_to_track="markers",
):
    # Problem parameters
    tau_min, tau_max, tau_init = -100, 100, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    excitation_min, excitation_max, excitation_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, target=excitations_ref)
    if use_residual_torque:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE)
    if kin_data_to_track == "markers":
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=100, target=markers_ref)
    elif kin_data_to_track == "q":
        objective_functions.add(
            Objective.Lagrange.TRACK_STATE, weight=100, target=q_ref, states_idx=range(biorbd_model.nbQ())
        )
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_residual_torque:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    else:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    x_bounds[0].min[[0, 1], :] = -2 * np.pi
    x_bounds[0].max[[0, 1], :] = 2 * np.pi

    # Add muscle to the bounds
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Initial guess
    x_init = InitialConditionsList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0] * biorbd_model.nbMuscles())

    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialConditionsList()
    if use_residual_torque:
        u_bounds.add(
            [
                [tau_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscles(),
                [tau_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscles(),
            ]
        )
        u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscles())
    else:
        u_bounds.add([[excitation_min] * biorbd_model.nbMuscles(), [excitation_max] * biorbd_model.nbMuscles()])
        u_init.add([excitation_init] * biorbd_model.nbMuscles())
    # ------------- #

    return OptimalControlProgram(
        biorbd_model, dynamics, nb_shooting, final_time, x_init, u_init, x_bounds, u_bounds, objective_functions,
    )

# Define the problem
biorbd_model = biorbd.Model(MP.model)
final_time = MP.final_time
n_shooting_points = MP.n_shooting_points
use_residual_torque = MP.use_residual_torque

# Recuperation donner Attibute.py
pathRef = MP.pathRef
DEN = MP.DataEssaieName
DRef = MP.DataRef
RefData = np.ones((len(DEN), len(DRef)))
RefData2 = RefData.tolist()
for i, D in enumerate(DEN):
    for j, R in enumerate(DRef):
        path2 = os.path.join(pathRef, D, R)
        if os.path.exists(path2):
            RefData2[i][j] = np.load(path2)

# Prepare the creation of New Data
pathRamd = MP.pathRamd
Rmd_E = MP.Ramd_E
a = 'DR'
c = MP.c

for i in range(len(DEN)):                          # Jeu de donner brut : 1, 2, 3
    # Data Ref
    muscle_excitations_ref = RefData2[i][0]
    markers_ref = RefData2[i][1]
    x_ref = RefData2[i][2]
    muscle_activations_ref = RefData2[i][3]
    t = RefData2[i][4]
    I = f'{i + 1}'

    for pc_E, PourCent_E in enumerate(Rmd_E):  # Jeu de bruit Excitation: 0, 5, 10, 15, 20
        if not os.path.exists(pathRamd + '/' + PourCent_E):
            os.mkdir(pathRamd + '/' + PourCent_E)
        for Essaie in c:  # Jeu d'essaie : 1, 2, 3, 4, 5
            # Generate random data to fit
            NewExcitation = Ramdomisation(nb_shooting=n_shooting_points, nivR_E=(pc_E) * 5,
                                          Data_E=muscle_excitations_ref[:-1])

            # # See the new data compare to the older one
            # NE = np.append(NewExcitation, NewExcitation[-1:, :], axis=0)
            # plt.figure("Muscle excitations visual")
            # plt.step(np.linspace(0, final_time, n_shooting_points + 1), muscle_excitations_ref, "k", where="post")
            # plt.step(np.linspace(0, final_time, n_shooting_points + 1), NE, "g*", where="post")
            # plt.xlabel("Time")
            # plt.ylabel("Excitation values")
            # plt.show()

            # Track these data
            biorbd_model = biorbd.Model("arm26_modifie.bioMod")  # To allow for non free variable, the model must be reloaded
            ocp = prepare_ocp(
                biorbd_model,
                final_time,
                n_shooting_points,
                markers_ref,
                NewExcitation,
                x_ref[: biorbd_model.nbQ(), :],
                use_residual_torque=use_residual_torque,
                kin_data_to_track="markers",
            )

            # --- Solve the program --- #
            sol = ocp.solve(show_online_optim=True)

            # --- Show the results --- #
            states_sol, controls_sol = Data.get_data(ocp, sol["x"])
            q = states_sol["q"]
            q_dot = states_sol["q_dot"]
            activations = states_sol["muscles"]
            if use_residual_torque:
                tau = controls_sol["tau"]
            excitations = controls_sol["muscles"]
            n_q = ocp.nlp[0]["model"].nbQ()
            # n_qdot = ocp.nlp[0]["model"].nbQdot()
            # n_mark = ocp.nlp[0]["model"].nbMarkers()
            # n_frames = q.shape[1]

            # --- Save the results --- #
            NewPath = pathRamd + '/' + PourCent_E + '/' + a + I + Essaie
            os.mkdir(NewPath)
            np.save(NewPath + '/r-q_ref.npy', x_ref[:n_q, :].T)
            np.save(NewPath + '/r-q.npy', q)
            np.save(NewPath + '/r-q_dot.npy', q_dot)
            np.save(NewPath + '/r-activations.npy', activations)
            np.save(NewPath + '/r-tau.npy', tau)
            np.save(NewPath + '/r-excitations.npy', excitations)
            np.save(NewPath + '/r-nexci.npy', NewExcitation)
