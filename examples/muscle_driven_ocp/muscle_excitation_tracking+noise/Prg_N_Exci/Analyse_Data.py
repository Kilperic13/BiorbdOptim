from scipy.integrate import solve_ivp
import numpy as np
import biorbd
from casadi import MX, Function
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


## Import Information ###

biorbd_model = biorbd.Model(MP.model)
final_time = MP.final_time
ns = MP.n_shooting_points
n_q = biorbd_model.nbQ()
n_qdot = biorbd_model.nbQdot()
n_mark = biorbd_model.nbMarkers()
n_muscl = biorbd_model.nbMuscleTotal()

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

# Recuperation  New Data
pathRamd = MP.pathRamd
Rmd_E = MP.Ramd_E
a = 'DR'
c = MP.c
Donner = ['r-activations.npy', 'r-tau.npy', 'r-q_dot.npy', 'r-excitations.npy', 'r-q.npy', 'r-nexci.npy']

# Preparation files
NbDossierRamd = []
mda_errer_exci = []
mda_errer_q = []
mda_errer_markx = []
mda_errer_marky = []
mda_errer_markz = []
errer_exci = []
errer_exci_virtuel = []
errer_q = []
errer_mark = []
errer_mark_virtuel = []
# errer_markx = []
# errer_marky = []
# errer_markz = []


### Funstions used to analyse data ###

# Erreur Excitation Creer et optimise
def cal_err_exci(exc_ref, exc):
    ErreurExci = [sum(np.abs([exc_ref[k][i] - exc[i][k] for i in range(n_muscl)])) for k in range(ns+1)]
    # print(f'Error Ex - MDA: {sum(ErreurExci)/len(ErreurExci)} - RMSD: {(np.dot(ErreurExci, ErreurExci)/t[-1])**(1/2)}')
    return ErreurExci[:-1]

# Erreur Q Creer et optimise
def cal_err_q(x_ref, q):
    ErreurQ = [sum(np.abs([x_ref[:n_q, :][i][k] - q[i][k] for i in range(n_q)])) for k in range(ns+1)]
    # print(f'Error Q - MDA: {sum(ErreurQ)/len(ErreurQ)} - RMSD: {(np.dot(ErreurQ, ErreurQ)/(ns+1))**(1/2)}')
    return ErreurQ

# Recuperation markers
def recup_mark(q):
    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = Function(
        "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"],
    ).expand()
    n_frames = q.shape[1]
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])
    return markers

# Erreur Markeur Creer et optimise
def cal_err_mark(mark_ref, mark, i = 'True'):
    # if i:
    #     return 'Problem'
    ErreurX = [sum(np.abs([mark_ref[0][i][k] - mark[0][i][k] for i in range(n_mark)])) for k in range(ns+1)]         #Erreur a chaque pas des marqueurs suivant X
    ErreurY = [sum(np.abs([mark_ref[1][i][k] - mark[1][i][k] for i in range(n_mark)])) for k in range(ns+1)]         #Erreur a chaque pas des marqueurs suivant Y
    ErreurZ = [sum(np.abs([mark_ref[2][i][k] - mark[2][i][k] for i in range(n_mark)])) for k in range(ns+1)]         #Erreur a chaque pas des marqueurs suivant Z
    # print(f'Erreur X - MDA: {sum(ErreurX)/len(ErreurX)} - RMSD: {(np.dot(ErreurX, ErreurX)/t[-1])**(1/2)}')
    # print(f'Erreur Y - MDA: {sum(ErreurY)/len(ErreurY)} - RMSD: {(np.dot(ErreurY, ErreurY)/t[-1])**(1/2)}')
    # print(f'Erreur Z - MDA: {sum(ErreurZ)/len(ErreurZ)} - RMSD: {(np.dot(ErreurZ, ErreurZ)/t[-1])**(1/2)}')
    if i == 0:
        return ErreurX
    elif i == 1:
        return ErreurY
    elif i == 2:
        return ErreurZ


### Calcul des erreurs ###

for i in range(len(DEN)):                          # Jeu de donner brut : 1, 2, 3
    muscle_excitations_ref = RefData2[i][0]
    markers_ref = RefData2[i][1]
    x_ref = RefData2[i][2]
    muscle_activations_ref = RefData2[i][3]
    t = RefData2[i][4]
    I = f'{i + 1}'
    # LNbDossier = []
    for PourCent in Rmd_E:                               # Jeu de bruit : 0, 5, 10, 15, 20
        # NbDossier = 0
        for Essaie in c:                                # Jeu d'essaie : 1, 2, 3, 4, 5
            List_Data_Ramd = [[0]]
            for d, D in enumerate(Donner):
                pathRamd2 = os.path.join(pathRamd, PourCent, a + I + Essaie, D)
                if os.path.exists(pathRamd2):
                    List_Data_Ramd[d] = np.load(pathRamd2)
                    List_Data_Ramd.append([0])
            q = List_Data_Ramd[4]
            q_dot = List_Data_Ramd[2]
            activations = List_Data_Ramd[0]
            tau = List_Data_Ramd[1]
            excitations = List_Data_Ramd[3]
            new_exci = List_Data_Ramd[5:]
            # NbDossier += 1
            errer_exci.append(cal_err_exci(muscle_excitations_ref, excitations))
            NE = np.append(new_exci, new_exci[-1:, :], axis=0)
            errer_exci_virtuel.append(cal_err_exci(NE, excitations))
            markers = recup_mark(q)
            for i in range(3):
                errer_mark.append(cal_err_mark(markers_ref, markers, i))
            errer_q.append(cal_err_q(x_ref, q))


### Analyse Data to save ###

np.save('RE_er_exci.npy', errer_exci)
np.save('RE_er_exci_vir.npy', errer_exci_virtuel)
np.save('RE_er_q.npy', errer_q)

ErrMX = errer_mark[::3]
ErrMY = errer_mark[1::3]
ErrMZ = errer_mark[2::3]
np.save('RE_er_mark_X.npy', ErrMX)
np.save('RE_er_mark_Y.npy', ErrMY)
np.save('RE_er_mark_Z.npy', ErrMZ)