import numpy as np
import biorbd
from casadi import MX, Function
from matplotlib import pyplot as plt
import pylab
import os
import MainPgr as MP

## Import Information ###

biorbd_model = biorbd.Model(MP.model)
ns = MP.n_shooting_points
n_q = biorbd_model.nbQ()
n_qdot = biorbd_model.nbQdot()
n_mark = biorbd_model.nbMarkers()
n_muscl = biorbd_model.nbMuscleTotal()
PctB = MP.PourcentBase
PctM = MP.PorcentMany
DEN = MP.DataEssaieName
Nb_DEN = len(DEN)
PctE = MP.PorcentEssaie
tf = MP.final_time

# Load error Data
Er_Exci = np.load('RE_er_exci.npy') * (100/n_muscl)             # 100 : Pour avoir des pourcentage
#Er_Exci_Vir = np.load('RE_er_exci_vir.npy') * (100/n_muscl)     # 100 : Pour avoir des pourcentage
EE = Er_Exci.tolist()
#EEV = Er_Exci_Vir.tolist()
Er_Q = np.load('RE_er_q.npy')
EQ = Er_Q.tolist()
Er_Mak_X = np.load('RE_er_mark_X.npy') * (100/n_mark)           # 100 : Pour avoir des pourcentage
Er_Mak_Y = np.load('RE_er_mark_Y.npy') * (100/n_mark)
Er_Mak_Z = np.load('RE_er_mark_Z.npy') * (100/n_mark)
EMX = Er_Mak_X.tolist()
EMY = Er_Mak_Y.tolist()
EMZ = Er_Mak_Z.tolist()

def Rang(i):
    R = ['1st', '2nd', '3rd']
    return R[i]

def Add_List(L, K, Start = 0):
    k = 0
    l = []
    while k < K:
        l += L[Start + k]
        k +=1
    return l

### Star the plot ###

# Boxplot for Excitation
Data = []
Data2 = []
BoxName = [f'{i * PctB}%' for i in range(PctM)]
for j in range(Nb_DEN):
    plt.subplot(1, Nb_DEN, 1 + j)
    data = []
    for i in range(PctM):
        data.append(Add_List(EE, PctE, Start = i*5 + j*20))
    plt.boxplot(data)
    pylab.xticks([h+1 for h in range(PctM)], BoxName)
    plt.title(f'Data set : {Rang(j)} best out of 28', fontsize='20')
    plt.ylabel("% of error of excitation by Muscle and by Increment", fontsize='14')
    plt.xlabel("Levels of noises", fontsize='14')
    Data.append(data)
    Data2.append(np.array(data).T)
# plt.savefig('Exci_noises-EE.png')
plt.gcf().subplots_adjust(left = 0.05, bottom = 0.07, right = 0.99, top = 0.95, wspace = 0.175, hspace = 0.25)
plt.show()

D2 = np.array(Data2)
D = np.array(Data)
# MeanD = [D[:, i].mean() for i in range(4)]
plt.figure()
plt.boxplot(np.concatenate((D2), axis = 0))
pylab.xticks([h+1 for h in range(PctM)], BoxName)
plt.title(f'Thrit best out of 28 data set', fontsize='20')
plt.ylabel("% of error of excitation by Muscle and by Increment", fontsize='14')
plt.xlabel("Levels of noises", fontsize='14')
plt.show()

# Boxplot for Q
BoxName = ['Erreur of Q']
data = [Er_Q[0]]
plt.boxplot(data)
# plt.ylim(-0.1, 1.1)
pylab.xticks([1], BoxName)
# plt.savefig('MultipleBoxPlot02.png')
# plt.title('Velocity of the optimisation')
plt.show()

# Boxplot for error markeurs x, y, z

BoxName = ['Error Markers - X axies', 'Error Markers - Y axies', 'Error Markers - Z axies']
data = [Er_Mak_X[0], Er_Mak_Y[0], Er_Mak_Z[0]]

plt.boxplot(data)
# plt.ylim(10, 80)
pylab.xticks([1, 2, 3], BoxName)
# plt.savefig('MultipleBoxPlot02.png')
# plt.title('Velocity of the optimisation')
plt.show()

BoxName = [f'{i * PctB}%' for i in range(PctM)]
Data3 = []
for j in range(Nb_DEN):
    plt.subplot(1, Nb_DEN, 1 + j)
    data = []
    for i in range(PctM):
        data.append(Add_List(EMX, PctE, Start = i*5 + j*20))
    plt.boxplot(data)
    pylab.xticks([h+1 for h in range(PctM)], BoxName)
    plt.title(f'Data set : {Rang(j)} best out of 28')
    if j == 0:
        plt.ylabel("% of error of X-axis for each marker by Increment")
    plt.xlabel("Levels of noises")
    Data3.append(np.array(data).T)
# plt.savefig('Exci_noises-EE.png')
plt.gcf().subplots_adjust(left = 0.07, bottom = 0.07, right = 0.975, top = 0.95, wspace = 0.18, hspace = 0.25)
plt.show()

D3 = np.array(Data3)
# MeanD = [D[:, i].mean() for i in range(4)]
plt.figure()
plt.boxplot(np.concatenate((D3), axis = 0))
pylab.xticks([h+1 for h in range(PctM)], BoxName)
plt.title(f'Thrit best out of 28 data set', fontsize='20')
plt.ylabel("% of error of X-axis for each marker by Increment", fontsize='14')
plt.xlabel("Levels of noises", fontsize='14')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
plt.show()


##   - Plot Excitation pour presentation -   ##

# Chemin Originaux

pathRef = MP.pathRef + '/' + DEN[0]
DRef = MP.DataRef
RefData = np.ones((len(DRef)))
RefData2 = RefData.tolist()

# Ref part

for j, R in enumerate(DRef):
    path2 = os.path.join(pathRef, R)
    if os.path.exists(path2):
        RefData2[j] = np.load(path2)

pathData = MP.pathRef + '/' + DEN[0] +'/DR1.5'

List_Data_Ramd = [[0]]
Donner = ['r-activations.npy', 'r-tau.npy', 'r-q_dot.npy', 'r-excitations.npy', 'r-q.npy', 'r-nexci.npy']
for d, D in enumerate(Donner):
    pathRamd2 = os.path.join(pathData, D)
    if os.path.exists(pathRamd2):
        List_Data_Ramd[d] = np.load(pathRamd2)
        List_Data_Ramd.append([0])
q = List_Data_Ramd[4]
q_dot = List_Data_Ramd[2]
activations = List_Data_Ramd[0]
tau = List_Data_Ramd[1]
excitations = List_Data_Ramd[3]
new_exci = List_Data_Ramd[5:]
new_exci2 = np.concatenate(([new_exci[0][0, 1]], new_exci[0][:-1, 1]), axis=0)

muscle_excitations_ref = RefData2[0]

plt.figure("Muscle excitations")
plt.step(np.linspace(0, tf, ns + 1), muscle_excitations_ref[:, 1], "k", where="post", label = 'Initial Input')
plt.step(np.linspace(0, tf, ns + 1), new_exci[0][:, 1], 'g*', where="post", label = 'Noises Input')
plt.step(np.linspace(0, tf, ns + 1), new_exci2, 'g*', where="post")
plt.step(np.linspace(0, tf, ns + 1), excitations.T[:, 1], color='r', ls ='--', where="post", label = 'Optimisation Resust')
plt.legend(loc = 'best', fontsize='12')
plt.xlabel("Time (s)", fontsize='14')
plt.ylabel("Excitation values", fontsize='14')
plt.title(f'Muscle excitation', fontsize='20')
plt.show()