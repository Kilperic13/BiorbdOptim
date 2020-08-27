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
PctBE = MP.PourcentBaseE
PctBM = MP.PourcentBaseM
PctME = MP.PorcentManyE
PctMM = MP.PorcentManyM
DEN = MP.DataEssaieName
Nb_DEN = len(DEN)
PctEss = MP.PorcentEssaie
tf = MP.final_time

# Load error Data
Er_Exci = np.load('RE_er_exci.npy') * (100/n_muscl)             # 100 : Pour avoir des pourcentage
Er_Exci_Vir = np.load('RE_er_exci_vir.npy') * (100/n_muscl)     # 100 : Pour avoir des pourcentage
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
Er_Mak_Vir_X = np.load('RE_er_mark_vir_X.npy') * (100/n_mark)   # 100 : Pour avoir des pourcentage
Er_Mak_Vir_Y = np.load('RE_er_mark_vir_Y.npy') * (100/n_mark)
Er_Mak_Vir_Z = np.load('RE_er_mark_vir_Z.npy') * (100/n_mark)
EMVX = Er_Mak_Vir_X.tolist()
EMVY = Er_Mak_Vir_Y.tolist()
EMVZ = Er_Mak_Vir_Z.tolist()

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
BoxName = [f'{i * PctBE}%' for i in range(PctME)]
for j in range(Nb_DEN):
    plt.subplot(1, Nb_DEN, 1 + j)
    data = []
    for i in range(PctME):
        data.append(Add_List(EE, PctEss, Start = (i*PctEss + j*(PctME*PctEss*PctMM))))
        # On veut l'ecart avec les differents % de bruits de E, symbolisé par i, avec les 3 données bruts, j, chacune écarté (Nb Jeu de bruit M (4) * Nb Jeu de bruit E (3)) * Nb Jeu d'essaie (3) = 36)
    plt.boxplot(data)
    pylab.xticks([h+1 for h in range(PctME)], BoxName)
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
pylab.xticks([h+1 for h in range(PctME)], BoxName)
plt.title(f'Thrit best out of 28 data set', fontsize='20')
plt.ylabel("% of error of excitation by Muscle and by Increment", fontsize='14')
plt.xlabel("Levels of noises", fontsize='14')
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

BoxName = [f'{i * PctBM}%' for i in range(PctMM)]
Data3 = []
for j in range(Nb_DEN):
    plt.subplot(1, Nb_DEN, 1 + j)
    data = []
    for i in range(PctMM):
        data.append(Add_List(EMX, PctEss, Start = (i*(PctEss*PctME) + j*(PctME*PctEss*PctMM))))
        # L'ecart pour i, entre 2 bruit de marqueur, est de (Nb de Jeux * Nb de Bruit Excitation)
    plt.boxplot(data)
    pylab.xticks([h+1 for h in range(PctMM)], BoxName)
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
pylab.xticks([h+1 for h in range(PctMM)], BoxName)
plt.title(f'Thrit best out of 28 data set', fontsize='20')
plt.ylabel("% of error of X-axis for each marker by Increment", fontsize='14')
plt.xlabel("Levels of noises", fontsize='14')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
plt.show()


## Pour être sur qu'on ai un ecart entre les marqueurs bruité et ceux optimisé
## On voit que c'est linéaire, avec une distribution uniforme, comme le bruit imposé, donc parfait
## Sert a verifier qu'on applique bien le bon bruit


BoxName = [f'{i * PctBM}%' for i in range(PctMM)]
Data4 = []
for j in range(Nb_DEN):
    plt.subplot(1, Nb_DEN, 1 + j)
    data = []
    for i in range(PctMM):
        data.append(Add_List(EMVX, PctEss, Start = (i*(PctEss*PctME) + j*(PctME*PctEss*PctMM))))
    plt.boxplot(data)
    pylab.xticks([h+1 for h in range(PctMM)], BoxName)
    plt.title(f'Data set : {Rang(j)} best out of 28')
    if j == 0:
        plt.ylabel("% of error of X-axis for each marker by Increment")
    plt.xlabel("Levels of noises")
    Data4.append(np.array(data).T)
# plt.savefig('Exci_noises-EE.png')
plt.gcf().subplots_adjust(left = 0.07, bottom = 0.07, right = 0.975, top = 0.95, wspace = 0.18, hspace = 0.25)
# plt.show()

D4 = np.array(Data4)
# MeanD = [D[:, i].mean() for i in range(4)]
plt.figure()
plt.boxplot(np.concatenate((D4), axis = 0))
pylab.xticks([h+1 for h in range(PctMM)], BoxName)
plt.title(f'Error of X-axis for each marker by Increment compare to input noises data', fontsize='20')
plt.ylabel("% of error of X-axis for each marker by Increment", fontsize='14')
plt.xlabel("Levels of noises", fontsize='14')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
plt.ylim(-0.1, 4)
# plt.show()


