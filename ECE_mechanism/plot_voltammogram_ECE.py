import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('viridis', 12)
from potential_applied import *
from EDP_solver_ECE import *

# main programm for linear sweep voltammetry
def main_LSV_ECE(cst_all):
    Nt, Nx, DM, Lambda, L_cuve, Dx = cst_all[3]
    F_norm = cst_all[0][3]
    E, tk = rampe(cst_all[4][0], cst_all[4][1], cst_all[4][4])
    k_p, k_m = cst_all[2][6], cst_all[2][7]
    ## time step
    Dt = tk/Nt

    print("DM = ", DM, "and lambda = ", Lambda)
    
    ## profil de concentration inital
    C_new = np.append([cst_all[1][0] for i in range(Nx)], [cst_all[1][1] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all[1][2] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all[1][3] for i in range(Nx)])
   
    
    ## propagation temporelle
    fig, ax = plt.subplots(5, figsize=(10, 30))
    
    (M_new_constant, M_old) = Matrix_constant_ECE(Nx, Dt, 4, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_ECE_boundaries(M_new_constant, t, E, Lambda, Nx, F_norm, cst_all[2])
        C_new = compute_Cnew(M_new, M_old, C_old, cst_all[1], Nx)
        I = np.append(I, compute_I_ECE(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            

    ax[4].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[0].title.set_text('Profil de concentration de A en fonction du temps')
    ax[1].title.set_text('Profil de concentration de B en fonction du temps')
    ax[2].title.set_text('Profil de concentration de C en fonction du temps')
    ax[3].title.set_text('Profil de concentration de D en fonction du temps')
    titre_i_E = f"Courbe intensité potentiel ECE E1 = {cst_all[2][0]} V et E2 = {cst_all[2][9]} V."
    ax[4].title.set_text(titre_i_E)
    plt.savefig('ECE.png')
    plt.show()
    
    return(I)

# main programm for cyclic staircase voltammetry
def main_CSV_E(cst_all):
    (Nt, Nx, DM, Lambda, L_cuve, Dx) = cst_all[3]
    F_norm = cst_all[0][3]
    (E, tk) = CSV(cst_all[4][0], cst_all[4][1], cst_all[4][3], cst_all[4][4])
    
    ## time step
    Dt = tk/Nt

    print("DM = ", DM, "and lambda = ", Lambda)
    
    ## profil de concentration inital
    C_new = np.append([cst_all[1][0] for i in range(Nx)],[cst_all[1][1] for i in range(Nx)])

    ## propagation temporelle
    fig, ax = plt.subplots(4, figsize=(20, 40))
    red = Color("red")
    colors = list(red.range_to(Color("blue"),10))
    
    (M_new_constant, M_old) = Matrix_constant(Nx, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_E_Non_Nernst_boundaries(M_new_constant, t, E, Lambda, Nx, F_norm, cst_all[2])
        C_new = compute_Cnew(M_new, M_old, C_old, cst_all[1], Nx)
        I = np.append(I, compute_I(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:Nx], label= 'time = %is' %(i*Dt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:], label= 'time = %is' %(i*Dt))

    ax[2].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[3].plot([i*Dt for i in range(Nt)],[E(i*(Dt)) for i in range(Nt)])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
    return(I)

# main programm for square wave voltammetry
def main_SWV_E(cst_all):
    (Nt, Nx, DM, Lambda, L_cuve, Dx) = cst_all[3]
    F_norm = cst_all[0][3]
    (E, E_sweep, tk) = SWV(cst_all[4][0], cst_all[4][1], cst_all[4][2], cst_all[4][3], cst_all[4][5])
    
    ## time step
    Dt = tk/Nt

    print("DM = ", DM, "and lambda = ", Lambda)
    print("Dt = ", Dt, "and T = 2Pi/f = ", 2*np.pi/cst_all[4][5])
    
    # arbitrary criteria to check if the time step is small enough compared to the time step of the SW
    if 20*Dt > 2*np.pi/cst_all[4][5]:
        print("YOU SHOULD INCREASE THE NUMBER OF TIME STEPS TO GET MEANINGFUL RESULTS !")
    
    ## profil de concentration inital
    C_new = np.append([cst_all[1][0] for i in range(Nx)],[cst_all[1][1] for i in range(Nx)])

    ## propagation temporelle
    fig, ax = plt.subplots(4, figsize=(10, 20))
    (M_new_constant, M_old) = Matrix_constant(Nx, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_E_Non_Nernst_boundaries(M_new_constant, t, E, Lambda, Nx, F_norm, cst_all[2])
        C_new = compute_Cnew(M_new, M_old, C_old, cst_all[1], Nx)
        I = np.append(I, compute_I(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:Nx], label= 'time = %is' %(i*Dt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:], label= 'time = %is' %(i*Dt))

    ax[2].plot([E_sweep(i*Dt) for i in range(Nt)], I)
    ax[3].plot([i*Dt for i in range(Nt)],[E(i*(Dt)) for i in range(Nt)])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    

    plt.show()
    
    return(I)

# exctraction des courants for rev et differentiels de la SWV
def plot_SWV(cst_all, I):
    (E, E_sweep, tk) = SWV(cst_all[4][0], cst_all[4][1], cst_all[4][2], cst_all[4][3], cst_all[4][5])
    Nt = cst_all[3][0]
    Dt = tk/Nt
    I_for = []
    E_for = []
    I_rev = []
    E_rev = []

    i = 1
    count = 0
    while i < Nt:
        if E((i-1)*Dt) == E(i*Dt):
            i += 1
        else:
            if count == 0:
                I_for.append(I[i-1])
                E_for.append(E((i-1)*Dt))
                count = 1
            else:
                I_rev.append(I[i-1])
                E_rev.append(E((i-1)*Dt))
                count = 0
            i += 1    

    plt.plot(E_for, I_for, label = 'i_for') 
    plt.plot(E_rev, I_rev, label = 'i_rev')

    if len(I_for) == len(I_rev):
        Delta_I = np.array(I_for) - np.array(I_rev)
        plt.plot(E_for, Delta_I, label = 'delta_i') 
    elif len(I_rev) == len(I_for)-1:
        I_rev_new = np.append(0, np.array(I_rev))
        Delta_I = np.array(I_for) - np.array(I_rev_new)
        plt.plot(E_for, Delta_I, label = 'delta_i') 
    else:
        print("Il y a une couille dans le potage")
    
    plt.legend()
    plt.figure(figsize=(10,10))
    plt.show()
    plt.savefig('ECE.png')
    return(E_for, I_for, E_rev, I_rev, Delta_I)
