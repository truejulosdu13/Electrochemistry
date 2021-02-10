import math
import numpy as np
import matplotlib.pyplot as plt
from potential_applied import *
from EDP_solver import *

# main programm for linear sweep voltammetry
def main_LSV_Non_Nernstian(cst_all):
    (Nt, Nx, DM, Lambda, L_cuve, Dx) = cst_all[3]
    F_norm = cst_all[0][3]
    (E, tk) = rampe(cst_all[4][0], cst_all[4][1], cst_all[4][4])
    
    ## time step
    Dt = tk/Nt

    print("DM = ", DM, "and lambda = ", Lambda)
    
    ## profil de concentration inital
    C_new = np.append([cst_all[1][0] for i in range(Nx)],[cst_all[1][1] for i in range(Nx)])

    ## propagation temporelle
    fig, ax = plt.subplots(3)
    (M_new_constant, M_old) = Matrix_constant(Nx, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_E_Non_Nernst_boundaries(M_new_constant, t, E, Lambda, Nx, F_norm, cst_all[2])
        C_new = compute_Cnew(M_new, M_old, C_old, cst_all[1], Nx)
        I = np.append(I, compute_I(C_new, cst_all))
        if i % 100 == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:Nx], label= 'time = %is' %(i*Dt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:], label= 'time = %is' %(i*Dt))

    ax[2].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
    return(I)
