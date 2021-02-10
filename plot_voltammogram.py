import math
import numpy as np
import matplotlib.pyplot as plt
from potential_applied import *
from EDP_solver_CN import *

# main programm for linear sweep voltammetry
def main_LSV_Non_Nernstian(LLambda, DM):
    Lambda = LLambda
    
    (E, tk) = rampe(E_i, E_f, v)
    print(E(0))
    ## Pas Dx et Dt
    Dt = tk/Nt  

    print("DM = ", DM, "and lambda = ", Lambda)
    
    ## profil de concentration inital
    C_new = np.append([C_ox_val for i in range(Nx)],[C_red_val for i in range(Nx)])

    ## propagation temporelle
    fig, ax = plt.subplots(3)
    (M_new_constant, M_old) = Matrix_constant(Nx, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_E_Non_Nernst_boundaries(M_new_constant, Nx, t, E, Lambda)
        C_new = compute_Cnew(M_new, M_old, C_old)
        I = np.append(I, compute_I(C_new))
        if i % 100 == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:Nx], label= 'time = %is' %(i*Dt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:], label= 'time = %is' %(i*Dt))

    ax[2].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
    return(I)
