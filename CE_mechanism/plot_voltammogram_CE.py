import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from potential_applied import *
from EDP_solver_CE import *

# main programm for linear sweep voltammetry
def main_LSV_CE(cst_all):
    F_norm = cst_all["F_norm"]
    Nx = cst_all["Nx"]
    Nt = cst_all["Nt"]
    DM = cst_all["DM"]
    Dx = cst_all["Dx"]
    (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
    k_p, k_m = cst_all["k_p"], cst_all["k_m"]
    
    ## time step
    Dt = tk/cst_all["Nt"]

    print("DM = ", cst_all["DM"], "and lambda = ", cst_all["Lambda"])
    
    ## profil de concentration inital
    # calcul de l'equilibre entre a et b (a+b =a0+b0 et a/b=K)
    (C_a_eq, C_b_eq) = compute_equilibrium(cst_all["C_a"], cst_all["C_b"], cst_all["K"]) 
    cst_conc_eq = (C_a_eq, C_b_eq, cst_all["C_c"])
    C_new = np.append([C_a_eq for i in range(Nx)], [C_b_eq for i in range(Nx)])
    C_new = np.append(C_new, [cst_conc_eq[2] for i in range(Nx)])
   
    
    ## propagation temporelle
    fig, ax = plt.subplots(4, figsize=(20, 10))
    (M_new_constant, M_old) = Matrix_constant_CE(Nx, Dt, 3, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_CE_boundaries(M_new_constant, 
                                     t, 
                                     E, 
                                     cst_all["Lambda"], 
                                     Nx, 
                                     F_norm, 
                                     cst_all["E_0_1"], 
                                     cst_all["alpha"],
                                     cst_all["n"])
        C_new = compute_Cnew_CE(M_new, M_old, C_old, cst_conc_eq, Nx)
        I = np.append(I, compute_I_CE(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-2*Nx], label= 'time = %is' %(i*Dt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-Nx], label= 'time = %is' %(i*Dt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:], label= 'time = %is' %(i*Dt))

    ax[3].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
    return(I)

# main programm for cyclic staircase voltammetry
def main_CSV_CE(cst_all):
    F_norm = cst_all["F_norm"]
    Nx = cst_all["Nx"]
    Nt = cst_all["Nt"]
    DM = cst_all["DM"]
    Dx = cst_all["Dx"]
    (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
    k_p, k_m = cst_all["k_p"], cst_all["k_m"]
    
    ## time step
    Dt = tk/cst_all["Nt"]

    print("DM = ", cst_all["DM"], "and lambda = ", cst_all["Lambda"])
    
    ## profil de concentration inital
    # calcul de l'equilibre entre a et b (a+b =a0+b0 et a/b=K)
    (C_a_eq, C_b_eq) = compute_equilibrium(cst_all["C_a"], cst_all["C_b"], cst_all["K"]) 
    cst_conc_eq = (C_a_eq, C_b_eq, cst_all["C_c"])
    C_new = np.append([C_a_eq for i in range(Nx)], [C_b_eq for i in range(Nx)])
    C_new = np.append(C_new, [cst_conc_eq[2] for i in range(Nx)])
   
    
    ## propagation temporelle
    fig, ax = plt.subplots(4, figsize=(20, 10))
    (M_new_constant, M_old) = Matrix_constant_CE(Nx, Dt, 3, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_CE_boundaries(M_new_constant, 
                                     t, 
                                     E, 
                                     cst_all["Lambda"], 
                                     Nx, 
                                     F_norm, 
                                     cst_all["E_0_1"], 
                                     cst_all["alpha"],
                                     cst_all["n"])
        C_new = compute_Cnew_CE(M_new, M_old, C_old, cst_conc_eq, Nx)
        I = np.append(I, compute_I_CE(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-2*Nx], label= 'time = %is' %(i*Dt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-Nx], label= 'time = %is' %(i*Dt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:], label= 'time = %is' %(i*Dt))

    ax[3].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    
    return(I)

# main programm for square wave voltammetry
def main_SWV_CE(cst_all):
    F_norm = cst_all["F_norm"]
    Nx = cst_all["Nx"]
    Nt = cst_all["Nt"]
    DM = cst_all["DM"]
    Dx = cst_all["Dx"]
    (E, E_sweep, tk) = SWV(cst_all["E_i"], 
                           cst_all["E_ox"], 
                           cst_all["E_red"], 
                           cst_all["E_SW"], 
                           cst_all["Delta_E"], 
                           cst_all["f"], 
                           cst_all["Ox"])
    k_p, k_m = cst_all["k_p"], cst_all["k_m"]
    
    ## time step
    Dt = tk/cst_all["Nt"]

    print("DM = ", cst_all["DM"], "and lambda = ", cst_all["Lambda"])
    print("Dt = ", Dt, "and T = 2Pi/f = ", 2*np.pi/cst_all["f"])
    
    # arbitrary criteria to check if the time step is small enough compared to the time step of the SW
    if 20*Dt > 2*np.pi/cst_all["f"]:
        print("YOU SHOULD INCREASE THE NUMBER OF TIME STEPS TO GET MEANINGFUL RESULTS !")
    
    ## profil de concentration inital
    # calcul de l'equilibre entre a et b (a+b =a0+b0 et a/b=K)
    (C_a_eq, C_b_eq) = compute_equilibrium(cst_all["C_a"], cst_all["C_b"], cst_all["K"]) 
    cst_conc_eq = (C_a_eq, C_b_eq, cst_all["C_c"])
    C_new = np.append([C_a_eq for i in range(Nx)], [C_b_eq for i in range(Nx)])
    C_new = np.append(C_new, [cst_conc_eq[2] for i in range(Nx)])


    ## propagation temporelle
    fig, ax = plt.subplots(5, figsize=(10, 20))
    (M_new_constant, M_old) = Matrix_constant_CE(Nx, Dt, 3, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_CE_boundaries(M_new_constant, 
                                     t, 
                                     E, 
                                     cst_all["Lambda"], 
                                     Nx, 
                                     F_norm, 
                                     cst_all["E_0_1"], 
                                     cst_all["alpha"],
                                     cst_all["n"])
        C_new = compute_Cnew_CE(M_new, M_old, C_old, cst_conc_eq, Nx)
        I = np.append(I, compute_I_CE(C_new, cst_all))
        if i % math.floor(cst_all["Nt"]/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-2*Nx], label= 'time = %is' %(i*Dt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-Nx], label= 'time = %is' %(i*Dt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:], label= 'time = %is' %(i*Dt))

    ax[3].plot([E_sweep(i*Dt) for i in range(cst_all["Nt"])], I)
    ax[4].plot([i*Dt for i in range(cst_all["Nt"])],[E(i*(Dt)) for i in range(cst_all["Nt"])])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.show()
    return(I)

# exctraction des courants for rev et differentiels de la SWV
def plot_SWV(cst_all, I):
    (E, E_sweep, tk) = SWV(cst_all["E_i"], 
                           cst_all["E_ox"], 
                           cst_all["E_red"], 
                           cst_all["E_SW"], 
                           cst_all["Delta_E"], 
                           cst_all["f"], 
                           cst_all["Ox"])
    Nt = cst_all["Nt"]
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
    
    return(E_for, I_for, E_rev, I_rev, Delta_I)
