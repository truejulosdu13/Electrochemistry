import sys
sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('viridis', 12)
from potential_applied import *
from ECE_mechanism.EDP_solver_ECE import *

# main programm for linear sweep voltammetry
def main_LSV_ECE_red(cst_all):
    F_norm = cst_all["F_norm"]
    Nx = cst_all["Nx"]
    Nt = cst_all["Nt"]
    DM = cst_all["DM"]
    Dx = cst_all["Dx"]
    k_p, k_m = cst_all["k_p"], cst_all["k_m"]
    (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])

    ## time step
    Dt = tk/cst_all["Nt"]

    print("DM = ", cst_all["DM"], "and lambda = ", cst_all["Lambda"])
    
    ## profil de concentration inital
    C_new = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_c"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_d"] for i in range(Nx)])
   
    
    ## propagation temporelle
    fig, ax = plt.subplots(6, figsize=(10, 30))
    
    (M_new_constant, M_old) = Matrix_constant_ECE(Nx, Dt, 4, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_ECE_boundaries_red(M_new_constant, t, E, 
                                          cst_all["Lambda"],
                                          Nx, 
                                          F_norm, 
                                          cst_all["E_0_1"],
                                          cst_all["E_0_2"],
                                          cst_all["alpha"], 
                                          cst_all["n"])
        C_new = compute_Cnew_ECE(M_new, M_old, C_old, 
                                 cst_all["C_a"],
                                 cst_all["C_b"],
                                 cst_all["C_c"],
                                 cst_all["C_d"],
                                 Nx)
        I = np.append(I, compute_I_ECE_red(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            

    ax[4].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[5].plot([i*Dt for i in range(Nt)], [E(i*(Dt)) for i in range(Nt)])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[0].title.set_text('Profil de concentration de A en fonction du temps')
    ax[1].title.set_text('Profil de concentration de B en fonction du temps')
    ax[2].title.set_text('Profil de concentration de C en fonction du temps')
    ax[3].title.set_text('Profil de concentration de D en fonction du temps')
    titre_i_E = f"Courbe intensité potentiel ECE E1 = {cst_all['E_0_1']} V et E2 = {cst_all['E_0_2']} V."
    ax[4].title.set_text(titre_i_E)
    ax[5].title.set_text('E(t)')
    plt.savefig('ECE.png')
    plt.show()
    
    return(I)

# main programm for cyclic staircase voltammetry
def main_CSV_ECE_red(cst_all):
    F_norm = cst_all["F_norm"]
    Nx = cst_all["Nx"]
    Nt = cst_all["Nt"]
    DM = cst_all["DM"]
    Dx = cst_all["Dx"]
    k_p, k_m = cst_all["k_p"], cst_all["k_m"]
    (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
    
    
    ## time step
    Dt = tk/cst_all["Nt"]

    print("DM = ", cst_all["DM"], "and lambda = ", cst_all["Lambda"])
    
    ## profil de concentration inital
    C_new = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_c"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_d"] for i in range(Nx)])
   
    
    ## propagation temporelle
    fig, ax = plt.subplots(6, figsize=(10, 30))
    
    (M_new_constant, M_old) = Matrix_constant_ECE(Nx, Dt, 4, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_ECE_boundaries_red(M_new_constant, t, E, 
                                          cst_all["Lambda"],
                                          Nx, 
                                          F_norm, 
                                          cst_all["E_0_1"],
                                          cst_all["E_0_2"],
                                          cst_all["alpha"], 
                                          cst_all["n"])
        C_new = compute_Cnew_ECE(M_new, M_old, C_old, 
                                 cst_all["C_a"],
                                 cst_all["C_b"],
                                 cst_all["C_c"],
                                 cst_all["C_d"],
                                 Nx)
        I = np.append(I, compute_I_ECE_red(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            

    ax[4].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[5].plot([i*Dt for i in range(Nt)], [E(i*(Dt)) for i in range(Nt)])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[0].title.set_text('Profil de concentration de A en fonction du temps')
    ax[1].title.set_text('Profil de concentration de B en fonction du temps')
    ax[2].title.set_text('Profil de concentration de C en fonction du temps')
    ax[3].title.set_text('Profil de concentration de D en fonction du temps')
    titre_i_E = f"Courbe intensité potentiel ECE E1 = {cst_all['E_0_1']} V et E2 = {cst_all['E_0_2']} V."
    ax[4].title.set_text(titre_i_E)
    ax[5].title.set_text('E(t)')
    plt.savefig('ECE.png')
    plt.show()
    
    return(I)

# main programm for cyclic staircase voltammetry
def main_CSV_ECE_ox(cst_all):
    F_norm = cst_all["F_norm"]
    Nx = cst_all["Nx"]
    Nt = cst_all["Nt"]
    DM = cst_all["DM"]
    Dx = cst_all["Dx"]
    k_p, k_m = cst_all["k_p"], cst_all["k_m"]
    (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])

    ## time step
    Dt = tk/cst_all["Nt"]

    print("DM = ", cst_all["DM"], "and lambda = ", cst_all["Lambda"])
    
    ## profil de concentration inital
    C_new = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_c"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_d"] for i in range(Nx)])
   
    
    ## propagation temporelle
    fig, ax = plt.subplots(6, figsize=(10, 30))
    
    (M_new_constant, M_old) = Matrix_constant_ECE(Nx, Dt, 4, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_ECE_boundaries_ox(M_new_constant, t, E, 
                                          cst_all["Lambda"],
                                          Nx, 
                                          F_norm, 
                                          cst_all["E_0_1"],
                                          cst_all["E_0_2"],
                                          cst_all["alpha"], 
                                          cst_all["n"])
        C_new = compute_Cnew_ECE(M_new, M_old, C_old, 
                                 cst_all["C_a"],
                                 cst_all["C_b"],
                                 cst_all["C_c"],
                                 cst_all["C_d"],
                                 Nx)
        I = np.append(I, compute_I_ECE_red(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            

    ax[4].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[5].plot([i*Dt for i in range(Nt)], [E(i*(Dt)) for i in range(Nt)])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[0].title.set_text('Profil de concentration de A en fonction du temps')
    ax[1].title.set_text('Profil de concentration de B en fonction du temps')
    ax[2].title.set_text('Profil de concentration de C en fonction du temps')
    ax[3].title.set_text('Profil de concentration de D en fonction du temps')
    titre_i_E = f"Courbe intensité potentiel ECE E1 = {cst_all['E_0_1']} V et E2 = {cst_all['E_0_2']} V."
    ax[4].title.set_text(titre_i_E)
    ax[5].title.set_text('E(t)')
    plt.savefig('ECE.png')
    plt.show()
    
    return(I)
        

# main programm for square wave voltammetry
def main_SWV_ECE_red(cst_all):
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
    Dt = tk/Nt

    print("DM = ", DM, "and lambda = ", cst_all["Lambda"])
    print("Dt = ", Dt, "and T = 2Pi/f = ", 2*np.pi/cst_all["f"])
    
    # arbitrary criteria to check if the time step is small enough compared to the time step of the SW
    if 20*Dt > 2*np.pi/cst_all["f"]:
        print("YOU SHOULD INCREASE THE NUMBER OF TIME STEPS TO GET MEANINGFUL RESULTS !")
    
        ## profil de concentration inital
    C_new = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_c"] for i in range(Nx)])
    C_new = np.append(C_new, [cst_all["C_d"] for i in range(Nx)])
   
    
    ## propagation temporelle
    fig, ax = plt.subplots(6, figsize=(10, 30))
    
    (M_new_constant, M_old) = Matrix_constant_ECE(Nx, Dt, 4, k_p, k_m, DM)
    I = np.array(())

    for i in range(Nt):
        C_old = C_new
        t = i*Dt 
        M_new = Matrix_ECE_boundaries_red(M_new_constant, t, E, 
                                          cst_all["Lambda"],
                                          Nx, 
                                          F_norm, 
                                          cst_all["E_0_1"],
                                          cst_all["E_0_2"],
                                          cst_all["alpha"], 
                                          cst_all["n"])
        C_new = compute_Cnew_ECE(M_new, M_old, C_old, 
                                 cst_all["C_a"],
                                 cst_all["C_b"],
                                 cst_all["C_c"],
                                 cst_all["C_d"],
                                 Nx)
        I = np.append(I, compute_I_ECE_red(C_new, cst_all))
        if i % math.floor(Nt/10) == 0:
            ax[0].plot([j*Dx for j in range(Nx)], C_new[:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            

    ax[4].plot([E(i*(Dt)) for i in range(Nt)], I)
    ax[5].plot([i*Dt for i in range(Nt)], [E(i*(Dt)) for i in range(Nt)])
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax[0].title.set_text('Profil de concentration de A en fonction du temps')
    ax[1].title.set_text('Profil de concentration de B en fonction du temps')
    ax[2].title.set_text('Profil de concentration de C en fonction du temps')
    ax[3].title.set_text('Profil de concentration de D en fonction du temps')
    titre_i_E = f"Courbe SWV ECE E1 = {cst_all['E_0_1']} V et E2 = {cst_all['E_0_2']} V."
    ax[4].title.set_text(titre_i_E)
    ax[5].title.set_text('E(t)')
    plt.savefig('ECE.png')
    plt.show()
    
    return(I)