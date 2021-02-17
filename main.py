import math
#import sys
#sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from potential_applied import *
from EDP_solver_tools import *


def initialise(cst_all):
    if cst_all["expe_type"] == 'LSV':
        if cst_all["mechanism"] == 'E':
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(cts_all["Nx"])],[cst_all["C_b"] for i in range(cst_all["Nx"])])
            (M_new_constant, M_old) = Matrix_constant_E(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_E_Non_Nernst_boundaries_red
            else:
                fun = Matrix_E_Non_Nernst_boundaries_ox     
        
        elif cst_all["mechanism"] == 'CE':
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            (C_a_eq, C_b_eq) = compute_equilibrium_CE(cst_all["C_a"], cst_all["C_b"], cst_all["K"]) 
            cst_conc_eq = (C_a_eq, C_b_eq, cst_all["C_c"])
            C_init = np.append([C_a_eq for i in range(Nx)], [C_b_eq for i in range(Nx)])
            C_init = np.append(C_new, [cst_conc_eq[2] for i in range(Nx)])
            (M_new_constant, M_old) = Matrix_constant_CE(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_CE_boundaries_red
            else:
                fun = Matrix_CE_boundaries_ox
            
        
        elif cst_all["mechanism"] == 'ECE':
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_new, [cst_all["C_c"] for i in range(Nx)])
            C_init = np.append(C_new, [cst_all["C_d"] for i in range(Nx)])
            (M_new_constant, M_old) = Matrix_constant_ECE(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_ECE_boundaries_red
            else:
                fun = Matrix_ECE_boundaries_ox
    
    elif cst_all["expe_type"] == 'CSV':
        if cst_all["mechanism"] == 'E':
            (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(cts_all["Nx"])],[cst_all["C_b"] for i in range(cst_all["Nx"])])
            (M_new_constant, M_old) = Matrix_constant_E(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_E_Non_Nernst_boundaries_red
            else:
                fun = Matrix_E_Non_Nernst_boundaries_ox  
        
        elif cst_all["mechanism"] == 'CE':
            (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            (C_a_eq, C_b_eq) = compute_equilibrium_CE(cst_all["C_a"], cst_all["C_b"], cst_all["K"]) 
            cst_conc_eq = (C_a_eq, C_b_eq, cst_all["C_c"])
            C_init = np.append([C_a_eq for i in range(Nx)], [C_b_eq for i in range(Nx)])
            C_init = np.append(C_new, [cst_conc_eq[2] for i in range(Nx)])
            (M_new_constant, M_old) = Matrix_constant_CE(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_CE_boundaries_red
            else:
                fun = Matrix_CE_boundaries_ox
        
        elif cst_all["mechanism"] == 'ECE':
            (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_new, [cst_all["C_c"] for i in range(Nx)])
            C_init = np.append(C_new, [cst_all["C_d"] for i in range(Nx)])
            (M_new_constant, M_old) = Matrix_constant_ECE(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_ECE_boundaries_red
            else:
                fun = Matrix_ECE_boundaries_ox
    
    elif cst_all["expe_type"] == 'SWV':
        if cst_all["mechanism"] == 'E':
            (E, E_sweep, tk) = SWV(cst_all["E_i"], 
                           cst_all["E_ox"], 
                           cst_all["E_red"], 
                           cst_all["E_SW"], 
                           cst_all["Delta_E"], 
                           cst_all["f"], 
                           cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(cts_all["Nx"])],[cst_all["C_b"] for i in range(cst_all["Nx"])])
            (M_new_constant, M_old) = Matrix_constant_E(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_E_Non_Nernst_boundaries_red
            else:
                fun = Matrix_E_Non_Nernst_boundaries_ox  
        
        elif cst_all["mechanism"] == 'CE':
            (E, E_sweep, tk) = SWV(cst_all["E_i"], 
                           cst_all["E_ox"], 
                           cst_all["E_red"], 
                           cst_all["E_SW"], 
                           cst_all["Delta_E"], 
                           cst_all["f"], 
                           cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            (C_a_eq, C_b_eq) = compute_equilibrium_CE(cst_all["C_a"], cst_all["C_b"], cst_all["K"]) 
            cst_conc_eq = (C_a_eq, C_b_eq, cst_all["C_c"])
            C_init = np.append([C_a_eq for i in range(Nx)], [C_b_eq for i in range(Nx)])
            C_init = np.append(C_new, [cst_conc_eq[2] for i in range(Nx)])
            (M_new_constant, M_old) = Matrix_constant_CE(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_CE_boundaries_red
            else:
                fun = Matrix_CE_boundaries_ox
        
        elif cst_all["mechanism"] == 'ECE':
            (E, E_sweep, tk) = SWV(cst_all["E_i"], 
                           cst_all["E_ox"], 
                           cst_all["E_red"], 
                           cst_all["E_SW"], 
                           cst_all["Delta_E"], 
                           cst_all["f"], 
                           cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_new, [cst_all["C_c"] for i in range(Nx)])
            C_init = np.append(C_new, [cst_all["C_d"] for i in range(Nx)])
            (M_new_constant, M_old) = Matrix_constant_ECE(Nx, DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_ECE_boundaries_red
            else:
                fun = Matrix_ECE_boundaries_ox
        
    return(cst_all, E, C_init, M_new_constant, M_old, fun)

def compute_equilibrium_CE(C_a, C_b, K):
    C_a_eq = (C_a + C_b)/(1+K)
    C_b_eq = (C_a + C_b)*K/(1+K)
    print("C_a_eq = ", C_a_eq, "C_b_eq = ", C_b_eq, "K =", K)
    return(C_a_eq, C_b_eq)

def calculate_I(cst_all, E, C_init, M_new_constant, M_old, fun):
    I = np.array(())
    Potential = np.array(())
    Time = np.array(())
    C_new = C_init
    for i in range(cst_all["Nt"]):
        C_old = C_new
        t = i*Dt 
        M_new = fun(M_new_constant, t, E, cst_all)
        C_new = compute_Cnew_E(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["Nx"])
        I = np.append(I, compute_I_E(C_new, cst_all))
        Potential = np.append(Potential, E(t))
        Time = np.append(Time, t)
    return(I, Potential, Time)

def plot_concentration_profiles(cst_all, E, C_init, M_new_constant, M_old, fun):
    num_species = 0
    if cst_all["mecanism"] == 'E':
        num_species = 2
    elif cst_all["mecanism"] == 'CE':
        num_species = 3
    elif cst_all["mecanism"] == 'ECE':
        num_species = 4
        
    fig, ax = plt.subplots(num_species, figsize=(20, 10))
    C_new = C_init
    for i in range(cst_all["Nt"]):
        C_old = C_new
        t = i*Dt 
        M_new = fun(M_new_constant, t, E, cst_all)
        C_new = compute_Cnew_E(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["Nx"])
        
        if cst_all["mecanism"] == 'E':
            if i % math.floor(Nt/10) == 0:
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[0].title.set_text('Profil de concentration de A en fonction du temps')
            ax[1].title.set_text('Profil de concentration de B en fonction du temps')
            plt.savefig('ECE.png')
            plt.show()
            
            
        elif cst_all["mecanism"] == 'CE':
            if i % math.floor(Nt/10) == 0:
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[0].title.set_text('Profil de concentration de A en fonction du temps')
            ax[1].title.set_text('Profil de concentration de B en fonction du temps')
            ax[2].title.set_text('Profil de concentration de C en fonction du temps')
            plt.savefig('ECE.png')
            plt.show()
            
            
        elif cst_all["mecanism"] == 'ECE':
            if i % math.floor(Nt/10) == 0:
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            ax[0].title.set_text('Profil de concentration de A en fonction du temps')
            ax[1].title.set_text('Profil de concentration de B en fonction du temps')
            ax[2].title.set_text('Profil de concentration de C en fonction du temps')
            ax[3].title.set_text('Profil de concentration de D en fonction du temps')
            plt.savefig('ECE.png')
            plt.show() 
