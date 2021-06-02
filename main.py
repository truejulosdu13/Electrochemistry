import math
#import sys
#sys.path.append('../')
import numpy as np
import matplotlib.pyplot as plt
from potential_applied import *
from EDP_solver_tools import *


def initialise(cst_all):
    Nx = cst_all["Nx"]
    cst_all["K"]  = cst_all["k_p"]/cst_all["k_m"]
    cst_all["K2"]  = cst_all["k_p_2"]/cst_all["k_m_2"]
#    DM = cst_all["DM"]
    
    if cst_all["expe_type"] == 'LSV':
        if cst_all["mechanism"] == 'E':
            cst_all["num_species"] = 2
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(cst_all["Nx"])],[cst_all["C_b"] for i in range(cst_all["Nx"])])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_E(Nx, cst_all["num_species"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_E_Non_Nernst_boundaries_red
                fun_I = compute_I_E_red
            else:
                fun = Matrix_E_Non_Nernst_boundaries_ox 
                fun_I = compute_I_E_ox
        
        elif cst_all["mechanism"] == 'CE':
            cst_all["num_species"] = 3
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            cst_all = compute_equilibrium_CE(cst_all) 
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_c"] for i in range(Nx)])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_CE(Nx, cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_CE_boundaries_red
                fun_I = compute_I_CE_red
            else:
                fun = Matrix_CE_boundaries_ox
                fun_I = compute_I_CE_ox
                
        elif cst_all["mechanism"] == 'EC':
            cst_all["num_species"] = 3
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_c"] for i in range(Nx)])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_EC(Nx, cst_all["Dt"], cst_all["num_species"], cst_all["kc"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_EC_boundaries_red
                fun_I = compute_I_EC_red
            else:
                fun = Matrix_EC_boundaries_ox
                fun_I = compute_I_EC_ox
            
        
        elif cst_all["mechanism"] == 'ECE':
            cst_all["num_species"] = 4
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_c"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_d"] for i in range(Nx)])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_ECE(Nx, cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_ECE_boundaries_red
                fun_I = compute_I_ECE_red
            else:
                fun = Matrix_ECE_boundaries_ox
                fun_I = compute_I_ECE_ox
                
        elif cst_all["mechanism"] == 'AL':
            cst_all["num_species"] = 5
            (E, tk) = rampe(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["v"], cst_all["Ox"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            cst_all = compute_equilibrium_AL(cst_all) 
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_c"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_d"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_e"] for i in range(Nx)])
            (M_new_constant, M_old) = (0,0) 
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            if cst_all["Reductible"] == True:
                fun = Matrix_AL_boundaries_red
                fun_I = compute_I_ECE_red
            else:
                fun = Matrix_AL_boundaries_ox
                fun_I = compute_I_ECE_ox
    
    elif cst_all["expe_type"] == 'CSV':
        if cst_all["mechanism"] == 'E':
            cst_all["num_species"] = 2
            (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(cst_all["Nx"])],[cst_all["C_b"] for i in range(cst_all["Nx"])])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_E(Nx, cst_all["num_species"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_E_Non_Nernst_boundaries_red
                fun_I = compute_I_E_red
            else:
                fun = Matrix_E_Non_Nernst_boundaries_ox
                fun_I = compute_I_E_ox
        
        elif cst_all["mechanism"] == 'CE':
            cst_all["num_species"] = 3
            (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            (C_a_eq, C_b_eq) = compute_equilibrium_CE(cst_all["C_a"], cst_all["C_b"], cst_all["K"]) 
            cst_conc_eq = (C_a_eq, C_b_eq, cst_all["C_c"])
            C_init = np.append([C_a_eq for i in range(Nx)], [C_b_eq for i in range(Nx)])
            C_init = np.append(C_init, [cst_conc_eq[2] for i in range(Nx)])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_CE(Nx, cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_CE_boundaries_red
                fun_I = compute_I_CE_red
            else:
                fun = Matrix_CE_boundaries_ox
                fun_I = compute_I_CE_ox
        
        elif cst_all["mechanism"] == 'ECE':
            cst_all["num_species"] = 4
            (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
            cst_all["exp_time"] = tk
            cst_all["Dt"] = tk/cst_all["Nt"]
            C_init = np.append([cst_all["C_a"] for i in range(Nx)], [cst_all["C_b"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_c"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_d"] for i in range(Nx)])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_ECE(Nx, cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_ECE_boundaries_red
                fun_I = compute_I_ECE_red
            else:
                fun = Matrix_ECE_boundaries_ox
                fun_I = compute_I_ECE_ox
    
    elif cst_all["expe_type"] == 'SWV':
        if cst_all["mechanism"] == 'E':
            cst_all["num_species"] = 2
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
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_E(Nx, cst_all["num_species"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_E_Non_Nernst_boundaries_red
                fun_I = compute_I_E_red
            else:
                fun = Matrix_E_Non_Nernst_boundaries_ox 
                fun_I = compute_I_E_ox
        
        elif cst_all["mechanism"] == 'CE':
            cst_all["num_species"] = 3
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
            C_init = np.append(C_init, [cst_conc_eq[2] for i in range(Nx)])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_CE(Nx, cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_CE_boundaries_red
                fun_I = compute_I_CE_red
            else:
                fun = Matrix_CE_boundaries_ox
                fun_I = compute_I_CE_ox
        
        elif cst_all["mechanism"] == 'ECE':
            cst_all["num_species"] = 4
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
            C_init = np.append(C_init, [cst_all["C_c"] for i in range(Nx)])
            C_init = np.append(C_init, [cst_all["C_d"] for i in range(Nx)])
            cst_all["L_cuve"] = math.floor((cst_all["D"]*tk)**0.5) + 0.1
            cst_all["Dx"] = cst_all["L_cuve"]/cst_all["Nx"] 
            DM = cst_all["DM"] = cst_all["D"]*cst_all["Dt"]/cst_all["Dx"]**2
            (M_new_constant, M_old) = Matrix_constant_ECE(Nx, cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], DM)
            if cst_all["Reductible"] == True:
                fun = Matrix_ECE_boundaries_red
                fun_I = compute_I_ECE_red
            else:
                fun = Matrix_ECE_boundaries_ox
                fun_I = compute_I_ECE_ox
                
                     # pas d'espace
    
    cst_all["Lambda"] = cst_all["k0"]*cst_all["Dx"]/cst_all["D"]
                
#    cst_all["Dx"] = np.sqrt(cst_all["D"]*cst_all["Dt"]/DM)
    print("$\lambda$ = ", cst_all["Lambda"], "DM = ", cst_all["DM"], "v = ", cst_all["v"])       
    return(cst_all, E, C_init, M_new_constant, M_old, fun, fun_I)

def compute_equilibrium_CE(cst_all):
    C_a = cst_all["C_a"]
    C_b = cst_all["C_b"]
    K   = cst_all["K"]
    C_a_eq = (C_a + C_b)/(1+K)
    C_b_eq = (C_a + C_b)*K/(1+K)
    print("C_a_eq = ", C_a_eq, "C_b_eq = ", C_b_eq, "K =", K)
    cst_all["C_a"] = C_a_eq
    cst_all["C_b"] = C_b_eq
    return(cst_all)

def compute_equilibrium_AL(cst_all):
    print("C_a = ", cst_all["C_a"], "C_b = ", cst_all["C_b"], "C_c = ", cst_all["C_c"], "K =", cst_all["K"])
    C_a = cst_all["C_a"]
    C_b = cst_all["C_b"]
    K   = cst_all["K"]
    # C_a : ester, C_b : AL, C_c : adduit
    Delta = (K*(C_a - C_b) + 1)**2 + 4*K*C_b
    x = (K*(C_a + C_b) + 1 + np.sqrt(Delta))/(2*K)
    if x <= min(C_a, C_b) and x >= 0 :
        C_a_eq = C_a - x
        C_b_eq = C_b - x
        C_c_eq = x
    else:
        x = (K*(C_a + C_b) + 1 - np.sqrt(Delta))/(2*K)
        C_a_eq = C_a - x
        C_b_eq = C_b - x
        C_c_eq = x
        
    print("C_a_eq = ", C_a_eq, "C_b_eq = ", C_b_eq, "C_c_eq = ", C_c_eq, "K =", K)
    cst_all["C_a"] = C_a_eq
    cst_all["C_b"] = C_b_eq
    cst_all["C_c"] = C_c_eq
    return(cst_all)

def calculate_I(cst_all, E, C_init, M_new_constant, M_old, fun, fun_I):
    I = np.array(())
    Potential = np.array(())
    Time = np.array(())
    C_new = C_init
    for i in range(cst_all["Nt"]):
        C_old = C_new
        t = i*cst_all["Dt"] 
        if cst_all["mechanism"] == 'AL':
            (M_new_constant, M_old) = Matrix_constant_AL(cst_all["Nx"], cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], cst_all["k_p_2"], cst_all["k_m_2"], cst_all["kc"], cst_all["kc2"], cst_all["DM"], C_old)
            M_new = fun(M_new_constant, t, E, cst_all)    
            C_new = compute_Cnew_AL(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["C_d"], cst_all["C_e"], cst_all["Nx"]) 
        else:    
            M_new = fun(M_new_constant, t, E, cst_all)    
            if cst_all["mechanism"] == 'ECE':
                C_new = compute_Cnew_ECE(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["C_d"], cst_all["Nx"])
            elif cst_all["mechanism"] == 'CE':
                C_new = compute_Cnew_CE(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["Nx"])
            elif cst_all["mechanism"] == 'EC':
                C_new = compute_Cnew_EC(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["Nx"])
            elif cst_all["mechanism"] == 'E':
                C_new = compute_Cnew_E(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["Nx"])    
        I = np.append(I, fun_I(C_new, cst_all))
        Potential = np.append(Potential, E(t))
        Time = np.append(Time, t)
    return(I, Potential, Time)

    
def plot_concentration_profiles(cst_all, E, C_init, M_new_constant, M_old, fun, fun_I):
    N_plots = 10
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    viridis = cm.get_cmap('viridis', 12)
    Nx = cst_all["Nx"]
    Nt = cst_all["Nt"]
    Dx = cst_all["Dx"]
    Dt = cst_all["Dt"]
    fig, ax = plt.subplots(cst_all["num_species"], figsize=(10, cst_all["num_species"]*10))
    C_new = C_init
    for i in range(cst_all["Nt"]):
        C_old = C_new
        t = i*cst_all["Dt"]
        if cst_all["mechanism"] == 'AL':
            (M_new_constant, M_old) = Matrix_constant_AL(cst_all["Nx"], cst_all["Dt"], cst_all["num_species"], cst_all["k_p"], cst_all["k_m"], cst_all["k_p_2"], cst_all["k_m_2"], cst_all["kc"], cst_all["kc2"], cst_all["DM"], C_old)
            M_new = fun(M_new_constant, t, E, cst_all)    
            C_new = compute_Cnew_AL(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["C_d"], cst_all["C_e"], cst_all["Nx"]) 
        else:    
            M_new = fun(M_new_constant, t, E, cst_all)    
            if cst_all["mechanism"] == 'ECE':
                C_new = compute_Cnew_ECE(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["C_d"], cst_all["Nx"])
            elif cst_all["mechanism"] == 'CE':
                C_new = compute_Cnew_CE(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["Nx"])
            elif cst_all["mechanism"] == 'EC':
                C_new = compute_Cnew_EC(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["C_c"], cst_all["Nx"])
            elif cst_all["mechanism"] == 'E':
                C_new = compute_Cnew_E(M_new, M_old, C_old, cst_all["C_a"], cst_all["C_b"], cst_all["Nx"]) 
                
        if i % math.floor(Nt/N_plots) == 0:
            if cst_all["mechanism"] == 'E':
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                
            elif cst_all["mechanism"] == 'EC':
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))   
            
            elif cst_all["mechanism"] == 'CE':
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))          
            
            elif cst_all["mechanism"] == 'ECE':
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                
            elif cst_all["mechanism"] == 'AL':
                ax[0].plot([j*Dx for j in range(Nx)], C_new[:-4*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[1].plot([j*Dx for j in range(Nx)], C_new[Nx:-3*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[2].plot([j*Dx for j in range(Nx)], C_new[2*Nx:-2*Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[3].plot([j*Dx for j in range(Nx)], C_new[3*Nx:-Nx], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
                ax[4].plot([j*Dx for j in range(Nx)], C_new[4*Nx:], label= 'time = %is' %(i*Dt), color = viridis(i/Nt))
            
    if cst_all["mechanism"] == 'E':
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[0].title.set_text('Profil de concentration de A en fonction du temps')
        ax[1].title.set_text('Profil de concentration de B en fonction du temps')
        plt.savefig('ECE.png')
        plt.show()
        
    elif cst_all["mechanism"] == 'EC':
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[0].title.set_text('Profil de concentration de A en fonction du temps')
        ax[1].title.set_text('Profil de concentration de B en fonction du temps')
        ax[2].title.set_text('Profil de concentration de C en fonction du temps')
        plt.savefig('ECE.png')
        plt.show()
        
    elif cst_all["mechanism"] == 'CE':
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[0].title.set_text('Profil de concentration de A en fonction du temps')
        ax[1].title.set_text('Profil de concentration de B en fonction du temps')
        ax[2].title.set_text('Profil de concentration de C en fonction du temps')
        plt.savefig('ECE.png')
        plt.show()
        
    elif cst_all["mechanism"] == 'ECE':
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
        
    elif cst_all["mechanism"] == 'AL':
        ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax[0].title.set_text('Profil de concentration de E en fonction du temps')
        ax[1].title.set_text('Profil de concentration de LA en fonction du temps')
        ax[2].title.set_text('Profil de concentration de E-LA en fonction du temps')
        ax[3].title.set_text('Profil de concentration de E-LA_red en fonction du temps')
        ax[4].title.set_text('Profil de concentration de E_red en fonction du temps')
        plt.savefig('AL.png')
        plt.show() 
