from potential_applied import SWV
from potential_applied import CSV
import numpy as np

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

# exctraction du courant mesuré expérimentalement en CSV à partir de I_total_CSV
def extract_expe_like_CSV(cst_all, I):
    (E, tk) = CSV(cst_all["E_i"], cst_all["E_ox"], cst_all["E_red"], cst_all["Delta_E"], cst_all["v"])
    ## time step
    Nt = cst_all["Nt"]
    Dt = tk/Nt
    
    Expe_Potential = [E(0)]
    Expe_Intensity = [I[0]]

    i = 1
    while i < Nt:
        if E((i-1)*Dt) == E(i*Dt):
            i += 1
        else:
            Expe_Intensity.append(I[i-1])
            Expe_Potential.append(E((i-1)*Dt))
            i += 1   
            
    return(Expe_Potential, Expe_Intensity)