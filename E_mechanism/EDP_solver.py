# this programmes numerically solves the partial derivative equations of the diffusion problem with bulk and electrode conditions with the Crank-Nicolson technique

import math
import numpy as np
import matplotlib.pyplot as plt

def Laplacian_Matrix_E(Nx):                              ## def Laplacien
    L = np.zeros((Nx,Nx))
    for i in range(Nx-2):
        L[i+1,i+1] = -2
        L[i+1,i]   = +1
        L[i+1,i+2] = +1
    return(L)

# a first constant over time Matrix is defined
def Matrix_constant_E(Nx, DM):
    # bloc building of Matrix_new and Matrix_old
    Identite_tronquee = np.concatenate((np.zeros((1,Nx-2)), np.eye(Nx-2), np.zeros((1,Nx-2))), axis=0)
    Identite_tronquee = np.concatenate((np.zeros((Nx,1)), Identite_tronquee, np.zeros((Nx,1))), axis=1)
    M_new_0_0 = M_new_1_1 = Identite_tronquee - 0.5*DM*Laplacian_Matrix_E(Nx)
    M_old_0_0 = M_old_1_1 = Identite_tronquee + 0.5*DM*Laplacian_Matrix_E(Nx)
    M_new_1_0 = M_new_0_1 = M_old_1_0 = M_old_0_1 = np.zeros((Nx,Nx))
    
    # assemblage des blocs
    M_new_0 = np.concatenate((M_new_0_0, M_new_0_1), axis=1)
    M_new_1 = np.concatenate((M_new_1_0, M_new_1_1), axis=1)
    M_old_0 = np.concatenate((M_old_0_0, M_old_0_1), axis=1)
    M_old_1 = np.concatenate((M_old_1_0, M_old_1_1), axis=1)
    
    M_new_constant = np.concatenate((M_new_0, M_new_1), axis=0)
    M_old_constant = np.concatenate((M_old_0, M_old_1), axis=0)
    
    return(M_new_constant, M_old_constant)
  
# the time dependend Matrix is defined thanks to the constant Matrix and the boundaries conditions of the electrochemical problem
def Matrix_E_Non_Nernst_boundaries_red(M_new_constant, t, E, Lambda, Nx, F_norm, E_0_1, alpha, n):       
    M_new = M_new_constant

    # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = 1
    
    # current condition on Cox
    M_new[0, 0] = + 1 + Lambda*k_red(t, E, n, F_norm, E_0_1, alpha)
    M_new[0, 1] = - 1
    M_new[0,Nx] = - Lambda*k_ox(t, E, n, F_norm, E_0_1, alpha)
    
    # equality of the concentration flux at the electrode
    M_new[Nx, 0]  = + 1
    M_new[Nx, 1]  = - 1
    M_new[Nx, Nx] = + 1
    M_new[Nx, Nx+1] = -1
    
    return(M_new)  

def Matrix_E_Non_Nernst_boundaries_ox(M_new_constant, t, E, Lambda, Nx, F_norm, E_0_1, alpha, n):       
    M_new = M_new_constant
    # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = 1 
    # current condition on Cred
    M_new[0, 0] = + 1 + Lambda*k_ox(t, E, n, F_norm, E_0_1, alpha)
    M_new[0, 1] = - 1
    M_new[0,Nx] = - Lambda*k_red(t, E, n, F_norm, E_0_1, alpha)    
    # equality of the concentration flux at the electrode
    M_new[Nx, 0]  = + 1
    M_new[Nx, 1]  = - 1
    M_new[Nx, Nx] = + 1
    M_new[Nx, Nx+1] = -1
    return(M_new)

# the dime dependent Matrix requires the definition of Butler-Volmer kinetic constants :
def sigma(t, E, n, F_norm, E_0_1):                                        ## definition de sigma
    s = np.exp(n*F_norm*(E(t) - E_0_1))       
    return(s)  

def k_red(t, E, n, F_norm, E_0_1, alpha):                                        ## definition de k_red
    s = np.exp(-n*alpha*F_norm*(E(t) - E_0_1))      
    return(s)  

def k_ox(t, E, n, F_norm, E_0_1, alpha):                                       ## definition de k_ox
    s = np.exp((1-alpha)*n*F_norm*(E(t) - E_0_1))       
    return(s)  

# calculating the right hand side term for the E mecanism
def RHS_E(M_old, C_old, C_a, C_b, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = C_a
    RHS[Nx]     = 0
    RHS[2*Nx-1] = C_b 
    return(RHS)

# temporal propagation of the concentration profile
def compute_Cnew_E(M_new, M_old, C_old, C_a, C_b, Nx):
    RHS = RHS_E(M_old, C_old, C_a, C_b, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)

# computing the current for any concentration profile
def compute_I_E(C, cst_all):
    # reference = Heinze 1993
    # I = S*sum(zi*fi) 
    # where zi is the algebraic charge of the current, the convention of positiv anodic current is taken
    # fi stands for the flux of the specie i at the electrode 
    # first approximation of the flux is taken here as the first partial derivative along space
    # 
    # Note :Intensity = n*S*(k_ox(t, E)*C[Nx] - k_red(t,E)*C[0]) cette expression n'est pas stable du au fait 
    # que les concentration à l'electrode sont trop faibles. on obtient un signal bruité.
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*(1*(C[1]-C[0])-1*(C[Nx+1]-C[Nx]))/Dx
    return(Intensity)


