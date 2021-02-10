# this programmes numerically solves the partial derivative equations of the diffusion problem with bulk and electrode conditions with the Crank-Nicolson technique

import math
import numpy as np
import matplotlib.pyplot as plt

def Laplacian_Matrix(Nx, n):                              ## def Laplacien troncated
    L = np.zeros((n*Nx,n*Nx))
    for j in range(n):
        for i in range(Nx-2):
            L[j*Nx+i+1,j*Nx+i+1] = -2
            L[j*Nx+i+1,j*Nx+i]   = +1
            L[j*Nx+i+1,j*Nx+i+2] = +1
    return(L)

def Identity_troncated(Nx, n):
    Identity_Mat = np.zeros((n*Nx,n*Nx))
    for j in range(n):
        for i in range(Nx-2):
            Identity_Mat[j*Nx+i+1,j*Nx+i+1] = 1
    return(Identity_Mat)

def Equilibrium_Matrix(Nx, n, k_p, k_m):
    Eq_Mat = np.zeros((n*Nx,n*Nx))  
    for i in range(Nx-2):
        Eq_Mat[i+1,i+1]       = -k_p 
        Eq_Mat[i+1,Nx+i+1]    = +k_m
        Eq_Mat[Nx+i+1,i+1]    = -k_p
        Eq_Mat[Nx+i+1,Nx+i+1] = -k_m       
    return(Eq_Mat)

# a first constant over time Matrix is defined
def Matrix_constant_CE(Nx, Dt, n, k_p, k_m, DM):
    M_new_constant = Identity_troncated(Nx, n) - 0.5*DM*Laplacian_Matrix(Nx, n) - 0.5*Dt*Equilibrium_Matrix(Nx, n, k_p, k_m)
    M_old_constant = Identity_troncated(Nx, n) + 0.5*DM*Laplacian_Matrix(Nx, n) + 0.5*Dt*Equilibrium_Matrix(Nx, n, k_p, k_m)  
    return(M_new_constant, M_old_constant)

# the time dependend Matrix is defined thanks to the constant Matrix and the boundaries conditions of the electrochemical problem
def Matrix_CE_boundaries(M_new_constant, t, E, Lambda, Nx, F_norm, cst_syst): 
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = 1
    
        # current condition on C_b
    M_new[Nx, Nx]  = + 1 + Lambda*k_red(t, E, cst_syst, F_norm)
    M_new[Nx, Nx+1] = - 1
    M_new[Nx, 2*Nx] = - Lambda*k_ox(t, E, cst_syst, F_norm)
    
        # equality of the concentration flux at the electrode
    M_new[2*Nx, Nx]     = + 1
    M_new[2*Nx, Nx+1]   = - 1
    M_new[2*Nx, 2*Nx]   = + 1
    M_new[2*Nx, 2*Nx+1] = - 1
    
    return(M_new)

def compute_equilibrium(cst_conc, cst_syst):
    K = cst_syst[6]/cst_syst[7]
    C_a_eq = (cst_conc[0] + cst_conc[1])/(1+K)
    C_b_eq = (cst_conc[0] + cst_conc[1])*K/(1+K)
    return(C_a_eq, C_b_eq)

# the dime dependent Matrix requires the definition of Butler-Volmer kinetic constants :
def sigma(t, E, cst_syst):                                        ## definition de sigma
    s = np.exp(n*F*(E(t) - E_0)/(R*T))       
    return(s)  

def k_red(t, E, cst_syst, F_norm):                                        ## definition de k_red
    s = np.exp(-cst_syst[3]*cst_syst[2]*F_norm*(E(t) - cst_syst[0]))      
    return(s)  

def k_ox(t, E, cst_syst, F_norm):                                        ## definition de k_ox
    s = np.exp((1-cst_syst[3])*cst_syst[2]*F_norm*(E(t) - cst_syst[0]))       
    return(s)  

# calculating the right hand side term for the E mecanism
def RHS_E(M_old, C_old, cst_conc, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = cst_conc[0]
    RHS[Nx]     = 0
    RHS[2*Nx-1] = cst_conc[1] 
    RHS[2*Nx]     = 0
    RHS[3*Nx-1] = cst_conc[2] 
    return(RHS)

# temporal propagation of the concentration profile
def compute_Cnew(M_new, M_old, C_old, cst_conc, Nx):
    RHS = RHS_E(M_old, C_old, cst_conc, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)

# computing the current for any concentration profile
def compute_I_CE(C, cst_all):
    # reference = Heinze 1993
    # I = S*sum(zi*fi) 
    # where zi is the algebraic charge of the current, the convention of positiv anodic current is taken
    # fi stands for the flux of the specie i at the electrode 
    # first approximation of the flux is taken here as the first partial derivative along space
    # 
    # Note :Intensity = n*S*(k_ox(t, E)*C[Nx] - k_red(t,E)*C[0]) cette expression n'est pas stable du au fait 
    # que les concentration à l'electrode sont trop faibles. on obtient un signal bruité.
    n, F, D, S, Nx, Dx = cst_all[2][2], cst_all[0][2], cst_all[2][1], cst_all[0][4], cst_all[3][1], cst_all[3][5]
    Intensity = -n*F*D*S*(1*(C[Nx+1]-C[Nx+0])-1*(C[2*Nx+1]-C[2*Nx]))/Dx
    return(Intensity)

    
    