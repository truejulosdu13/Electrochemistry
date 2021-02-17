# this programmes numerically solves the partial derivative equations of the diffusion problem with bulk and electrode conditions with the Crank-Nicolson technique

import math
import numpy as np
import matplotlib.pyplot as plt

def Lapl_Mat_CE(Nx, n):                              ## def Laplacien troncated
    L = np.zeros((n*Nx,n*Nx))
    for j in range(n):
        for i in range(Nx-2):
            L[j*Nx+i+1,j*Nx+i+1] = -2
            L[j*Nx+i+1,j*Nx+i]   = +1
            L[j*Nx+i+1,j*Nx+i+2] = +1
    return(L)

def Id_tronc_CE(Nx, n):
    Identity_Mat = np.zeros((n*Nx,n*Nx))
    for j in range(n):
        for i in range(Nx-2):
            Identity_Mat[j*Nx+i+1,j*Nx+i+1] = 1
    return(Identity_Mat)

def Eq_Mat_CE(Nx, n, k_p, k_m):
    Eq_Mat = np.zeros((n*Nx,n*Nx))  
    for i in range(Nx-2):
        Eq_Mat[i+1,i+1]       = - k_p 
        Eq_Mat[i+1,Nx+i+1]    = + k_m
        Eq_Mat[Nx+i+1,i+1]    = + k_p
        Eq_Mat[Nx+i+1,Nx+i+1] = - k_m       
    return(Eq_Mat)

# a first constant over time Matrix is defined
def Matrix_constant_CE(Nx, Dt, n, k_p, k_m, DM):
    M_new_constant = Id_tronc_CE(Nx, n) - 0.5*DM*Lapl_Mat_CE(Nx, n) - 0.5*Dt*Eq_Mat_CE(Nx, n, k_p, k_m)
    M_old_constant = Id_tronc_CE(Nx, n) + 0.5*DM*Lapl_Mat_CE(Nx, n) + 0.5*Dt*Eq_Mat_CE(Nx, n, k_p, k_m) 
    return(M_new_constant, M_old_constant)

# the time dependend Matrix is defined thanks to the constant Matrix and the boundaries conditions of the electrochemical problem
def Matrix_CE_boundaries_red(M_new_constant, t, E, Lambda, Nx, F_norm, E_0_1, alpha, n): 
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = 1
    
        # zero flux condition on C_a
    M_new[0,0]  = + 1
    M_new[0,1]  = - 1 
    
        # current condition on C_b
    M_new[Nx, Nx]  = + 1 + Lambda*k_red(t, E, n, F_norm, E_0_1, alpha)
    M_new[Nx, Nx+1] = - 1
    M_new[Nx, 2*Nx] = - Lambda*k_ox(t, E, n, F_norm, E_0_1, alpha)
    
        # equality of the concentration flux at the electrode
    M_new[2*Nx, Nx]     = + 1
    M_new[2*Nx, Nx+1]   = - 1
    M_new[2*Nx, 2*Nx]   = + 1
    M_new[2*Nx, 2*Nx+1] = - 1
    
    return(M_new)

def Matrix_CE_boundaries_ox(M_new_constant, t, E, Lambda, Nx, F_norm, E_0_1, alpha, n): 
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = 1
    
        # zero flux condition on C_a
    M_new[0,0]  = + 1
    M_new[0,1]  = - 1 
    
        # current condition on C_b
    M_new[Nx, Nx]  = + 1 + Lambda*k_ox(t, E, n, F_norm, E_0_1, alpha)
    M_new[Nx, Nx+1] = - 1
    M_new[Nx, 2*Nx] = - Lambda*k_red(t, E, n, F_norm, E_0_1, alpha)
    
        # equality of the concentration flux at the electrode
    M_new[2*Nx, Nx]     = + 1
    M_new[2*Nx, Nx+1]   = - 1
    M_new[2*Nx, 2*Nx]   = + 1
    M_new[2*Nx, 2*Nx+1] = - 1
    
    return(M_new)


def compute_equilibrium(C_a, C_b, K):
    C_a_eq = (C_a + C_b)/(1+K)
    C_b_eq = (C_a + C_b)*K/(1+K)
    print("C_a_eq = ", C_a_eq, "C_b_eq = ", C_b_eq, "K =", K)
    return(C_a_eq, C_b_eq)

# the dime dependent Matrix requires the definition of Butler-Volmer kinetic constants :
def sigma(t, E, cst_syst):                                        ## definition de sigma
    s = np.exp(n*F*(E(t) - E_0)/(R*T))       
    return(s)  

def k_red(t, E, n, F_norm, E_0_1, alpha):                                        ## definition de k_red
    s = np.exp(-n*alpha*F_norm*(E(t) - E_0_1))      
    return(s)  

def k_ox(t, E, n, F_norm, E_0_1, alpha):                                       ## definition de k_ox
    s = np.exp((1-alpha)*n*F_norm*(E(t) - E_0_1))       
    return(s)   

# calculating the right hand side term for the E mecanism
def RHS_CE(M_old, C_old, cst_conc, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = cst_conc[0]
    RHS[Nx]     = 0
    RHS[2*Nx-1] = cst_conc[1] 
    RHS[2*Nx]     = 0
    RHS[3*Nx-1] = cst_conc[2] 
    return(RHS)

# temporal propagation of the concentration profile
def compute_Cnew_CE(M_new, M_old, C_old, cst_conc, Nx):
    RHS = RHS_CE(M_old, C_old, cst_conc, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)

# computing the current for any concentration profile
def compute_I_CE_red(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*(1*(C[Nx+1]-C[Nx+0])-1*(C[2*Nx+1]-C[2*Nx]))/Dx
    return(Intensity)

def compute_I_CE_ox(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*(1*(C[Nx+1]-C[Nx+0])-1*(C[2*Nx+1]-C[2*Nx]))/Dx
    return(Intensity)


    
    