# this programmes numerically solves the partial derivative equations of the diffusion problem with bulk and electrode conditions with the Crank-Nicolson technique

import math
import numpy as np

## Calculate the n+1 concentration profile

# temporal propagation of the concentration profile for the E mechanism
def compute_Cnew_E(M_new, M_old, C_old, C_a, C_b, Nx):
    RHS = RHS_E(M_old, C_old, C_a, C_b, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)

# temporal propagation of the concentration profile for the CE mechanism
def compute_Cnew_EC(M_new, M_old, C_old, C_a, C_b, C_c, Nx):
    RHS = RHS_EC(M_old, C_old, C_a, C_b, C_c, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)

# temporal propagation of the concentration profile for the CE mechanism
def compute_Cnew_CE(M_new, M_old, C_old, C_a, C_b, C_c, Nx):
    RHS = RHS_CE(M_old, C_old, C_a, C_b, C_c, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)

# temporal propagation of the concentration profile for the ECE mechanism
def compute_Cnew_ECE(M_new, M_old, C_old, C_a, C_b, C_c, C_d, Nx):
    RHS = RHS_ECE(M_old, C_old, C_a, C_b, C_c, C_d, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)

# temporal propagation of the concentration profile for the AL mechanism
def compute_Cnew_AL(M_new, M_old, C_old, C_a, C_b, C_c, C_d, C_e, Nx):
    RHS = RHS_AL(M_old, C_old, C_a, C_b, C_c, C_d, C_e, Nx)
    C_new = np.linalg.solve(M_new, RHS)
    return(C_new)










## M_new matrices :

# the time dependend Matrix is defined thanks to the constant Matrix and 
# the boundaries conditions of the electrochemical problem

# for an E mechanism with a chemical that can be reduced
def Matrix_E_Non_Nernst_boundaries_red(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
    # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = 1
    # current condition on Cox
    M_new[0, 0] = + 1 + cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1] = - 1
    M_new[0,Nx] = - cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    # equality of the concentration flux at the electrode
    M_new[Nx, 0]  = + 1
    M_new[Nx, 1]  = - 1
    M_new[Nx, Nx] = + 1
    M_new[Nx, Nx+1] = -1 
    return(M_new)  

# for an E mechanism with a chemical that can be oxidized
def Matrix_E_Non_Nernst_boundaries_ox(M_new_constant, t, E, cst_all):    
    Nx = cst_all["Nx"]
    M_new = M_new_constant
    # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = 1 
    # current condition on Cred
    M_new[0, 0] = + 1 + cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1] = - 1
    M_new[0,Nx] = - cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])    
    # equality of the concentration flux at the electrode
    M_new[Nx, 0]  = + 1
    M_new[Nx, 1]  = - 1
    M_new[Nx, Nx] = + 1
    M_new[Nx, Nx+1] = -1
    return(M_new)

# for an CE mechanism with a chemical that can be oxidized
def Matrix_EC_boundaries_red(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = 1   
        # current condition on C_a
    M_new[0, 0]  = + 1 + cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1] = - 1
    M_new[0, Nx] = - cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])   
        # equality of the concentration flux at the electrode
    M_new[Nx, 0]      = + 1
    M_new[Nx, 1]      = - 1
    M_new[Nx, Nx]     = + 1
    M_new[Nx, Nx+1]   = - 1
         # zero flux condition on C_c
    M_new[2*Nx,2*Nx]    = + 1
    M_new[2*Nx,2*Nx+1]  = - 1 
    return(M_new)

# for an CE mechanism with a chemical that can be oxidized
def Matrix_EC_boundaries_ox(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = 1   
        # current condition on C_a
    M_new[0, 0]  = + 1 + cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1] = - 1
    M_new[0, Nx] = - cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])   
        # equality of the concentration flux at the electrode
    M_new[Nx, 0]      = + 1
    M_new[Nx, 1]      = - 1
    M_new[Nx, Nx]     = + 1
    M_new[Nx, Nx+1]   = - 1
         # zero flux condition on C_c
    M_new[2*Nx,2*Nx]    = + 1
    M_new[2*Nx,2*Nx+1]  = - 1 
    return(M_new)


# for an CE mechanism with a chemical that can be reduced
def Matrix_CE_boundaries_red(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = 1 
        # zero flux condition on C_a
    M_new[0,0]  = + 1
    M_new[0,1]  = - 1  
        # current condition on C_b
    M_new[Nx, Nx]  = + 1 + cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[Nx, Nx+1] = - 1
    M_new[Nx, 2*Nx] = - cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
        # equality of the concentration flux at the electrode
    M_new[2*Nx, Nx]     = + 1
    M_new[2*Nx, Nx+1]   = - 1
    M_new[2*Nx, 2*Nx]   = + 1
    M_new[2*Nx, 2*Nx+1] = - 1
    return(M_new)

# for an CE mechanism with a chemical that can be oxidized
def Matrix_CE_boundaries_ox(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = 1   
        # zero flux condition on C_a
    M_new[0,0]  = + 1
    M_new[0,1]  = - 1 
        # current condition on C_b
    M_new[Nx, Nx]  = + 1 + cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[Nx, Nx+1] = - 1
    M_new[Nx, 2*Nx] = - cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])   
        # equality of the concentration flux at the electrode
    M_new[2*Nx, Nx]     = + 1
    M_new[2*Nx, Nx+1]   = - 1
    M_new[2*Nx, 2*Nx]   = + 1
    M_new[2*Nx, 2*Nx+1] = - 1
    return(M_new)


# for an ECE mechanism with a chemical that can be reduced
def Matrix_ECE_boundaries_red(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = M_new[4*Nx-1, 4*Nx-1] = 1
       # current condition on C_c
    M_new[2*Nx, 2*Nx]   = + 1 + cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"])
    M_new[2*Nx, 2*Nx+1] = - 1
    M_new[2*Nx, 3*Nx]   = - cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"])
        # current condition on C_a
    M_new[0, 0]         = + 1 + cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1]         = - 1
    M_new[0, Nx]        = - cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
        # equality of the concentration flux at the electrode for A/B
    M_new[Nx, 0]        = + 1
    M_new[Nx, 1]        = - 1
    M_new[Nx, Nx]       = + 1
    M_new[Nx, Nx+1]     = - 1
        # equality of the concentration flux at the electrode for C/D
    M_new[3*Nx, 2*Nx]    = + 1
    M_new[3*Nx, 2*Nx+1]  = - 1
    M_new[3*Nx, 3*Nx]    = + 1
    M_new[3*Nx, 3*Nx+1]  = - 1 
    return(M_new)

# for an ECE mechanism with a chemical that can be oxidized
def Matrix_ECE_boundaries_ox(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = M_new[4*Nx-1, 4*Nx-1] = 1
       # current condition on C_c
    M_new[2*Nx, 2*Nx]   = + 1 + cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"])
    M_new[2*Nx, 2*Nx+1] = - 1
    M_new[2*Nx, 3*Nx]   = - cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"]) 
        # current condition on C_a
    M_new[0, 0]         = + 1 + cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1]         = - 1
    M_new[0, Nx]        = - cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
        # equality of the concentration flux at the electrode for A/B
    M_new[Nx, 0]        = + 1
    M_new[Nx, 1]        = - 1
    M_new[Nx, Nx]       = + 1
    M_new[Nx, Nx+1]     = - 1
        # equality of the concentration flux at the electrode for C/D
    M_new[3*Nx, 2*Nx]    = + 1
    M_new[3*Nx, 2*Nx+1]  = - 1
    M_new[3*Nx, 3*Nx]    = + 1
    M_new[3*Nx, 3*Nx+1]  = - 1
    return(M_new)


def Matrix_AL_boundaries_red(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = M_new[4*Nx-1, 4*Nx-1] = M_new[5*Nx-1, 5*Nx-1] = 1
       # current condition on ELA/ELA-
    M_new[2*Nx, 2*Nx]   = + 1 + cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"])
    M_new[2*Nx, 2*Nx+1] = - 1
    M_new[2*Nx, 3*Nx]   = - cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"]) 
        # current condition on E/E-
    M_new[0, 0]         = + 1 + cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1]         = - 1
    M_new[0, 4*Nx]        = - cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
        # equality of the concentration flux at the electrode for E/E-
    M_new[Nx, 0]        = + 1
    M_new[Nx, 1]        = - 1
    M_new[Nx, 4*Nx]       = + 1
    M_new[Nx, 4*Nx+1]     = - 1
        # equality of the concentration flux at the electrode for ELA/ELA-
    M_new[3*Nx, 2*Nx]    = + 1
    M_new[3*Nx, 2*Nx+1]  = - 1
    M_new[3*Nx, 3*Nx]    = + 1
    M_new[3*Nx, 3*Nx+1]  = - 1
        # zero flux of LA at electrode
    M_new[4*Nx, Nx]    = + 1
    M_new[4*Nx, Nx+1]  = - 1
    return(M_new)

def Matrix_AL_boundaries_ox(M_new_constant, t, E, cst_all): 
    Nx = cst_all["Nx"]
    M_new = M_new_constant
        # boundaries conditions at bulk :
    M_new[Nx-1,Nx-1] = M_new[2*Nx-1, 2*Nx-1] = M_new[3*Nx-1, 3*Nx-1] = M_new[4*Nx-1, 4*Nx-1] = M_new[5*Nx-1, 5*Nx-1] = 1
       # current condition on ELA/ELA-
    M_new[2*Nx, 2*Nx]   = + 1 + cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"])
    M_new[2*Nx, 2*Nx+1] = - 1
    M_new[2*Nx, 3*Nx]   = - cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_2"], cst_all["alpha"]) 
        # current condition on E/E-
    M_new[0, 0]         = + 1 + cst_all["Lambda"]*k_ox(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
    M_new[0, 1]         = - 1
    M_new[0, 4*Nx]        = - cst_all["Lambda"]*k_red(t, E, cst_all["n"], cst_all["F_norm"], cst_all["E_0_1"], cst_all["alpha"])
        # equality of the concentration flux at the electrode for E/E-
    M_new[Nx, 0]        = + 1
    M_new[Nx, 1]        = - 1
    M_new[Nx, 4*Nx]       = + 1
    M_new[Nx, 4*Nx+1]     = - 1
        # equality of the concentration flux at the electrode for ELA/ELA-
    M_new[3*Nx, 2*Nx]    = + 1
    M_new[3*Nx, 2*Nx+1]  = - 1
    M_new[3*Nx, 3*Nx]    = + 1
    M_new[3*Nx, 3*Nx+1]  = - 1
        # zero flux of LA at electrode
    M_new[4*Nx, Nx]    = + 1
    M_new[4*Nx, Nx+1]  = - 1
    return(M_new)








## Matrices used for the construction of M_new and M_old :

# Constants Matrices (or M_old and M_new_constant_part)
def Matrix_constant_ECE(Nx, Dt, num_species, k_p, k_m, DM):
    M_new_constant = Id_tronc(Nx, num_species) - 0.5*DM*Lapl_Mat(Nx, num_species) - 0.5*Dt*Eq_Mat_ECE(Nx, num_species, k_p, k_m)
    M_old_constant = Id_tronc(Nx, num_species) + 0.5*DM*Lapl_Mat(Nx, num_species) + 0.5*Dt*Eq_Mat_ECE(Nx, num_species, k_p, k_m) 
    return(M_new_constant, M_old_constant)

def Matrix_constant_CE(Nx, Dt, num_species, k_p, k_m, DM):
    M_new_constant = Id_tronc(Nx, num_species) - 0.5*DM*Lapl_Mat(Nx, num_species) - 0.5*Dt*Eq_Mat_CE(Nx, num_species, k_p, k_m)
    M_old_constant = Id_tronc(Nx, num_species) + 0.5*DM*Lapl_Mat(Nx, num_species) + 0.5*Dt*Eq_Mat_CE(Nx, num_species, k_p, k_m) 
    return(M_new_constant, M_old_constant)

def Matrix_constant_EC(Nx, Dt, num_species, k_c, DM):
    M_new_constant = Id_tronc(Nx, num_species) - 0.5*DM*Lapl_Mat(Nx, num_species) - 0.5*Dt*React_Mat_EC(Nx, num_species, k_c)
    M_old_constant = Id_tronc(Nx, num_species) + 0.5*DM*Lapl_Mat(Nx, num_species) + 0.5*Dt*React_Mat_EC(Nx, num_species, k_c) 
    return(M_new_constant, M_old_constant)

def Matrix_constant_E(Nx, num_species, DM):
    M_new_constant = Id_tronc(Nx, num_species) - 0.5*DM*Lapl_Mat(Nx, num_species) 
    M_old_constant = Id_tronc(Nx, num_species) + 0.5*DM*Lapl_Mat(Nx, num_species) 
    return(M_new_constant, M_old_constant)

def Matrix_constant_AL(Nx, Dt, num_species, k_p, k_m, k_p_2, k_m_2, kc, kc2, DM, C):
    M_new_constant = Id_tronc(Nx, num_species) - 0.5*DM*Lapl_Mat(Nx, num_species) - 0.5*Dt*Eq_Mat_kin_cst_LA(Nx, num_species, k_p, k_m, k_p_2, k_m_2, kc, kc2) - 0.5*Dt*Eq_Mat_kin_td_LA(Nx, num_species, k_p, k_m, k_p_2, k_m_2, kc, kc2, C)
    M_old_constant = Id_tronc(Nx, num_species) + 0.5*DM*Lapl_Mat(Nx, num_species) + 0.5*Dt*Eq_Mat_kin_cst_LA(Nx, num_species, k_p, k_m, k_p_2, k_m_2, kc, kc2)
    return(M_new_constant, M_old_constant)









# Laplacian Matrices :
def Lapl_Mat(Nx, num_species):                              ## def Laplacien troncated meca ECE
    L = np.zeros((num_species*Nx,num_species*Nx))
    for j in range(num_species):
        for i in range(Nx-2):
            L[j*Nx+i+1,j*Nx+i+1] = -2
            L[j*Nx+i+1,j*Nx+i]   = +1
            L[j*Nx+i+1,j*Nx+i+2] = +1
    return(L)



# Chemical Equilibrium matrices :
def Eq_Mat_kin_cst_LA(Nx, num_species, k_p, k_m, k_p_2, k_m_2, kc, kc2):
    Eq_Mat = np.zeros((num_species*Nx,num_species*Nx))  
    for i in range(Nx-2):
        Eq_Mat[i+1,2*Nx+i+1]         = + k_p 
        Eq_Mat[Nx+i+1,2*Nx+i+1]      = + k_p
        Eq_Mat[Nx+i+1,3*Nx+i+1]      = + k_m_2 + kc2
        Eq_Mat[2*Nx+i+1,2*Nx+i+1]    = - k_p
        Eq_Mat[3*Nx+i+1,3*Nx+i+1]    = - k_m_2 - kc2
        Eq_Mat[4*Nx+i+1,3*Nx+i+1]    = + k_m_2 
        Eq_Mat[4*Nx+i+1,4*Nx+i+1]    = - kc
    return(Eq_Mat)

def Eq_Mat_kin_td_LA(Nx, num_species, k_p, k_m, k_p_2, k_m_2, kc, kc2, C):
    Eq_Mat = np.zeros((num_species*Nx,num_species*Nx))  
    for i in range(Nx-2):
        Eq_Mat[i+1,i+1]            = - k_p*C[Nx+i+1] 
        Eq_Mat[i+1,Nx+i+1]         = - k_p*C[i+1]
        Eq_Mat[Nx+i+1,i+1]         = - k_p*C[Nx+i+1] 
        Eq_Mat[Nx+i+1,Nx+i+1]      = - k_p*C[i+1] - k_p_2*C[4*Nx+i+1]
        Eq_Mat[Nx+i+1,4*Nx+i+1]    = - k_p_2*C[Nx+i+1]
        Eq_Mat[2*Nx+i+1,i+1]       = + k_p*C[Nx+i+1] 
        Eq_Mat[2*Nx+i+1,Nx+i+1]    = + k_p*C[i+1]
        Eq_Mat[3*Nx+i+1,Nx+i+1]    = + k_p_2*C[4*Nx+i+1]
        Eq_Mat[3*Nx+i+1,4*Nx+i+1]  = + k_p_2*C[Nx+i+1]
        Eq_Mat[4*Nx+i+1,Nx+i+1]    = - k_p_2*C[4*Nx+i+1]
        Eq_Mat[4*Nx+i+1,4*Nx+i+1]  = - k_p_2*C[Nx+i+1]
    return(Eq_Mat)

def Eq_Mat_ECE(Nx, num_species, k_p, k_m):
    Eq_Mat = np.zeros((num_species*Nx,num_species*Nx))  
    for i in range(Nx-2):
        Eq_Mat[Nx+i+1,Nx+i+1]      = -k_p 
        Eq_Mat[Nx+i+1,2*Nx+i+1]    = +k_m
        Eq_Mat[2*Nx+i+1,Nx+i+1]    = +k_p
        Eq_Mat[2*Nx+i+1,2*Nx+i+1]  = -k_m       
    return(Eq_Mat)

def Eq_Mat_CE(Nx, num_species, k_p, k_m):
    Eq_Mat = np.zeros((num_species*Nx,num_species*Nx))  
    for i in range(Nx-2):
        Eq_Mat[i+1,i+1]       = - k_p 
        Eq_Mat[i+1,Nx+i+1]    = + k_m
        Eq_Mat[Nx+i+1,i+1]    = + k_p
        Eq_Mat[Nx+i+1,Nx+i+1] = - k_m       
    return(Eq_Mat)

def React_Mat_EC(Nx, num_species, k_c):
    Eq_Mat = np.zeros((num_species*Nx,num_species*Nx))  
    for i in range(Nx-2):
        Eq_Mat[Nx+i+1,Nx+i+1]       = - k_c 
        Eq_Mat[2*Nx+i+1,Nx+i+1]     = + k_c    
    return(Eq_Mat)

# Troncated Identity (identity all over except on the boundaries)
def Id_tronc(Nx, num_species):
    Identity_Mat = np.zeros((num_species*Nx,num_species*Nx))
    for j in range(num_species):
        for i in range(Nx-2):
            Identity_Mat[j*Nx+i+1,j*Nx+i+1] = 1
    return(Identity_Mat)








# Kinetics used for the construction of the Matrices :
# the dime dependent Matrix requires the definition of Butler-Volmer kinetic constants :
def sigma(t, E, cst_syst):                                        ## definition de sigma
    s = np.exp(n*F*(E(t) - E_0)/(R*T))       
    return(s)  

def k_red(t, E, n, F_norm, E_0_1, alpha):                                        ## definition de k_red
    s = np.exp(-n*alpha*F_norm*(E(t) - E_0_1))      
    return(s)  

def k_ox(t, E, n, F_norm, E_0_1, alpha):                                         ## definition de k_ox
    s = np.exp((1-alpha)*n*F_norm*(E(t) - E_0_1))       
    return(s)  







## RHS Definition:
# calculating the right hand side term for the AL mecanism
def RHS_AL(M_old, C_old, C_a, C_b, C_c, C_d, C_e, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = C_a
    RHS[Nx]     = 0
    RHS[2*Nx-1] = C_b 
    RHS[2*Nx]   = 0
    RHS[3*Nx-1] = C_c
    RHS[3*Nx]   = 0
    RHS[4*Nx-1] = C_d
    RHS[4*Nx]   = 0
    RHS[5*Nx-1] = C_e
    return(RHS)

# calculating the right hand side term for the ECE mecanism
def RHS_ECE(M_old, C_old, C_a, C_b, C_c, C_d, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = C_a
    RHS[Nx]     = 0
    RHS[2*Nx-1] = C_b 
    RHS[2*Nx]   = 0
    RHS[3*Nx-1] = C_c
    RHS[3*Nx]   = 0
    RHS[4*Nx-1] = C_d
    return(RHS)

# calculating the right hand side term for the CE mecanism
def RHS_CE(M_old, C_old, C_a, C_b, C_c, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = C_a
    RHS[Nx]     = 0
    RHS[2*Nx-1] = C_b 
    RHS[2*Nx]     = 0
    RHS[3*Nx-1] = C_c
    return(RHS)

# calculating the right hand side term for the CE mecanism
def RHS_EC(M_old, C_old, C_a, C_b, C_c, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = C_a
    RHS[Nx]     = 0
    RHS[2*Nx-1] = C_b 
    RHS[2*Nx]     = 0
    RHS[3*Nx-1] = C_c
    return(RHS)

# calculating the right hand side term for the E mecanism
def RHS_E(M_old, C_old, C_a, C_b, Nx):
    RHS = np.dot(M_old, C_old)
    RHS[0]      = 0
    RHS[Nx-1]   = C_a
    RHS[Nx]     = 0
    RHS[2*Nx-1] = C_b 
    return(RHS)












## Calculate the current :

    # reference = Heinze 1993
    # I = S*sum(zi*fi) 
    # where zi is the algebraic charge of the current, the convention of positiv anodic current is taken
    # fi stands for the flux of the specie i at the electrode 
    # first approximation of the flux is taken here as the first partial derivative along space
    # 
    # Note :Intensity = n*S*(k_ox(t, E)*C[Nx] - k_red(t,E)*C[0]) cette expression n'est pas stable du au fait 
    # que les concentration à l'electrode sont trop faibles. on obtient un signal bruité.
    
# computing the current for AL mechanism
# dans le cas où l'espèce étudiée est réduite
def compute_I_AL_red(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*((C[1]-C[0]) - (C[4*Nx+1]-C[4*Nx]) + (C[2*Nx+1]-C[2*Nx]) - (C[3*Nx+1]-C[3*Nx]))/Dx
    return(Intensity)

# dans le cas où l'espèce étudiée est oxydée
def compute_I_AL_ox(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = n*F*D*S*((C[1]-C[0]) - (C[4*Nx+1]-C[4*Nx]) + (C[2*Nx+1]-C[2*Nx]) - (C[3*Nx+1]-C[3*Nx]))/Dx
    return(Intensity)

# computing the current for ECE mechanism
# dans le cas où l'espèce étudiée est réduite
def compute_I_ECE_red(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*((C[1]-C[0]) - (C[Nx+1]-C[Nx]) + (C[2*Nx+1]-C[2*Nx]) - (C[3*Nx+1]-C[3*Nx]))/Dx
    return(Intensity)

# dans le cas où l'espèce étudiée est oxydée
def compute_I_ECE_ox(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = n*F*D*S*((C[1]-C[0]) - (C[Nx+1]-C[Nx]) + (C[2*Nx+1]-C[2*Nx]) - (C[3*Nx+1]-C[3*Nx]))/Dx
    return(Intensity)

# computing the current for CE mechanism
# dans le cas où l'espèce étudiée est réduite
def compute_I_CE_red(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*((C[Nx+1]-C[Nx+0]) - (C[2*Nx+1]-C[2*Nx]))/Dx
    return(Intensity)

# dans le cas où l'espèce étudiée est oxydée
def compute_I_CE_ox(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*(1*(C[Nx+1]-C[Nx+0]) - (C[2*Nx+1]-C[2*Nx]))/Dx
    return(Intensity)

# computing the current for CE mechanism
# dans le cas où l'espèce étudiée est réduite
def compute_I_EC_red(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*((C[1]-C[0]) - (C[Nx+1]-C[Nx]))/Dx
    return(Intensity)

# dans le cas où l'espèce étudiée est oxydée
def compute_I_EC_ox(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*((C[1]-C[0]) - (C[Nx+1]-C[Nx]))/Dx
    return(Intensity)

# computing the current for E mechanism
# dans le cas où l'espèce étudiée est réduite
def compute_I_E_red(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = -n*F*D*S*(1*(C[1]-C[0])-1*(C[Nx+1] - C[Nx]))/Dx
    return(Intensity)

# dans le cas où l'espèce étudiée est oxydée
def compute_I_E_ox(C, cst_all):
    n, F, D, S, Nx, Dx = cst_all["n"], cst_all["F"], cst_all["D"], cst_all["S"], cst_all["Nx"], cst_all["Dx"]
    Intensity = +n*F*D*S*(1*(C[1]-C[0]) - (C[Nx+1]-C[Nx]))/Dx
    return(Intensity)

