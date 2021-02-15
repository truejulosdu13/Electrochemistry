import math
import numpy as np

# this functions take as input the experimental parameters that you fix depending on the kind of experiment you use ; 
# for example for a linear sweep potential you have to fix : start potential, end potential and sweep rate.

def rampe(E_i, E_f, v):                   # rampe de potentiel
    ## DERIVED CONSTANTS = function du signal imposé
    tk  = 2*abs(E_i - E_f)/v            
    
    def E(t):
        t_12 = abs(E_f - E_i)/v
        signe = abs(E_f - E_i)/(E_f - E_i)
        if t < t_12 :
            E_t = E_i + signe*v*t
        else :
            E_t = E_f - signe*v*(t - t_12)
        return(E_t)
    
    return(E, tk)    

def creneau(E_1, E_2, tau):              # creneau (chrono_double_saut)
    ## DERIVED CONSTANTS = function du signal imposé
    tk  = 10*tau            
    
    def E(t):
        if (math.floor(t/tau) % 2) == 0:
            E_t = E_1
        else:
            E_t = E_2
        return(E_t)
    return(E, tk)

def CSV(E_i, E_ox, E_red, DeltaE, v):       # fonction marche (staircase_volta)
    if E_i > E_ox or E_i < E_red :
        print("E_i > E_ox or E_i < E_red : change your experimental parameters !")
    ## DERIVED CONSTANTS = function du signal imposé
    tk  = 2*abs(E_red - E_ox)/v            
    
    signe = abs(DeltaE)/DeltaE
    def E(t):
        if signe > 0:      
            t_inv1 = (E_ox - E_i)/v
            t_inv2 = (E_ox - E_i + E_ox - E_red)/v
            if t < t_inv1 :
                E_t = E_i + math.floor(v*t/DeltaE)*DeltaE
            elif t < t_inv2 :
                E_t = E_ox - math.floor(v*(t-t_inv1)/DeltaE)*DeltaE
            else:
                E_t = E_red + math.floor(v*(t-t_inv2)/DeltaE)*DeltaE
        
        if signe < 0:      
            t_inv1 = (E_i - E_red)/v
            t_inv2 = (E_i - E_red + E_ox - E_red)/v
            if t < t_inv1 :
                E_t = E_i + math.floor(v*t/abs(DeltaE))*DeltaE
            elif t < t_inv2 :
                E_t = E_red - math.floor(v*(t-t_inv1)/abs(DeltaE))*DeltaE
            else:
                E_t = E_ox + math.floor(v*(t-t_inv2)/abs(DeltaE))*DeltaE
        return(E_t)
    return(E, tk)

def SWV(E_i, E_f, ESW, Delta_E, f):       # fonction marche (staircase_volta)
    ## DERIVED CONSTANTS = function du signal imposé
    tk  = 2*np.pi*abs(E_i - E_f)/(Delta_E*f)            
    
    signe = abs(E_f - E_i)/(E_f - E_i)
    def E(t):
        tau = 2*np.pi/f
        E_t = E_i + signe*math.floor(t/tau)*Delta_E
        if (math.floor(2*t/tau) % 2) == 0:
            E_t += signe*ESW  
        return(E_t)
    def E_sweep(t):
        E_sweep_t = E_i + Delta_E*f*t/(2*np.pi) 
        return(E_sweep_t)
    return(E, E_sweep, tk)


