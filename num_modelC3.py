# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:34:45 2020

@author: Cesar Rodriguez

Modifed by Rigel:
- Made bounds go from L0, L0 + lambda/2 so that piezo voltage values don't go negative.

Updated the SOP, and reverted Rigel's change as we can miss a mode below the initial length.

Importing the code for the numerical matrix simulation
"""
import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from lmfit import Model
import lmfit
from lmfit.model import save_modelresult
from scipy.optimize import minimize_scalar, newton, minimize
from scipy.interpolate import interp1d

#Import the design stacks of C3 mirrors
λ_design = 0.615 #mu
# mirror_df = pd.read_csv(r'../../../../Manuals, Datasheets and SOPs/Laseroptik/Mirror Coatings/C3 stack length/flatMC3.csv', index_col=0)
# fiber_df = pd.read_csv(r'../../../../Manuals, Datasheets and SOPs/Laseroptik/Mirror Coatings/C3 stack length/fiberC.csv', index_col=0)
mirror_df = pd.read_csv(r'flatMC3.csv',index_col=0)
fiber_df = pd.read_csv(r'fiberC3.csv',index_col=0)


#Define index of refraction 
n_subs = 1.4633
n_Ta2O5 = 2.125#2.1306
n_SiO2 = 1.482#1.4588
n_dia = 2.417#2.4183
n_air = 1

def layer_lengths(material,lengths,λd):
    r = np.zeros(lengths.size)
    condition1 = np.where(material=='Ta2O5')
    condition2 = np.where(material=='SiO2')
    r[condition1] = lengths.values[condition1]*λd/(4*n_Ta2O5)
    r[condition2] = lengths.values[condition2]*λd/(4*n_SiO2)
    return r

# x1000 to convert to nm
fiber_stack_length_nm = layer_lengths(fiber_df['Material'],fiber_df['Length [lambda/4n]'],λ_design)*1000
mirror_stack_LI_length_nm = layer_lengths(mirror_df['Material'],mirror_df['Length [lambda/4n]'],λ_design)*1000
mirror_stack_HI_length_nm = layer_lengths(fiber_df['Material'],fiber_df['Length [lambda/4n]'],λ_design)*1000

def arr_to_matrix(v1,v2):
    M = np.zeros((v1.size,v2.size))
    return M

def vec_x_vec_matrix(v1,v2):
    if type(v1) != np.ndarray:
        v1 = np.array([v1])
    if type(v2) != np.ndarray:
        v2 = np.array([v2])
    r = arr_to_matrix(v1,v2)
    i=0
    for v in v1:
        r[i,:] = v*v2 
        i=i+1
    return r

def prop_matrix(a):
    a = np.exp(-1j*a)
    x = np.reshape(a,a.size)
    M = np.zeros((a.size,2,2),dtype=complex)
    n_rows = a[0,:].size
    n_columns = a[:,0].size
    for (i,x0) in enumerate(x):
        M[i] = np.array([[x0,0],[0,x0.conjugate()]])
    M = np.reshape(M,(n_columns,n_rows,2,2))
    return M

def index_change(n1,n2):
    #From a material with index n1 to a material with index n2.
    n1 = n1.real
    n2 = n2.real
    a = (n1+n2)/(2*n2)
    b = (-n1+n2)/(2*n2)
    c = (-n1+n2)/(2*n2)
    d = (n1+n2)/(2*n2)
    M = np.array([[a, b],[c, d]])
    return M

def planewave_propagation(n,L,λ):
    k = 2*np.pi*n/λ
    a = vec_x_vec_matrix(k,L)
    M = prop_matrix(a)
    return M

#Create matrices
Ta2O5_to_SiO2 = index_change(n_Ta2O5,n_SiO2)
SiO2_to_Ta2O5 = index_change(n_SiO2,n_Ta2O5)
SiO2_to_air = index_change(n_SiO2,n_air)
air_to_SiO2 = index_change(n_air,n_SiO2)
air_to_Ta2O5 = index_change(n_air,n_Ta2O5)
Ta2O5_to_air = index_change(n_Ta2O5,n_air)
subs_to_Ta2O5 = index_change(n_subs,n_Ta2O5)
Ta2O5_to_subs = index_change(n_Ta2O5,n_subs)

def flat_mirror_LI_matrix(material,stack_length_nm,λ):
    mat_Ta2O5 = np.where(material=='Ta2O5')
    mat_SiO2 = np.where(material=='SiO2')
    number_of_matrix = stack_length_nm.size*2+1
    #Create the matrix
    if type(λ) == float:
        λ = np.asarray([λ])
    n_λ = λ.size
    transfer_matrix_mirror_stack = np.zeros((n_λ,number_of_matrix,2,2),dtype=complex) #datatype 2x2 arrays
    M_s_to_air = np.zeros((n_λ,2,2),dtype=complex)
    M_air_to_s = np.zeros((n_λ,2,2),dtype=complex)
    #Move the index for the matrix
    propagation_Ta205 = 2*mat_Ta2O5[0]+1
    propagation_SiO2 = 2*mat_SiO2[0]+1
    #Populate the matrix
    #Propagation
    transfer_matrix_mirror_stack[:,propagation_Ta205] = planewave_propagation(n=n_Ta2O5,L=stack_length_nm[mat_Ta2O5],λ=λ)
    transfer_matrix_mirror_stack[:,propagation_SiO2]= planewave_propagation(n=n_SiO2,L=stack_length_nm[mat_SiO2],λ=λ)
    #Index change
    transfer_matrix_mirror_stack[:,propagation_Ta205+1] = Ta2O5_to_SiO2
    transfer_matrix_mirror_stack[:,propagation_SiO2+1] = SiO2_to_Ta2O5
    #First and Final matrix when it goes from the substrate and into air
    transfer_matrix_mirror_stack[:,0] = subs_to_Ta2O5
    transfer_matrix_mirror_stack[:,-1] = SiO2_to_air
    for j in range(n_λ):
        λ0 = λ[j]
        #Multiply them
        A = np.array([[1,0],[0,1]],dtype=complex)
        for T in transfer_matrix_mirror_stack[j,:]:
            A = np.matmul(T,A)
        M_s_to_air[j] = A
    
    #We now calculate the inverse matrix when the light goes from the air to the susbtrate
    #Reverse substrates
    #Index change
    transfer_matrix_mirror_stack[:,propagation_Ta205+1] = SiO2_to_Ta2O5
    transfer_matrix_mirror_stack[:,propagation_SiO2+1] = Ta2O5_to_SiO2
    transfer_matrix_mirror_stack[:,0] = Ta2O5_to_subs
    transfer_matrix_mirror_stack[:,-1] = air_to_SiO2
    air_to_subs_matrix_mirror_stack = np.flip(transfer_matrix_mirror_stack,axis=1)
    for j in range(n_λ):
        λ0 = λ[j]
        #Multiply them
        A = np.array([[1,0],[0,1]],dtype=complex)
        for T in air_to_subs_matrix_mirror_stack[j,:]:
            A = np.matmul(T,A)
        M_air_to_s[j] = A

    return M_s_to_air, M_air_to_s

def fiber_mirror_matrix(material,stack_length_nm,λ):
    mat_Ta2O5 = np.where(material=='Ta2O5')
    mat_SiO2 = np.where(material=='SiO2')
    number_of_matrix = stack_length_nm.size*2+1
    #Create the matrix
    if type(λ) == float:
        λ = np.asarray([λ])
    n_λ = λ.size
    transfer_matrix_mirror_stack = np.zeros((n_λ,number_of_matrix,2,2),dtype=complex) #datatype 2x2 arrays
    M_s_to_air = np.zeros((n_λ,2,2),dtype=complex)
    M_air_to_s = np.zeros((n_λ,2,2),dtype=complex)
    #Move the index for the matrix
    propagation_Ta205 = 2*mat_Ta2O5[0]+1
    propagation_SiO2 = 2*mat_SiO2[0]+1
    #Populate the matrix
    #Propagation
    transfer_matrix_mirror_stack[:,propagation_Ta205] = planewave_propagation(n=n_Ta2O5,L=stack_length_nm[mat_Ta2O5],λ=λ)
    transfer_matrix_mirror_stack[:,propagation_SiO2]= planewave_propagation(n=n_SiO2,L=stack_length_nm[mat_SiO2],λ=λ)
    #Index change
    transfer_matrix_mirror_stack[:,propagation_Ta205+1] = Ta2O5_to_SiO2
    transfer_matrix_mirror_stack[:,propagation_SiO2+1] = SiO2_to_Ta2O5
    #First and Final matrix when it goes from the substrate and into air
    transfer_matrix_mirror_stack[:,0] = subs_to_Ta2O5
    transfer_matrix_mirror_stack[:,-1] = Ta2O5_to_air
    for j in range(n_λ):
        λ0 = λ[j]
        #Multiply them
        A = np.array([[1,0],[0,1]],dtype=complex)
        for T in transfer_matrix_mirror_stack[j,:]:
            A = np.matmul(T,A)
        M_s_to_air[j] = A
    
    #We now calculate the inverse matrix when the light goes from the air to the susbtrate
    #Reverse substrates
    #Index change
    transfer_matrix_mirror_stack[:,propagation_Ta205+1] = SiO2_to_Ta2O5
    transfer_matrix_mirror_stack[:,propagation_SiO2+1] = Ta2O5_to_SiO2
    transfer_matrix_mirror_stack[:,0] = Ta2O5_to_subs
    transfer_matrix_mirror_stack[:,-1] = air_to_Ta2O5
    air_to_subs_matrix_mirror_stack = np.flip(transfer_matrix_mirror_stack,axis=1)
    for j in range(n_λ):
        λ0 = λ[j]
        #Multiply them
        A = np.array([[1,0],[0,1]],dtype=complex)
        for T in air_to_subs_matrix_mirror_stack[j,:]:
            A = np.matmul(T,A)
        M_air_to_s[j] = A

    return M_s_to_air, M_air_to_s

def transmission_reflection_from_matrix(n1,n2,mirror_matrix):
    mm = mirror_matrix
    if mirror_matrix.shape == (2,2):
        term1 = mm[1,0]/mm[1,1]
        term2 = mm[0,0]-mm[0,1]*term1
    else:
        term1 = mm[:,1,0]/mm[:,1,1]
        term2 = mm[:,0,0]-mm[:,0,1]*term1
    T = n2/n1*np.abs(term2)**2
    R = np.abs(term1)**2
    return T,R

#Interface matrix going from diamond to air and viceversa
def Interface_ad(λ,td,L,R0,k):
    z1 = td
    ta = L -td
    w0d = np.sqrt(λ/np.pi)*((ta+td/n_dia)*(R0-(ta+td/n_dia)))**(1/4)
    # Convert z1 to nm to match lambda
    a = np.exp(-1j*(2*n_dia*np.pi*(z1*1000)/λ - (1+k) * np.arctan(z1*λ/(n_dia*np.pi*w0d**2))))
    M = np.zeros((a.size,2,2),dtype=complex)
    for (i,a0) in enumerate(a):
        M[i] = np.array([[a0,0],[0,a0.conjugate()]])
    return M

def Interface_da(λ,td,L,R0,k):
    z1 = td
    z2 = td*(1-1/n_dia)
    ta = L - td
    w0a = np.sqrt(λ/np.pi)*((ta+td/n_dia)*(R0-(ta+td/n_dia)))**(1/4)
    # conver to nm to match lambda
    a = np.exp(-1j*(2*n_air*np.pi*((L-z1)*1000)/λ - (1+k) * (np.arctan((L-z2)*λ/(n_air*np.pi*w0a**2)) - np.arctan((-z2+z1)*λ/(n_air*np.pi*w0a**2)))  ))
    M = np.zeros((a.size,2,2),dtype=complex)
    for (i,a0) in enumerate(a):
        M[i] = np.array([[a0,0],[0,a0.conjugate()]])
    return M


def r12(n1,n2,λ,σ):
    n1 = n1.real
    n2 = n2.real
    r0 = (n2-n1)/(n1+n2)
    return r0*np.exp(-2*(2*np.pi*σ*n2/λ)**2)

def t12(n1,n2,λ,σ):
    n1 = n1.real
    n2 = n2.real
    t0 = 2*n1/(n1+n2)
    return t0*np.exp(-1/2*(2*np.pi*σ*(n1-n2)/λ)**2)

def displacement_air_diam(λ,σ):
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    r120 = r12(n_dia,n_air,λ,σ)
    r210 = r12(n_air,n_dia,λ,σ)

    t120 = t12(n_air,n_dia,λ,σ)
    t210 = t12(n_dia,n_air,λ,σ)
    
    a = t120 - r120*r210/t210
    b = r210/t210
    c = -r120/t210
    d = 1/t210
    
    M = np.zeros((n_λ,2,2),dtype=complex)
    if a.size >= 1:
        for (i,a0) in enumerate(a):
            M[i] = np.array([[a[i],b[i]],[c[i],d[i]]])
    else:
        M = np.array([[a,b],[c,d]])
    return M

def displacement_diam_air(λ,σ):
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    r43 = r12(n_dia,n_air,λ,σ)
    r34 = r12(n_air,n_dia,λ,σ)
    t43 = t12(n_air,n_dia,λ,σ)
    t34 = t12(n_dia,n_air,λ,σ)
    
    a = t34 - r34*r43/t43
    b = r43/t43
    c = -r34/t43
    d = 1/t43
    
    M = np.zeros((n_λ,2,2),dtype=complex)
    if a.size >= 1:
        for (i,a0) in enumerate(a):
            M[i] = np.array([[a[i],b[i]],[c[i],d[i]]])
    else:
        M = np.array([[a,b],[c,d]])
    return M

def cavity_transfer_matrix(mirror,fiber_inv,Dad,Dda,λ,td,L,R0,k):
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    if type(L) != np.ndarray:
        L = np.asarray([L])
    n_L = L.size
    transfer_matrix = np.zeros((n_L,n_λ,2,2),dtype=complex) #datatype 2x2 arrays
    R = np.zeros((n_L,n_λ))
    T = np.zeros((n_L,n_λ))
    for i in range(n_L):
        T1 = Interface_ad(λ,td,L[i],R0,k)
        T2 = Interface_da(λ,td,L[i],R0,k)
        A = np.matmul(Dad,mirror)
        B = np.matmul(T1,A)
        C = np.matmul(Dda,B)
        D = np.matmul(T2,C)
        E = np.matmul(fiber_inv,D)
        transfer_matrix[i] = E
        T[i],R[i] = transmission_reflection_from_matrix(1,1,transfer_matrix[i])
    return T,R, transfer_matrix

def resonant_length_LI(λ_test,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
                       L0,λ,td,R0,k):
    index = np.where(λ_test == λ)[0][0]
    k = np.round(k)
    bounds = [L0-λ/4,L0+λ/4]
    f = lambda L: cavity_transfer_matrix(mirror_LI_matrix_result[index],fiber_inv_matrix_result[index],
                                         Dad[index],Dda[index],λ,td,L,R0,k)[1]
    value = minimize_scalar(f, bounds=bounds, method='bounded', options={'xatol':1e-8})
    return value

def resonant_length_HI(λ_test,mirror_HI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
                       L0,λ,td,R0,k):
    index = np.where(λ_test == λ)[0][0]
    k = np.round(k)
    bounds = [L0-λ/4,L0+λ/4]
    f = lambda L: cavity_transfer_matrix(mirror_HI_matrix_result[index],fiber_inv_matrix_result[index],
                                         Dad[index],Dda[index],λ,td,L,R0,k)[1]
    value = minimize_scalar(f, bounds=bounds, method='bounded', options={'xatol':1e-8})
    return value

#Fitting only one mode
def Linmode_HI(λ,mirror_HI_matrix_result,fiber_inv_matrix_result,Dad,Dda,td,R0,L0,k):
    k = np.round(k)
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    m = np.zeros(n_λ)
    L = L0
    for i in range(n_λ):
        m[i] = resonant_length_HI(λ,mirror_HI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
         L,λ[i],td,R0,k).x
        L = m[i]
    return(m)
    
def Linmode_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,td,R0,L0,k):
    k = np.round(k)
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    m = np.zeros(n_λ)
    L = L0
    for i in range(n_λ):
        m[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
         L,λ[i],td,R0,k).x
        L = m[i]
    return(m)


#Fitting more than one mode simultaenously
#Follows the following nomenclature: Linmodes_xm_yk where 'x' is the number of modes and 'y' is the number of higher order modes 
#per mode.
def Linmodes_2m_1k(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,td,R0,L0):
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    m0k0 = np.zeros(n_λ)
    m0k1 = np.zeros(n_λ)
    m1k0 = np.zeros(n_λ)
    m1k1 = np.zeros(n_λ)
    L = L0
    for i in range(n_λ):
        m0k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L,λ[i],td,R0,0).x
        L = m0k0[i]
        m0k1[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L,λ[i],td,R0,1).x
        m1k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L+λ[i]/2,λ[i],td,R0,0).x
        L1 = m1k0[i]
        m1k1[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L1,λ[i],td,R0,1).x
    
    return(np.array([m0k0,m0k1,m1k0,m1k1]))
    
def Linmodes_2m_0k(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,td,R0,L0):
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    m0k0 = np.zeros(n_λ)
    m1k0 = np.zeros(n_λ)
    #mt = np.zeros(n_λ)
    L = L0
    for i in range(n_λ):
        m0k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L,λ[i],td,R0,0).x
        L = m0k0[i]
        m1k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L+λ[i]/2,λ[i],td,R0,0).x
        L1 = m1k0[i]
    return(np.array([m0k0,m1k0]))

def Linmodes_4m_0k(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,td,R0,L0):
    if type(λ) != np.ndarray:
        λ = np.asarray([λ])
    n_λ = λ.size
    m0k0 = np.zeros(n_λ)
    m1k0 = np.zeros(n_λ)
    m2k0 = np.zeros(n_λ)
    m3k0 = np.zeros(n_λ)
    #mt = np.zeros(n_λ)
    L = L0
    for i in range(n_λ):
        m0k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L,λ[i],td,R0,0).x
        L = m0k0[i]
        m1k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L+λ[i]/2,λ[i],td,R0,0).x
        L1 = m1k0[i]
        m2k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L1+λ[i]/2,λ[i],td,R0,0).x
        L2 = m2k0[i]
        m3k0[i] = resonant_length_LI(λ,mirror_LI_matrix_result,fiber_inv_matrix_result,Dad,Dda,
            L2+λ[i]/2,λ[i],td,R0,0).x
        L3 = m3k0[i]
    return(np.array([m0k0,m1k0,m2k0,m3k0]))

if __name__ == "__main__":
    lambda_test = 602.0
    td_test = 0.8
    L_test = 10.0
    R0_test = 19.8
    sigma_test = 0E-9

    M_s_to_air, M_air_to_s = flat_mirror_LI_matrix(mirror_df['Material'],
                                                   mirror_stack_LI_length_nm,
                                                   lambda_test)
    F_s_to_air, F_air_to_s = fiber_mirror_matrix(fiber_df['Material'],
                                                 fiber_stack_length_nm,
                                                 lambda_test)
    Dad = displacement_air_diam(lambda_test,sigma_test)
    Dda = displacement_diam_air(lambda_test,sigma_test)
    T,R,cav = cavity_transfer_matrix(M_s_to_air,F_air_to_s,Dad,Dda,lambda_test,td_test,L_test,R0_test,k=0)

    print(M_s_to_air[0])
    print(la.det(M_s_to_air[0]))
    print(F_air_to_s[0])
    print(la.det(F_air_to_s[0]))
    print(cav[0,0])
    print(la.det(cav[0,0]))
    print(T,R)