#!/usr/bin/env python3
#
# Python equivalent of Transport + EoS from PeleC
# SPECIFICALLY for pure CO2, SRK EoS, Simple Transport
# Input data should be fixed resolution 2D slices in *.npz dictionary format
#

# ==========================================================================
#
# Imports
#
# ==========================================================================

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time
from datetime import timedelta

# ==========================================================================
#
# Constants
#
# ==========================================================================

avogadro = 6.02214199e23
boltzmann = 1.3806503e-16 # we work in CGS
Rcst = 83.144598 # in bar [CGS] !

#
# From pureCO2 mechanism.cpp
#

# egtransetWT - molecular weight in g/mol
WT  = 4.40099500e1

# egtransetEPS - lennard-jones potential well depth eps/kb in K
EPS = 2.44000000e2

# egtransetSIG - lennard-jones collision diameter in Angstroms
SIG = 3.76300000

# egtransetDIP - dipole moment in Debye
DIP = 0.00000000

# egtransetPOL - polarizability in cubic Angstroms
POL = 2.65000000

# egtransetZROT - rotational relaxation collision number at 298 K
ZROT = 2.10000000

# egtransetNLIN - 0: monoatomic, 1: linear, 2: nonlinear
NLIN = 1 

# egtransetCOFETA - Poly fits for the viscosities, dim NO*KK
COFETA = np.empty(4)
COFETA[0] = -2.28110345e1
COFETA[1] = 4.62954710
COFETA[2] = -5.00689001e-1
COFETA[3] = 2.10012969e-2

# egtransetCOFLAM - Poly fits for the conductivities, dim NO*KK
COFLAM = np.empty(4)
COFLAM[0] = -8.74831432
COFLAM[1] = 4.79275291
COFLAM[2] = -4.18685061e-1
COFLAM[3] = 1.35210242e-2

# egtransetCOFD - Poly fits for the diffusion coefficients, dim NO*KK*KK
COFD = np.empty(4)
COFD[0] = -2.05810669e1
COFD[1] = 5.07469434
COFD[2] = -4.25340301e-1
COFD[3] = 1.76800795e-2

# inverse molecular weight
iWT = 1./WT

# Critical Parameters for CO2 - imported from NIST
Tc = 304.120000
a = 1e6 * 0.42748 * np.power(Rcst,2.0) * np.power(Tc,2.0) / (np.power(44.009950,2.0) * 73.740000); 
b = 0.08664 * Rcst * Tc / (44.009950 * 73.740000); 
acentric = 0.225000

# SRK Constants
f0 = 0.48508e+0
f1 = 1.5517e+0
f2 = -0.151613e+0
convCrit = 1e-4

oneOverTc = 1.0 / Tc
sqrtOneOverTc = np.sqrt(oneOverTc)
sqrtAsti = np.sqrt(a)
Fomega = f0 + acentric * (f1 + f2 * acentric)


# Chung Nonideal Transport Coefficient Parameters

Afac = np.empty(40)
Afac[0] = 6.32402
Afac[1] = 50.41190
Afac[2] = -51.68010
Afac[3] = 1189.020
Afac[4] = 0.12102e-2
Afac[5] = -0.11536e-2
Afac[6] = -0.62571e-2
Afac[7] = 0.37283e-1
Afac[8] = 5.28346
Afac[9] = 254.209
Afac[10] = -168.481
Afac[11] = 3898.27
Afac[12] = 6.62263
Afac[13] = 38.09570
Afac[14] = -8.46414
Afac[15] = 31.41780
Afac[16] = 19.74540
Afac[17] = 7.63034
Afac[18] = -14.35440
Afac[19] = 31.52670
Afac[20] = -1.89992
Afac[21] = -12.5367
Afac[22] = 4.98529
Afac[23] = -18.15070
Afac[24] = 24.27450
Afac[25] = 3.44945
Afac[26] = -11.29130
Afac[27] = 69.34660
Afac[28] = 0.79716
Afac[29] = 1.11764
Afac[30] = 0.12348e-1
Afac[31] = -4.11661
Afac[32] = -0.23816;
Afac[33] = 0.67695e-1;
Afac[34] = -0.81630;
Afac[35] = 4.02528;
Afac[36] = 0.68629e-1;
Afac[37] = 0.34793;
Afac[38] = 0.59256;
Afac[39] = -0.72663;

Bfac = np.empty(28)
Bfac[0] = 2.41657
Bfac[1] = 0.74824
Bfac[2] = -0.91858
Bfac[3] = 121.721
Bfac[4] = -0.50924
Bfac[5] = -1.50936
Bfac[6] = -49.99120
Bfac[7] = 69.9834
Bfac[8] = 6.61069
Bfac[9] = 5.62073
Bfac[10] = 64.75990
Bfac[11] = 27.0389
Bfac[12] = 14.54250
Bfac[13] = -8.91387
Bfac[14] = -5.63794
Bfac[15] = 74.3435
Bfac[16] = 0.79274
Bfac[17] = 0.82019
Bfac[18] = -0.69369
Bfac[19] = 6.31734
Bfac[20] = -5.86340
Bfac[21] = 12.80050
Bfac[22] = 9.58926
Bfac[23] = -65.5292
Bfac[24] = 81.17100
Bfac[25] = 114.15800
Bfac[26] = -60.84100
Bfac[27] = 466.7750

Kappa = 0.0

#
# SIMPLIFICATIONS FOR 1 SPECIES CASE:
#
# tparm->sqrtT2ij[idx] = SIG
# tparm->sqrtEpsilonij[idx] = EPS
# tparm->sqrtMWij[idx] = np.sqrt(WT);
# tparm->sqrtKappaij[idx] = Kappa
#


# Physics Constants

gamma = 1.4
RU = 8.31446261815324e7
RUC = 1.98721558317399615845
PATM = 1.01325e6
AIRMW = 28.97
Avna = 6.022140857e23

M_PI = 3.1415926535897932384626433832795





# ==========================================================================
#
# TRANSPORT Functions
#
# ==========================================================================

def transport(tc, rholoc):

    # Yloc and Xloc = 1 for 1 species mixture, so those have been eliminated

    Tloc = tc[1]
    Dim = Tloc.shape
    
    logT_hold = np.log(Tloc)

    logT = np.array([logT_hold, np.power(logT_hold, 2.0), np.power(logT_hold, 3.0)])

    wbar = 1.0/(iWT)

    # get shear (mu) and bulk (xi) viscosity

    muloc = COFETA[0] + COFETA[1]*logT[0] + COFETA[2]*logT[1] + COFETA[3]*logT[2]
    mu = np.exp(muloc)

    xiloc = comp_pure_bulk(tc, mu)
    xi = xiloc

    # get conductivity (lamda)

    lamloc = COFLAM[0] + COFLAM[1]*logT[0] + COFLAM[2]*logT[1] + COFLAM[3]*logT[2]
    lam = np.exp(lamloc)

    # add non-ideal Chung corrections

    mu, lam = NonIdealChungCorrections(tc, rholoc, wbar, mu, lam)

    # get diffusion coefficients
    # actually not necessary since CO2 case is single species

    # WRITE FUNCTION -> BinaryDiff(Yloc, logT, rholoc, Tloc, Ddiag)

    return mu, xi, lam

def comp_pure_bulk(tc, muloc):

    Tloc = tc[1]
    pi3_2 = np.power(M_PI, 1.5)

    epskoverTstd = EPS/298.0
    epskoverT = EPS/Tloc

    cvk = CKCVMS(tc)

    cvkint = cvk*WT/RU - 1.50
    cvkrot = 1.0

    Fnorm = 1.0 + 0.50*pi3_2*np.sqrt(epskoverTstd) + (2.0 + .50*M_PI*M_PI)*epskoverTstd + pi3_2*np.sqrt(epskoverTstd)*epskoverTstd
    FofT = 1.0 + 0.50*pi3_2*np.sqrt(epskoverT) + (2.0 + .50*M_PI*M_PI)*epskoverT + pi3_2*np.sqrt(epskoverT)*epskoverT

    xiloc = 0.250*M_PI*(cvkint/(cvkint + 1.50)*cvkint/(cvkint + 1.50))*ZROT/cvkrot*Fnorm/FofT*muloc
    return xiloc

# function for creating temp cache given temp
def temp_cache(T):

    tc = np.array([np.zeros(T.shape), T, T*T, T*T*T, T*T*T*T])

    return tc

# compute Cv/R at the given temperature
# tc contains precomputed powers of T, tc[0] = log(T) 
def cv_R(tc):

    T = tc[1]
    Dim = T.shape
    species = np.empty(Dim)

    species = np.where(T < 1000,
                       1.35677352e+00 * np.ones(Dim) + 8.98459677e-03 * tc[1] - 7.12356269e-06 * tc[2] + 2.45919022e-09 * tc[3] - 1.43699548e-13 * tc[4],
                       2.85746029e+00 * np.ones(Dim) + 4.41437026e-03 * tc[1] - 2.21481404e-06 * tc[2] + 5.23490188e-10 * tc[3] - 4.72084164e-14 * tc[4])
    return species

# Returns the specific heats at constant volume in mass units    
def CKCVMS(tc):

    cvms = cv_R(tc)

    cvms *= 1.889234139098090e+06 # multiply by R/molecularweight for CO2
    return cvms
    
def NonIdealChungCorrections(tc, rholoc, wbar, mu, lam):

    Tloc = tc[1]
    T2 = SIG*SIG
    T3 = SIG*T2

    Epsilon_M = T3*EPS

    Omega_M = T3*acentric

    MW_m = EPS*T2*np.sqrt(WT)

    DP_m_4 = 0.0
    KappaM = 0.0

    MW_m *= MW_m

    InvSigma1 = 1.0/SIG
    InvSigma3 = 1.0/T3
    Epsilon_M *= InvSigma3

    Tstar = Tloc/Epsilon_M
    Tcm = 1.2593 * Epsilon_M
    Vcm = 1.8887 * T3

    Omega_M *= InvSigma3
    MW_m = MW_m * InvSigma3 * InvSigma1 / (Epsilon_M * Epsilon_M)

    # DP_m_4 = DP_m_4 * T3 * Epsilon_M # DP_m_4 = 0 still for CO2
    DP_red_4 = 0.0 # 297.2069113e6 * DP_m_4 / (Vcm * Vcm * Tcm * Tcm) # consequently this also yields 0

    y = Vcm * rholoc / (6.0 * wbar)
    G1 = (1.0 - 0.5 * y) / ((1.0 - y) * (1.0 - y) * (1.0 - y))

    # set nonideal viscosity

    A = np.empty(10)

    for i in range(10):
        A[i] = Afac[i*4] + Afac[i*4 + 1]*Omega_M + Afac[i*4 + 2]*DP_red_4 + Afac[i*4 + 3]*KappaM

    G2 = (A[0] * (1.0 - np.exp(-A[3] * y)) / y + A[1] * G1 * np.exp(A[4] * y) + A[2] * G1) / (A[0] * A[3] + A[1] + A[2])
    eta_P = (36.344e-6 * np.sqrt(MW_m * Tcm) / np.power(Vcm, 2.0 / 3.0)) * A[6] * y * y * G2 * np.exp(A[7] + A[8] / Tstar + A[9] / (Tstar * Tstar))

    updated_mu = mu * (1.0 / G2 + A[5] * y) + eta_P

    mu = np.where(updated_mu > 0, updated_mu, mu)

    # set nonideal conductivity

    B = np.empty(7)

    for i in range(7):
        B[i] = Bfac[i * 4] + Bfac[i * 4 + 1] * Omega_M + Bfac[i * 4 + 2] * DP_red_4 + Bfac[i * 4 + 3] * KappaM

    H2 = (B[0] * (1.0 - np.exp(-B[3] * y)) / y + B[1] * G1 * np.exp(B[4] * y) + B[2] * G1) / (B[0] * B[3] + B[1] + B[2])
    lambda_p = 3.039e-4 * np.sqrt(Tcm / MW_m) / np.power(Vcm, 2.0 / 3.0) * B[6] * y * y * H2 * np.sqrt(Tstar)

    lambda_p *= 4.184e+7 # erg/(cm s K)
    beta = 1.0 / H2 + B[5] * y

    updated_lam = lam * beta + lambda_p

    lam = np.where(updated_lam > 0, updated_lam, lam)

    return mu, lam



# ==========================================================================
#
# SRK Functions
#
# ==========================================================================

def RTY2E(R, tc):

    T = tc[1]
    # Calculate ideal gas portion
    E = CKUMS(tc)

    # Add in non-ideal portion
    am, bm = MixingRuleAmBm(T)
    dAmdT = Calc_dAmdT(T)
    # below was log(1 + bm/tau), tau = 1/R so this is simpler)
    K1 = (1.0 / bm) * np.log1p(bm * R)
    E += (T * dAmdT - am) * K1

    return E
    
def RTY2Hi(R, tc):

    T = tc[1]
    inv_mwt = iWT

    # Ideal gas part
    Hi = CKHMS(tc)

    # Non-ideal part: Could be optimized a bit more by combining all derivative calls
    bm = b

    am, dAmdT, d2AmdT2, dAmdY, d2AmdTY = Calc_Am_and_derivs(T)

    wbar = 1.0/iWT
    
    tau = 1.0 / R
    K1 = (1.0 / bm) * np.log1p(bm * R)
    InvEosT1Denom = 1.0 / (tau - bm)
    InvEosT2Denom = 1.0 / (tau * (tau + bm))
    InvEosT3Denom = 1.0 / (tau + bm)
    Rm = RU / wbar

    dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom + am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom
    dhmdtau = -(T * dAmdT - am) * InvEosT2Denom + am * InvEosT3Denom * InvEosT3Denom - Rm * T * bm * InvEosT1Denom * InvEosT1Denom

    Rmk = RU * iWT
    dpdYk = Rmk * T * InvEosT1Denom - dAmdY * InvEosT2Denom + b * (Rm * T * InvEosT1Denom * InvEosT1Denom + am * InvEosT2Denom * InvEosT3Denom)
    dhmdYk = Hi + (T * d2AmdTY - dAmdY) * K1 - b * (T * dAmdT - am) * (K1 / bm - InvEosT3Denom / bm) + am * b * InvEosT3Denom * InvEosT3Denom - InvEosT3Denom * dAmdY + Rmk * T * bm * InvEosT1Denom + Rm * T * b * (InvEosT1Denom + bm * InvEosT1Denom * InvEosT1Denom)
    Hi = dhmdYk - (dhmdtau / dpdtau) * dpdYk

    return Hi

    
def RTY2Cs(R, tc, Cp, P):

    G = get_G(R, tc, Cp, P)
    Cs = np.sqrt(G * P / R)

    return Cs

def get_G(R, tc, Cp, P):

    T = tc[1]
    wbar = 1.0/iWT
    bm = b
    am, dAmdT, d2AmdT2, dAmdY, d2AmdTY = Calc_Am_and_derivs(T)

    tau = 1.0 / R
    K1 = (1.0 / bm) * np.log1p(bm * R)

    InvEosT1Denom = 1.0 / (tau - bm)
    InvEosT2Denom = 1.0 / (tau * (tau + bm))
    InvEosT3Denom = 1.0 / (tau + bm)
    Rm = RU / wbar

    dpdT = Rm * InvEosT1Denom - dAmdT * InvEosT2Denom
    dpdtau = -Rm * T * InvEosT1Denom * InvEosT1Denom + am * (2.0 * tau + bm) * InvEosT2Denom * InvEosT2Denom

    Cv = CKCVBS(tc)
    Cv += T * d2AmdT2 * K1

    G = -tau * Cp * dpdtau / (P * Cv)

    return G

    
def Calc_CompressFactor_Z(P, T):

    Wbar = 1.0/iWT

    am, bm = MixingRuleAmBm(T)

    RmT = RU / Wbar * T
    B1 = bm * P / RmT
    R1 = RmT
    R2 = R1 * RmT
    R3 = R2 * RmT
    alpha = -1.0
    beta = (am * P - bm * P * bm * P) / R2 - B1
    gamma = -(am * bm * P * P) / R3
    Q = (alpha * alpha - 3.0 * beta) / 9.0
    R = (2.0 * alpha * alpha * alpha - 9.0 * alpha * beta + 27.0 * gamma) / 54.0

    # Multiple roots of cubic
    third = 1.0 / 3.0
    Z = np.where((Q * Q * Q - R * R) > 0,
                 Z_if_true(Q, R),
                 -np.copysign(1.0, R) * (np.power((np.sqrt(R * R - Q * Q * Q) + np.absolute(R)), third) + Q / (np.power(np.sqrt(R * R - Q * Q * Q) + np.absolute(R), third))) - alpha * third)

    return Z

def Z_if_true(Q, R):

    third = 1.0/3.0
    alpha = -1.0
    
    sqrtQ = np.sqrt(Q)
    theta = np.arccos(R / (Q * sqrtQ))
    Z1 = -2.0 * sqrtQ * np.cos(theta * third) - alpha * third
    Z2 = -2.0 * sqrtQ * np.cos((theta + 2.0 * M_PI) * third) - alpha * third
    Z3 = -2.0 * sqrtQ * np.cos((theta + 4.0 * M_PI) * third) - alpha * third
    Z = np.maximum(Z1, Z2)
    Z = np.maximum(Z, Z3)

    return Z

def Calc_Themal_Diffusivity(k, cp, rho):

    thermdiff = k / (cp * rho)
    return thermdiff

    
def CKUMS(tc):

    
    tT = tc[1]
    RT = 8.31451e+07*tT # R*T 

    ums = speciesInternalEnergy(tc)

    ums *= RT*iWT

    return ums

def CKHMS(tc):
    tT = tc[1] # temporary temperature 
    RT = 8.31451e+07*tT # R*T 
    hms = speciesEnthalpy(tc)

    hms *= RT*iWT

    return hms

def CKCVBS(tc):

    result = 0.0 
    cvor = cv_R(tc)
    # multiply by y/molecularweight
    result += cvor*iWT # CO2 

    cvbs = result * 8.31451e+07

    return cvbs
    

def MixingRuleAmBm(T):
    am = 0.0
    bm = 0.0
    
    sqrtT = np.sqrt(T)
    amloc = 0.0

    amloc = (1.0 + Fomega * (1.0 - sqrtT * sqrtOneOverTc)) * sqrtAsti
    bm += b
    am += amloc*amloc

    return am, bm

def Calc_dAmdT(T):

    dAmdT = 0.0
    # AMREX_ASSERT(T > 0.0);
    oneOverT = 1.0 / T
    sqrtT = np.sqrt(T)
    amloc = 0.0
    amlocder = 0.0

    amloc = (1.0 + Fomega * (1.0 - sqrtT * sqrtOneOverTc)) * sqrtAsti
    amlocder = -0.5 * Fomega * sqrtAsti * oneOverT * sqrtT * sqrtOneOverTc

    dAmdT += amloc * amlocder + amloc * amlocder

    return dAmdT

def Calc_Am_and_derivs(T):

    oneOverT = 1.0 / T
    tmp1 = -0.5 * oneOverT
    sqrtT = np.sqrt(T)
    amloc = 0.0
    amlocder = 0.0

    # Compute species-dependent intermediates
    amloc = (1.0 + Fomega * (1.0 - sqrtT * sqrtOneOverTc)) * sqrtAsti
    amlocder = Fomega * sqrtAsti * sqrtOneOverTc # *-0.5*oneOverT*sqrtT

    am = amloc * amloc
    dAmdT = 2.0 * (amloc * amlocder)
    d2AmdT2 = amlocder * amlocder
    dAmdY = 2.0 * (amloc * amloc)
    d2AmdTY = 2.0 * (amloc * amlocder + amloc * amlocder)

    # factor in constants
    dAmdT *= tmp1 * sqrtT
    d2AmdT2 = -d2AmdT2
    d2AmdT2 += dAmdT
    d2AmdT2 *= tmp1
    d2AmdTY *= tmp1 * sqrtT

    return am, dAmdT, d2AmdT2, dAmdY, d2AmdTY
    
def speciesInternalEnergy(tc):

    # temperature
    T = tc[1]
    invT = 1/T

    Dim = T.shape
    species = np.empty(Dim)


    # species with midpoint at T=1000 kelvin 
    species = np.where(T < 1000,
                      1.35677352e+00 * np.ones(Dim) + 4.49229839e-03 * tc[1] - 2.37452090e-06 * tc[2] + 6.14797555e-10 * tc[3] - 2.87399096e-14 * tc[4] - 4.83719697e+04 * invT,
                      2.85746029e+00 * np.ones(Dim) + 2.20718513e-03 * tc[1] - 7.38271347e-07 * tc[2] + 1.30872547e-10 * tc[3] - 9.44168328e-15 * tc[4] - 4.87591660e+04 * invT)

    return species
            
def speciesEnthalpy(tc):

    # temperature
    T = tc[1]
    invT = 1 / T

    Dim = T.shape
    species = np.empty(Dim)

    # species with midpoint at T=1000 kelvin 
    species = np.where(T < 1000,
                       2.35677352e+00 * np.ones(Dim) + 4.49229839e-03 * tc[1] - 2.37452090e-06 * tc[2] + 6.14797555e-10 * tc[3] - 2.87399096e-14 * tc[4] - 4.83719697e+04 * invT,
                       3.85746029e+00 * np.ones(Dim) + 2.20718513e-03 * tc[1] - 7.38271347e-07 * tc[2] + 1.30872547e-10 * tc[3] - 9.44168328e-15 * tc[4] - 4.87591660e+04 * invT)

    return species


# ==========================================================================
#
# TRANS TEST FROM PELEPHYSICS
#
# ==========================================================================

dTemp = 5.0
dRho = 0.005
# y = plo[1] + (j + 0.5) * dx[1]
# x = plo[0] + (i + 0.5) * dx[0]
pi = 3.1415926535897932
#   amrex::GpuArray<amrex::Real, 3> L;
#   amrex::GpuArray<amrex::Real, 3> P;
#  amrex::GpuArray<amrex::Real, NUM_SPECIES> Y_lo;
#  amrex::GpuArray<amrex::Real, NUM_SPECIES> Y_hi;

def test_transport(npts): 
    
    
    y = np.linspace(-1., 1., npts)
    y = np.array(y)
    
    domain = np.full((npts, npts), y)
    domain = np.transpose(domain)

    temp = 313.99999999770193 + dTemp * np.sin(2.0 * pi * (domain + 1.) / 0.5)
    rho = 0.5127417019181719 + dRho * np.sin(2.0 * pi * (domain + 1.) / 0.5)

    return temp, rho






# ==========================================================================
#
# MAIN
#
# ==========================================================================

if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments                                                                                                                                                                                      
    parser = argparse.ArgumentParser(description="Get additiona QOI for sCO2 Project")
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        help="Folder containing slice files",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-d",
        "--direction",
        dest="slice_dir",
        help="Direction of slice: normal, vertical, centerline",
        type=str,
        default="normal",
    )
    parser.add_argument(
        "-t",
        "--testing",
        dest="testing",
        help="Runs test case that matches Transport test function in PelePhysics",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-p",
        "--plot",
        dest="plot_check",
        help="Saves plots of data for visualization",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    # Setup

    if(args.slice_dir == "centerline"):
        hold = "centerline"
    else:
        hold = "slice"
    
    fdirs = sorted(glob.glob(os.path.join(args.folder, "plt*_"+hold+"*.npz")))
    odir = os.path.join(args.folder, "Additional_QoI")
    if not os.path.exists(odir):
        os.makedirs(odir)

    testing = args.testing
    plot_check = args.plot_check
    
    # Testing
    if testing:

        
        npts = 128
        Dim = np.array([npts, npts])
        temp, rho = test_transport(npts)

        x = np.linspace(-1., 1., npts)
        z = np.linspace(-1., 1., npts)

        grid = np.meshgrid(x, z)

        extents = [-1., 1., -1., 1.]
        
        mu = np.empty(Dim)
        xi = np.empty(Dim)
        lam = np.empty(Dim)

        tc = temp_cache(temp)

        mu, xi, lam = transport(tc, rho)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axs[0].imshow(mu, origin="lower", extent=extents, cmap = "turbo")
        axs[0].set_title(r"$\mu$")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("z")
        ticks = np.linspace(mu.min(), mu.max(), 6, endpoint=True)
        fig.colorbar(im0, ax=axs[0], ticks=ticks)

        im1 = axs[1].imshow(xi, origin="lower", extent=extents, cmap = "turbo")
        axs[1].set_title(r"$\xi$")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("z")
        ticks = np.linspace(xi.min(), xi.max(), 6, endpoint=True)
        fig.colorbar(im1, ax=axs[1], ticks=ticks)

        im2 = axs[2].imshow(lam, origin="lower", extent=extents, cmap = "turbo")
        axs[2].set_title("$\lambda$")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("z")
        ticks = np.linspace(lam.min(), lam.max(), 6, endpoint=True)
        fig.colorbar(im2, ax=axs[2], ticks=ticks)

        plt.savefig(os.path.join(odir, "transport_test_image_" + str(1)), dpi=300)
        plt.close("all")

        sys.exit()


    
    for fdir in fdirs:
        slc = np.load(fdir)

        step = slc["step"]
        extents = slc["extents"]
    
        islice = 0

        if(args.slice_dir == "centerline"):

            cp = slc["cp_cl"]
            pressure = slc["pressure_cl"]
            rho = slc["rho_cl"]
            temp = slc["temp_cl"]

            tc = temp_cache(temp)

            y = slc["y"]
            Dim = cp.shape

            mu = np.empty(Dim)
            xi = np.empty(Dim)
            lam = np.empty(Dim)

            mu, xi, lam = transport(tc, rho)

            E = np.empty(Dim)
            Hi = np.empty(Dim)
            Cs = np.empty(Dim)
            Z = np.empty(Dim)
            alpha = np.empty(Dim)

            E = RTY2E(rho, tc)
            Hi = RTY2Hi(rho, tc)
            Cs = RTY2Cs(rho, tc, cp, pressure)

            Z = Calc_CompressFactor_Z(pressure, temp)
            alpha = Calc_Themal_Diffusivity(lam, cp, rho)

            rho_E = rho*E
            rho_e = rho*Hi

            rho_E_max = np.max(rho_E)
            rho_E_min = np.min(rho_E)

            test_min = -4.687e10
            test_max = -4.324e10

            Cs_min = 2.612e4
            Cs_max = 3.133e4
            
            # print(rho_E_max, rho_E_min)
            # print(np.max(rho*Hi), np.min(rho*Hi))
            # print(np.max(Cs), np.min(Cs))
            # print()
            # print(np.max(rho), np.min(rho))
            # print()
            # print("Find matching density when T=314:\n")

            Temp_diffs = np.abs(temp-314)

            Temp_index = np.unravel_index(np.argmin(Temp_diffs, axis=None), Temp_diffs.shape) 
            # print(Temp_index)
            # T_i = Temp_index[0][0]
            # T_j = Temp_index[1][0]

            # Rho_val = rho[T_i][T_j]
            # Temp_val = temp[T_i][T_j]

            # print(temp[Temp_index], rho[Temp_index])
            # print()
            
            # Save the data
            # Save the data
            pfx = f"plt{step:05d}_slice_{islice:04d}_aQoI"
            uname = os.path.join(odir, pfx)
            np.savez_compressed(
                uname,
                fdir = args.folder,
                step = step,
                y = y,
                mu = mu,
                xi = xi,
                lam = lam,
                E = E,
                Hi = Hi,
                Cs = Cs,
                Z = Z,
                alpha = alpha,
                extents = extents
            )


            if plot_check:
                fig, axs = plt.subplots(3, 3, figsize=(14, 18))
                im0 = axs[0,0].plot(y, rho_E)
                axs[0,0].set_ylabel(r"$\rho$E")
                axs[0,0].set_xlabel("y")

                im1 = axs[0,1].plot(y, rho_e)
                axs[0,1].set_ylabel(r"$\rho$e")
                axs[0,1].set_xlabel("y")

                im2 = axs[0,2].plot(y, Cs)
                axs[0,2].set_ylabel("Cs") # Sound speed
                axs[0,2].set_xlabel("y")

                im3 = axs[1,0].plot(y, lam)
                axs[1,0].set_ylabel(r"$\lambda$") # Thermal Conductivity
                axs[1,0].set_xlabel("y")

                im4 = axs[1,1].plot(y, mu)
                axs[1,1].set_ylabel(r"$\mu$") # Shear Viscosity
                axs[1,1].set_xlabel("y")

                im5 = axs[1,2].plot(y, xi)
                axs[1,2].set_ylabel(r"$xi$") # Bulk Viscosity
                axs[1,2].set_xlabel("y")

                im6 = axs[2,0].plot(y, alpha)
                axs[2,0].set_ylabel(r"$\alpha$") # Thermal Diffusivity
                axs[2,0].set_xlabel("y")

                im7 = axs[2,1].plot(y, Z)
                axs[2,1].set_ylabel("Z") # Compressibility
                axs[2,1].set_xlabel("y")

                im8 = axs[2,2].plot(y, rho)
                axs[2,2].set_ylabel(r"$\rho$") # Density
                axs[2,2].set_xlabel("y")

            
                fig.suptitle("Slice at y = {0:.6f}".format(islice))
                plt.savefig(os.path.join(odir, pfx + "_centerline_check_" + str(1)), dpi=300)
                plt.close("all")


        else:    
            if(args.slice_dir == "normal"):
                index = fdir.find("slice_")
                islice = int(fdir[index+9:-4])

            print("On plt",step, "slice",islice+1)

                
            cp = slc["cp"]
            pressure = slc["pressure"]
            rho = slc["rho"]
            temp = slc["temp"]

            if(args.slice_dir == "normal"):
                cp = np.transpose(cp)
                pressure = np.transpose(pressure)
                rho = np.transpose(rho)
                temp = np.transpose(temp)
            
            tc = temp_cache(temp)

            x = slc["x"]
            z = slc["z"]
        
            Dim = cp.shape

            mu = np.empty(Dim)
            xi = np.empty(Dim)
            lam = np.empty(Dim)

            mu, xi, lam = transport(tc, rho)

            E = np.empty(Dim)
            Hi = np.empty(Dim)
            Cs = np.empty(Dim)
            Z = np.empty(Dim)
            alpha = np.empty(Dim)

            E = RTY2E(rho, tc)
            Hi = RTY2Hi(rho, tc)
            Cs = RTY2Cs(rho, tc, cp, pressure)

            Z = Calc_CompressFactor_Z(pressure, temp)
            alpha = Calc_Themal_Diffusivity(lam, cp, rho)

            rho_E = rho*E
            rho_e = rho*Hi

            rho_E_max = np.max(rho_E)
            rho_E_min = np.min(rho_E)

            test_min = -4.687e10
            test_max = -4.324e10

            Cs_min = 2.612e4
            Cs_max = 3.133e4
            
            # print(rho_E_max, rho_E_min)
            # print(np.max(rho*Hi), np.min(rho*Hi))
            # print(np.max(Cs), np.min(Cs))
            # print()
            # print(np.max(rho), np.min(rho))
            # print()
            # print("Find matching density when T=314:\n")

            Temp_diffs = np.abs(temp-314)

            Temp_index = np.unravel_index(np.argmin(Temp_diffs, axis=None), Temp_diffs.shape) 
            # print(Temp_index)
            # T_i = Temp_index[0][0]
            # T_j = Temp_index[1][0]

            # Rho_val = rho[T_i][T_j]
            # Temp_val = temp[T_i][T_j]

            # print(temp[Temp_index], rho[Temp_index])
            # print()
            
            # Save the data
            pfx = f"plt{step:05d}_slice_{islice:04d}_aQoI"
            uname = os.path.join(odir, pfx)
            np.savez_compressed(
                uname,
                fdir = args.folder,
                step = step,
                x = x,
                z = z,
                mu = mu,
                xi = xi,
                lam = lam,
                E = E,
                Hi = Hi,
                Cs = Cs,
                Z = Z,
                alpha = alpha,
                extents = extents
            )


            if plot_check:
                fig, axs = plt.subplots(3, 3, figsize=(14, 18))
                im0 = axs[0,0].imshow(rho_E, origin="lower", extent=extents, vmin=rho_E_min, vmax=rho_E_max,  cmap = "turbo")
                axs[0,0].set_title(r"$\rho$E")
                axs[0,0].set_xlabel("x")
                axs[0,0].set_ylabel("z")
                fig.colorbar(im0, ax=axs[0,0])

                im1 = axs[0,1].imshow(rho_e, origin="lower", extent=extents, vmin=test_min, vmax=test_max, cmap = "turbo")
                axs[0,1].set_title(r"$\rho$e")
                axs[0,1].set_xlabel("x")
                axs[0,1].set_ylabel("z")
                fig.colorbar(im1, ax=axs[0,1])

                im2 = axs[0,2].imshow(Cs, origin="lower", extent=extents, vmin=Cs_min, vmax=Cs_max, cmap = "turbo")
                axs[0,2].set_title("Speed of Sound")
                axs[0,2].set_xlabel("x")
                axs[0,2].set_ylabel("z")
                fig.colorbar(im2, ax=axs[0,2])

                im3 = axs[1,0].imshow(lam, origin="lower", extent=extents, cmap = "turbo")
                axs[1,0].set_title("Thermal Conductivity")
                axs[1,0].set_xlabel("x")
                axs[1,0].set_ylabel("z")
                fig.colorbar(im3, ax=axs[1,0])

                im4 = axs[1,1].imshow(mu, origin="lower", extent=extents, cmap = "turbo")
                axs[1,1].set_title("Sheer Viscosity")
                axs[1,1].set_xlabel("x")
                axs[1,1].set_ylabel("z")
                fig.colorbar(im4, ax=axs[1,1])

                im5 = axs[1,2].imshow(xi, origin="lower", extent=extents, cmap = "turbo")
                axs[1,2].set_title("Bulk Viscosity")
                axs[1,2].set_xlabel("x")
                axs[1,2].set_ylabel("z")
                fig.colorbar(im5, ax=axs[1,2])

                im6 = axs[2,0].imshow(alpha, origin="lower", extent=extents, cmap = "turbo")
                axs[2,0].set_title("Thermal Diffusivity")
                axs[2,0].set_xlabel("x")
                axs[2,0].set_ylabel("z")
                fig.colorbar(im6, ax=axs[2,0])

                im7 = axs[2,1].imshow(Z, origin="lower", extent=extents, cmap = "turbo")
                axs[2,1].set_title("Compressibility")
                axs[2,1].set_xlabel("x")
                axs[2,1].set_ylabel("z")
                fig.colorbar(im7, ax=axs[2,1])

                im8 = axs[2,2].imshow(rho, origin="lower",extent=extents, vmin=np.min(rho), vmax=np.max(rho), cmap="turbo")
                axs[2,2].set_title("Density")
                axs[2,2].set_xlabel("x")
                axs[2,2].set_ylabel("z")
                fig.colorbar(im8, ax=axs[2,2])

            
                fig.suptitle("Slice at y = {0:.6f}".format(islice))
                plt.savefig(os.path.join(odir, pfx + "_image_" + str(1)), dpi=300)
                plt.close("all")

        
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
