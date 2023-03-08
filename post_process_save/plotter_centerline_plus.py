#!/usr/bin/env python3
"""Make some plots
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import yt
import glob
import time
from datetime import timedelta
import matplotlib.pyplot as plt


# ========================================================================
#
# Some defaults variables
#
# ========================================================================
plt.rc("text", usetex=True)
cmap_med = [
    "#F15A60",
    "#7AC36A",
    "#5A9BD4",
    "#FAA75B",
    "#9E67AB",
    "#CE7058",
    "#D77FB4",
    "#737373",
]
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [5, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]



# ========================================================================
#
# Functions
#
# ========================================================================
def radial_profile(data):
    nx, ny = data.shape
    y, x = np.indices((data.shape))
    r = np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2).astype(np.int)

    middle = int(len(r[0])/2)

    rmax = int(r[middle][-1])
    
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile_corners = tbin / nr

    radialprofile = radialprofile_corners[0:rmax]
    
    return radialprofile


def r_half_scaled(averages):
    compare = averages[1:]
    max_value = np.max(averages)
    min_value = np.min(averages)
    full_width = max_value-min_value
    r_val = (np.abs(compare - (min_value + 0.5*full_width))).argmin()
    return r_val + 1

def r_half_raw(averages):
    compare = averages[1:]
    centerline = averages[0]
    r_val = (np.abs(compare - 0.5*centerline)).argmin()
    return r_val + 1

# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple plotting tool")
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        help="Parent folder for vertical slice data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Directory where plots will be saved",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    fdir = os.path.abspath(args.folder)

    QoIs = ["E", "Hi", "Cs", "Z", "alpha", "mu", "xi", "lam"]

    Titles ={"E": "$E$ (Erg)", "Hi": r"$Hi$ (erg)", "Cs": "$Cs$ (cm/s)", "Z": "$Z$ ", "alpha": r"$\alpha$ (St)", "mu": r"$\mu$ (P)", "xi": r"$\xi$ (P)", "lam": r"$\lambda$ (Erg/(cm*s*K))"}
    Titles_scaled ={"E": r"$E^+ = \frac{E - E_{0}}{E_{in} - E_{0}}$", "Hi": r"$Hi^+ = \frac{Hi - Hi_{0}}{Hi_{in} - Hi_{0}}$", "Cs": r"$Cs^{\dag} = \frac{Cs - Cs_{0}}{Cs_{in}-Cs_{0}}$", "Z": r"$Z^+= \frac{Z - Z_{0}}{Z_{in} - Z_{0}}$", "alpha": r"$\alpha^{\dag} = \frac{\alpha - \alpha_{0}}{\alpha_{in} - \alpha_{0}}$", "mu": r"$\alpha^{\dag} = \frac{\mu - \mu_{0}}{\mu_{in} - \mu_{0}}$", "xi": r"$\xi^{\dag} = \frac{\xi - \xi_{0}}{\xi_{in} - \xi_{0}}$", "lam": r"$\lambda^{\dag} = \frac{\lambda - \lambda_{0}}{\lambda_{in} - \lambda_{0}}$" }

    cases = ["314", "same", "350"]
    cmap_presets = {"E": "plasma", "Hi": "plasma", "Cs": "viridis", "Z": "magma_r", "alpha": "plasma", "mu": "viridis", "xi": "cividis", "lam": "magma"}
    scale_fix = {"same": 200/202, "314": 1, "350": 200/203}

    E_amb = {"same": -89793077270.07535, "314": -91390644616.69522, "350": -88979571952.02536}
    Hi_amb = {"same": -89460853629.9153, "314": -91193033149.04276, "350": -88537118697.93825}
    Cs_amb = {"same": 25931.69773318637, "314": 31312.358345590612, "350": 26605.275185467108}
    Z_amb = {"same": 0.5328877740537954, "314": 0.33311062034458827, "350": 0.6690949698835132}
    alpha_amb = {"same": 0.0003876062647505032, "314": 0.00021753227789008593, "350": 0.0007916836135729608}
    mu_amb = {"same": 0.00027475776708225393, "314": 0.0004110209276541549, "350": 0.00024453042856215497}
    xi_amb = {"same": 0.00010292596665761674, "314": 9.145564014980209e-05, "350": 0.00011883401868464863}
    lam_amb = {"same": 3965.950996184853, "314": 6360.926280141722, "350": 3442.6586170884834}

    odir = os.path.abspath(args.output)

    vals = []
    label = []
    count = 0
    
    for case in cases:

        fname = os.path.join(fdir, case+"_ambient", "centerline_slices/Additional_QoI/avg_centerline_aQoI.npz")
        slc = np.load(fname)

        if case == "same":
            leg_label = " 330"
        else:
            leg_label = " "+case

        for field in slc.keys():
            print(field)
    
        E = slc["E"]*scale_fix[case]
        E_scaled = (E - E_amb[case])/(E[0] - E_amb[case])

        Hi = slc["Hi"]*scale_fix[case]
        Hi_scaled = (Hi - Hi_amb[case])/(Hi[0] - Hi_amb[case])

        Cs = slc["Cs"]*scale_fix[case]
        Cs_max = np.max(Cs)
        Cs_min = np.min(Cs)
        Cs_scaled = (Cs - Cs_amb[case])/(Cs[0] - Cs_amb[case])

        lam = slc["lam"]*scale_fix[case]
        lam_max = np.max(lam)
        lam_min = np.min(lam)
        lam_scaled = (lam - lam_amb[case])/(lam[0] - lam_amb[case])


        Z = slc["Z"]*scale_fix[case]
        Z_scaled = (Z - Z_amb[case])/(Z[0] - Z_amb[case])

        alpha = slc["alpha"]*scale_fix[case]
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)
        alpha_scaled = (alpha - alpha_amb[case])/(alpha[0] - alpha_amb[case])

        mu = slc["mu"]*scale_fix[case]
        mu_min = np.min(mu)
        mu_max = np.max(mu)
        mu_scaled = (mu - mu_amb[case])/(mu[0] - mu_amb[case])

        xi = slc["xi"]*scale_fix[case]
        xi_min = np.min(xi)
        xi_max = np.max(xi)
        xi_scaled = (xi - xi_amb[case])/(xi[0] - xi_amb[case])

        centerline = slc["y"]/slc["diameter"]
        
        # Make plots including same_ambient case

        p1 = plt.figure(1)
        plt.plot(centerline, E_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["E"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.axis([0,60,0,max(E_scaled)+.325*max(E_scaled)])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(E_scaled)
        label.append(f"$T_0 = $" + leg_label)

        p2 = plt.figure(2)
        plt.plot(centerline, E, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["E"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        # plt.axis([0,60,310,350])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(E)
        label.append(f"$T_0 = $" + leg_label)

        p3 = plt.figure(3)
        plt.plot(centerline, Hi_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["Hi"])
        plt.axis([0,60,0,max(Hi_scaled)+.25*max(Hi_scaled)])
        vals.append(Hi_scaled)
        label.append(f"T_0 = "+leg_label)

        p4 = plt.figure(4)
        plt.plot(centerline, Hi, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["Hi"])
        # plt.axis([0,60,min(rho) - .25*min(rho),max(rho)+.25*max(rho)])
        vals.append(Hi)
        label.append(f"T_0 = "+leg_label)

        p5 = plt.figure(5)
        plt.plot(centerline, Cs_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["Cs"])
        # plt.axis([0,60,0,1.35])
        vals.append(Cs_scaled)
        label.append(f"T_0 = "+leg_label)

        p6 = plt.figure(6)
        plt.plot(centerline, Cs, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["Cs"])
        # plt.axis([0,60,min(cp)-.25*min(cp),max(cp) + .25*max(cp)])
        vals.append(Cs)
        label.append(f"T_0 = "+leg_label)

        p7 = plt.figure(7)
        plt.plot(centerline, Z_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["Z"])
        # plt.axis([0,60,0,max(Z_scaled)+.25*max(Z_scaled)])
        vals.append(Z_scaled)
        label.append(f"T_0 = "+leg_label)

        p8 = plt.figure(8)
        plt.plot(centerline, Z, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["Z"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(Z)
        label.append(f"T_0 = "+leg_label)

        p9 = plt.figure(9)
        plt.plot(centerline, alpha_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["alpha"])
        # plt.axis([0,60,0,max(alpha_scaled)+.25*max(alpha_scaled)])
        vals.append(alpha_scaled)
        label.append(f"T_0 = "+leg_label)

        p10 = plt.figure(10)
        plt.plot(centerline, alpha, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["alpha"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(alpha)
        label.append(f"T_0 = "+leg_label)

        p11 = plt.figure(11)
        plt.plot(centerline, mu_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["mu"])
        # plt.axis([0,60,0,max(mu_scaled)+.25*max(mu_scaled)])
        vals.append(mu_scaled)
        label.append(f"T_0 = "+leg_label)

        p12 = plt.figure(12)
        plt.plot(centerline, mu, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["mu"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(mu)
        label.append(f"T_0 = "+leg_label)

        p13 = plt.figure(13)
        plt.plot(centerline, xi_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["xi"])
        # plt.axis([0,60,0,max(xi_scaled)+.25*max(xi_scaled)])
        vals.append(xi_scaled)
        label.append(f"T_0 = "+leg_label)

        p14 = plt.figure(14)
        plt.plot(centerline, xi, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["xi"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(xi)
        label.append(f"T_0 = "+leg_label)

        p15 = plt.figure(15)
        plt.plot(centerline, lam_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["lam"])
        # plt.axis([0,60,0,max(lam_scaled)+.25*max(lam_scaled)])
        vals.append(lam_scaled)
        label.append(f"T_0 = "+leg_label)

        p16 = plt.figure(16)
        plt.plot(centerline, lam, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["lam"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(lam)
        label.append(f"T_0 = "+leg_label)
        
        count += 1

    plt.figure(1)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"E_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(2)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"E_centerline_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(3)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Hi_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(4)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Hi_centerline_same.pdf"), format="pdf", dpi=300
    )

    plt.figure(5)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Cs_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(6)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Cs_centerline_same.pdf"), format="pdf", dpi=300
    )

    plt.figure(7)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Z_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(8)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Z_centerline_same.pdf"), format="pdf", dpi=300
    )

    plt.figure(9)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"alpha_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(10)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"alpha_centerline_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(11)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"mu_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(12)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"mu_centerline_same.pdf"), format="pdf", dpi=300
    )
    plt.figure(13)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"xi_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(14)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"xi_centerline_same.pdf"), format="pdf", dpi=300
    )
    plt.figure(15)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"lam_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(16)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"lam_centerline_same.pdf"), format="pdf", dpi=300
    )

    
    plt.close("all")

    cases.remove("same")

    print(cases)
    
    count = 0

    for case in cases:

        fname = os.path.join(fdir, case+"_ambient", "centerline_slices/Additional_QoI/avg_centerline_aQoI.npz")
        slc = np.load(fname)

        if case == "same":
            leg_label = " 330"
        else:
            leg_label = " "+case

        E = slc["E"]*scale_fix[case]
        E_scaled = (E - E_amb[case])/(E[0] - E_amb[case])

        Hi = slc["Hi"]*scale_fix[case]
        Hi_scaled = (Hi - Hi_amb[case])/(Hi[0] - Hi_amb[case])

        Cs = slc["Cs"]*scale_fix[case]
        Cs_max = np.max(Cs)
        Cs_min = np.min(Cs)
        Cs_scaled = (Cs - Cs_amb[case])/(Cs[0] - Cs_amb[case])

        lam = slc["lam"]*scale_fix[case]
        lam_max = np.max(lam)
        lam_min = np.min(lam)
        lam_scaled = (lam - lam_amb[case])/(lam[0] - lam_amb[case])


        Z = slc["Z"]*scale_fix[case]
        Z_scaled = (Z - Z_amb[case])/(Z[0] - Z_amb[case])

        alpha = slc["alpha"]*scale_fix[case]
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)
        alpha_scaled = (alpha - alpha_amb[case])/(alpha[0] - alpha_amb[case])

        mu = slc["mu"]*scale_fix[case]
        mu_min = np.min(mu)
        mu_max = np.max(mu)
        mu_scaled = (mu - mu_amb[case])/(mu[0] - mu_amb[case])

        xi = slc["xi"]*scale_fix[case]
        xi_min = np.min(xi)
        xi_max = np.max(xi)
        xi_scaled = (xi - xi_amb[case])/(xi[0] - xi_amb[case])

            
        centerline = slc["y"]/slc["diameter"]
        
        # Make plots including same_ambient case

        p1 = plt.figure(1)
        plt.plot(centerline, E_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["E"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.axis([0,60,0,max(E_scaled)+.325*max(E_scaled)])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(E_scaled)
        label.append(f"$T_0 = $" + leg_label)

        p2 = plt.figure(2)
        plt.plot(centerline, E, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["E"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        # plt.axis([0,60,310,350])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(E)
        label.append(f"$T_0 = $" + leg_label)

        p3 = plt.figure(3)
        plt.plot(centerline, Hi_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["Hi"])
        # plt.axis([0,60,0,max(Hi_scaled)+.25*max(Hi_scaled)])
        vals.append(Hi_scaled)
        label.append(f"T_0 = "+leg_label)

        p4 = plt.figure(4)
        plt.plot(centerline, Hi, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["Hi"])
        # plt.axis([0,60,min(rho) - .25*min(rho),max(rho)+.25*max(rho)])
        vals.append(Hi)
        label.append(f"T_0 = "+leg_label)

        p5 = plt.figure(5)
        plt.plot(centerline, Cs_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["Cs"])
        # plt.axis([0,60,0,1.35])
        vals.append(Cs_scaled)
        label.append(f"T_0 = "+leg_label)

        p6 = plt.figure(6)
        plt.plot(centerline, Cs, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["Cs"])
        # plt.axis([0,60,min(cp)-.25*min(cp),max(cp) + .25*max(cp)])
        vals.append(Cs)
        label.append(f"T_0 = "+leg_label)

        p7 = plt.figure(7)
        plt.plot(centerline, Z_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["Z"])
        # plt.axis([0,60,0,max(Z_scaled)+.25*max(Z_scaled)])
        vals.append(Z_scaled)
        label.append(f"T_0 = "+leg_label)

        p8 = plt.figure(8)
        plt.plot(centerline, Z, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["Z"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(Z)
        label.append(f"T_0 = "+leg_label)

        p9 = plt.figure(9)
        plt.plot(centerline, alpha_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["alpha"])
        # plt.axis([0,60,0,max(alpha_scaled)+.25*max(alpha_scaled)])
        vals.append(alpha_scaled)
        label.append(f"T_0 = "+leg_label)

        p10 = plt.figure(10)
        plt.plot(centerline, alpha, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["alpha"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(alpha)
        label.append(f"T_0 = "+leg_label)

        p11 = plt.figure(11)
        plt.plot(centerline, mu_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["mu"])
        # plt.axis([0,60,0,max(mu_scaled)+.25*max(mu_scaled)])
        vals.append(mu_scaled)
        label.append(f"T_0 = "+leg_label)

        p12 = plt.figure(12)
        plt.plot(centerline, mu, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["mu"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(mu)
        label.append(f"T_0 = "+leg_label)

        p13 = plt.figure(13)
        plt.plot(centerline, xi_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["xi"])
        # plt.axis([0,60,0,max(xi_scaled)+.25*max(xi_scaled)])
        vals.append(xi_scaled)
        label.append(f"T_0 = "+leg_label)

        p14 = plt.figure(14)
        plt.plot(centerline, xi, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["xi"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(xi)
        label.append(f"T_0 = "+leg_label)

        p15 = plt.figure(15)
        plt.plot(centerline, lam_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["lam"])
        # plt.axis([0,60,0,max(lam_scaled)+.25*max(lam_scaled)])
        vals.append(lam_scaled)
        label.append(f"T_0 = "+leg_label)

        p16 = plt.figure(16)
        plt.plot(centerline, lam, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["lam"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(lam)
        label.append(f"T_0 = "+leg_label)
        
        count += 1

    plt.figure(1)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"E_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(2)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"E_centerline.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(3)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Hi_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(4)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Hi_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(5)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Cs_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(6)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Cs_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(7)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Z_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(8)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"Z_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(9)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"alpha_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(10)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"alpha_centerline.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(11)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"mu_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(12)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"mu_centerline.pdf"), format="pdf", dpi=300
    )
    plt.figure(13)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"xi_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(14)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"xi_centerline.pdf"), format="pdf", dpi=300
    )
    plt.figure(15)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"lam_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(16)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"lam_centerline.pdf"), format="pdf", dpi=300
    )

    
    plt.close("all")
