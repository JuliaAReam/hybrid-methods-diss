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

    QoIs = ["ptemp_rms_rad", "prho_rms_rad", "cp_rad"]
    Titles ={"temp": "$T$ (K)", "rho": r"$\rho$ (g/cm$^{3}$)", "cp": "$C_p$ (J/g*K)", "v": "$v$ (cm/s)" }
    cases = ["314", "350"]
    slices = ["avg_slice_0000.npz", "avg_slice_0001.npz"]
    scale_fix = {"same": 200/202, "314": 1, "350": 200/203}

    T_amb = {"same": 329.9900916653577, "314": 313.99981302039595, "350": 350.01795947600795}
    # T_amb = {"same": 330, "314": 314, "350": 350}
    rho_amb = {"same": 0.3019277174113144, "314": 0.5127411755780334, "350": 0.2256708202157278}

    odir = os.path.abspath(args.output)

    vals = []
    label = []
    lab = 0
    count = 0

    for sl in slices:
        count = 0
        for case in cases:

            fname = os.path.join(fdir, case+"_ambient", "slices_3", sl)
            slc = np.load(fname)

            if case == "same":
                leg_label = " 330"
            else:
                leg_label = " "+case

            # for field in slc.keys():
                # print(field)


            Temp = slc["temp_pert"]

            # print(Temp.shape)
        
            rms=0
            for i in range(199):
                # print("i is: ", i)
                mult = Temp[i]*Temp[i]
                rms += mult
            rms /= 199
            rms = np.sqrt(rms)

            rms_rad = radial_profile(rms)
            rms_scaled = rms_rad/T_amb[case]
        

            r_d = slc["r"]/slc["diameter"]

            dist = np.ceil(slc["y"]/slc["diameter"])
        
            # Make plot
                                                                            
            p1 = plt.plot(r_d, rms_scaled, color=cmap[count%8], label="$T_0 = $"+leg_label, lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/d$")
            plt.ylabel("$T_{rms}^{\prime}/\overline{T}_{0}$")
            # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                    
            plt.axis([0,6,0,max(rms_scaled)+.325*max(rms_scaled)])
            # plt.text(.2, max(rms_scaled)+.325*max(rms_scaled)-.005, '$y/d = $'+str(dist))
            
            vals.append(rms_scaled)
            label.append(f"$T_0 = $" + leg_label)
            count += 1

        plt.text(.2, max(rms_scaled)+.255*max(rms_scaled), '$y/d = $'+str(dist))
        plt.legend(loc='best', handlelength=3)
        # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"temp_pert_rad_"+str(lab)+".pdf"), format="pdf", dpi=300
        )
        plt.close("all")
        lab += 1


    lab = 0
    for sl in slices:
        count = 0
        ymax = 0
        for case in cases:

            fname = os.path.join(fdir, case+"_ambient", "slices_3", sl)
            slc = np.load(fname)

            if case == "same":
                leg_label = " 330"
            else:
                leg_label = " "+case

            # for field in slc.keys():
                # print(field)


            rho = slc["rho_pert"]

            # print(rho.shape)
        
            rms=0
            for i in range(199):
                # print("i is: ", i)
                mult = rho[i]*rho[i]
                rms += mult
            rms /= 199
            rms = np.sqrt(rms)

            rms_rad = radial_profile(rms)
            rms_scaled = rms_rad/rho_amb[case]
        

            r_d = slc["r"]/slc["diameter"]

            dist = np.ceil(slc["y"]/slc["diameter"])
            ymax = max(np.max(rms_scaled), ymax)
        
            # Make plot
                                                                            
            p1 = plt.plot(r_d, rms_scaled, color=cmap[count%8], label="$T_0 = $"+leg_label, lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/d$")
            plt.ylabel(r"$\rho_{rms}^{\prime}/\overline{\rho}_{0}$")
            # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                    
            plt.axis([0,6,0,ymax+.325*ymax])
            # plt.text(.2, max(rms_scaled)+.325*max(rms_scaled)-.005, '$y/d = $'+str(dist))
            
            vals.append(rms_scaled)
            label.append(f"$T_0 = $" + leg_label)
            count += 1

        plt.text(.2, ymax+.255*ymax, '$y/d = $'+str(dist))
        plt.legend(loc='best', handlelength=3)
        # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"rho_pert_rad_"+str(lab)+".pdf"), format="pdf", dpi=300
        )
        plt.close("all")
        lab += 1

    
    count = 0
    lab = 0
    ymax = 0
    for sl in slices:
        for case in cases:
        
            fname = os.path.join(fdir, case+"_ambient", "slices_3", slices[0])
            slc = np.load(fname)

            if case == "same":
                leg_label = " 330"
            else:
                leg_label = " "+case

            cp = slc["cp_rad"]
            cp = cp*1E-7 # convert Erg to J
            # cp = cp*44.0095 # convert g^-1 to mol^-1

            r = slc["r"]/slc["diameter"]
            dist = np.ceil(slc["y"]/slc["diameter"])

            ymax = max(np.max(cp), ymax)

            p1 = plt.plot(r, cp, color=cmap[count%8], label="$T_0 = $"+leg_label, lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/d$")
            plt.ylabel(Titles["cp"])
            plt.axis([0,6,0,ymax+.25*ymax])
            vals.append(cp)
            label.append(f"T_0 = "+leg_label)

            count += 1

        plt.text(.2, ymax+.2*ymax, '$y/d = $'+str(dist))
        plt.legend(loc='best', handlelength=3)
        plt.savefig(
            os.path.join(odir, f"cp_rad_comp"+str(lab)+".pdf"), format="pdf", dpi=300
        )
        plt.close("all")
        lab += 1
