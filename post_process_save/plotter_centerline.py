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

    QoIs = ["temp", "rho", "cp", "v"]
    Titles_scaled ={"temp": r"$\overline{T}^+=(\overline{T} - \overline{T}_{0})/(\overline{T}_{in} - \overline{T}_{0})$", "rho": r"$\overline{\rho}^+=(\overline{\rho} - \overline{\rho}_{0})/(\overline{\rho}_{in} - \overline{\rho}_{0})$", "cp": r"$\overline{c_p}^+ = (\overline{c_p} - \overline{c_p}_{in})/(\overline{c_p}_{0} - \overline{c_p}_{in})$", "v": r"$v/v_{in}$" }
    Titles = {"temp": "$\overline{T}$ (K)", "rho": r"$\overline{\rho}$ (g/cm$^{3}$)", "cp": "$\overline{c_p}$ (J/mol*K)", "v": "$\overline{v}$ (cm/s)" }
    cases = ["314", "same", "350"]
    scale_fix = {"same": 200/202, "314": 1, "350": 200/203}

    T_amb = {"same": 329.9900916653577, "314": 313.99981302039595, "350": 350.01795947600795}
    # T_amb = {"same": 330, "314": 314, "350": 350}
    rho_amb = {"same": 0.3019277174113144, "314": 0.5127411755780334, "350": 0.2256708202157278}
    cp_amb = {"same": 33891229.97658293, "314": 57029238.5417072, "350": 19269307.297705736} # in Erg/g*K

    odir = os.path.abspath(args.output)

    vals = []
    label = []
    count = 0
    
    for case in cases:

        fname = os.path.join(fdir, case+"_ambient", "centerline_slices/avg_centerline.npz")
        slc = np.load(fname)

        if case == "same":
            leg_label = " 330"
        else:
            leg_label = " "+case

        # for field in slc.keys():
        #     print(field)

        Temp = slc["temp_cl"]*scale_fix[case]
        Temp_scaled = (Temp - T_amb[case])/(Temp[0] - T_amb[case])

        rho = slc["rho_cl"]*scale_fix[case]
        rho_scaled = (rho - rho_amb[case])/(rho[0] - rho_amb[case])

        cp = slc["cp_cl"]*scale_fix[case]
        cp = cp*1E-7*44.01 # convert to J/mol*K
        cp_a = cp_amb[case]*1E-7*44.01
        cp_scaled = (cp - cp[0])/(cp_a - cp[0])

        v = slc["v_cl"]*scale_fix[case]
        # v_scaled = v/slc["v_in"]
        print(v[0])
        print(slc["v_in"])

        v_scaled = v/v[0]

        centerline = slc["y"]/slc["diameter"]
        
        # Make plots including same_ambient case

        p1 = plt.figure(1)
        plt.plot(centerline, Temp_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["temp"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.axis([0,60,0,max(Temp_scaled)+.325*max(Temp_scaled)])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(Temp_scaled)
        label.append(f"$T_0 = $" + leg_label)

        p2 = plt.figure(2)
        plt.plot(centerline, Temp, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["temp"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.axis([0,60,310,350])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(Temp)
        label.append(f"$T_0 = $" + leg_label)

        p3 = plt.figure(3)
        plt.plot(centerline, rho_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["rho"])
        plt.axis([0,60,0,max(rho_scaled)+.25*max(rho_scaled)])
        vals.append(rho_scaled)
        label.append(f"T_0 = "+leg_label)

        p4 = plt.figure(4)
        plt.plot(centerline, rho, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["rho"])
        # plt.axis([0,60,min(rho) - .25*min(rho),max(rho)+.25*max(rho)])
        vals.append(rho)
        label.append(f"T_0 = "+leg_label)

        p5 = plt.figure(5)
        plt.plot(centerline, cp_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["cp"])
        plt.axis([0,60,0,1.35])
        vals.append(cp_scaled)
        label.append(f"T_0 = "+leg_label)

        p6 = plt.figure(6)
        plt.plot(centerline, cp, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["cp"])
        # plt.axis([0,60,min(cp)-.25*min(cp),max(cp) + .25*max(cp)])
        vals.append(cp)
        label.append(f"T_0 = "+leg_label)

        p7 = plt.figure(7)
        plt.plot(centerline, v_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["v"])
        plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(v_scaled)
        label.append(f"T_0 = "+leg_label)

        p8 = plt.figure(8)
        plt.plot(centerline, v, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["v"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(v)
        label.append(f"T_0 = "+leg_label)
        
        count += 1

    plt.figure(1)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"temp_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(2)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"temp_centerline_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(3)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"rho_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(4)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"rho_centerline_same.pdf"), format="pdf", dpi=300
    )

    plt.figure(5)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"cp_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(6)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"cp_centerline_same.pdf"), format="pdf", dpi=300
    )

    plt.figure(7)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"v_centerline_scaled_same.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(8)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"v_centerline_same.pdf"), format="pdf", dpi=300
    )
    
    plt.close("all")

    cases.remove("same")

    print(cases)
    
    count = 0

    for case in cases:

        fname = os.path.join(fdir, case+"_ambient", "centerline_slices/avg_centerline.npz")
        slc = np.load(fname)

        if case == "same":
            leg_label = " 330"
        else:
            leg_label = " "+case

        # for field in slc.keys():
        #     print(field)

        Temp = slc["temp_cl"]*scale_fix[case]
        Temp_scaled = (Temp - T_amb[case])/(Temp[0] - T_amb[case])

        rho = slc["rho_cl"]*scale_fix[case]
        rho_scaled = (rho - rho_amb[case])/(rho[0] - rho_amb[case])

        cp = slc["cp_cl"]*scale_fix[case]
        cp = cp*1E-7*44.01 # convert to J/mol*K
        cp_a = cp_amb[case]*1E-7*44.01
        cp_scaled = (cp - cp[0])/(cp_a - cp[0])

        v = slc["v_cl"]*scale_fix[case]
        # v_scaled = v/slc["v_in"]
        print(v[0])
        print(slc["v_in"])

        v_scaled = v/v[0]

        centerline = slc["y"]/slc["diameter"]
        
        # Make plots including same_ambient case

        p1 = plt.figure(1)
        plt.plot(centerline, Temp_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["temp"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.axis([0,60,0,max(Temp_scaled)+.325*max(Temp_scaled)])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(Temp_scaled)
        label.append(f"$T_0 = $" + leg_label)

        p2 = plt.figure(2)
        plt.plot(centerline, Temp, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["temp"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.axis([0,60,310,350])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(Temp)
        label.append(f"$T_0 = $" + leg_label)

        p3 = plt.figure(3)
        plt.plot(centerline, rho_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["rho"])
        plt.axis([0,60,0,max(rho_scaled)+.25*max(rho_scaled)])
        vals.append(rho_scaled)
        label.append(f"T_0 = "+leg_label)

        p4 = plt.figure(4)
        plt.plot(centerline, rho, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["rho"])
        # plt.axis([0,60,min(rho) - .25*min(rho),max(rho)+.25*max(rho)])
        vals.append(rho)
        label.append(f"T_0 = "+leg_label)

        p5 = plt.figure(5)
        plt.plot(centerline, cp_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["cp"])
        plt.axis([0,60,0,1.35])
        vals.append(cp_scaled)
        label.append(f"T_0 = "+leg_label)

        p6 = plt.figure(6)
        plt.plot(centerline, cp, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["cp"])
        # plt.axis([0,60,min(cp)-.25*min(cp),max(cp) + .25*max(cp)])
        vals.append(cp)
        label.append(f"T_0 = "+leg_label)

        p7 = plt.figure(7)
        plt.plot(centerline, v_scaled, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["v"])
        plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(v_scaled)
        label.append(f"T_0 = "+leg_label)

        p8 = plt.figure(8)
        plt.plot(centerline, v, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles["v"])
        # plt.axis([0,60,0,max(v_scaled)+.25*max(v_scaled)])
        vals.append(v)
        label.append(f"T_0 = "+leg_label)
        
        count += 1

    plt.figure(1)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"temp_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(2)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"temp_centerline.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(3)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"rho_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(4)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"rho_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(5)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"cp_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(6)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"cp_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(7)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"v_centerline_scaled.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(8)
    plt.legend(loc='best', handlelength=3)
    plt.savefig(
        os.path.join(odir, f"v_centerline.pdf"), format="pdf", dpi=300
    )
    
    plt.close("all")

