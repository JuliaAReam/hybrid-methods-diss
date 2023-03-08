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
    Titles_scaled ={"TKE": r"$\overline{\overline{TKE}}/v_{in}^2$", "u_fa": r"$ \left(\overline{\overline{u^{\prime\prime} u^{\prime\prime}}}\right)^{1/2}/v_{in}$", "v_fa": r"$ \left(\overline{\overline{v^{\prime\prime} v^{\prime\prime}}}\right)^{1/2}/v_{in}$", "w_fa": r"$ \left(\overline{\overline{w^{\prime\prime} w^{\prime\prime}}}\right)^{1/2}/v_{in}$"}
    Titles = {"TKE": "$\overline{T}$ (K)", "rho": r"$\overline{\rho}$ (g/cm$^{3}$)", "cp": "$\overline{c_p}$ (J/mol*K)", "v": "$\overline{v}$ (cm/s)" }
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

        fname = os.path.join(fdir, case+"_ambient", "centerline_slices/favre_avg_centerline.npz")
        slc = np.load(fname)

        if case == "same":
            leg_label = " 330"
        else:
            leg_label = " "+case

        TKE = slc["TKE_fa"]*scale_fix[case]
        v_in = slc["v_in"]

        u_fa = slc["u_fa"]*scale_fix[case]
        v_fa = slc["v_fa"]*scale_fix[case]
        w_fa = slc["w_fa"]*scale_fix[case]

        TKE_max = np.argmax(TKE)
        u_max = np.argmax(u_fa)
        v_max = np.argmax(v_fa)
        w_max = np.argmax(w_fa)

        centerline = slc["y"]/slc["diameter"]

        TKE_shift = centerline - centerline[TKE_max]
        u_shift = centerline - centerline[u_max]
        v_shift = centerline - centerline[v_max]
        w_shift = centerline - centerline[w_max]

        TKE_plot = TKE/(np.power(v_in, 2.0))
        u_plot = u_fa/v_in
        v_plot = v_fa/v_in
        w_plot = w_fa/v_in
        
        # Make plots including same_ambient case

        p1 = plt.figure(1)
        plt.plot(centerline, TKE_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["TKE"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.xlim(0,40)
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(TKE_plot)
        label.append(f"$T_0 = $" + leg_label)

        p2 = plt.figure(2)
        plt.plot(centerline, u_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["u_fa"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        # plt.xlim(0,40)
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(u_plot)
        label.append(f"$T_0 = $" + leg_label)

        p3 = plt.figure(3)
        plt.plot(centerline, v_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["v_fa"])
        # plt.axis([0,60,0,max(rho_scaled)+.25*max(rho_scaled)])
        vals.append(v_plot)
        label.append(f"T_0 = "+leg_label)

        p4 = plt.figure(4)
        plt.plot(centerline, w_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["w_fa"])
        # plt.axis([0,60,min(rho) - .25*min(rho),max(rho)+.25*max(rho)])
        vals.append(w_plot)
        label.append(f"T_0 = "+leg_label)

        p5 = plt.figure(5)
        plt.plot(TKE_shift, TKE_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["TKE"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        plt.xlim(TKE_shift[0],40)
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(TKE_plot)
        label.append(f"$T_0 = $" + leg_label)

        p6 = plt.figure(6)
        plt.plot(u_shift, u_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # plt.set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["u_fa"])
        # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)                                                                                                     
        # plt.axis([0,60,310,350])
        # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))                                                                                                                                               
        vals.append(u_plot)
        label.append(f"$T_0 = $" + leg_label)

        p7 = plt.figure(7)
        plt.plot(v_shift, v_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["v_fa"])
        # plt.axis([0,60,0,max(rho_scaled)+.25*max(rho_scaled)])
        vals.append(v_plot)
        label.append(f"T_0 = "+leg_label)

        p8 = plt.figure(8)
        plt.plot(w_shift, w_plot, color=cmap[count%8], dashes=dashseq[count%7], label="$T_0 = $"+leg_label, lw=1)
        # p1[0].set_dashes(dashseq[count%7])
        plt.xlabel("$y/d$")
        plt.ylabel(Titles_scaled["w_fa"])
        # plt.axis([0,60,min(rho) - .25*min(rho),max(rho)+.25*max(rho)])
        vals.append(w_plot)
        label.append(f"T_0 = "+leg_label)
        
        count += 1

    plt.figure(1)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"TKE_centerline.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(2)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"u_fa_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(3)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"v_fa_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(4)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"w_fa_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(5)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"TKE_shift_centerline.pdf"), format="pdf", dpi=300
    )
    
    plt.figure(6)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"u_fa_shift_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(7)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"v_fa_shift_centerline.pdf"), format="pdf", dpi=300
    )

    plt.figure(8)
    plt.legend(loc='best', handlelength=3)
    # plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
    plt.savefig(
        os.path.join(odir, f"w_fa_shift_centerline.pdf"), format="pdf", dpi=300
    )
    
    plt.close("all")
    
    count = 0
