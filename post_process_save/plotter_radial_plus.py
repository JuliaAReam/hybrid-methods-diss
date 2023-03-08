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
        "--folders",
        dest="folder",
        help="Parent folder for all slice data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--slices",
        dest="slices",
        help="Slice inteval in axial direction; either 3 or 5",
        type=int,
        default=3,
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
    odir = os.path.abspath(args.output)
    
    # on Eagle
    # Pope_folder = "/home/jream/slice_test/Pope_Data"

    # on personal laptop
    Pope_folder = "Pope_Data"

    # Pope
    fnames_p1 = sorted(glob.glob(os.path.join(Pope_folder, "In_Scale_xd_*.csv")))
    fnames_p2 = sorted(glob.glob(os.path.join(Pope_folder, "Cent_Scale_xd_*.csv")))
    fnames_p3 = sorted(glob.glob(os.path.join(Pope_folder, "Rey_Str_*.csv")))
    
    dataset_p1 = [30, 60, 100]
    dataset_p2 = [40, 50, 60, 75, 97.5]
    dataset_p3 = ["Pope $<u^2>$", "Pope $<uv>$", "Pope $<v^2>$", "Pope $<w^2>$"]

    QoIs = ["temp_rad", "rho_rad", "cp_rad", "v_c_rad"]
    Titles_scaled ={"temp_rad": r"$\overline{T}^+=(\overline{T} - \overline{T}_{0})/(\overline{T}_{in} - \overline{T}_{0})$", "rho_rad": r"$\overline{\rho}^+=(\overline{\rho} - \overline{\rho}_{0})/(\overline{\rho}_{in} - \overline{\rho}_{0})$", "cp_rad": r"$\overline{c_p}^+ = (\overline{c_p} - \overline{c_p}_{in})/(\overline{c_p}_{0} - \overline{c_p}_{in})$", "v_c_rad": r"$v/v_{in}$" }
    Titles = {"temp_rad": "$\overline{T}$ (K)", "rho_rad": r"$\overline{\rho}$ (g/cm$^{3}$)", "cp_rad": "$\overline{c_p}$ (J/mol*K)", "v_c_rad": "$\overline{v}$ (cm/s)" }
    cases = ["314", "same", "350"]
    scale_fix = {"same": 200/202, "314": 1, "350": 200/203}
    
    T_amb = {"same": 329.9900916653577, "314": 313.99981302039595, "350": 350.01795947600795}
    rho_amb = {"same": 0.3019277174113144, "314": 0.5127411755780334, "350": 0.2256708202157278}
    cp_amb = {"same": 33891229.97658293, "314": 57029238.5417072, "350": 19269307.297705736} # in Erg/g*K

    inj = {"temp_rad": 329.9900916653577, "rho_rad": 0.3019277174113144, "cp_rad": 33891229.97658293*1E-7*44.01, "v_c_rad": 1800}

    fnames = ["avg_slice_0000.npz", "avg_slice_0001.npz", "avg_slice_0002.npz", "avg_slice_0003.npz", "avg_slice_0004.npz", "avg_slice_0005.npz"]
    fnames_favre = ["favre_avg_slice_0000.npz", "favre_avg_slice_0001.npz", "favre_avg_slice_0002.npz", "favre_avg_slice_0003.npz", "favre_avg_slice_0004.npz", "favre_avg_slice_0005.npz"]
    fnames_plus = ["avg_slice_0000_aQoI.npz", "avg_slice_0001_aQoI.npz", "avg_slice_0002_aQoI.npz", "avg_slice_0003_aQoI.npz", "avg_slice_0004_aQoI.npz", "avg_slice_0005_aQoI.npz"]
    
    if args.slices == 5:
        fnames += ["avg_slice_0006.npz", "avg_slice_0007.npz", "avg_slice_0008.npz", "avg_slice_0009.npz"]
        fnames_favre += ["favre_avg_slice_0006.npz", "favre_avg_slice_0007.npz", "favre_avg_slice_0008.npz", "favre_avg_slice_0009.npz"]
        fnames_plus += ["avg_slice_0006.npz_aQoI", "avg_slice_0007_aQoI.npz", "avg_slice_0008_aQoI.npz", "avg_slice_0009_aQoI.npz"]

    count = 0

    for fname in fnames:

        # cases for given slice
        fname_same = os.path.join(fdir, "same_ambient/slices_"+str(args.slices), fname)
        fname_350 = os.path.join(fdir, "350_ambient/slices_"+str(args.slices), fname)
        fname_314 = os.path.join(fdir, "314_ambient/slices_"+str(args.slices), fname)
        
        vals = []
        label = []

        slc314 = np.load(fname_314)
        slcsame = np.load(fname_same)
        slc350 = np.load(fname_350)
        
        # if args.slices == 3:
        #     fnames.pop(0)
        #     fnames.pop(0)
        # elif args.slices == 5:
        #     fnames.pop(0)
        
        # print("fnames are: ", fnames)

        cc = 0
        for QOI in QoIs:
            
            # for field in slc.iterkeys():
            #     print(field)
            
            Q314 = slc314[QOI]
            Q350 = slc350[QOI]
            Qsame = slcsame[QOI]

            max314 = np.max(Q314)
            max350 = np.max(Q350)
            max330 = np.max(Qsame)

            min314 = np.min(Q314)
            min350 = np.min(Q350)
            min330 = np.min(Qsame)

            ambient314 = Q314[-1]
            ambient350 = Q350[-1]
            ambient330 = Qsame[-1]

            center314 = Q314[0]
            center350 = Q350[0]
            center330 = Qsame[0]

            scaled314 = (Q314 - ambient314)/(inj[QOI] - ambient314)
            scaled350 = (Q350 - ambient350)/(inj[QOI] - ambient350)
            scaled330 = (Qsame - ambient330)/(inj[QOI] - ambient330)

            if QOI == "cp_rad":
                # convert everything to J/mol*K
                Q314 = Q314*1E-7*44.01
                Q350 = Q350*1E-7*44.01
                Qsame = Qsame*1E-7*44.01

                max314 = max314*1E-7*44.01
                max350 = max350*1E-7*44.01
                max330 = max330*1E-7*44.01

                ambient314 = ambient314*1E-7*44.01
                ambient350 = ambient350*1E-7*44.01
                ambient330 = ambient330*1E-7*44.01

                center314 = center314*1E-7*44.01
                center350 = center350*1E-7*44.01
                center330 = center330*1E-7*44.01

                # scaled314 = (Q314 - ambient314)/(inj[QOI] - ambient314)
                # scaled350 = (Q350 - ambient350)/(inj[QOI] - ambient350)
                # scaled330 = (Qsame - ambient330)/(inj[QOI] - ambient330)

                scaled314 = (Q314 - inj[QOI])/(ambient314 - inj[QOI])
                scaled350 = (Q350 - inj[QOI])/(ambient350 - inj[QOI])
                
            r_D = slc314["r"]/slc314["diameter"]
            l = slc314["y"]/slc314["diameter"]

            p1 = plt.figure(1)
            plt.plot(r_D, Q314, color=cmap[(cc+1)%8], dashes=dashseq[(cc+1)%7], label="$T_0 = 314$", lw=1)
            plt.plot(r_D, Q350, color=cmap[(cc+3)%8], dashes=dashseq[(cc+3)%7], label="$T_0 = 350$", lw=1)
            plt.xlabel("$r/D$")
            plt.ylabel(Titles[QOI])
            plt.xlim(0,4)
            # plt.axis([0,60,0,max(Temp_scaled)+.325*max(Temp_scaled)])
            # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))
            # vals.append(Q314)
            # label.append(f"$T_0 = $" + leg_label)
            
            plt.legend(loc='best', handlelength=3, title='$y/d=$ '+str(int(np.ceil(l))))
            plt.savefig(
                os.path.join(odir, QOI+f"_comp_{count:04d}.pdf"), format="pdf", dpi=300
            )
            plt.close("all")

            p2 = plt.figure(2)
            plt.plot(r_D, scaled314, color=cmap[(cc+1)%8], dashes=dashseq[(cc+1)%7], label="$T_0 = 314$", lw=1)
            # plt.plot(r_D, scaled330, color=cmap[(cc+2)%8], dashes=dashseq[(cc+2)%7], label="$T_0 = 330$", lw=1)
            plt.plot(r_D, scaled350, color=cmap[(cc+3)%8], dashes=dashseq[(cc+3)%7], label="$T_0 = 350$", lw=1)
            plt.xlabel("$r/D$")
            plt.ylabel(Titles_scaled[QOI])
            plt.xlim(0,4)
            # plt.axis([0,60,0,max(Temp_scaled)+.325*max(Temp_scaled)])
            # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))
            # vals.append(Q314)
            # label.append(f"$T_0 = $" + leg_label)
            
            plt.legend(loc='best', handlelength=3, title='$y/d=$ '+str(int(np.ceil(l))))
            plt.savefig(
                os.path.join(odir, QOI+f"_scaled_{count:04d}.pdf"), format="pdf", dpi=300
            )
            plt.close("all")

            cc += 1


        count += 1
