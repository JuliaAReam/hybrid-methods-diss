#!/usr/bin/env python3
"""Get some averages
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time
from datetime import timedelta


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


def r_half(averages):
    compare = averages[1:]
    max_value = np.max(averages)
    min_value = np.min(averages)
    full_width = max_value-min_value
    r_val = (np.abs(compare - (min_value + 0.5*full_width))).argmin()
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
    parser = argparse.ArgumentParser(description="A simple averaging tool")
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        help="Folder containing slice files",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-n", 
        "--navg",
        dest="navg",
        help="Number of time steps to average over", 
        type=int, 
        default=1,
    )
    args = parser.parse_args()
        
    # Setup
    fdir = os.path.abspath(args.folder)
    fnames = sorted(glob.glob(os.path.join(fdir, "plt*_slice_*.npz")))
    df = pd.DataFrame({"fname": fnames})
    df["step"] = df.fname.apply(
        lambda x: [int(x) for x in re.findall(r"\d+", os.path.basename(x))][0]
    )
    df["slice"] = df.fname.apply(
        lambda x: [int(x) for x in re.findall(r"\d+", os.path.basename(x))][1]
    )

    # Keep only last navg steps                                                                                                                                                                         
    steps_to_keep = np.unique(df.step)[-args.navg :]
    df = df[df.step >= np.min(steps_to_keep)]
    step_scale = len(steps_to_keep)
    print(step_scale)

    fields = ["u_c", "v_c", "w_c"]

    for index, group in df.groupby("slice"):
        slc_num = group.slice.iloc[0]
        avgs = {}
        fnames_slice = group["fname"]
        print(slc_num)
        print()
    
        rho_avg = 0
        u_avg = 0
        v_avg = 0
        w_avg = 0
        # Get density weighted averages and extra quantities
        for fname in fnames_slice:
            slc = np.load(fname, allow_pickle=True)
            y = slc["y"]
            x = slc["x"]
            z = slc["z"]
            ics = slc["ics"]
            v_in = slc["v_in"]

            rho = np.array(slc["rho"])
            
            u_avg += np.array(slc["u_c"])*rho
            v_avg += np.array(slc["v_c"])*rho
            w_avg += np.array(slc["w_c"])*rho
            rho_avg += rho
            
        rho_avg /= step_scale
        u_avg /= step_scale
        v_avg /= step_scale
        w_avg /= step_scale

        # Get Favre averages: <rho*u>/<rho>, <rho*v>/<rho>, <rho*w>/<rho>
        u_barbar = u_avg/rho_avg
        v_barbar = v_avg/rho_avg
        w_barbar = w_avg/rho_avg

        # Get perturbations for slice at each time step, i.e. u'' = u - u_barbar
        # Then get weighted Reynolds averages, i.e. <rho*u''*u''> (needed for TKE_barbar and (u''u''_barbar)^1/2)
        wR_u_pert = 0
        wR_v_pert = 0
        wR_w_pert = 0
        wR_cross_pert = 0
        u_pert = 0
        v_pert = 0
        w_pert = 0
        cross_pert = 0
        for fname in fnames_slice:
            slc = np.load(fname, allow_pickle=True)
            rho = np.array(slc["rho"])
            u_prime = np.array(slc["u_c"]) - u_barbar
            v_prime = np.array(slc["v_c"]) - v_barbar
            w_prime = np.array(slc["w_c"]) - w_barbar

            # print(np.max(u_pert))
            # print(np.max(v_pert))
            # print(np.max(cross_pert))
            # print()
            u_pert = u_prime*u_prime*rho
            v_pert = v_prime*v_prime*rho
            w_pert = w_prime*w_prime*rho
            cross_pert = u_prime*v_prime*rho
            
            wR_u_pert += u_pert
            wR_v_pert += v_pert
            wR_w_pert += w_pert
            wR_cross_pert += cross_pert

        # print(np.max(u_pert))
        # print(np.max(v_pert))
        # print(np.max(cross_pert))
        # break    
        
        wR_u_pert /= step_scale
        wR_v_pert /= step_scale
        wR_w_pert /= step_scale
        wR_cross_pert /= step_scale

        
        # Get TKE_barbar
        # TKE_barbar = 0.5*(wR_u_pert + wR_v_pert + wR_w_pert)/np.array(avgs["rho"])

        # Get Favre Reynolds Stresses = <rho*u''*u''>/<rho>
        u_fa_rey = wR_u_pert/rho_avg
        v_fa_rey = wR_v_pert/rho_avg
        w_fa_rey = wR_w_pert/rho_avg
        cross_fa_rey = wR_cross_pert/rho_avg

        # Get radial distributions
        u_fa_r = radial_profile(u_fa_rey)
        v_fa_r = radial_profile(v_fa_rey)
        w_fa_r = radial_profile(w_fa_rey)
        cross_fa_r = radial_profile(cross_fa_rey)

        r = np.linspace(0, x[0][-1], len(u_fa_r))
        rmax = slc["rmax"]
        
        u_index = r_half(u_fa_r)
        rhalf_u = r[u_index]

        v_index = r_half(v_fa_r)
        rhalf_v = r[v_index]

        w_index = r_half(w_fa_r)
        rhalf_w = r[w_index]

        cross_index = r_half(cross_fa_r)
        rhalf_cross = r[cross_index]

        uname = os.path.join(fdir, f"favre_avg_slice_{slc_num:04d}")
        np.savez_compressed(
            uname,
            fdir=args.folder,
            steps=steps_to_keep,
            ics=ics,
            y=y,
            x=x,
            z=z,
            diameter=0.01,
            rmax = rmax,
            rhalf_u = rhalf_u,
            rhalf_v = rhalf_v,
            rhalf_w = rhalf_w,
            rhalf_cross = rhalf_cross,
            v_in=v_in,
            u_fa = u_fa_r,
            v_fa = v_fa_r,
            w_fa = w_fa_r,
            cross_fa = cross_fa_r
        )

        # plot for reference
        print(v_in)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        im0 = axs[0][0].plot(r/rhalf_u, u_fa_r/np.power(v_in,2.0))
        axs[0][0].set_title("Favre Reynolds Stress for u")
        axs[0][0].set_xlabel("r/r_half")
        axs[0][0].set_ylabel("<u''u''>/v_in^2")

        im1 = axs[0][1].plot(r/rhalf_v, v_fa_r/np.power(v_in,2.0))
        axs[0][1].set_title("Favre Reynolds Stress for v")
        axs[0][1].set_xlabel("r/r_half")
        axs[0][1].set_ylabel("<v''v''>/v_in^2")
        
        im2 = axs[1][0].plot(r/rhalf_w, w_fa_r/np.power(v_in,2.0))
        axs[1][0].set_title("Favre Reynolds Stress for w")
        axs[1][0].set_xlabel("r/r_half")
        axs[1][0].set_ylabel("<w''w''>/v_in^2")

        im3 = axs[1][1].plot(r/rhalf_cross, cross_fa_r/np.power(v_in,2.0))
        axs[1][1].set_title("Favre Reynolds Stress for uv")
        axs[1][1].set_xlabel("r/r_half")
        axs[1][1].set_ylabel("<u''v''>/v_in^2")

        fig.suptitle("Favre Average Reynolds Stress Values")
        plt.savefig(os.path.join(fdir, "favre_avg_check" + "_" + str(slc_num)), dpi=300)
        plt.close("all")

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
    
