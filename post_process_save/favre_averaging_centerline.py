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
    fnames = sorted(glob.glob(os.path.join(fdir, "plt*_centerline.npz")))
    
    df = pd.DataFrame({"fname": fnames})
    df["step"] = df.fname.apply(
        lambda x: [int(x) for x in re.findall(r"\d+", os.path.basename(x))][0]
    )

    # Keep only last navg steps
    steps_to_keep = np.unique(df.step)[-args.navg :]
    df = df[df.step >= np.min(steps_to_keep)]
    step_scale = len(steps_to_keep)
    print(step_scale)

    fields = ["u_cl", "v_cl", "w_cl"]

    avgs = {}
    rho_avg = 0
    # Get <rho> and centerline measurements
    for fname in fnames:
        slc = np.load(fname, allow_pickle=True)
        y = slc["y"]
        ics = slc["ics"]
        v_in = slc["v_in"]
        rho_avg += slc["rho_cl"]
    rho_avg /= step_scale
    avgs["rho_cl"] = rho_avg 
        
    # Get average of centerlines: <rho*u>, <rho*v>, <rho*w>
    for field in fields:
        avg = 0
        for fname in fnames:
            slc = np.load(fname, allow_pickle=True)
            rho_quant = np.array(slc[field])*np.array(slc["rho_cl"]) 
            avg += rho_quant
        avg /= step_scale
        avgs[field] = avg

    # Get Favre averages: <rho*u>/<rho>, <rho*v>/<rho>, <rho*w>/<rho>
    u_barbar = np.array(avgs["u_cl"])/np.array(avgs["rho_cl"])
    v_barbar = np.array(avgs["v_cl"])/np.array(avgs["rho_cl"])
    w_barbar = np.array(avgs["w_cl"])/np.array(avgs["rho_cl"])

    # Get perturbations for slice at each time step, i.e. u'' = u - u_barbar
    # Then get weighted Reynolds averages, i.e. <rho*u''*u''> (needed for TKE_barbar and (u''u''_barbar)^1/2)
    wR_u_pert = 0
    wR_v_pert = 0
    wR_w_pert = 0
    for fname in fnames:
        slc = np.load(fname, allow_pickle=True)
        rho = np.array(slc["rho_cl"])
        u_pert = np.array(slc["u_cl"]) - u_barbar
        v_pert = np.array(slc["v_cl"]) - v_barbar
        w_pert = np.array(slc["w_cl"]) - w_barbar
        
        u_pert *= u_pert*rho
        v_pert *= v_pert*rho
        w_pert *= w_pert*rho
        
        wR_u_pert += u_pert
        wR_v_pert += v_pert
        wR_w_pert += w_pert
        
    wR_u_pert /= step_scale
    wR_v_pert /= step_scale
    wR_w_pert /= step_scale

    # Get TKE_barbar
    TKE_barbar = 0.5*(wR_u_pert + wR_v_pert + wR_w_pert)/np.array(avgs["rho_cl"])

    # Get u_fa_rms = <rho*u''*u''>/<rho>^1/2
    u_fa_rms = np.sqrt(wR_u_pert/np.array(avgs["rho_cl"]))
    v_fa_rms = np.sqrt(wR_v_pert/np.array(avgs["rho_cl"]))
    w_fa_rms = np.sqrt(wR_w_pert/np.array(avgs["rho_cl"]))

    uname = os.path.join(fdir, f"favre_avg_centerline")
    np.savez_compressed(
        uname,
        fdir=args.folder,
        steps=steps_to_keep,
        ics=ics,
        y=y,
        diameter=0.01,
        v_in=v_in,
        TKE_fa = TKE_barbar,
        u_fa = u_fa_rms,
        v_fa = v_fa_rms,
        w_fa = w_fa_rms
    )

    # Centerline plots for reference
    print(v_in)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    im0 = axs[0][0].plot(y/0.01, TKE_barbar/(np.power(v_in,2.0)))
    axs[0][0].set_title("Favre Averaged TKE along centerline")
    axs[0][0].set_xlabel("y/D")
    axs[0][0].set_ylabel("TKE_barbar")

    im1 = axs[0][1].plot(y/0.01, u_fa_rms/v_in)
    axs[0][1].set_title("Favre Averaged rms u perturbations")
    axs[0][1].set_xlabel("y/D")
    axs[0][1].set_ylabel("sqrt(FA(u''u''))")

    im2 = axs[1][0].plot(y/0.01, v_fa_rms/v_in)
    axs[1][0].set_title("Favre Averaged rms v perturbations")
    axs[1][0].set_xlabel("y/D")
    axs[1][0].set_ylabel("sqrt(FA(v''v''))")

    im3 = axs[1][1].plot(y/0.01, w_fa_rms/v_in)
    axs[1][1].set_title("Favre Averaged rms w perturbations")
    axs[1][1].set_xlabel("y/D")
    axs[1][1].set_ylabel("sqrt(FA(w''w''))")

    fig.suptitle("Favre Average Centerline Values")
    plt.savefig(os.path.join(fdir, "favre_centerline_check" + "_" + str(1)), dpi=300)
    plt.close("all")

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
    
