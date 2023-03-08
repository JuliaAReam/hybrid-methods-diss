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
    parser.add_argument(
        "-t",
        "--slice_type",
        dest="slice_type",
        help="Type of slices being averaged. Either 'centerline' or 'axial'",
        type=str,
        default="axial",
    )
    args = parser.parse_args()

    if args.slice_type == 'centerline':
        
        # Setup
        fdir = os.path.abspath(args.folder)
        fnames = sorted(glob.glob(os.path.join(fdir, "plt*_aQoI.npz")))
    
        df = pd.DataFrame({"fname": fnames})
        df["step"] = df.fname.apply(
            lambda x: [int(x) for x in re.findall(r"\d+", os.path.basename(x))][0]
        )

        # Keep only last navg steps
        steps_to_keep = np.unique(df.step)[-args.navg :]
        df = df[df.step >= np.min(steps_to_keep)]

        fields = ["E", "Hi", "Cs", "Z", "alpha", "mu", "xi", "lam"]

        # Get average of centerlines
        avgs = {}
        for field in fields:
            avg = 0
            for fname in df["fname"]:
                slc = np.load(fname, allow_pickle=True)
                y = slc["y"]
                avg += slc[field]
            avg /= len(steps_to_keep)
            avgs[field] = avg

        uname = os.path.join(fdir, f"avg_centerline_aQoI")
        np.savez_compressed(
            uname,
            fdir=args.folder,
            steps=steps_to_keep,
            y=y,
            diameter=0.01,
            **avgs,
        )

        # Centerline plots for reference

        fig, axs = plt.subplots(1, 3, figsize=(14, 6))
        im0 = axs[0].plot(y, avgs["Z"])
        axs[0].set_title("Compressibility")
        axs[0].set_xlabel("y")
        axs[0].set_ylabel("Z")

        im1 = axs[1].plot(y, avgs["Hi"])
        axs[1].set_title("Enthalphy")
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("Hi")

        im2 = axs[2].plot(y, avgs["mu"])
        axs[2].set_title("Shear Viscosity")
        axs[2].set_xlabel("y")
        axs[2].set_ylabel("$\mu$")

        fig.suptitle("Average Centerline Values")
        plt.savefig(os.path.join(fdir, "cl_avg_check" + "_aQoI"), dpi=300)
        plt.close("all")

    elif args.slice_type == 'axial':

        # Setup
        fdir = os.path.abspath(args.folder)
        fnames = sorted(glob.glob(os.path.join(fdir, "plt*_aQoI.npz")))
        
        df = pd.DataFrame({"fname": fnames})
        df["step"] = df.fname.apply(
            lambda x: [int(x) for x in re.findall(r"\d+", os.path.basename(x))][0]
        )
    
        # Keep only last navg steps
        steps_to_keep = np.unique(df.step)[-args.navg :]
        df = df[df.step >= np.min(steps_to_keep)]
        
        fields = ["E", "Hi", "Cs", "Z", "alpha", "mu", "xi", "lam"]

        # Get average of vertical slices
        avgs = {}
        for field in fields:
            avg = 0
            for fname in df["fname"]:
                slc = np.load(fname, allow_pickle=True)
                extents = slc["extents"]
                avg += slc[field]
            avg /= len(steps_to_keep)
            avgs[field] = avg
            print("done averaging ",field)
        
        uname = os.path.join(fdir, f"avg_vertical")
        np.savez_compressed(
            uname,
            fdir=args.folder,
            steps=steps_to_keep,
            diameter=0.01,
            **avgs,
            extents=extents
        )

        # Make a plot for reference

        fig, axs = plt.subplots(1, 3, figsize=(14, 6))
        im0 = axs[0].imshow(avgs["mu"], origin="lower", extent=extents, cmap="turbo")
        axs[0].set_title("Sheer Viscosity")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(avgs["Hi"], origin="lower", extent=extents, cmap="turbo")
        axs[1].set_title("Enthalpy")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        fig.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(avgs["alpha"], origin="lower", extent=extents, cmap="turbo")
        axs[2].set_title("Thermal Diffusivity")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        fig.colorbar(im2, ax=axs[2])

        fig.suptitle("Slice at z = 0.0")
        plt.savefig(os.path.join(fdir, "avg_vertical_check" + "_" + str(1)), dpi=300)
        plt.close("all")

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
