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
        help="Type of slices being averaged. Either 'centerline', 'axial', or 'normal'",
        type=str,
        default="normal",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Folder to output averages to",
        type=str,
        default=".",
    )
    args = parser.parse_args()

    if args.slice_type == 'normal':
        
        # Setup
        fdir = os.path.abspath(args.folder)
        odir = os.path.abspath(args.output)
        fnames = sorted(glob.glob(os.path.join(fdir, "plt*_slice_*_aQoI.npz")))
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

        fields = ["E", "Hi", "mu", "xi", "lam", "alpha", "Z", "Cs"]

        for index, group in df.groupby("slice"):
            slc_num = group.slice.iloc[0]
            print("on slice:", slc_num)
            avgs = {}
            
            # Get averages for slice data
            for field in fields:
                avg = 0
                for index, row in group.iterrows():
                    slc = np.load(row.fname, allow_pickle=True)
                    extents = slc["extents"]
                    x = slc["x"]
                    avg += slc[field]
                avg /= len(steps_to_keep)
                avgs[field] = avg
                print("done with", field)
            print()

            # Get radial profile for averages
            avgs_rad = {}
            for key in avgs:
                new_key = key + "_rad"
                avgs_rad[new_key] = radial_profile(avgs[key])

            r = np.linspace(0, x[0][-1], len(avgs_rad["E_rad"]))
            
            # Get scaling parameters
            centerline = {}
            r_h = {}
            for key in avgs_rad:
                new_key_cl = key + "_cl"
                new_key_r = key + "_r"
                centerline[new_key_cl] = avgs_rad[key][0]
                r_h_index = r_half(avgs_rad[key])
                r_h[new_key_r] = r[r_h_index]
                
            uname = os.path.join(fdir, f"avg_slice_{slc_num:04d}_aQoI")
            np.savez_compressed(
                uname,
                fdir=args.folder,
                step=df.step,
                slc=slc_num,
                x=slc["x"],
                z=slc["z"],
                r=r,
                extents=extents,
                **avgs,
                **avgs_rad,
                **centerline,
                **r_h,
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

            fig.suptitle("Slice at y = "+str(slc_num))
            plt.savefig(os.path.join(fdir, f"avg_normal_check_{slc_num:04d}"), dpi=300)
            plt.close("all")


    
    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
