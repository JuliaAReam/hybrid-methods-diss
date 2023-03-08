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
import imageio
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
        "-t",
        "--slice_type",
        dest="slice_type",
        help="Type of slices being averaged. Either 'centerline', 'axial', or 'normal'",
        type=str,
        default="normal",
    )
    args = parser.parse_args()

    if args.slice_type == 'centerline':
        
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

        fields = ["u_cl", "v_cl", "w_cl", "temp_cl", "rho_cl", "cp_cl", "pressure_cl"]

        # Get average of centerlines
        avgs = {}
        for field in fields:
            avg = 0
            for fname in fnames:
                slc = np.load(fname, allow_pickle=True)
                y = slc["y"]
                ics = slc["ics"]
                v_in = slc["v_in"]
                avg += slc[field]
            avg /= len(steps_to_keep)
            avgs[field] = avg

        # Get RMS averages for slice data
        rms = {}
        for field in fields:
            avg = 0
            for fname in fnames:
                slc = np.load(fname, allow_pickle=True)
                avg += np.array(slc[field])*np.array(slc[field])
            avg /= len(steps_to_keep)
            new_key = field + "_rms"
            rms[new_key] = np.sqrt(avg)

        # Get perturbations for slice at each time step
        perts = {}
        for field in fields:
            pert = []
            new_key = field + "_pert"
            for fname in fnames:
                slc = np.load(fname, allow_pickle=True)
                pert.append(slc[field] - avgs[field])
            perts[new_key] = pert

        # Get averages of perturbations over time
        pert_avgs = {}
        for key in perts:
            avg = 0
            count = 0
            Pert_slice = perts[key]
            new_key = key + "_avg"
            for fname in fnames:
                avg += Pert_slice[count]
                count += 1
            avg /= len(steps_to_keep)
            pert_avgs[new_key] = avg

        # Start RMS averaging of turbulent perturbation components (Reynold's stresses squared for velocity) 
        Rey = {}
        Rey["uu"] = np.array(perts["u_cl_pert"]) * np.array(perts["u_cl_pert"])
        Rey["vv"] = np.array(perts["v_cl_pert"]) * np.array(perts["v_cl_pert"])
        Rey["ww"] = np.array(perts["w_cl_pert"]) * np.array(perts["w_cl_pert"])
        Rey["ptemp"] = np.array(perts["temp_cl_pert"]) * np.array(perts["temp_cl_pert"])
        Rey["prho"] = np.array(perts["rho_cl_pert"]) * np.array(perts["rho_cl_pert"])
        Rey["ppressure"] = np.array(perts["pressure_cl_pert"]) * np.array(perts["pressure_cl_pert"])
        Rey["pcp"] = np.array(perts["cp_cl_pert"]) * np.array(perts["cp_cl_pert"])

            
        # Complete RMS Averaging of Perturbation components
        pert_fields = ["uu", "vv", "ww", "ptemp", "prho", "ppressure", "pcp"]
        pert_rms = {}
        for field in pert_fields:
            new_key = field + "_rms"
            avg = 0
            count = 0
            pert_slice = Rey[field]
            for fname in fnames:
                avg += pert_slice[count]
                count += 1
            avg /= len(steps_to_keep)
            pert_rms[new_key] = np.sqrt(avg)

        uname = os.path.join(fdir, f"avg_centerline")
        np.savez_compressed(
            uname,
            fdir=args.folder,
            steps=steps_to_keep,
            ics=ics,
            y=y,
            diameter=0.01,
            v_in=v_in,
            **avgs,
            **rms,
            **perts,
            **pert_avgs,
            **pert_rms,
        )

        # Centerline plots for reference

        fig, axs = plt.subplots(1, 3, figsize=(14, 6))
        im0 = axs[0].plot(y, avgs["temp_cl"])
        axs[0].set_title("Temperature along centerline")
        axs[0].set_xlabel("y")
        axs[0].set_ylabel("T")

        im1 = axs[1].plot(y, avgs["rho_cl"])
        axs[1].set_title("Density along centerline")
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("rho")

        im2 = axs[2].plot(y, avgs["cp_cl"])
        axs[2].set_title("Specific Heat along centerline")
        axs[2].set_xlabel("y")
        axs[2].set_ylabel("cp")

        fig.suptitle("Centerline Values")
        plt.savefig(os.path.join(fdir, "centerline_check" + "_" + str(1)), dpi=300)
        plt.close("all")


    elif args.slice_type == 'axial':

        # Setup
        fdir = os.path.abspath(args.folder)
        fnames = sorted(glob.glob(os.path.join(fdir, "plt*_vertical_slice.npz")))

        # Get average of centerlines
        for fname in fnames:
            break
            slc = np.load(fname, allow_pickle=True)
            temp = np.transpose(slc["temp"])
            rho = slc["rho"]
            cp = slc["cp"]
            x = slc["x"]
            y = slc["y"]
            ics = slc["ics"]
            v_in = slc["v_in"]
            extents = slc["extents"]
            print("extents are: ", extents)

            # Make a plot for reference
            im0 = plt.imshow(temp, origin="lower")
            plt.axis('off')

            plt.savefig(os.path.join(fdir, str(slc["step"])+"_vertical_temp"), dpi=300, transparent=True, bbox_inches='tight')
            plt.close("all")

        gif_pngs = sorted(glob.glob(os.path.join(fdir, "*_vertical_temp.png")))
        gif_name = "vert_temp"
            
        with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
            for gif_png in gif_pngs:
                image = imageio.imread(gif_png)
                writer.append_data(image)
            
    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
