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
    args = parser.parse_args()

    if args.slice_type == 'normal':
        
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

        fields = ["u_c", "v_c", "w_c", "temp", "rho", "cp", "cv", "gamma", "pressure"]

        for index, group in df.groupby("slice"):
            slc_num = group.slice.iloc[0]
            avgs = {}
            
            # Get averages for slice data
            for field in fields:
                avg = 0
                for index, row in group.iterrows():
                    slc = np.load(row.fname, allow_pickle=True)
                    x = slc["x"]
                    avg += slc[field]
                avg /= len(steps_to_keep)
                avgs[field] = avg

            # Get radial profile for averages
            avgs_rad = {}
            for key in avgs:
                new_key = key + "_rad"
                avgs_rad[new_key] = radial_profile(avgs[key])

            rms = {}
            # Get RMS averages for slice data
            for field in fields:
                avg = 0
                for index, row in group.iterrows():
                    slc = np.load(row.fname, allow_pickle=True)
                    avg += np.array(slc[field])*np.array(slc[field])
                avg /= len(steps_to_keep)
                new_key = field + "_rms"
                rms[new_key] = np.sqrt(avg)

            # Get radial profiles for RMS averages
            rms_rad = {}
            for key in rms:
                new_key = key + "_rad"
                rms_rad[new_key] = radial_profile(rms[key])

                
            rmax = slc["rmax"]
            r = np.linspace(0, x[0][-1], len(avgs_rad["u_c_rad"]))

            # Get scaling parameters
            centerline = {}
            r_h = {}
            for key in avgs_rad:
                new_key_cl = key + "_cl"
                new_key_r = key + "_r"
                centerline[new_key_cl] = avgs_rad[key][0]
                r_h_index = r_half(avgs_rad[key])
                r_h[new_key_r] = r[r_h_index]

            # Get perturbations for slice at each time step
            perts = {}
            for field in fields:
                pert = []
                new_key = field + "_pert"
                for index, row in group.iterrows():
                    slc = np.load(row.fname, allow_pickle=True)
                    pert.append(slc[field] - avgs[field])
                perts[new_key] = pert

            # Get averages of perturbations over time
            pert_avgs = {}
            for key in perts:
                avg = 0
                count = 0
                Pert_slice = perts[key]
                new_key = key + "_avg"
                for index, row in group.iterrows():
                    avg += Pert_slice[count]
                    count += 1
                avg /= len(steps_to_keep)
                pert_avgs[new_key] = avg

            # Get radial profiles of average perturbations over time
            pert_rad = {}
            for key in pert_avgs:
                new_key = key + "_rad"
                pert_rad[new_key] = radial_profile(pert_avgs[key])

            # Start RMS averaging of turbulent perturbation components (Reynold's stresses squared for velocity) 
            Rey = {}
            Rey["uu"] = np.array(perts["u_c_pert"]) * np.array(perts["u_c_pert"])
            Rey["vv"] = np.array(perts["v_c_pert"]) * np.array(perts["v_c_pert"])
            Rey["ww"] = np.array(perts["w_c_pert"]) * np.array(perts["w_c_pert"])
            Rey["uv"] = np.array(perts["u_c_pert"]) * np.array(perts["v_c_pert"])
            Rey["ptemp"] = np.array(perts["temp_pert"]) * np.array(perts["temp_pert"])
            Rey["prho"] = np.array(perts["rho_pert"]) * np.array(perts["rho_pert"])
            Rey["ppressure"] = np.array(perts["pressure_pert"]) * np.array(perts["pressure_pert"])
            Rey["pcp"] = np.array(perts["cp_pert"]) * np.array(perts["cp_pert"])
            Rey["pcv"] = np.array(perts["cv_pert"]) * np.array(perts["cv_pert"])
            Rey["pgamma"] = np.array(perts["gamma_pert"]) * np.array(perts["gamma_pert"])

            # Get Reynolds stresses
            Reynolds_fields = ["uu", "vv", "ww", "uv"]
            Reynolds_stress = {}
            for field in Reynolds_fields:
                avg = 0
                count = 0
                Reynolds_slice = Rey[field]
                for index, row in group.iterrows():
                    avg += Reynolds_slice[count]
                    count += 1
                avg /= len(steps_to_keep)
                Reynolds_stress[field] = avg

            # Get radial profiles of Reynolds stresses
            Reynolds_stress_rad = {}
            for key in Reynolds_stress:
                new_key = key + "_rad"
                Reynolds_stress_rad[new_key] = radial_profile(Reynolds_stress[key])
            
            # Complete RMS Averaging of Perturbation components
            pert_fields = ["uu", "vv", "ww", "ptemp", "prho", "ppressure", "pcp", "pcv", "pgamma"]
            pert_rms = {}
            for field in pert_fields:
                new_key = field + "_rms"
                avg = 0
                count = 0
                pert_slice = Rey[field]
                for index, row in group.iterrows():
                    avg += pert_slice[count]
                    count += 1
                avg /= len(steps_to_keep)
                pert_rms[new_key] = np.sqrt(avg)

            # Get radial profiles for RMS perturbations
            pert_rms_rad = {}
            for key in pert_rms:
                new_key = key + "_rad"
                pert_rms_rad[new_key] = radial_profile(pert_rms[key])

            uname = os.path.join(fdir, f"avg_slice_{slc_num:04d}")
            np.savez_compressed(
                uname,
                fdir=args.folder,
                step=df.step,
                ics=slc["ics"],
                y=slc["y"],
                slc=slc_num,
                x=slc["x"],
                z=slc["z"],
                rmax=slc["rmax"],
                r=r,
                diameter=slc["diameter"],
                v_in=slc["v_in"],
                **avgs,
                **avgs_rad,
                **rms,
                **rms_rad,
                **centerline,
                **r_h,
                **perts,
                **pert_avgs,
                **pert_rad,
                **Reynolds_stress,
                **Reynolds_stress_rad,
                **pert_rms,
                **pert_rms_rad,
            )

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
