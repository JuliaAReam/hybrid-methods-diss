#!/usr/bin/env python3
"""Get some slices
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
import yt
import glob
import time
from datetime import timedelta
import matplotlib.pyplot as plt


# ========================================================================
#
# Functions
#
# ========================================================================
def radial_profile(data):
    nx, ny = data.shape
    y, x = np.indices((data.shape))
    r = np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2).astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def perturbations(radialprofile, data):
    nx, ny = data.shape
    y, x = np.indices((data.shape))
    r = np.sqrt((x - nx // 2) ** 2 + (y - ny // 2) ** 2).astype(np.int)

    pert = np.empty([nx, ny])
    for index, value in enumerate(radialprofile):
        pert = np.where(r < index, pert, data - value)

    return pert


def stress(data1, data2):
    stress12 = data1 * data2

    return stress12


def r_half(averages):
    compare = averages[1:]
    centerline = averages[0]
    r_val = (np.abs(compare - 0.5 * centerline)).argmin()
    return r_val + 1


def cart_to_cyl(u, w, x, z):
    # (x,y,z) ---> (r,y,theta)
    # (u,v,w) ---> (u_c,v_c,w_c)
    u_c = (x * u + z * w) / (np.sqrt(x * x + z * z))
    w_c = (z * u - x * w) / (x * x + z * z) * np.sqrt(x * x + z * z)
    return u_c, w_c


# Make sure Prob_nd.F90 and probin.jet are in same file as plots
def parse_ic(fdir):

    ics = {"u_jet": 0.0, "r_jet": 0.0}

    # Load defaults
    fname = os.path.join(fdir, "Prob_nd.F90")
    try:
        with open(fname, "r") as f:
            for line in f:
                for key in ics:
                    if line.lstrip(" ").startswith(key):
                        ics[key] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
    except FileNotFoundError:
        pass

    # Load probin overrides
    pname = os.path.join(fdir, "probin.jet")
    with open(pname, "r") as f:
        for line in f:
            for key in ics:
                if line.lstrip(" ").startswith(key):
                    ics[key] = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])

    return ics


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple slicing tool")
    parser.add_argument(
        "-f",
        "--folder",
        dest="folder",
        help="Folder containing plot files",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-n",
        "--nslices",
        dest="nslices",
        help="Number of slices to take in x-direction",
        type=int,
        default=7,
    )
    parser.add_argument(
        "-ms",
        "--max_slice",
        dest="max_slice",
        help="Maximum distance y/D in axial direction to take final slice",
        type=int,
        default=18,
    )
    args = parser.parse_args()

    # Setup
    ics = parse_ic(args.folder)
    fdirs = sorted(glob.glob(os.path.join(args.folder, "plt*")))
    # odir = os.path.join(args.folder, "slices")
    slice_dist = args.max_slice/(args.nslices-1)
    label = int(slice_dist)
    odir = f"slices_{label:01d}"
    if not os.path.exists(odir):
        os.makedirs(odir)

    # Load the data
    for fdir in fdirs:

        print(f"Slicing {fdir}")
        ds = yt.load(fdir)
        step = int(re.search(r"\d+", os.path.basename(fdir)).group())
        max_level = ds.index.max_level
        # max_level = 0
        ref = int(np.product(ds.ref_factors[0:max_level]))
        L = (ds.domain_right_edge - ds.domain_left_edge).d
        N = ds.domain_dimensions * ref
        dxmin = ds.index.get_smallest_dx()
        # dxmin = L[0]/N[0]
        extents = np.array(
            [
                ds.domain_left_edge.d[0],
                ds.domain_right_edge.d[0],
                ds.domain_left_edge.d[2],
                ds.domain_right_edge.d[2],
            ]
        )
        rmax = 0.5 * np.sqrt(
            ((extents[1] - extents[0]) ** 2 + (extents[3] - extents[2]) ** 2)
        )
        diameter = 2.0 * ics["r_jet"]
        v_in = ics["u_jet"]


        # slice parameters
        width = L[0]
        res = [N[0], N[2]]
        x_slc, z_slc = np.meshgrid(
            np.linspace(extents[0], extents[1], res[0]),
            np.linspace(extents[2], extents[3], res[1]),
        )

        eps = 1e-6
        # Slice values; adjust for y/d=15,30,45,60
        slices = ds.domain_left_edge.d[1] + np.linspace(0 - eps, 0.01*args.max_slice - eps, args.nslices)
        slices = np.delete(slices, 0)
        

        # Take the slices
        for i, islice in enumerate(slices):

            # Take a slice
            print("... taking a slice at y = {0:f}".format(islice))
            # slc = ds.r[:, (islice, "cm"), :]
            slc = ds.slice("y", islice)
            frb = slc.to_frb(width, res)

            # Get arrays
            u_slc = np.array(frb["velocity_x"])
            v_slc = np.array(frb["velocity_y"])
            w_slc = np.array(frb["velocity_z"])
            temp_slc = np.array(frb["Temp"])
            rho_slc = np.array(frb["density"])
            cp_slc = np.array(frb["cp"])
            cv_slc = np.array(frb["cv"])
            pressure_slc = np.array(frb["pressure"])
            gamma_slc = cp_slc / cv_slc

            # Get cylindrical velocities (x,y,z)--->(r,y,theta)
            u_c, w_c = cart_to_cyl(u_slc, w_slc, x_slc, z_slc)
            v_c = v_slc

            # Save the slices
            pfx = f"plt{step:05d}_slice_{i:04d}"
            uname = os.path.join(odir, pfx)
            np.savez_compressed(
                uname,
                fdir=args.folder,
                ics=ics,
                y=islice,
                step=step,
                t=ds.current_time.d,
                u=u_slc,
                v=v_slc,
                w=w_slc,
                x=x_slc,
                z=z_slc,
                u_c=u_c,
                v_c=v_c,
                w_c=w_c,
                temp=temp_slc,
                rho=rho_slc,
                cp=cp_slc,
                cv=cv_slc,
                pressure=pressure_slc,
                gamma=gamma_slc,
                diameter=diameter,
                v_in=v_in,
                rmax=rmax,
                extents=extents,
            )

            # Make a plot for reference

            fig, axs = plt.subplots(1, 3, figsize=(14, 6))
            im0 = axs[0].imshow(temp_slc, origin="lower", extent=extents)
            axs[0].set_title("temperature")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("z")
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(rho_slc, origin="lower", extent=extents)
            axs[1].set_title("density")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("z")
            fig.colorbar(im1, ax=axs[1])

            im2 = axs[2].imshow(cp_slc, origin="lower", extent=extents)
            axs[2].set_title("specific heat")
            axs[2].set_xlabel("x")
            axs[2].set_ylabel("z")
            fig.colorbar(im2, ax=axs[2])

            fig.suptitle("Slice at y = {0:.6f}".format(islice))
            plt.savefig(os.path.join(odir, pfx + "_" + str(1)), dpi=300)
            plt.close("all")

            # Everything in cylindrical coordinates (r,y,theta) --> (u_c,v_c,w_c)
            fig, axs = plt.subplots(1, 3, figsize=(14, 6))
            im0 = axs[0].imshow(u_c, origin="lower", extent=extents)
            axs[0].set_title("u")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("z")
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(v_c, origin="lower", extent=extents)
            axs[1].set_title("v")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("z")
            fig.colorbar(im1, ax=axs[1])

            im2 = axs[2].imshow(w_c, origin="lower", extent=extents)
            axs[2].set_title("w")
            axs[2].set_xlabel("x")
            axs[2].set_ylabel("z")
            fig.colorbar(im2, ax=axs[2])

            fig.suptitle("Slice at y = {0:.6f}".format(islice))
            plt.savefig(os.path.join(odir, pfx + "cyl_" + str(1)), dpi=300)
            plt.close("all")

    #            p1 = plt.imshow(v_slc, extent=extents)
    #            plt.xlabel("x")
    #            plt.ylabel("z")
    #            plt.title("v")
    #            plt.colorbar()

    #            plt.savefig(os.path.join(odir, pfx + "_" + str(1)), dpi=300)
    #            plt.close("all")

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
