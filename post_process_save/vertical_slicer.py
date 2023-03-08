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
    args = parser.parse_args()

    # Setup
    ics = parse_ic(args.folder)
    fdirs = sorted(glob.glob(os.path.join(args.folder, "plt*")))
    # odir = os.path.join(args.folder, "slices")
    odir_v = "vertical_slices"
    if not os.path.exists(odir_v):
        os.makedirs(odir_v)
    odir_c = "centerline_slices"
    if not os.path.exists(odir_c):
        os.makedirs(odir_c)

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
                ds.domain_left_edge.d[1],
                ds.domain_right_edge.d[1],
            ]
        )

        v_in = ics["u_jet"]


        # slice parameters
        width = L[0]
        res = [N[0], N[1]]

        x_vert, y_vert = np.meshgrid(
            np.linspace(extents[0], extents[1], res[0]),
            np.linspace(extents[2], extents[3], res[1]),
        )

        y_centerline = np.linspace(extents[2], extents[3], res[1])

        eps = 1e-6
        
        # Set up centerline conditions
        start_cl = [eps, eps, eps]
        end_cl = [eps, extents[3], eps]
        ray = ds.r[start_cl:end_cl:N[1]*1j]
        srt = np.argsort(ray["y"])

        print("extents[3] is ", extents[3])

        # Extract centerline values
        u_cl = np.array(ray["velocity_x"][srt])
        v_cl = np.array(ray["velocity_y"][srt])
        w_cl = np.array(ray["velocity_z"][srt])
        y_cl = np.array(ray["y"][srt])
        temp_cl = np.array(ray["Temp"][srt])
        rho_cl = np.array(ray["density"][srt])
        cp_cl = np.array(ray["cp"][srt])
        pressure_cl = np.array(ray["pressure"][srt])

        # Save centerlines
        pfx_cl = f"plt{step:05d}_centerline"
        uname_cl = os.path.join(odir_c, pfx_cl)
        np.savez_compressed(
            uname_cl,
            fdir=args.folder,
            ics=ics,
            step=step,
            t=ds.current_time.d,
            u_cl=u_cl,
            v_cl=v_cl,
            w_cl=w_cl,
            y=y_cl,
            temp_cl=temp_cl,
            rho_cl=rho_cl,
            cp_cl=cp_cl,
            pressure_cl=pressure_cl,
            v_in=v_in,
            extents=extents,
        )

        # Take vertical slice for visualization
        print("... taking a vertical slice at z=0.0")
        slc = ds.slice("z", eps)
        frb = slc.to_frb(width, res, height=L[1])

        # Get arrays

        u_slc = np.array(frb["velocity_x"])
        v_slc = np.array(frb["velocity_y"])
        w_slc = np.array(frb["velocity_z"])
        temp_slc = np.array(frb["Temp"])
        rho_slc = np.array(frb["density"])
        cp_slc = np.array(frb["cp"])
        cv_slc = np.array(frb["cv"])
        magvort_slc = np.array(frb["magvort"])
        pressure_slc = np.array(frb["pressure"])
        gamma_slc = cp_slc / cv_slc


        # Save the slices
        pfx_v = f"plt{step:05d}_vertical_slice"
        uname = os.path.join(odir_v, pfx_v)
        np.savez_compressed(
            uname,
            fdir=args.folder,
            ics=ics,
            z=0.0,
            step=step,
            t=ds.current_time.d,
            u=u_slc,
            v=v_slc,
            w=w_slc,
            x=x_vert,
            y=y_vert,
            temp=temp_slc,
            rho=rho_slc,
            cp=cp_slc,
            cv=cv_slc,
            magvort=magvort_slc,
            pressure=pressure_slc,
            gamma=gamma_slc,
            v_in=v_in,
            extents=extents,
        )

        # Make a plot for reference

        fig, axs = plt.subplots(1, 3, figsize=(14, 6))
        im0 = axs[0].imshow(temp_slc, origin="lower", extent=extents)
        axs[0].set_title("temperature")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(rho_slc, origin="lower", extent=extents)
        axs[1].set_title("density")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        fig.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(magvort_slc, origin="lower", extent=extents)
        axs[2].set_title("|vorticity|")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        fig.colorbar(im2, ax=axs[2])

        fig.suptitle("Slice at z = 0.0")
        plt.savefig(os.path.join(odir_v, pfx_v + "_" + str(1)), dpi=300)
        plt.close("all")

        # Centerline plots for reference

        fig, axs = plt.subplots(1, 3, figsize=(14, 6))
        im0 = axs[0].plot(y_cl, temp_cl)
        axs[0].set_title("Temperature along centerline")
        axs[0].set_xlabel("y")
        axs[0].set_ylabel("T")

        im1 = axs[1].plot(y_cl, rho_cl)
        axs[1].set_title("Density along centerline")
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("rho")

        im2 = axs[2].plot(y_cl, cp_cl)
        axs[2].set_title("Specific Heat along centerline")
        axs[2].set_xlabel("y")
        axs[2].set_ylabel("cp")

        fig.suptitle("Centerline Values")
        plt.savefig(os.path.join(odir_c, pfx_cl + "_" + str(1)), dpi=300)
        plt.close("all")

        print("length of ray y: ", y_cl.shape)
        print("length of ray temp: ", temp_cl.shape)
        print("length of y_vert: ", y_vert.shape)
        print("length of x_vert: ", x_vert.shape)

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
