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

    QoIs = ["temp", "rho", "magvort", "cp", "pressure", "v"]
    same_QoIs = ["magvort", "pressure", "v"]
    Zoom = ["temp", "rho", "v", "cp"]
    Titles ={"temp": "$T$ (K)", "rho": r"$\rho$ (g/cm$^3$)", "magvort": "$|\omega|$ (1/s)", "cp": "$c_p$ (J/mol*K)", "pressure": "$p$ (MPa)", "v": "$v$ (cm/s)" }
    Titles_scaled ={"temp": r"$T^+ = \frac{T - T_{in}}{T_{0} - T_{in}}$", "rho": r"$\rho^+ = \frac{\rho - \rho_{in}}{\rho_{0} - \rho_{in}}$", "magvort": r"$|\omega|^{\dag} = \frac{|\omega| - |\omega|_{min}}{|\omega|_{max}-|\omega|_{min}}$", "cp": r"$c_p^+ = \frac{c_p - c_{p_{in}}}{c_{p_{0}} - c_{p_{in}}}$", "pressure": r"$p^{\dag} = \frac{p - p_{0}}{p_{max} - p_{0}}$", "v": r"$v/v_{in}$" }
    cases = ["314", "same", "350"]
    cmap_presets = {"temp": "plasma", "rho": "viridis", "magvort": "magma_r", "cp": "plasma", "pressure": "RdBu", "v": "cividis"}
    times = []
    all_times = {}
    for case in cases:
        if case == "same":
            t1 = "plt67808_vertical_slice.npz"
            t2 = "plt70526_vertical_slice.npz"
            t3 = "plt73243_vertical_slice.npz"
            t4 = "plt75358_vertical_slice.npz"
            times = [t1, t2, t3, t4] 
        elif case == "314":
            t1 = "plt77018_vertical_slice.npz"
            t2 = "plt79877_vertical_slice.npz"
            t3 = "plt82743_vertical_slice.npz"
            t4 = "plt85091_vertical_slice.npz"
            times =	[t1, t2, t3, t4]
        elif case == "350":
            t1 = "plt67788_vertical_slice.npz"
            t2 = "plt70508_vertical_slice.npz"
            t3 = "plt73229_vertical_slice.npz"
            t4 = "plt75344_vertical_slice.npz"
            times =	[t1, t2, t3, t4]
        all_times[case] = times

    # print(all_times)
    
    # Zooms in on jet edge to look at vortical structures better
    for case in cases:
        break
        for QoI in QoIs:
            

            fdir = os.path.abspath(args.folder)

            fnameA = os.path.join(fdir, case+"_ambient/vertical_slices", all_times[case][0])
            fnameB = os.path.join(fdir, case+"_ambient/vertical_slices", all_times[case][1])
            fnameC = os.path.join(fdir, case+"_ambient/vertical_slices", all_times[case][2])
            fnameD = os.path.join(fdir, case+"_ambient/vertical_slices", all_times[case][3])

            odir = os.path.abspath(args.output)

            slc1 = np.load(fnameA)
            slc2 = np.load(fnameB)
            slc3 = np.load(fnameC)
            slc4 = np.load(fnameD)

            QoI1 = slc1[QoI]
            QoI2 = slc2[QoI]
            QoI3 = slc3[QoI]
            QoI4 = slc4[QoI]

            # Note: Vertical slices are sideways for some reason. No clue how I did that on accident
            inflow_loc = QoI1.shape[1]//2

            QoI_inflow = QoI1[0][inflow_loc]
            QoI_ambient = QoI1[0][-1]

            QoI_inflow_check = QoI3[0][inflow_loc]
            QoI_ambient_check = QoI3[0][-1]


            if QoI == "pressure":
                QoI1 = 1e-7*QoI1
                QoI2 = 1e-7*QoI2
                QoI3 = 1e-7*QoI3
                QoI4 = 1e-7*QoI4

                QoI_inflow = QoI_inflow*1e-7
                QoI_ambient = QoI_ambient*1e-7

                minQoI1 = np.min(QoI1)
                maxQoI1 = np.max(QoI1)

                minQoI2 = np.min(QoI2)
                maxQoI2 = np.max(QoI2)
                
                minQoI3 = np.min(QoI3)
                maxQoI3 = np.max(QoI3)
                
                minQoI4 = np.min(QoI4)
                maxQoI4 = np.max(QoI4)

                QoI1_scaled = (QoI1 - QoI_ambient)/(maxQoI1 - QoI_ambient)
                QoI2_scaled = (QoI2 - QoI_ambient)/(maxQoI2 - QoI_ambient)
                QoI3_scaled = (QoI3 - QoI_ambient)/(maxQoI3 - QoI_ambient)
                QoI4_scaled = (QoI4 - QoI_ambient)/(maxQoI4 - QoI_ambient)

                clb_min_scaled = -1
                clb_max_scaled = 1.

                # QoI3 has weird blip of blowup for cp. don't know why
                clb_min = min(np.min(QoI1), np.min(QoI2), np.min(QoI4))
                clb_max = max(np.max(QoI1), np.max(QoI2), np.max(QoI4))


            elif QoI == "cp":
                QoI1 = QoI1*1E-7*44.01
                QoI2 = QoI2*1E-7*44.01
                QoI3 = QoI3*1E-7*44.01
                QoI4 = QoI4*1E-7*44.01

                QoI_inflow = QoI_inflow*1E-7*44.01
                QoI_ambient = QoI_ambient*1E-7*44.01
                QoI_inflow_check = QoI_inflow_check*1E-7*44.01
                QoI_ambient_check = QoI_ambient_check*1E-7*44.01

                minQoI = np.min(QoI4)
                maxQoI = np.max(QoI4)

                print("QoI_inflow is: ", QoI_inflow)
                print("QoI_ambient is: ", QoI_ambient)
                print()
                print(QoI_ambient, " compared to ", QoI_ambient_check)
                print(QoI_inflow, " compared to ", QoI_inflow_check)

                QoI1_scaled = (QoI1 - QoI_inflow)/(QoI_ambient - QoI_inflow)
                QoI2_scaled = (QoI2 - QoI_inflow)/(QoI_ambient - QoI_inflow)
                QoI3_scaled = (QoI3 - QoI_inflow)/(QoI_ambient - QoI_inflow)
                QoI4_scaled = (QoI4 - QoI_inflow)/(QoI_ambient - QoI_inflow)

                print()
                print(np.max(QoI3_scaled))
                print(np.min(QoI3_scaled))
                print()

                clb_min_scaled = min(np.min(QoI1_scaled), np.min(QoI2_scaled), np.min(QoI4_scaled))
                clb_max_scaled = max(np.max(QoI1_scaled), np.max(QoI2_scaled), np.max(QoI4_scaled))

                # QoI3 has weird blip of blowup for cp. don't know why
                clb_min = min(np.min(QoI1), np.min(QoI2), np.min(QoI4))
                clb_max = max(np.max(QoI1), np.max(QoI2), np.max(QoI4))


            elif QoI == "magvort":
                minQoI1 = np.min(QoI1)
                maxQoI1 = np.max(QoI1)

                minQoI2 = np.min(QoI2)
                maxQoI2 = np.max(QoI2)
                
                minQoI3 = np.min(QoI3)
                maxQoI3 = np.max(QoI3)
                
                minQoI4 = np.min(QoI4)
                maxQoI4 = np.max(QoI4)
                
                QoI1_scaled = (QoI1 - minQoI1)/(maxQoI1 - minQoI1)
                QoI2_scaled = (QoI2 - minQoI2)/(maxQoI2 - minQoI2)
                QoI3_scaled = (QoI3 - minQoI3)/(maxQoI3 - minQoI3)
                QoI4_scaled = (QoI4 - minQoI4)/(maxQoI4 - minQoI4)

                clb_min_scaled = min(np.min(QoI1_scaled), np.min(QoI2_scaled), np.min(QoI4_scaled))
                clb_max_scaled = max(np.max(QoI1_scaled), np.max(QoI2_scaled), np.max(QoI4_scaled))

                # QoI3 has weird blip of blowup for cp. don't know why
                clb_min = min(np.min(QoI1), np.min(QoI2), np.min(QoI4))
                clb_max = max(np.max(QoI1), np.max(QoI2), np.max(QoI4))

                
            else:
                QoI1_scaled = (QoI1 - QoI_inflow)/(QoI_ambient - QoI_inflow)
                QoI2_scaled = (QoI2 - QoI_inflow)/(QoI_ambient - QoI_inflow)
                QoI3_scaled = (QoI3 - QoI_inflow)/(QoI_ambient - QoI_inflow)
                QoI4_scaled = (QoI4 - QoI_inflow)/(QoI_ambient - QoI_inflow)

                clb_min_scaled = min(np.min(QoI1_scaled), np.min(QoI2_scaled), np.min(QoI4_scaled))
                clb_max_scaled = max(np.max(QoI1_scaled), np.max(QoI2_scaled), np.max(QoI4_scaled))

                # QoI3 has weird blip of blowup for cp. don't know why
                clb_min = min(np.min(QoI1), np.min(QoI2), np.min(QoI4))
                clb_max = max(np.max(QoI1), np.max(QoI2), np.max(QoI4))


    
            extents = slc1["extents"]/0.01
            # print(extents)

            # clb_min = np.min(QoI4)
            # clb_max = np.max(QoI4)
            print("************************")
            print("for QoI: ", QoI)
            print("max is: ", clb_max)
            print("min is: ", clb_min)
            print("************************")
            print()
            fig, axs = plt.subplots(1, 4, figsize=(15, 6))
            im0 = axs[0].imshow(QoI1, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QoI])
            axs[0].set_title("$t = 6.25$ ms")
            axs[0].set_xlabel("x/d")
            axs[0].set_ylabel("y/d")
            axs[0].set_xlim(0, 1.2)
            axs[0].set_ylim(3, 6)
            axs[0].set_box_aspect(1)

            im1 = axs[1].imshow(QoI2, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QoI])
            axs[1].set_title("$t = 6.50$ ms")
            axs[1].set_xlabel("x/d")
            # axs[1].set_ylabel("y")
            axs[1].yaxis.set_visible(False)
            axs[1].set_xlim(0, 1.2)
            axs[1].set_ylim(3, 6)
            axs[1].set_box_aspect(1)

            im2 = axs[2].imshow(QoI3, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QoI])
            axs[2].set_title("$t = 6.75$ ms")
            axs[2].set_xlabel("x/d")
            # axs[2].set_ylabel("y")
            axs[2].yaxis.set_visible(False)
            axs[2].set_xlim(0, 1.2)
            axs[2].set_ylim(3, 6)
            axs[2].set_box_aspect(1)
            # axs[2].set_aspect('equal')

            im3 = axs[3].imshow(QoI4, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QoI])
            axs[3].set_title("$t = 6.94$ ms")
            axs[3].set_xlabel("x/d")
            # axs[3].set_ylabel("y")
            axs[3].yaxis.set_visible(False)
            axs[3].set_xlim(0, 1.2)
            axs[3].set_ylim(3, 6)
            axs[3].set_box_aspect(1)
            # axs[3].set_aspect(abs((3-0)/(6-3)))

            clb = fig.colorbar(im3, fraction=0.035*6/15, ax=axs.ravel().tolist())
            clb.ax.set_ylabel(Titles[QoI], rotation=360, labelpad=25)

            plt.savefig(
	        os.path.join(odir, case+"_"+QoI+"_zoom.pdf"), format="pdf", dpi=300, bbox_inches='tight'
            )
            plt.close("all")

            # Scaled plots
            

            fig, axs = plt.subplots(1, 4, figsize=(15, 6))
            im0 = axs[0].imshow(QoI1_scaled, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QoI])
            axs[0].set_title("$t = 6.25$ ms")
            axs[0].set_xlabel("x/d")
            axs[0].set_ylabel("y/d")
            axs[0].set_xlim(0, 1.2)
            axs[0].set_ylim(3, 6)
            axs[0].set_box_aspect(1)

            im1 = axs[1].imshow(QoI2_scaled, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QoI])
            axs[1].set_title("$t = 6.50$ ms")
            axs[1].set_xlabel("x/d")
            # axs[1].set_ylabel("y")
            axs[1].yaxis.set_visible(False)
            axs[1].set_xlim(0, 1.2)
            axs[1].set_ylim(3, 6)
            axs[1].set_box_aspect(1)

            im2 = axs[2].imshow(QoI3_scaled, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QoI])
            axs[2].set_title("$t = 6.75$ ms")
            axs[2].set_xlabel("x/d")
            # axs[2].set_ylabel("y")
            axs[2].yaxis.set_visible(False)
            axs[2].set_xlim(0, 1.2)
            axs[2].set_ylim(3, 6)
            axs[2].set_box_aspect(1)

            im3 = axs[3].imshow(QoI4_scaled, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QoI])
            axs[3].set_title("$t = 6.94$ ms")
            axs[3].set_xlabel("x/d")
            # axs[3].set_ylabel("y")
            axs[3].yaxis.set_visible(False)
            axs[3].set_xlim(0, 1.2)
            axs[3].set_ylim(3, 6)
            axs[3].set_box_aspect(1)

            clb = fig.colorbar(im3, fraction=0.035*6/15, ax=axs.ravel().tolist())
            clb.ax.set_ylabel(Titles_scaled[QoI], rotation=360, labelpad=40)

            plt.savefig(
	        os.path.join(odir, case+"_"+QoI+"_zoom_scaled.pdf"), format="pdf", dpi=300, bbox_inches='tight'
            )
            plt.close("all")

        

    for QOI in QoIs:
        # break
        num_plots = 2
        fig_width = 8

        #if QOI in same_QoIs:
        #    num_plots = 3
        #    fig_width = 12

        # print(time)
        fdir = os.path.abspath(args.folder)
        
        fnameA = os.path.join(fdir, "314_ambient/vertical_slices", all_times["314"][3])
        fnameB = os.path.join(fdir, "350_ambient/vertical_slices", all_times["350"][3])
        fnameC = os.path.join(fdir, "same_ambient/vertical_slices", all_times["same"][3])
        
        odir = os.path.abspath(args.output)

        # print("314fname is: ", fnameA)
        # print("350fname is: ", fnameB)
        # print("samefname is: ", fnameC)
        
        slc314 = np.load(fnameA)
        slc350 = np.load(fnameB)
        slcsame = np.load(fnameC)

        print(slc314["x"].shape)
        print(slc314[QOI].shape)
        
        Q314 = slc314[QOI]
        Q350 = slc350[QOI]
        Qsame = slcsame[QOI]

        max314 = np.max(Q314)
        max350 = np.max(Q350)

        min314 = np.min(Q314)
        min350 = np.min(Q350)

        ambient314 = Q314[0][-1]
        ambient350 = Q350[0][-1]
        ambient330 = Qsame[0][-1]

        center314 = Q314[0][639]
        center350 = Q350[0][639]

        print("for QoI: ", QOI)
        print("ambient for 314 is: ", ambient314)
        print("ambient for 350 is: ", ambient350)
        print("ambient for same is: ", ambient330)

        scaled314 = (Q314 - center314)/(ambient314 - center314)
        scaled350 = (Q350 - center350)/(ambient350 - center350)

        if QOI == "pressure":
            Q314 = 1e-7*Q314
            Q350 = 1e-7*Q350
            Qsame = 1e-7*Qsame

            min314 = min314*1e-7
            max314 = max314*1e-7

            min350 = min350*1e-7
            max350 = max350*1e-7

            ambient314 = ambient314*1e-7
            ambient350 = ambient350*1e-7

            scaled314 = (Q314 - ambient314)/(max314 - ambient314)
            scaled350 = (Q350 - ambient350)/(max350 - ambient350)

            clb_min_scaled = -1.
            clb_max_scaled = 1.

            clb_min = min(np.min(Q314), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Q350))

            
        elif QOI == "cp":
            Q314 = Q314*1E-7*44.01
            Q350 = Q350*1E-7*44.01
            Qsame = Qsame*1E-7*44.01

            max314 = max314*1E-7*44.01
            max350 = max350*1E-7*44.01

            ambient314 = ambient314*1E-7*44.01
            ambient350 = ambient350*1E-7*44.01

            center314 = center314*1E-7*44.01
            center350 = center350*1E-7*44.01
            
            # scaled314 = (Q314 - ambient314)/(center314 - ambient314)
            # scaled350 = (Q350 - ambient350)/(center350 - ambient350)

            scaled314 = (Q314 - center314)/(ambient314 - center314)
            scaled350 = (Q350 - center350)/(ambient350 - center350)

            clb_min_scaled = min(np.min(scaled314), np.min(scaled350))
            clb_max_scaled = max(np.max(scaled314), np.max(scaled350))

            clb_min = min(np.min(Q314), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Q350))
            
        elif QOI == "magvort":
            scaled314 = (Q314 - min314)/(max314 - min314)
            scaled350 = (Q350 - min350)/(max350 - min350)

            clb_min = min(np.min(Q314), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Q350))

            clb_min_scaled = min(np.min(scaled314), np.min(scaled350))
            clb_max_scaled = max(np.max(scaled314), np.max(scaled350))

        else:    
        
            clb_min = min(np.min(Q314), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Q350))

            clb_min_scaled = min(np.min(scaled314), np.min(scaled350))
            clb_max_scaled = max(np.max(scaled314), np.max(scaled350))

        extents = slcsame["extents"]/0.01
        # extents_transpose = [0, 62.5, -12.5, 12.5]

            
        # if QOI == "pressure":
        #     clb_max = 10.15
        #     clb_min = 10.10
        # elif QOI == "magvort":
        #     clb_max = 1E6
        # elif QOI == "v":
        #     clb_max = 1500
        #     clb_min = 0
        # elif QOI == "cp":
        #     clb_max = 5

        print("inst min for ", QOI, ": ", clb_min)
        print("inst max for ", QOI, ": ", clb_max)

        fig, axs = plt.subplots(1, num_plots, figsize=(fig_width, 6))
        im0 = axs[0].imshow(Q314, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QOI])
        axs[0].set_title("$T_{0} = 314 K$")
        # axs[0].axes.xaxis.set_visible(False)
        # axs[0].axes.yaxis.set_visible(False)
        axs[0].set_xlabel("x/d")
        axs[0].set_ylabel("y/d")
        # clb0 = fig.colorbar(im0, ax=axs[0])
        # clb0.ax.set_title(Titles[QOI], loc="left", x=2.0)

        if num_plots == 3:

            im1 = axs[1].imshow(Qsame, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QOI])
            axs[1].set_title("$T_0 = 330 K$")
            axs[1].set_xlabel("x/d")
            axs[1].set_ylabel("y/d")
            # clb1 = fig.colorbar(im1, ax=axs[1])
            # clb1.ax.set_title(Titles[QOI], loc="left", x=2.0)

        im2 = axs[num_plots-1].imshow(Q350, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QOI])
        axs[num_plots-1].set_title("$T_{0} = 350 K$")
        axs[num_plots-1].set_xlabel("x/d")
        axs[num_plots-1].set_ylabel("y/d")
        # clb2 = fig.colorbar(im2, ax=axs[2])
        # clb2.ax.set_title(Titles[QOI], loc="left", x=2.0)

        clb = fig.colorbar(im2, ax=axs.ravel().tolist())
        clb.ax.set_ylabel(Titles[QOI], rotation=360, labelpad=25)

        # fig.suptitle("Average slice at y = {0:.6f}".format(int(np.ceil(l))))
        plt.savefig(
            os.path.join(odir, QOI+"_vert.pdf"), format="pdf", dpi=300, bbox_inches='tight' 
        )
        plt.close("all")
        
        
        fig, axs = plt.subplots(1, num_plots, figsize=(fig_width, 6))
        im0 = axs[0].imshow(scaled314, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QOI])
        axs[0].set_title("$T_{0} = 314 K$")
        # axs[0].axes.xaxis.set_visible(False)
        # axs[0].axes.yaxis.set_visible(False)
        axs[0].set_xlabel("x/d")
        axs[0].set_ylabel("y/d")
        # clb0 = fig.colorbar(im0, ax=axs[0])
        # clb0.ax.set_title(Titles[QOI], loc="left", x=2.0)

        if num_plots == 3:

            im1 = axs[1].imshow(Qsame, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QOI])
            axs[1].set_title("$T_0 = 330 K$")
            axs[1].set_xlabel("x/d")
            axs[1].set_ylabel("y/d")
            # clb1 = fig.colorbar(im1, ax=axs[1])
            # clb1.ax.set_title(Titles[QOI], loc="left", x=2.0)

        im2 = axs[num_plots-1].imshow(scaled350, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QOI])
        axs[num_plots-1].set_title("$T_{0} = 350 K$")
        axs[num_plots-1].set_xlabel("x/d")
        axs[num_plots-1].set_ylabel("y/d")
        # clb2 = fig.colorbar(im2, ax=axs[2])
        # clb2.ax.set_title(Titles[QOI], loc="left", x=2.0)

        clb = fig.colorbar(im2, ax=axs.ravel().tolist())
        clb.ax.set_ylabel(Titles_scaled[QOI], rotation=360, labelpad=50)

        # fig.suptitle("Average slice at y = {0:.6f}".format(int(np.ceil(l))))
        plt.savefig(
            os.path.join(odir, QOI+"_scaled_vert.pdf"), format="pdf", dpi=300, bbox_inches='tight' 
        )
        plt.close("all")

                                                                                                                                                                   

    for QOI in QoIs:
        break
        num_plots = 2
        fig_width = 8

        # if QOI in same_QoIs:
        #     num_plots = 3
        #     fig_width = 12
       
        # print(time)
       
        fdir = os.path.abspath(args.folder)

        fnameA = os.path.join(fdir, "314_ambient/vertical_slices/avg_vertical.npz")
        fnameB = os.path.join(fdir, "350_ambient/vertical_slices/avg_vertical.npz")
        fnameC = os.path.join(fdir, "same_ambient/vertical_slices/avg_vertical.npz")

        odir = os.path.abspath(args.output)

        # print("314fname is: ", fnameA)                                                                                                                                                                   
        # print("350fname is: ", fnameB)
        # print("samefname is: ", fnameC)         
        
        slc314 = np.load(fnameA)
        slc350 = np.load(fnameB)
        slcsame = np.load(fnameC)

        Q314 = slc314[QOI]
        Q350 = slc350[QOI]*200/203
        Qsame = slcsame[QOI]*200/202

        # if QOI == "temp":
        #     print("314 ambient temp = ", Q314[0][-1])
        #     print("350 ambient temp = ", Q350[0][-1])
        #     print("same ambient temp = ", Qsame[0][-1])

        # if QOI == "rho":
        #     print("314 ambient density = ", Q314[0][-1])
        #     print("350 ambient density = ", Q350[0][-1])
        #     print("same ambient density = ", Qsame[0][-1])

        # if QOI == "pressure":
        #     clb_max = 10.15
        #     clb_min = 10.10
        # elif QOI == "magvort":
        #     clb_max = 1E6
        # elif QOI == "v":
        #     clb_max = 1500
        #     clb_min = 0
        # elif QOI == "cp":
        #     clb_max = 5

        # Scaled averages
        max314 = np.max(Q314)
        max350 = np.max(Q350)

        min314 = np.min(Q314)
        min350 = np.min(Q350)

        ambient314 = Q314[0][-1]
        ambient350 = Q350[0][-1]

        center314 = Q314[0][639]
        center350 = Q350[0][639]

        print(center314)
        print(center350)

        scaled314 = (Q314 - center314)/(ambient314 - center314)
        scaled350 = (Q350 - center350)/(ambient350 - center350)

        if QOI == "pressure":

            Q314 = 1e-7*Q314
            Q350 = 1e-7*Q350
            Qsame = 1e-7*Qsame

            ambient314 = ambient314*1e-7
            ambient350 = ambient350*1e-7

            max314 = max314*1e-7
            max350 = max350*1e-7

            scaled314 = (Q314 - ambient314)/(max314 - ambient314)
            scaled350 = (Q350 - ambient350)/(max350 - ambient350)

            clb_min = min(np.min(Q314), np.min(Qsame), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Qsame), np.max(Q350))
            
            clb_min_scaled = -1.
            clb_max_scaled = 1.


        elif QOI == "cp":
            Q314 = Q314*1E-7*44.01
            Q350 = Q350*1E-7*44.01
            Qsame = Qsame*1E-7*44.01

            center314 = center314*1E-7*44.01
            center350 = center350*1E-7*44.01

            ambient314 = ambient314*1E-7*44.01
            ambient350 = ambient350*1E-7*44.01
            
            
            scaled314 = (Q314 - center314)/(ambient314 - center314)
            scaled350 = (Q350 - center350)/(ambient350 - center350)

            clb_min = min(np.min(Q314), np.min(Qsame), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Qsame), np.max(Q350))
            
            clb_min_scaled = min(np.min(scaled314), np.min(scaled350))
            clb_max_scaled = max(np.max(scaled314), np.max(scaled350))

            
        elif QOI == "magvort":

            scaled314 = (Q314 - min314)/(max314 - min314)
            scaled350 = (Q350 - min350)/(max350 - min350)

            clb_min = min(np.min(Q314), np.min(Qsame), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Qsame), np.max(Q350))

            clb_min_scaled = min(np.min(scaled314), np.min(scaled350))
            clb_max_scaled = max(np.max(scaled314), np.max(scaled350))

        else:

            clb_min = min(np.min(Q314), np.min(Qsame), np.min(Q350))
            clb_max = max(np.max(Q314), np.max(Qsame), np.max(Q350))

            clb_min_scaled = min(np.min(scaled314), np.min(scaled350))
            clb_max_scaled = max(np.max(scaled314), np.max(scaled350))


        extents = slcsame["extents"]/0.01
        # extents_transpose = [0, 62.5, -12.5, 12.5]

        
        fig, axs = plt.subplots(1, num_plots, figsize=(fig_width, 6))
        im0 = axs[0].imshow(Q314, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QOI])
        axs[0].set_title("$T_0 = 314 K$")
        axs[0].set_xlabel("x/d")
        axs[0].set_ylabel("y/d")
        # clb0 = fig.colorbar(im0, ax=axs[0])
        # clb0.ax.set_title(Titles[QOI], loc="left", x=2.0)

        if num_plots == 3:
        
            im1 = axs[1].imshow(Qsame, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QOI])
            axs[1].set_title("$T_0 = 330 K$")
            axs[1].set_xlabel("x/d")
            axs[1].set_ylabel("y/d")
            # clb1 = fig.colorbar(im1, ax=axs[1])
            # clb1.ax.set_title(Titles[QOI], loc="left", x=2.0)

        im2 = axs[num_plots-1].imshow(Q350, origin="lower", extent=extents, vmin=clb_min, vmax=clb_max, cmap=cmap_presets[QOI])
        axs[num_plots-1].set_title("$T_0 = 350 K$")
        axs[num_plots-1].set_xlabel("x/d")
        axs[num_plots-1].set_ylabel("y/d")
        # clb2 = fig.colorbar(im2, ax=axs[2])
        # clb2.ax.set_title(Titles[QOI], loc="left", x=2.0)

        clb = fig.colorbar(im2, ax=axs.ravel().tolist())
        clb.ax.set_ylabel(Titles[QOI], rotation=360, labelpad=25)

        plt.savefig(
            os.path.join(odir, QOI+"_vert_avg_lim.pdf"), format="pdf", dpi=300
        )
        plt.close("all")


        
        fig, axs = plt.subplots(1, num_plots, figsize=(fig_width, 6))
        im0 = axs[0].imshow(scaled314, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QOI])
        axs[0].set_title("$T_0 = 314 K$")
        axs[0].set_xlabel("x/d")
        axs[0].set_ylabel("y/d")
        # clb0 = fig.colorbar(im0, ax=axs[0])
        # clb0.ax.set_title(Titles[QOI], loc="left", x=2.0)

        if num_plots == 3:
        
            im1 = axs[1].imshow(scaledsame, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QOI])
            axs[1].set_title("$T_0 = 330 K$")
            axs[1].set_xlabel("x/d")
            axs[1].set_ylabel("y/d")
            # clb1 = fig.colorbar(im1, ax=axs[1])
            # clb1.ax.set_title(Titles[QOI], loc="left", x=2.0)

        im2 = axs[num_plots-1].imshow(scaled350, origin="lower", extent=extents, vmin=clb_min_scaled, vmax=clb_max_scaled, cmap=cmap_presets[QOI])
        axs[num_plots-1].set_title("$T_0 = 350 K$")
        axs[num_plots-1].set_xlabel("x/d")
        axs[num_plots-1].set_ylabel("y/d")
        # clb2 = fig.colorbar(im2, ax=axs[2])
        # clb2.ax.set_title(Titles[QOI], loc="left", x=2.0)

        clb = fig.colorbar(im2, ax=axs.ravel().tolist())
        clb.ax.set_ylabel(Titles_scaled[QOI], rotation=360, labelpad=40)

        plt.savefig(
            os.path.join(odir, QOI+"_scaled_vert_avg_lim.pdf"), format="pdf", dpi=300
        )
        plt.close("all")
