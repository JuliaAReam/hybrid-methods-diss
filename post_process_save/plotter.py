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
        dest="folders",
        help="Folders containing average slice data",
        type=str,
        required=True,
        nargs="+",
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
    # on Eagle
    # Pope_folder = "/home/jream/slice_test/Pope_Data"

    # on personal laptop
    Pope_folder = "Pope_Data"
    
    dataset_p1 = [30, 60, 100]
    dataset_p2 = [40, 50, 60, 75, 97.5]
    dataset_p3 = ["Pope $<u^2>$", "Pope $<uv>$", "Pope $<v^2>$", "Pope $<w^2>$"]

    for i, folder in enumerate(args.folders):
        fdir = os.path.abspath(folder)
        fnames = sorted(glob.glob(os.path.join(fdir, "avg_slice_*.npz")))
        fnames_p1 = sorted(glob.glob(os.path.join(Pope_folder, "In_Scale_xd_*.csv")))
        fnames_p2 = sorted(glob.glob(os.path.join(Pope_folder, "Cent_Scale_xd_*.csv")))
        fnames_p3 = sorted(glob.glob(os.path.join(Pope_folder, "Rey_Str_*.csv")))
        count = 0
        vals = []
        label = []
        l = 0
        
        odir = os.path.abspath(args.output)

        if args.slices == 3:
            fnames.pop(0)
            fnames.pop(0)
        elif args.slices == 5:
            fnames.pop(0)
        
        print("fnames are: ", fnames)

        for fname in fnames:
            slc = np.load(fname)
            
            # for field in slc.iterkeys():
            #     print(field)
            
            l = slc["y"]/slc["diameter"]

            # Plot normalized radial temperature profiles
            Temp = slc["temp_rad"]
            Temp_scaled = (Temp-Temp[-1])/(Temp[0]-Temp[-1])

            print(Temp[-1])

            
            r_half_index = r_half_raw(Temp_scaled)
            r_half = slc["r"][r_half_index]
            r_rhalf = slc["r"]/(r_half)

            # Make ploty
            p1 = plt.plot(r_rhalf, Temp_scaled, color=cmap[count%8], label="y/d = " + str(int(np.ceil(l))), lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/r_{1/2}$")
            plt.ylabel("$T^*=(\overline{T} - \overline{T}_{0})/(\overline{T}_c - \overline{T}_{0})$")
            # plt.title("Normalized Radial Profiles of Temperature in the Turbulent Round Jet", y=1.08)
            plt.axis([0,4,0,1.25])
            # plt.text(1, 1, '$T_{\infty} = $'+str(Temp[-1]))
            vals.append(Temp_scaled)
            label.append(f"y/d =" + str(l))
            count += 1
            if count==6:
                break
        plt.legend(loc='best', handlelength=5)
        plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"r_vs_temp.pdf"), format="pdf", dpi=300
        )
        plt.close("all")

        
        count = 0
        for fname in fnames:
            slc = np.load(fname)

            l = slc["y"]/slc["diameter"]
            Temp = slc["temp_rad"]

            # Plot normalized radial temperature profiles
            
            v_rad = slc["v_c_rad"]
            v_rad_scaled = (v_rad-v_rad[-1])/(v_rad[0]-v_rad[-1])

            r_half_index = r_half_raw(v_rad_scaled)
            r_half = slc["r"][r_half_index]

            r_rhalf = slc["r"]/r_half
            
            # Make plot
            
            p1 = plt.plot(r_rhalf, v_rad_scaled, color=cmap[count%8], label="y/d =" + str(int(np.ceil(l))), lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/r_{1/2}$")
            plt.ylabel("$v^*=(\overline{v} - \overline{v}_{0})/(\overline{v}_c - \overline{v}_{0})$")
            # plt.title("Normalized Radial Profiles of Axial Velocity in the Turbulent Round Jet", y=1.08)
            plt.axis([0,4,0,1.25])
            
            vals.append(v_rad_scaled)
            label.append(f"y/d =" + str(l))
            count += 1
            if count==6:
                break
        plt.legend(loc='best', handlelength=5)
        plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"r_vs_v.pdf"), format="pdf", dpi=300
        )
        plt.close("all")

        count = 0
        for fname in fnames:
            slc = np.load(fname)
            
            l = slc["y"]/slc["diameter"]
            Temp = slc["temp_rad"]

            # Plot normalized radial temperature profiles
            
            rho = slc["rho_rad"]
            rho_scaled = (rho-rho[-1])/(rho[0]-rho[-1])

            r_half_index = r_half_raw(rho_scaled)
            r_half = slc["r"][r_half_index]
            r_rhalf = slc["r"]/r_half

            # Make plot
            
            p1 = plt.plot(r_rhalf, rho_scaled, color=cmap[count%8], label="y/d =" + str(int(np.ceil(l))), lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/r_{1/2}$")
            plt.ylabel(r"$\rho^*=(\overline{\rho} - \overline{\rho}_{0})/(\overline{\rho}_c - \overline{\rho}_{0})$")
            # plt.title("Normalized Radial Profiles of Density in the Turbulent Round Jet", y=1.08)
            plt.axis([0,4,0,1.25])
            
            vals.append(rho_scaled)
            label.append(f"y/d =" + str(l))
            count += 1
            if count==6:
                break
        plt.legend(loc='best', handlelength=5)
        plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"r_vs_rho.pdf"), format="pdf", dpi=300
        )
        plt.close("all")
        
        count = 0
        for fname in fnames:

            slc = np.load(fname)
            l = slc["y"]/slc["diameter"]
            Temp = slc["temp_rad"]

            # Plot radial cp profiles
            Cp_r = slc["cp_rad"]
            Cp_r_scaled = (Cp_r-Cp_r[-1])/(Cp_r[0]-Cp_r[-1])

            r_half_index = r_half_raw(Cp_r_scaled)
            r_half = slc["r"][r_half_index]

            r_rhalf = slc["r"]/r_half

            # Make plot
            p1 = plt.plot(r_rhalf, Cp_r_scaled, color=cmap[count%8], label="y/d =" + str(int(np.ceil(l))), lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/r_{jet}$")
            plt.ylabel("$c_p^*=(\overline{c_p} - \overline{c_p}_{0})/(\overline{c_p}_c - \overline{c_p}_{0})$")
            # plt.title("Normalized Radial Profiles of Specific Heat in the Turbulent Round Jet", y=1.08)
            plt.axis([0,4,0,1.25])
            vals.append(Cp_r_scaled)
            label.append(f"y/d =" + str(l))
            count += 1
            if count==6:
                break
        plt.legend(loc='best', handlelength=5)
        plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"r_vs_cp.pdf"), format="pdf", dpi=300
        )
        plt.close("all")


        count = 0
        for fname in fnames:
            slc = np.load(fname)
            l = slc["y"]/slc["diameter"]
            Temp = slc["temp_rad"]

            # Scale parameters of interest to match Pope scalings/labels
            r_d = slc["r"]/slc["diameter"]
            vr_vin = slc["v_c_rad"]/slc["v_in"]

            # Make plot
            p1 = plt.plot(r_d, vr_vin, color=cmap[count%8], label="y/d =" + str(int(np.ceil(l))), lw=1)
            p1[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/d$")
            plt.ylabel("$<V>/V_{in}$")
            # plt.title("Radial Profiles of Mean Axial Velocity in the Turbulent Round Jet")
            plt.axis([0,6,0,0.6])
            vals.append(vr_vin)
            label.append(f"y/d =" + str(l))
            count += 1
            if count==6:
                break

        # count = 0
        # for fname_p in fnames_p1:
            # slc = pd.read_csv(fname_p, sep=', ')
            # l = dataset_p1[count]

            # Scale parameters of interest to match Pope scalings/labels                                                                                                                                  
            # r_d = slc["r_d"]
            # vr_vin = slc["vr_vin"]

            # Make plot                                                                                                                                                                                   
            # p1 = plt.plot(r_d, vr_vin, color=cmap[(1+(2*count))%8], marker=markertype[count%5], markersize=3, label="Pope y/d =" + str(np.ceil(l)), linestyle='None')
            # p1[0].set_dashes(dashseq[i])
            # plt.xlabel("$r/d$")
            # plt.ylabel("$<V>/V_{in}$")
            # plt.title("Radial Profiles of Mean Axial Velocity in the Turbulent Round Jet")
            # plt.axis([0,17.5,0,.25])                                                                                                                                                                    
 
            # vals.append(vr_vin)
            # label.append(f"y/d =" + str(l))
            # count += 1
            

        plt.legend(loc='best', handlelength=5)
        plt.text(0.05, 0.575, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"ur_u_in_vs_r_d.pdf"), format="pdf", dpi=300
        )
        plt.close("all")
        
        count = 0
        for fname in fnames:
            slc = np.load(fname)
            l = slc["y"]/slc["diameter"]
            Temp = slc["temp_rad"]

            # Scale parameters of interest to match Pope scalings/labels
            r_rh = slc["r"]/slc["v_c_rad_r"]
            vr_centerline = slc["v_c_rad"]/slc["v_c_rad_cl"]

            # Make plots
            p2 = plt.plot(r_rh, vr_centerline, color=cmap[count%8],  label="y/d =" + str(int(np.ceil(l))), lw=1)
            p2[0].set_dashes(dashseq[count%7])
            plt.xlabel("$r/r_{1/2}$")
            plt.ylabel("$<V>/V_{0}$")
            plt.axis([0,4.0,0,1.25])
            # plt.xlim([0.0, 4.0])
            # plt.title("Mean Axial Velocity against Radial Distance in the Turbulent Round Jet")
            vals.append(vr_centerline)
            label.append(f"y/d =" + str(l))
            count += 1
            if count==6:
                break

        # for fname_p in fnames_p2:
            # slc = pd.read_csv(fname_p, sep=', ')

            # if args.slices == 3:
            #     rescale = 6
            # else:
            #     rescale = 10
                
            # l = dataset_p2[count-rescale]

            # Scale parameters of interest to match Pope scalings/labels                                                                                                                                  
 
            # r_rh = slc["r_rh"]
            # vr_centerline = slc["vr_centerline"]

            # Make plot                                                                                                                                                                                   
 
            # p2 = plt.plot(r_rh, vr_centerline, color=cmap[count%8], marker=markertype[count%5], markersize=3, label="Pope y/d =" + str(np.ceil(l)), linestyle='None')
            # p1[0].set_dashes(dashseq[i])                                                                                                                                                                 
            # plt.xlabel("$r/r_{1/2}$")
            # plt.ylabel("$<V>/V_{0}$")
            # plt.title("Mean Axial Velocity against Radial Distance in the Turbulent Round Jet")
            # plt.axis([0,17.5,0,.25])

            # vals.append(vr_centerline)
            # label.append(f"y/d =" + str(l))
            # count += 1

        plt.legend(loc='best', handlelength=5)
        plt.text(0.05, 1.2, '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"ur_u_0_vs_r_r_half.pdf"), format="pdf", dpi=300
        )
        plt.close("all")

        count = 0
        slice_y = []
        centerline_y = []
        for fname in fnames:
            slc = np.load(fname)
            Temp = slc["temp_rad"]
            slice_y.append(slc["y"]/slc["diameter"])
            centerline_y.append(slc["v_in"]/slc["v_c_rad_cl"])
            count += 1
            if count == 6:
                break
        # Make plot                                                                                                                                              
        p1 = plt.plot(slice_y, centerline_y, color=cmap[1], marker=markertype[i], linestyle='None', lw=2)
        plt.xlabel("$y/d$")
        plt.ylabel("$V_{in}/V_{0}$")
        # plt.title("Variation with Axial Distance of Mean Velocity Along Centerline")
        # plt.text(0.05, 1.2, '$T_{\infty} = $ '+str(int(Temp[-1])))
        plt.text(slice_y[0], centerline_y[-1], '$T_{0} = $ '+str(int(np.round(Temp[-1]))))
        plt.savefig(
            os.path.join(odir, f"uin_u0_vs_x_d.pdf"), format="pdf", dpi=300
        )
        plt.close("all")        

        for fname in fnames:
            slc = np.load(fname)
            
            # Scale parameters of interest to match Pope scalings/labels
            uu_scaled = slc["uu_rad"]/pow(slc["v_c_rad_cl"],2)
            vv_scaled = slc["vv_rad"]/pow(slc["v_c_rad_cl"],2)
            ww_scaled = slc["ww_rad"]/pow(slc["v_c_rad_cl"],2)
            uv_scaled = slc["uv_rad"]/pow(slc["v_c_rad_cl"],2)

            r_half_index_uu = r_half_raw(uu_scaled)
            r_half_uu = slc["r"][r_half_index_uu]
            r_rhalf_uu = slc["r"]/r_half_uu

            r_half_index_uv = r_half_raw(uv_scaled)
            r_half_uv = slc["r"][r_half_index_uv]
            r_rhalf_uv = slc["r"]/r_half_uv


            r_half_index_vv = r_half_raw(vv_scaled)
            r_half_vv = slc["r"][r_half_index_vv]
            r_rhalf_vv = slc["r"]/r_half_vv


            r_half_index_ww = r_half_raw(ww_scaled)
            r_half_ww = slc["r"][r_half_index_ww]
            r_rhalf_ww = slc["r"]/r_half_ww


            
            # r_rh_uu = slc["r"]/slc["v_c_rad_r"]

            # Make plot
            plt.figure(slc["slc"])
            label = ["$<u^2>$", "$<uv>$", "$<v^2>$", "$<w^2>$"]
            pope_keys = ['uu_scaled', 'uv_scaled', 'vv_scaled', 'ww_scaled']
            plt.plot(r_rhalf_uu, uu_scaled, label=label[0], color=cmap[0], dashes=dashseq[0], lw=1)
            plt.plot(r_rhalf_uv, uv_scaled, label=label[1], color=cmap[1], dashes=dashseq[1], lw=1)
            plt.plot(r_rhalf_vv, vv_scaled, label=label[2], color=cmap[2], dashes=dashseq[2], lw=1)
            plt.plot(r_rhalf_ww, ww_scaled, label=label[3], color=cmap[3], dashes=dashseq[3], lw=1)
            plt.xlabel("$r/r_{1/2}$")
            plt.ylabel("$<u_iu_j>/V_0^2$")
            # plt.axis([0,.3,0,.08])
            plt.xlim([0.0, 2.0])
            # plt.title("Profiles of Reynolds Stresses in the Round Jet")
            # if np.format_float_positional(slc['y'], precision=2)=='0.6':
                # count = 0
                # for fname_p in fnames_p3:
                    # slc2 = pd.read_csv(fname_p, sep=', ')
                    # print(slc2.keys())
                    # Scale parameters of interest to match Pope scalings/labels
         
                    # rey_scaled = slc2[pope_keys[count]]
                    # r_rh = slc2["r_rh"]

                    # Make plot                                                                                                                                                       
         
                    # plt.figure(slc["slc"])
                    # plt.plot(r_rh, rey_scaled, label=dataset_p3[count], color=cmap[count], marker=markertype[count%5], markersize=3, linestyle='None')
                    # plt.xlabel("$r/r_{1/2}$")
                    # plt.ylabel("$<u_iu_j>/V_0^2$")
                    # plt.axis([0,.3,0,.08])                                                                                                                                                              

                    # plt.xlim([0.0, 3.0])
                    # plt.title("Profiles of Reynolds Stresses in the Round Jet")
                    # count += 1

            l = slc['y']/slc['diameter']
            leg = plt.legend(loc='best', handlelength=5, title='$T_{0}=$ '+str(int(np.round(Temp[-1])))+', $y/d=$ '+str(int(np.ceil(l))))
            leg._legend_box.align = "right"
            
            plt.savefig(
                os.path.join(odir, f"Rey_Stress_{np.format_float_positional(slc['y'], precision=2)}.pdf"), format="pdf", dpi=300
            )
            plt.close("all")

        count = 0
        for fname in fnames:
            break
            slc = np.load(fname)
            temp = slc['temp']
            cp = slc['temp_pert']
            rho = slc['temp_pert_avg']
            y = slc['y']
            x = slc['x']
            z = slc['z']

            
            avg=0
            for i in range(199):
                print("i is: ", i)
                mult = cp[i]*cp[i]
                avg += mult
            avg /= 199
            avg = np.sqrt(avg)

            print(avg.shape)

            extents = np.array(
                [
                    x[0][0],
                    x[0][-1],
                    z[0][0],
                    z[-1][0]
                ]
            )


            fig, axs = plt.subplots(1, 3, figsize=(14, 6))
            im0 = axs[0].imshow(temp, origin="lower", extent=extents)
            axs[0].set_title("temperature")
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("z")
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(avg, origin="lower", extent=extents)
            axs[1].set_title("density")
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("z")
            fig.colorbar(im1, ax=axs[1])

            im2 = axs[2].imshow(cp[199], origin="lower", extent=extents)
            axs[2].set_title("specific heat")
            axs[2].set_xlabel("x")
            axs[2].set_ylabel("z")
            fig.colorbar(im2, ax=axs[2])

            fig.suptitle("Average slice at y = {0:.6f}".format(int(np.ceil(l))))
            plt.savefig(
                os.path.join(odir, f"Avg_check_{np.format_float_positional(slc['y'], precision=2)}.pdf"), format="pdf", dpi=300
            )
            plt.close("all")
            count += 1
            # sys.exit()
