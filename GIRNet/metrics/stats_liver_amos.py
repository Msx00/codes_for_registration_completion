import os
import json
import torch
import numpy as np
import pickle as pkl
try:
    from metrics import metrics
    from data import vtk_utils
except:
    from . import metrics
    from . import vtk_utils
from matplotlib import pyplot as plt
import vtk
from vtk.util import numpy_support
import torch


def stats_curator_amos(
        stats,
        output_folder=None,
):
    print(stats.keys())

    PRD_list = np.asarray(stats["PRD_list"])
    TRE_list = np.asarray(stats["TRE_list"])
    perlin_noise_list = np.asarray(stats["perlin_noise_list"])
    gaussian_noise_list = np.asarray(stats["gaussian_noise_list"])
    intraop_surface_area_list = np.asarray(stats["intraop_surface_area_list"])


    print("PRD_list.shape", PRD_list.shape)
    print("TRE_list.shape", TRE_list.shape)
    print("perlin_noise_list.shape", perlin_noise_list.shape)
    print("gaussian_noise_list.shape", gaussian_noise_list.shape)
    print("intraop_surface_area_list.shape", intraop_surface_area_list.shape)

    mean_PRD_list = PRD_list.mean(axis=0)
    mean_TRE_list = TRE_list.mean(axis=0)
    print(mean_PRD_list.shape)
    print(mean_TRE_list.shape)

    print("mean PRD no noise", mean_PRD_list[0])
    print("mean TRE no noise", mean_TRE_list[0])

    for idx_noise in range(len(mean_PRD_list)):
        print("Perlin noise", perlin_noise_list[0][idx_noise], "Gaussian noise", gaussian_noise_list[0][idx_noise])
        print("\tmean PRD", mean_PRD_list[idx_noise])
        print("\tmean TRE", mean_TRE_list[idx_noise])

    stats_curated = {
        "mean_PRD_list": mean_PRD_list,
        "mean_TRE_list": mean_TRE_list,
    }

    return stats_curated




def surface_distance_calculator(
        preop,
        intraop,
):
    """Calculate the surface distance between preop (deformed) volume and partial intraop surface

    Args:
        preop (vtk): preoperative volume
        intraop (vtk): intraoperative partial surface
    Returns:
        float: calculated surface distance
    """

    preop_surface = vtk_utils.extract_surface(mesh=preop)

    preop_np = numpy_support.vtk_to_numpy(preop_surface.GetPoints().GetData())
    intraop_np = numpy_support.vtk_to_numpy(intraop.GetPoints().GetData())

    # print("preop_np.shape", preop_np.shape)
    # print("intraop_np.shape", intraop_np.shape)

    # Calculate the surface distance between preop and intraop point clouds (original)
    chamfer_dist_intraop_to_preop = metrics.chamfer(
        prediction=torch.FloatTensor(preop_np).unsqueeze(0).to("cuda"),
        target=torch.FloatTensor(intraop_np).unsqueeze(0).to("cuda"),
        mode="T2P",
        reduction="mean",
    )

    return chamfer_dist_intraop_to_preop





def stat_curator_amos_threshold(
        stats,
        threshold=0.070,
):
    print(stats.keys())

    PRD_list = np.asarray(stats["PRD_list"])
    TRE_list = np.asarray(stats["TRE_list"])
    perlin_noise_list = np.asarray(stats["perlin_noise_list"])
    gaussian_noise_list = np.asarray(stats["gaussian_noise_list"])
    intraop_surface_area_list = np.asarray(stats["intraop_surface_area_list"])

    print("PRD_list.shape", PRD_list.shape)
    print("TRE_list.shape", TRE_list.shape)
    print("perlin_noise_list.shape", perlin_noise_list.shape)
    print("gaussian_noise_list.shape", gaussian_noise_list.shape)
    print("intraop_surface_area_list.shape", intraop_surface_area_list.shape)

    # exclude all PRD larger than threshold, then caculate the mean
    PRD_list[PRD_list > threshold] = np.nan
    mean_PRD_list = np.nanmean(PRD_list, axis=0)
    print(mean_PRD_list)

    TRE_list[PRD_list > threshold] = np.nan
    mean_TRE_list = np.nanmean(TRE_list, axis=0)
    print(mean_TRE_list)

    for idx_noise in range(len(mean_PRD_list)):
        print("Perlin noise", perlin_noise_list[0][idx_noise], "Gaussian noise", gaussian_noise_list[0][idx_noise])
        print("\tmean PRD", mean_PRD_list[idx_noise])
        print("\tmean TRE", mean_TRE_list[idx_noise])

    # group by perlin noise, draw lines of the mean TREs of different gaussian noise levels
    # x axis is perlin noise, y axis is mean TRE, lines are different gaussian noise levels
    perlin_noise_values = np.unique(perlin_noise_list[0])
    gaussian_noise_values = np.unique(gaussian_noise_list[0])
    plt.figure()
    # color the lines using different colors

    for idx_g_noise, g_noise in enumerate(gaussian_noise_values):
        idx_g_noise = np.where(gaussian_noise_list[0] == g_noise)
        plt.plot(perlin_noise_values, mean_TRE_list[idx_g_noise], label=f"Gaussian noise {g_noise}", marker='o')
    plt.xlabel("Perlin noise")
    plt.ylabel("Mean TRE")
    plt.title("Perlin noise vs Mean TRE grouped by Gaussian noise")
    plt.legend( )
    plt.grid()
    plt.savefig("/mnt/ceph/tco/TCO-Staff/Homes/liupeng/organ-deformation-net/visualization/test_plots/Perlin_noise_vs_Mean_TRE_grouped_by_Gaussian_noise.png")


    # group by gaussian noise, draw lines of the mean TREs of different perlin noise levels
    # x axis is gaussian noise, y axis is mean TRE, lines are different perlin noise levels
    # perlin_noise_values = np.unique(perlin_noise_list[0])
    # gaussian_noise_values = np.unique(gaussian_noise_list[0])

    plt.figure()
    for idx_p_noise, p_noise in enumerate(perlin_noise_values):
        idx_p_noise = np.where(perlin_noise_list[0] == p_noise)
        plt.plot(gaussian_noise_values, mean_TRE_list[idx_p_noise], label=f"Perlin noise {p_noise}", marker='o')
    plt.xlabel("Gaussian noise")
    plt.ylabel("Mean TRE")
    plt.title("Gaussian noise vs Mean TRE grouped by Perlin noise")
    plt.legend()
    plt.grid()
    plt.savefig("/mnt/ceph/tco/TCO-Staff/Homes/liupeng/organ-deformation-net/visualization/test_plots/Gaussian_noise_vs_Mean_TRE_grouped_by Perlin_noise.png")




if __name__ =="__main__":
    # stats_path ="/mnt/ceph/tco/TCO-Staff/Homes/liupeng/NeuralNetworks/OrganDeformNet/inference_v2s_net/inference_amos/2024-10-28_12-06-32_N100000+100000+100000_v5_NH_continue/2024-11-12_10-56-45/stats.pkl"
    stats_path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/NeuralNetworks/OrganDeformNet/inference_v2s_net/inference_amos/2024-10-28_12-06-32_N100000+100000+100000_v5_NH_continue/2024-11-12_14-43-09/stats.pkl"
    with open(stats_path, "rb") as f:
        stats = pkl.load(f)
    
    # stats_curator_amos(stats)
    stat_curator_amos_threshold(stats)


