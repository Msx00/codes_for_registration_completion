import os
import json
import torch
import numpy as np
try:
    from metrics import metrics
    from data import vtk_utils
except:
    from . import metrics
    from . import vtk_utils


def stats_calculator_liver_matchingcues(    
    preop, 
    intraop,
    displ_pred,
    displ_gt=None,
    preop_cues=None,
    intraop_cues=None,
    landmarks_preop=None,
    landmarks_intraop=None,
):

    # Calculate displacement error:
    displ_error, target_displ = None, None
    if displ_gt is not None:
        displ_error = metrics.MeanDisplacementError( displ_pred, displ_gt ).item()
        target_displ = metrics.MeanMagnitude( displ_gt, batch_reduce=True ).item()


    # Calculate chamfer distance:
    preop_deformed = preop + displ_pred

    # TODO remove dummy points
    mask = torch.abs(preop) < 1e3
    preop = preop[mask].view(1, 3, -1)
    mask = torch.abs(intraop) < 1e3
    intraop = intraop[mask].view(1, 3, -1)
    mask = torch.abs(preop_deformed) < 1e3
    preop_deformed = preop_deformed[mask].view(1, 3, -1)
    print("mask.shape", mask.shape, "preop.shape:", preop.shape, "intraop.shape:", intraop.shape)
    chamfer_dist_sym_pre_reg = metrics.chamfer(
        prediction=preop,
        target=intraop,
        mode="SYMMETRIC",
    ).item()
    chamfer_dist_sym_post_reg = metrics.chamfer(
        prediction=preop_deformed,
        target=intraop,
        mode="SYMMETRIC",
    ).item()


    chamfer_dist_p2t_pre_reg = metrics.chamfer(
        prediction=preop,
        target=intraop,
        mode="P2T",
    ).item()
    chamfer_dist_p2t_post_reg = metrics.chamfer(
        prediction=preop_deformed,
        target=intraop,
        mode="P2T",
    ).item()


    chamfer_dist_t2p_pre_reg = metrics.chamfer(
        prediction=preop,
        target=intraop,
        mode="T2P",
    ).item()
    chamfer_dist_t2p_post_reg = metrics.chamfer(
        prediction=preop_deformed,
        target=intraop,
        mode="T2P",
    ).item()


    res ={
        "TRE_post_reg": displ_error,
        "TRE_pre_reg": target_displ,
        "chamfer_dist_sym_pre_reg": chamfer_dist_sym_pre_reg,
        "chamfer_dist_sym_post_reg": chamfer_dist_sym_post_reg,
        "chamfer_dist_p2t_pre_reg": chamfer_dist_p2t_pre_reg,
        "chamfer_dist_p2t_post_reg": chamfer_dist_p2t_post_reg,
        "chamfer_dist_t2p_pre_reg": chamfer_dist_t2p_pre_reg,
        "chamfer_dist_t2p_post_reg": chamfer_dist_t2p_post_reg,
    }

    return res



def stats_curator_liver_matching_cues(
    stats_list,
    output_folder=None,
):

    print("\n\n==============Performing stats curation for liver matching cues...===========")

    TRE_post_reg_list = []
    TRE_pre_reg_list = []
    chamfer_dist_post_reg_list = []
    chamfer_dist_pre_reg_list = []
    for idx_s, stats in enumerate(stats_list):      
        # print(stats[0])  
        if "TRE_post_reg" in stats[0].keys():
            if idx_s == 0:
                TRE_post_reg_baseline = [s["TRE_post_reg"] for s in stats]
                TRE_pre_reg_baseline = [s["TRE_pre_reg"] for s in stats]
            else:
                # print(s)
                TRE_post_reg_list.append([s["TRE_post_reg"] for s in stats])
                TRE_pre_reg_list.append([s["TRE_pre_reg"] for s in stats]) # should be the same as TRE_pre_reg_baseline

        if "chamfer_dist_sym_post_reg" in stats[0].keys():
            if idx_s == 0:
                chamfer_dist_sym_post_reg_baseline = [s["chamfer_dist_sym_post_reg"] for s in stats]
                chamfer_dist_sym_pre_reg_baseline = [s["chamfer_dist_sym_pre_reg"] for s in stats]
            else:
                chamfer_dist_post_reg_list.append([s["chamfer_dist_sym_post_reg"] for s in stats])
                chamfer_dist_pre_reg_list.append([s["chamfer_dist_sym_pre_reg"] for s in stats]) # should be the same as chamfer_dist_sym_pre_reg_baseline
    

    TRE_post_reg_list = np.asarray(TRE_post_reg_list)
    TRE_pre_reg_list = np.asarray(TRE_pre_reg_list)
    chamfer_dist_post_reg_list = np.asarray(chamfer_dist_post_reg_list)
    chamfer_dist_pre_reg_list = np.asarray(chamfer_dist_pre_reg_list)

    N = 10

    ######################## Calculate TRE diff:########################
    print("\n-------TRE diff:-------")
    diff_TRE_list = TRE_post_reg_list - TRE_pre_reg_list
    mean_diff_TRE = np.mean(diff_TRE_list, axis=1)
    print("mean_diff_TRE:\n\t", mean_diff_TRE)
    # Top N samples with highest diff:
    idx_sorted =  np.argsort(diff_TRE_list, axis=1)
    print("idx_sorted.shape", idx_sorted.shape)
    idx_top_N = np.argsort(diff_TRE_list, axis=1)[:, :N]
    print("idx_top_N.shape:", idx_top_N.shape)
    diff_TRE_list_top_N = np.take_along_axis(diff_TRE_list, idx_top_N, axis=1)
    print("Top", N, "samples with highest diff:")
    for idx_b in range(diff_TRE_list_top_N.shape[0]):
        print("Cue:", idx_b)
        for idx_n in range(diff_TRE_list_top_N.shape[1]):
            print("\t{}:".format(idx_top_N[idx_b, idx_n]), diff_TRE_list_top_N[idx_b, idx_n])
    # print(diff_TRE_list_top_N)

    # Get samples with strictly increasing diff:
    # np.diff(diff_TRE_list, axis=0)
    idx_increasing = np.where(np.all(np.diff(diff_TRE_list, axis=0) < 0, axis=0))[0]
    print("idx_increasing.shape:", idx_increasing.shape)
    print("Sample with strictly increasing diff ({} in total):".format(len(idx_increasing)))
    for idx_i in idx_increasing:
        print("\t", idx_i, "\t", diff_TRE_list[:, idx_i])

    ######################## Calculate mean values:########################
    print("\n-------mean values:-------")
    mean_TRE_post_reg_baseline = sum(TRE_post_reg_baseline) / len(TRE_post_reg_baseline)
    mean_TRE_pre_reg_baseline = sum(TRE_pre_reg_baseline) / len(TRE_pre_reg_baseline)
    mean_chamfer_dist_sym_post_reg_baseline = sum(chamfer_dist_sym_post_reg_baseline) / len(chamfer_dist_sym_post_reg_baseline)
    mean_chamfer_dist_sym_pre_reg_baseline = sum(chamfer_dist_sym_pre_reg_baseline) / len(chamfer_dist_sym_pre_reg_baseline)
    # print("mean_TRE_post_reg_baseline:", mean_TRE_post_reg_baseline)
    # print("mean_TRE_pre_reg_baseline:", mean_TRE_pre_reg_baseline)
    # print("mean_chamfer_dist_sym_post_reg_baseline:", mean_chamfer_dist_sym_post_reg_baseline)
    # print("mean_chamfer_dist_sym_pre_reg_baseline:", mean_chamfer_dist_sym_pre_reg_baseline)
    print("mean TRE baseline:")
    print("\tpre reg:", mean_TRE_pre_reg_baseline)
    print("\tpost reg:", mean_TRE_post_reg_baseline)
    print("mean chamfer distance baseline:")
    print("\tpre reg:", mean_chamfer_dist_sym_pre_reg_baseline)
    print("\tpost reg:", mean_chamfer_dist_sym_post_reg_baseline)

    # print(TRE_post_reg_list)
    # print(TRE_pre_reg_list)
    # print("TRE_post_reg_list.shape:", TRE_post_reg_list.shape)
    # print("TRE_pre_reg_list.shape:", TRE_pre_reg_list.shape)

    mean_TRE_post_reg_list = np.mean(TRE_post_reg_list, axis=1)
    mean_TRE_pre_reg_list = np.mean(TRE_pre_reg_list, axis=1)
    mean_chamfer_dist_post_reg_list = np.mean(chamfer_dist_post_reg_list, axis=1)
    mean_chamfer_dist_pre_reg_list = np.mean(chamfer_dist_pre_reg_list, axis=1)

    # print("mean_TRE_post_reg_list:", mean_TRE_post_reg_list)
    # print("mean_TRE_pre_reg_list:", mean_TRE_pre_reg_list)
    # print("mean_chamfer_dist_post_reg_list:", mean_chamfer_dist_post_reg_list)
    # print("mean_chamfer_dist_pre_reg_list:", mean_chamfer_dist_pre_reg_list) 
    print("mean TRE:")
    print("\tpre reg:", mean_TRE_pre_reg_list)
    print("\tpost reg:", mean_TRE_post_reg_list)
    print("mean chamfer distance:")
    print("\tpre reg:", mean_chamfer_dist_pre_reg_list)
    print("\tpost reg:", mean_chamfer_dist_post_reg_list)



    ######################## Calculate ratios:########################
    print("\n-------mean ratios:-------")
    # 1. Ratio between mean values:
    ratio_mean_TRE = [t / mean_chamfer_dist_pre_reg_list[0] for t in mean_TRE_post_reg_list ]
    ratio_mean_chamfer_dist = [t / mean_chamfer_dist_pre_reg_list[0] for t in mean_chamfer_dist_post_reg_list ]
    print("ratio_mean_TRE:", ratio_mean_TRE)
    print("ratio_mean_chamfer_dist:", ratio_mean_chamfer_dist)


    # 2 Ratio between metrics of each sample:
    ratio_TRE_list = []
    for idx_s, TRE_list in enumerate(TRE_post_reg_list):
        ratio_TRE_list.append([t / TRE_post_reg_baseline[idx_t] for idx_t, t in enumerate(TRE_list)])
    ratio_chamfer_list = []
    for idx_s, chamfer_list in enumerate(chamfer_dist_post_reg_list):
        ratio_chamfer_list.append([c / chamfer_dist_sym_post_reg_baseline[idx_c] for idx_c, c in enumerate(chamfer_list)])

    ratio_TRE_list = np.asarray(ratio_TRE_list)
    ratio_chamfer_list = np.asarray(ratio_chamfer_list)

    # print("ratio_TRE_list.shape:", ratio_TRE_list.shape)
    # print("ratio_chamfer_list.shape:", ratio_chamfer_list.shape)

    mean_ratio_TRE_list = np.mean(ratio_TRE_list, axis=1)
    mean_ratio_chamfer_list = np.mean(ratio_chamfer_list, axis=1)

    print("mean_ratio_TRE_list:", mean_ratio_TRE_list)
    print("mean_ratio_chamfer_list:", mean_ratio_chamfer_list)


    stats_curated = {
        "mean_TRE_post_reg_baseline": mean_TRE_post_reg_baseline,
        "mean_TRE_pre_reg_baseline": mean_TRE_pre_reg_baseline,
        "mean_chamfer_dist_sym_post_reg_baseline" : mean_chamfer_dist_sym_post_reg_baseline,
        "mean_chamfer_dist_sym_pre_reg_baseline": mean_chamfer_dist_sym_pre_reg_baseline,

        "mean_TRE_post_reg_list": mean_TRE_post_reg_list,
        "mean_TRE_pre_reg_list": mean_TRE_pre_reg_list,
        "mean_chamfer_dist_post_reg_list": mean_chamfer_dist_post_reg_list,
        "mean_chamfer_dist_pre_reg_list": mean_chamfer_dist_pre_reg_list,

        "ratio_mean_TRE" : ratio_mean_TRE,
        "ratio_mean_chamfer_dist" : ratio_mean_chamfer_dist,
        "mean_ratio_TRE_list" : mean_ratio_TRE_list,
        "mean_ratio_chamfer_list" : mean_ratio_chamfer_list,
    }


    return stats_curated


