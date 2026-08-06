import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import pickle as pkl


def plot(
        intraop_area_list_total,
        intraop_area_list_total_percentage,
        TRE_est_errs_sample_list,
        TRE_target_errs_sample_list,
        output_folder,
        label_font_size = 14,
):
    
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # # c = np.array([
    # #     (255,0,0) if mean_orig_displ_err> TRE_est_errs_intraop_surfaces_list[i] else (0,0,255)  
    # #             for i in range(len(TRE_est_errs_intraop_surfaces_list))
    # #     ] )
    # ax[0].scatter(intraop_area_list_total, TRE_est_errs_sample_list, )
    # ax[0].set_title("TRE over visible amount of intraop surfaces for all samples")
    # ax[0].set_xlabel("visible surface amount [m^2]")
    # ax[0].set_ylabel("TRE")
    # # draw horizontal line at target TRE
    # # ax[0].axhline(mean_orig_displ_err, color='black', lw=2)
    # ax[1].scatter(intraop_area_list_total_percentage, TRE_est_errs_sample_list, )
    # ax[1].set_title("TRE over visible amount of intraop surfaces percentage for all samples")
    # ax[1].set_xlabel("visible surface amount percentage")
    # ax[1].set_ylabel("TRE")
    # # draw horizontal line at target TRE
    # # ax[1].axhline(mean_orig_displ_err, color='black', lw=2)
    # output_path = os.path.join(output_folder, "TRE_over_visible_surface_amount_all_samples.png")
    # plt.savefig(output_path)
    # print("saved plot to", output_path)


    # data_bin_indices = np.digitize(intraop_area_list_total, bins=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    # bins = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,]
    # step = np.max(intraop_area_list_total_percentage) / 7
    # bins = [i * step for i in range(8)]


    # diff = TRE_est_errs_sample_list - TRE_target_errs_sample_list




    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
    # ax[0].boxplot(TRE_est_errs_sample_list_binned)
    # # ax[0].set_title("TRE over visible amount of intraop surfaces for all samples")
    # ax[0].set_xlabel("visible surface amount percentage", fontsize=12)
    # ax[0].set_ylabel("Target registration error [m]", fontsize=12)
    # # set x-axis labels
    # # ax[0].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    # # set x-axis labels in percentage
    # ax[0].set_xticklabels([f"{bins[i-1]*100:.0f}%-{bins[i]*100:.0f}%" for i in range(1, len(bins))])


    # steps = 0.04 / 5
    # bins = [i * steps for i in range(6)]
    # print("bins", bins)
    # intraop_area_bin_indices = np.digitize(intraop_area_list_total, bins=bins)
    # # diff_binned = [diff[data_bin_indices == i] for i in range(1, len(bins))]
    # TRE_est_errs = [TRE_est_errs_sample_list[intraop_area_bin_indices == i] for i in range(1, len(bins))]


    # ax[0].boxplot(TRE_est_errs)
    # # ax[0].set_title("TRE over visible amount of intraop surfaces for all samples")
    # # ax[0].set_xlabel("visible surface amount [m^2]", fontsize=14)
    # ax[0].set_xlabel("visible surface amount percentage", fontsize=14)
    # ax[0].set_ylabel("TRE [m]", fontsize=14)
    # ax[0].axhline(0, color='grey', lw=2)
    # # set x-axis labels
    # # ax[0].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    # # set x-axis labels in percentage
    # # ax[0].set_xticklabels([f"{bins[i-1]*100:.0f}%-{bins[i]*100:.0f}%" for i in range(1, len(bins))])
    # # set x-axis labels in square meters
    # ax[0].set_xticklabels([f"{bins[i-1]:.3f}-{bins[i]:.3f}" for i in range(1, len(bins))])



    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    # bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    num_bins = 7
    steps = 0.5 / num_bins
    bins = [i * steps for i in range(num_bins + 1)]
    print("bins", bins)
    intraop_area_percentage_bin_indices = np.digitize(intraop_area_list_total_percentage, bins=bins)
    diff = TRE_est_errs_sample_list - TRE_target_errs_sample_list

    TRE_est_errs_binned = [TRE_est_errs_sample_list[intraop_area_percentage_bin_indices == i] for i in range(1, len(bins))] 
    diff_binned = [diff[intraop_area_percentage_bin_indices == i] for i in range(1, len(bins))]


    ax[0].boxplot(TRE_est_errs_binned,)# widths=(0.05) * len(bins))
    # ax[0].set_title("TRE over visible amount of intraop surfaces for all samples")
    # ax[0].set_xlabel("visible surface amount [m^2]", fontsize=14)
    ax[0].set_xlabel("visible surface amount", fontsize=label_font_size)
    ax[0].set_ylabel("TRE [mm]", fontsize=label_font_size, labelpad=-1)
    ax[0].set_xticklabels([f"{bins[i-1]*100:.0f}-{bins[i]*100:.0f}%" for i in range(1, len(bins))])


    # output_path = os.path.join(output_folder, "Boxplot_NKI_visible_surface_exp_est_TRE.png")
    # plt.savefig(output_path, bbox_inches='tight')
    # print("saved plot to", output_path)

    # output_path = os.path.join(output_folder, "Boxplot_NKI_visible_surface_exp_est_TRE.pdf")
    # plt.savefig(output_path, bbox_inches='tight')
    # print("saved plot to", output_path)




    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 5))
    ax[1].boxplot(diff_binned, )#widths=(0.05) * len(bins))
    # ax[1].set_title("TRE over visible amount of intraop surfaces percentage for all samples")
    ax[1].set_xlabel("visible surface amount", fontsize=label_font_size)
    ax[1].set_ylabel("TRE difference [mm]", fontsize=label_font_size, labelpad=-1)
    ax[1].axhline(0, color='grey', lw=2)
    # set x-axis labels
    # ax[1].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    ax[1].set_xticklabels([f"{bins[i-1]*100:.0f}-{bins[i]*100:.0f}%" for i in range(1, len(bins))])


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=None)



    # output_path = os.path.join(output_folder, "Boxplot_NKI_visible_surface_exp_diff_est_tgt_TRE.png")
    # plt.savefig(output_path, bbox_inches='tight')
    # print("saved plot to", output_path)

    # output_path = os.path.join(output_folder, "Boxplot_NKI_visible_surface_exp_diff_est_tgt_TRE.pdf")
    # plt.savefig(output_path, bbox_inches='tight')
    # print("saved plot to", output_path)




    output_path = os.path.join(output_folder, "Boxplot_NKI_visible_surface_exp-left_est_TRE-right_diff_est_tgt_TRE.png")
    plt.savefig(output_path, bbox_inches='tight')
    print("saved plot to", output_path)


    output_path = os.path.join(output_folder, "Boxplot_NKI_visible_surface_exp-left_est_TRE-right_diff_est_tgt_TRE.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    print("saved plot to", output_path)



    # diff = TRE_est_errs_sample_list - TRE_target_errs_sample_list
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # c = np.array([
    #     (255,0,0) if diff[i] < 0 else (0,0,255)  
    #             for i in range(len(diff))
    #     ] )
    # ax[0].scatter(intraop_area_list_total, diff, c=c/255.0,)
    # ax[0].set_title("Diff between est and tgt TRE over visible amount of intraop surfaces for all samples")
    # ax[0].set_xlabel("visible surface amount [m^2]")
    # ax[0].set_ylabel("TRE")
    # # draw horizontal line at target TRE
    # # ax[0].axhline(mean_orig_displ_err, color='black', lw=2)
    # ax[1].scatter(intraop_area_list_total_percentage, diff, c=c/255.0,)
    # ax[1].set_title("Diff between est and tgt TRE over visible amount of intraop surfaces percentage for all samples")
    # ax[1].set_xlabel("visible surface amount percentage")
    # ax[1].set_ylabel("TRE")
    # # draw horizontal line at target TRE
    # # ax[1].axhline(mean_orig_displ_err, color='black', lw=2)
    # output_path = os.path.join(output_folder, "Diff_between_est_and_tgt_TRE_over_visible_surface_amount_all_samples.png")
    # plt.savefig(output_path)
    # print("saved plot to", output_path)

    # # boxplot
    # print("diff.shape", diff.shape)
    # print("data_bin_indices.shape", data_bin_indices.shape)
    # diff_binned = [diff[data_bin_indices == i] for i in range(1, 6)]

    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # ax[0].boxplot(diff_binned)
    # ax[0].set_title("Diff between est and tgt TRE over visible amount of intraop surfaces for all samples")
    # ax[0].set_xlabel("visible surface amount [m^2]")
    # ax[0].set_ylabel("TRE")
    # # set x-axis labels
    # ax[0].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    # ax[0].axhline(0, color='black', lw=2)
    # ax[1].boxplot(diff_binned )
    # ax[1].set_title("Diff between est and tgt TRE over visible amount of intraop surfaces percentage for all samples")
    # ax[1].set_xlabel("visible surface amount percentage")
    # ax[1].set_ylabel("TRE")
    # # set x-axis labels
    # ax[1].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    # ax[1].axhline(0, color='black', lw=2)
    # output_path = os.path.join(output_folder, "Diff_between_est_and_tgt_TRE_over_visible_surface_amount_all_samples_boxplot.png")
    # plt.savefig(output_path)
    # print("saved plot to", output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_stats_path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/NeuralNetworks/OrganDeformNet/inference_v2s_net/2024-03-02_09-47-17_no_internals_switch_200e_N50000/2024-03-06_09-07-02/stats_all.pkl"
    parser.add_argument("--stats_path", type=str, default=default_stats_path)

    args = parser.parse_args()
    stats_path = args.stats_path

    with open(stats_path, "rb") as f:
        stats = pkl.load(f)
        f.close()
    print("loaded stats from", stats_path)

    intraop_area_list_total = stats["intraop_area_list_total"]
    intraop_area_list_total_percentage = np.asarray(stats["intraop_area_list_total_percentage"]).flatten()
    TRE_est_errs_sample_list = stats["TRE_est_errs_sample_list"]
    TRE_target_errs_sample_list = stats["TRE_target_errs_sample_list"]

    print("intraop_area_list_total.shape", intraop_area_list_total.shape)
    print("intraop_area_list_total_percentage.shape", intraop_area_list_total_percentage.shape)
    print("TRE_est_errs_sample_list.shape", TRE_est_errs_sample_list.shape)
    print("TRE_target_errs_sample_list.shape", TRE_target_errs_sample_list.shape)

    # print("max(intraop_area_list_total_percentage):", np.max(intraop_area_list_total_percentage))

    plot(
        intraop_area_list_total= intraop_area_list_total,
        intraop_area_list_total_percentage=intraop_area_list_total_percentage,
        TRE_est_errs_sample_list=TRE_est_errs_sample_list * 1e3 ,
        TRE_target_errs_sample_list=TRE_target_errs_sample_list * 1e3,
        # output_folder=os.path.dirname(stats_path)
        output_folder=None,
    )



