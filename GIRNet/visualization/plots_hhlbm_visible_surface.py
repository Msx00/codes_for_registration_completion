import numpy as np
import os
from matplotlib import pyplot as plt
import pickle
import yaml



def plot_single_sample(
        mean_orig_displ_err,
        TRE_est_errs_intraop_surfaces_list,
        dist_chamfer_intraop_surfaces_list,
        intraop_area_list,
        intraop_area_list_percentage,
        output_folder,
):
    # scatter plot TRE_est_errs_intraop_surfaces_list over intraop surface area
    # and scatter plot TRE_est_errs_intraop_surfaces_list over intraop surface area percentage
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    c = np.array([
        (255,0,0) if mean_orig_displ_err> TRE_est_errs_intraop_surfaces_list[i] else (0,0,255)  
                for i in range(len(TRE_est_errs_intraop_surfaces_list))
        ] )
    ax[0].scatter(intraop_area_list, TRE_est_errs_intraop_surfaces_list, c=c/255.0, )
    ax[0].set_title("TRE over visible amount of intraop surfaces")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("TRE")
    # draw horizontal line at target TRE
    ax[0].axhline(mean_orig_displ_err, color='black', lw=2)
    ax[1].scatter(intraop_area_list_percentage, TRE_est_errs_intraop_surfaces_list, c=c/255.0,)
    ax[1].set_title("TRE over visible amount of intraop surfaces percentage")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("TRE")
    # draw horizontal line at target TRE
    ax[1].axhline(mean_orig_displ_err, color='black', lw=2)
    output_path = os.path.join(output_folder, "TRE_over_visible_surface_amount.png")
    if not os.path.exists(output_folder):
        os.makedirs(os.path.join(output_folder,))
    plt.savefig(output_path)
    print("saved plot to", output_path)


    # # scatter plot difference between TRE_est_errs_intraop_surfaces_list and target TRE over intraop surface area
    # over intraop surface area
    diff = np.array(TRE_est_errs_intraop_surfaces_list) - mean_orig_displ_err
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    c = np.array([
        (255,0,0) if diff[i] < 0 else (0,0,255)  
                for i in range(len(diff))
        ] )
    ax[0].scatter(intraop_area_list, diff, c=c/255.0, )
    ax[0].set_title("diff between TRE and PRD over visible amount of intraop surfaces")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("Diff between TRE and PRD")
    # draw horizontal line at target TRE
    ax[0].axhline(0, color='black', lw=2)
    ax[1].scatter(intraop_area_list_percentage, diff, c=c/255.0,)
    ax[1].set_title("diff between TRE and PRD over visible amount of intraop surfaces percentage")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("Diff between TRE and PRD")
    # draw horizontal line at target TRE
    ax[1].axhline(0, color='black', lw=2)
    output_path = os.path.join(output_folder, "Diff_between_est_and_tgt_TRE_over_visible_surface_amount.png")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_path)
    print("saved plot to", output_path)

    
    # scatter plot dist_chamfer_intraop_surfaces_list over intraop surface area
    # and scatter plot dist_chamfer_intraop_surfaces_list over intraop surface area percentage
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].scatter(intraop_area_list, dist_chamfer_intraop_surfaces_list, )
    ax[0].set_title("Chamfer distance over visible amount of intraop surfaces")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("Chamfer distance [mm]")
    ax[1].scatter(intraop_area_list_percentage, dist_chamfer_intraop_surfaces_list, )
    ax[1].set_title("Chamfer distance over visible amount of intraop surfaces percentage")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("Chamfer distance [mm]")
    output_path = os.path.join(output_folder, "Chamfer_distance_over_visible_surface_amount.png")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_path)
    print("saved plot to", output_path)



def plot_all_samples(
    TRE_est_errs_sample_list,
    TRE_target_errs_sample_list,
    dist_chamfer_sample_list,
    intraop_area_list_total,
    intraop_area_list_total_percentage,
    output_folder_hhlbm,
):

    stats_all = {}

    TRE_est_errs_sample_list = np.array(TRE_est_errs_sample_list)
    TRE_target_errs_sample_list = np.array(TRE_target_errs_sample_list)
    dist_chamfer_sample_list = np.array(dist_chamfer_sample_list)
    intraop_area_list_total = np.array(intraop_area_list_total)
    intraop_area_list_total_percentage = np.array(intraop_area_list_total_percentage)

    print("TRE_est_errs_sample_list.shape", TRE_est_errs_sample_list.shape)
    print("TRE_target_errs_sample_list.shape", TRE_target_errs_sample_list.shape)
    print("intraop_area_list_total.shape", intraop_area_list_total.shape)
    print("intraop_area_list_total_percentage.shape", intraop_area_list_total_percentage.shape)

    stats_all["stats_origin"] = {
        "TRE_est_errs_sample_list" : TRE_est_errs_sample_list,
        "TRE_target_errs_sample_list" : TRE_target_errs_sample_list,
        "intraop_area_list_total" : intraop_area_list_total,
        "intraop_area_list_total_percentage" : intraop_area_list_total_percentage,
    }



    # plot lines of TRE per sample
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    markers = ['o', 's', 'D', '^', 'x']
    for i in range(len(TRE_est_errs_sample_list)):
        intraop_area_list_truncated_idx = intraop_area_list_total[i, ] > 0.01
        intraop_area_list_truncated = intraop_area_list_total[i, ][intraop_area_list_truncated_idx]
        TRE_est_errs_sample_list_truncated = TRE_est_errs_sample_list[i, ][intraop_area_list_truncated_idx]
        ax.plot(intraop_area_list_truncated, TRE_est_errs_sample_list_truncated, marker=markers[i % len(markers)], label="{}".format(i))
    ax.set_title("TRE over visible amount of intraop surfaces for all samples")
    ax.set_xlabel("visible surface amount [m^2]")
    ax.set_ylabel("TRE")
    # add legend for samples
    ax.legend()
    output_path = os.path.join(output_folder_hhlbm, "TRE_over_visible_surface_amount_all_samples_lines.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)


    # plot lines of diff TRE per sample
    diff_list_per_sample = TRE_est_errs_sample_list - TRE_target_errs_sample_list
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    markers = ['o', 's', 'D', '^', 'x']
    for i in range(len(diff_list_per_sample)):
        intraop_area_list_truncated_idx = intraop_area_list_total[i, ] > 0.01
        intraop_area_list_truncated = intraop_area_list_total[i, ][intraop_area_list_truncated_idx]
        diff_list_per_sample_truncated = diff_list_per_sample[i, ][intraop_area_list_truncated_idx]
        ax.plot(intraop_area_list_truncated, diff_list_per_sample_truncated, marker=markers[i % len(markers)], label="{}".format(i))
    ax.set_title("Diff between TRE and PRD over visible amount of intraop surfaces")
    ax.set_xlabel("visible surface amount [m^2]")
    ax.set_ylabel("Diff between TRE and PRD")
    # draw horizontal line at target TRE
    ax.axhline(0, color='black', lw=2)
    ax.legend()
    output_path = os.path.join(output_folder_hhlbm, "Diff_between_PRD_and_TRE_over_visible_surface_amount_all_samples_lines.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)



    # Plot lines for chamfer distances:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    markers = ['o', 's', 'D', '^', 'x']
    for i in range(len(dist_chamfer_sample_list)):
        intraop_area_list_truncated_idx = intraop_area_list_total[i, ] > 0.01
        intraop_area_list_truncated = intraop_area_list_total[i, ][intraop_area_list_truncated_idx]
        # print("i", i)
        # print("dist_chamfer_sample_list", len(dist_chamfer_sample_list))
        # print("dist_chamfer_sample_list[i]", dist_chamfer_sample_list[i])
        dist_chamfer_sample_list_truncated = dist_chamfer_sample_list[i][intraop_area_list_truncated_idx]
        ax.plot(intraop_area_list_truncated, dist_chamfer_sample_list_truncated, marker=markers[i % len(markers)], label="{}".format(i))
    ax.set_title("Chamfer distance over visible amount of intraop surfaces for all samples")
    ax.set_xlabel("visible surface amount [m^2]")
    ax.set_ylabel("Chamfer distance [mm]")
    ax.legend()
    output_path = os.path.join(output_folder_hhlbm, "Chamfer_distance_over_visible_surface_amount_all_samples_lines.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)


    TRE_est_errs_sample_list = TRE_est_errs_sample_list.flatten()
    TRE_target_errs_sample_list = TRE_target_errs_sample_list.flatten()
    intraop_area_list_total = intraop_area_list_total.flatten()
    intraop_area_list_total_percentage = intraop_area_list_total_percentage.flatten()

    valid_sample_idx = np.where(intraop_area_list_total_percentage > 0.01)
    TRE_est_errs_sample_list = TRE_est_errs_sample_list[valid_sample_idx]
    TRE_target_errs_sample_list = TRE_target_errs_sample_list[valid_sample_idx]
    intraop_area_list_total = intraop_area_list_total[valid_sample_idx]
    intraop_area_list_total_percentage = intraop_area_list_total_percentage[valid_sample_idx]

    stats_all["stats_flatten"] = {
        "TRE_est_errs_sample_list" : TRE_est_errs_sample_list,
        "TRE_target_errs_sample_list" : TRE_target_errs_sample_list,
        "intraop_area_list_total" : intraop_area_list_total,
        "intraop_area_list_total_percentage" : intraop_area_list_total_percentage,
    }



    # calcualte mean target / estimated TRE:
    mean_tgt_TRE = np.mean(TRE_target_errs_sample_list)
    mean_est_TRE = np.mean(TRE_est_errs_sample_list)
    print("Mean Displacement:")
    print("\tPRD", mean_tgt_TRE)
    print("\tTRE", mean_est_TRE)

    stats_all["mean_TRE"] = {
        "mean_tgt_TRE" : mean_tgt_TRE,
        "mean_est_TRE" : mean_est_TRE,
    }



    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # c = np.array([
    #     (255,0,0) if mean_orig_displ_err> TRE_est_errs_intraop_surfaces_list[i] else (0,0,255)  
    #             for i in range(len(TRE_est_errs_intraop_surfaces_list))
    #     ] )
    ax[0].scatter(intraop_area_list_total, TRE_est_errs_sample_list, )
    ax[0].set_title("TRE over visible amount of intraop surfaces for all samples")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("TRE")
    # draw horizontal line at target TRE
    # ax[0].axhline(mean_orig_displ_err, color='black', lw=2)
    ax[1].scatter(intraop_area_list_total_percentage, TRE_est_errs_sample_list, )
    ax[1].set_title("TRE over visible amount of intraop surfaces percentage for all samples")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("TRE")
    # draw horizontal line at target TRE
    # ax[1].axhline(mean_orig_displ_err, color='black', lw=2)
    output_path = os.path.join(output_folder_hhlbm, "TRE_over_visible_surface_amount_all_samples.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)



    bins_surface_area = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    data_bin_indices_surface_area = np.digitize(intraop_area_list_total, bins=bins_surface_area)
    data_binned_surface_area = [TRE_est_errs_sample_list[data_bin_indices_surface_area == i] for i in range(1, 6)]
    bins_surface_area_percentage = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    data_bin_indices_surface_area_percentage = np.digitize(intraop_area_list_total_percentage, bins=bins_surface_area_percentage)
    data_binned_surface_area_percentage = [TRE_est_errs_sample_list[data_bin_indices_surface_area_percentage == i] for i in range(1, 6)]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].boxplot(data_binned_surface_area)
    ax[0].set_title("TRE over visible amount of intraop surfaces for all samples")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("TRE")
    # set x-axis labels
    ax[0].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])

    ax[1].boxplot(data_binned_surface_area_percentage)
    ax[1].set_title("TRE over visible amount of intraop surfaces percentage for all samples")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("TRE")
    # set x-axis labels
    # ax[1].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    ax[1].set_xticklabels(["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"])

    output_path = os.path.join(output_folder_hhlbm, "TRE_over_visible_surface_amount_all_samples_boxplot.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)


    # dynamic bins, based on the min and max value of the data

    # exclude all surface amounts lower than 10% regarding the visible surface percentage
    # data_included = intraop_area_list_total_percentage > 0.1
    # intraop_area_list_total = intraop_area_list_total[data_included]
    # intraop_area_list_total_percentage = intraop_area_list_total_percentage[data_included]
    # TRE_est_errs_sample_list = TRE_est_errs_sample_list[data_included]
    # TRE_target_errs_sample_list = TRE_target_errs_sample_list[data_included]


    bins_surface_area = np.linspace(intraop_area_list_total.min(), intraop_area_list_total.max(), 10)
    data_bin_indices_surface_area = np.digitize(intraop_area_list_total, bins=bins_surface_area)
    data_binned_surface_area = [TRE_est_errs_sample_list[data_bin_indices_surface_area == i] for i in range(1, len(bins_surface_area))]
    bins_surface_area_percentage = np.linspace(intraop_area_list_total_percentage.min(), intraop_area_list_total_percentage.max(), 10)
    data_bin_indices_surface_area_percentage = np.digitize(intraop_area_list_total_percentage, bins=bins_surface_area_percentage)
    data_binned_surface_area_percentage = [TRE_est_errs_sample_list[data_bin_indices_surface_area_percentage == i] for i in range(1, len(bins_surface_area_percentage))]
    # calculate the mean of each bin
    data_binned_surface_area_mean = [np.mean(data_binned_surface_area[i]) for i in range(len(data_binned_surface_area))]
    data_binned_surface_area_percentage_mean = [np.mean(data_binned_surface_area_percentage[i]) for i in range(len(data_binned_surface_area_percentage))]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].boxplot(data_binned_surface_area)
    ax[0].set_title("TRE over visible amount of intraop surfaces for all samples")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("TRE")
    # set x-axis labels
    # ax[0].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    ax[0].set_xticklabels([f"{bins_surface_area[i]:.2f}-{bins_surface_area[i+1]:.2f}" for i in range(len(bins_surface_area)-1)])
    # plot the mean of each bin
    ax[0].plot(range(1, len(bins_surface_area)), data_binned_surface_area_mean, color='black', marker='o', label="mean")

    ax[1].boxplot(data_binned_surface_area_percentage)
    ax[1].set_title("TRE over visible amount of intraop surfaces percentage for all samples")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("TRE")
    # set x-axis labels
    # ax[1].set_xticklabels(["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"])
    ax[1].set_xticklabels([f"{bins_surface_area_percentage[i]:.2f}-{bins_surface_area_percentage[i+1]:.2f}" for i in range(len(bins_surface_area_percentage)-1)])
    # plot the mean of each bin
    ax[1].plot(range(1, len(bins_surface_area_percentage)), data_binned_surface_area_percentage_mean, color='black', marker='o', label="mean")
    output_path = os.path.join(output_folder_hhlbm, "TRE_over_visible_surface_amount_all_samples_boxplot_dynamic_bins.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)




    diff = TRE_est_errs_sample_list - TRE_target_errs_sample_list
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    c = np.array([
        (255,0,0) if diff[i] < 0 else (0,0,255)  
                for i in range(len(diff))
        ] )
    ax[0].scatter(intraop_area_list_total, diff, c=c/255.0,)
    ax[0].set_title("Diff between TRE and PRD over visible amount of intraop surfaces for all samples")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("Diff between TRE and PRD")
    # draw horizontal line at target TRE
    ax[0].axhline(0, color='black', lw=2)

    ax[1].scatter(intraop_area_list_total_percentage, diff, c=c/255.0,)
    ax[1].set_title("Diff between TRE and PRD over visible amount of intraop surfaces percentage for all samples")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("Diff between TRE and PRD")
    # draw horizontal line at target TRE
    ax[1].axhline(0, color='black', lw=2)
    output_path = os.path.join(output_folder_hhlbm, "Diff_between_TRE_and_PRD_over_visible_surface_amount_all_samples.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)


    # boxplot
    # print("diff.shape", diff.shape)
    # print("data_bin_indices.shape", data_bin_indices.shape)
    diff_binned_surface_area = [diff[data_bin_indices_surface_area == i] for i in range(1, len(bins_surface_area))]
    diff_binned_surface_area_percentage = [diff[data_bin_indices_surface_area_percentage == i] for i in range(1,  len(bins_surface_area_percentage))]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].boxplot(diff_binned_surface_area)
    ax[0].set_title("Diff between TRE and PRD over visible amount of intraop surfaces for all samples")
    ax[0].set_xlabel("visible surface amount [m^2]")
    ax[0].set_ylabel("Diff between TRE and PRD")
    # set x-axis labels
    # ax[0].set_xticklabels(["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05"])
    ax[0].set_xticklabels([f"{bins_surface_area[i]:.2f}-{bins_surface_area[i+1]:.2f}" for i in range(len(bins_surface_area)-1)])
    ax[0].axhline(0, color='black', lw=2)
    ax[1].boxplot(diff_binned_surface_area_percentage )
    ax[1].set_title("Diff between TRE and PRD over visible amount of intraop surfaces percentage for all samples")
    ax[1].set_xlabel("visible surface amount percentage")
    ax[1].set_ylabel("Diff between TRE and PRD")
    # set x-axis labels
    # ax[1].set_xticklabels(["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"])
    ax[1].axhline(0, color='black', lw=2)
    ax[1].set_xticklabels([f"{bins_surface_area_percentage[i]:.2f}-{bins_surface_area_percentage[i+1]:.2f}" for i in range(len(bins_surface_area_percentage)-1)])
    output_path = os.path.join(output_folder_hhlbm, "Diff_between_TRE_and_PRD_over_visible_surface_amount_all_samples_boxplot.png")
    plt.savefig(output_path)
    print("saved plot to", output_path)






    # save the stats
    stats_path = os.path.join(output_folder_hhlbm, "stats_all.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats_all, f)
        f.close()
    print("saved stats to", stats_path)

    stats_path = os.path.join(output_folder_hhlbm, "stats_all.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(stats_all, f)
        f.close()
    print("saved stats to", stats_path)





