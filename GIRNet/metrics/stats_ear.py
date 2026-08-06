import os
import json
import torch
try:
    from metrics import metrics
    from data import vtk_utils
except:
    from . import metrics
    from . import vtk_utils


def stats_calculator_ear(
    preop, 
    intraop,
    displ_pred,
    displ_gt=None,
    landmarks_preop=None,
    landmarks_intraop=None,
):
    # Calculate displacement error:
    displ_error, target_displ = None, None
    if displ_gt is not None:
        displ_error = metrics.MeanDisplacementError( displ_pred, displ_gt ).item()
        target_displ = metrics.MeanMagnitude( displ_gt, batch_reduce=True ).item()
    
    # preop_vtk, intraop_vtk, preop_internal_vtk, intraop_internal_vtk, landmarks_preop_vtk, \
    #     landmarks_intraop_vtk, landmarks_intraop_estimated = vtk_utils.save_output_as_vtk_dry(
    #         coords_pre=preop, 
    #         coords_intra=intraop, 
    #         displ=displ_pred, 
    #         displ_gt = displ_gt,
    #         # features_pre=None, features_intra=None,
    #         # coords_pre_internal=None, coords_intra_internal=None,
    #         # features_pre_internal=None, features_intra_internal=None,
    #         landmarks_preop=None, landmarks_intraop=None,
    #         preop_meshes=[], intraop_meshes=[],
    #         scale = 1, #1e-3
    #         folder = "tmp_out",
    #     )
    # Calculate Chamfer distance:
    print("preop.shape:", preop.shape)
    print("displ_pred.shape:", displ_pred.shape)
    preop_deformed = preop + displ_pred
    chamfer_dist_P2T_before_reg = metrics.chamfer(
        prediction=preop,
        target=intraop,
        mode="P2T",
    )
    chamfer_dist_P2T_after_reg = metrics.chamfer(
        prediction=preop_deformed,
        target=intraop,
        mode="P2T",
    )

    chamfer_dist_T2P_before_reg = metrics.chamfer(
        prediction=preop,
        target=intraop,
        mode="T2P",
    )
    chamfer_dist_T2P_after_reg = metrics.chamfer(
        prediction=preop_deformed,
        target=intraop,
        mode="T2P",
    )

    mean_chamfer_dist_P2T_before_reg = chamfer_dist_P2T_before_reg / preop.shape[2]
    mean_chamfer_dist_P2T_after_reg = chamfer_dist_P2T_after_reg / preop.shape[2]
    mean_chamfer_dist_T2P_before_reg = chamfer_dist_T2P_before_reg / intraop.shape[2]
    mean_chamfer_dist_T2P_after_reg = chamfer_dist_T2P_after_reg / intraop.shape[2]
    mean_chamfer_dist_SYM_before_reg = mean_chamfer_dist_P2T_before_reg + mean_chamfer_dist_T2P_before_reg
    mean_chamfer_dist_SYM_after_reg = mean_chamfer_dist_P2T_after_reg + mean_chamfer_dist_T2P_after_reg
    print("Chamfer distance:")
    print("\tBefore registration (P2T):", chamfer_dist_P2T_before_reg.item(), "mean:", mean_chamfer_dist_P2T_before_reg.item())
    print("\tAfter registration (P2T):", chamfer_dist_P2T_after_reg.item(), "mean:", mean_chamfer_dist_P2T_after_reg.item())
    print("\tBefore registration (T2P):", chamfer_dist_T2P_before_reg.item(), "mean:", mean_chamfer_dist_T2P_before_reg.item())
    print("\tAfter registration (T2P):", chamfer_dist_T2P_after_reg.item(), "mean:", mean_chamfer_dist_T2P_after_reg.item())
    print("\tBefore registration (SYM):", mean_chamfer_dist_SYM_before_reg.item())
    print("\tAfter registration (SYM):", mean_chamfer_dist_SYM_after_reg.item())


    # Caluclate landmarks error:
    landmark_error_list = []
    if landmarks_preop is not None and landmarks_intraop is not None:
        # print(landmarks_intraop)
        print("len(landmarks_intraop):", len(landmarks_intraop))
        for idx_l, landmark in enumerate(landmarks_intraop):
            # print(landmark)
            if landmark is not None and landmark != []:
                print(idx_l, landmark.shape, landmarks_preop[idx_l].shape)
                landmark_error = metrics.chamfer(
                    prediction=landmarks_preop[idx_l].float().permute(0, 2, 1),
                    target=landmark.float().permute(0, 2, 1),
                    mode="P2T",
                )
                mean_landmark_error = landmark_error.item() / landmarks_preop[idx_l].shape[1]
                print("idx_l:", idx_l, "mean_landmark_error:", mean_landmark_error, "landmarks_preop[idx_l].shape[1]", landmarks_preop[idx_l].shape[1])
                landmark_error_list.append(mean_landmark_error)
            # else: 
            #     landmark_error_list.append(None)
    mean_landmark_error_overall_landmarks = sum(landmark_error_list) / len(landmark_error_list)
    print("mean_landmark_error_overall_landmarks:", mean_landmark_error_overall_landmarks)

    res = {
        "displ_error": displ_error,
        "displ_target": target_displ,
        "chamfer_dist_P2T_target": chamfer_dist_P2T_before_reg.item(),
        "avg_chamfer_dist_P2T_target": mean_chamfer_dist_P2T_before_reg.item(),
        "chamfer_dist_P2T_error": chamfer_dist_P2T_after_reg.item(),
        "avg_chamfer_dist_P2T_error": mean_chamfer_dist_P2T_after_reg.item(),
        "chamfer_dist_T2P_target": chamfer_dist_T2P_before_reg.item(),
        "avg_chamfer_dist_T2P_target": mean_chamfer_dist_T2P_before_reg.item(),
        "chamfer_dist_T2P_error": chamfer_dist_T2P_after_reg.item(),
        "avg_chamfer_dist_T2P_error": mean_chamfer_dist_T2P_after_reg.item(),
        "chamfer_dist_SYM_target": mean_chamfer_dist_SYM_before_reg.item(),
        "chamfer_dist_SYM_error": mean_chamfer_dist_SYM_after_reg.item(),
        "mean_landmark_error": mean_landmark_error_overall_landmarks,
        "landmark_error_list": landmark_error_list,
    }
    
    return res


def stats_curator_ear_syn(
        displacement_error_list,
        displacement_target_list,
        output_folder,
):

    print("mean displacement error:", sum(displacement_error_list) / len(displacement_error_list))
    print("mean displacement target:", sum(displacement_target_list) / len(displacement_target_list))

    top_10_smallest_displ_err = torch.topk(torch.tensor(displacement_error_list), 10, largest=False, sorted=True)

    print("top 10 smallest displacement error:", top_10_smallest_displ_err.values.tolist())
    print("top 10 smallest displacement error indices:", top_10_smallest_displ_err.indices.tolist())

    top_10_largest_displ_err = torch.topk(torch.tensor(displacement_error_list), 10, largest=True, sorted=True)
    print("top 10 largest displacement error:", top_10_largest_displ_err.values.tolist())
    print("top 10 largest displacement error indices:", top_10_largest_displ_err.indices.tolist())

    res = {
        "displacement_error": displacement_error_list,
        "displacement_target": displacement_target_list,
        "mean_displacement_error": sum(displacement_error_list) / len(displacement_error_list),
        "mean_displacement_target": sum(displacement_target_list) / len(displacement_target_list),
        "top_10_smallest_displ_err": {
            "values": top_10_smallest_displ_err.values.tolist(),
            "indices": top_10_smallest_displ_err.indices.tolist(),
        }
    }

    with open(os.path.join(output_folder, "results.json"), "w") as f:
        json.dump(res, f)
    print("Saved results to:", os.path.join(output_folder, "results.json"))




def stats_curator_ear(
    stats,
    output_folder,
):
    if stats[0]["displ_error"] is not None:
        displacement_error_list = [s["displ_error"] for s in stats]
        displacement_target_list = [s["displ_target"] for s in stats]

    if stats[0]["chamfer_dist_P2T_target"] is not None:
        chamfer_dist_P2T_target_list = [s["chamfer_dist_P2T_target"] for s in stats]
        mean_chamfer_dist_P2T_target_list = [s["avg_chamfer_dist_P2T_target"] for s in stats]
        chamfer_dist_P2T_error_list = [s["chamfer_dist_P2T_error"] for s in stats]
        mean_chamfer_dist_P2T_error_list = [s["avg_chamfer_dist_P2T_error"] for s in stats]

        chamfer_dist_T2P_target_list = [s["chamfer_dist_T2P_target"] for s in stats]
        mean_chamfer_dist_T2P_target_list = [s["avg_chamfer_dist_T2P_target"] for s in stats]
        chamfer_dist_T2P_error_list = [s["chamfer_dist_T2P_error"] for s in stats]
        mean_chamfer_dist_T2P_error_list = [s["avg_chamfer_dist_T2P_error"] for s in stats]

        chamfer_dist_SYM_target_list = [s["chamfer_dist_SYM_target"] for s in stats]
        chamfer_dist_SYM_error_list = [s["chamfer_dist_SYM_error"] for s in stats]

        print("Mean Chamfer distance P2T target:", sum(chamfer_dist_P2T_target_list) / len(chamfer_dist_P2T_target_list))
        print("Mean average Chamfer distance P2T target:", sum(mean_chamfer_dist_P2T_target_list) / len(mean_chamfer_dist_P2T_target_list))
        print("Mean Chamfer distance P2T error:", sum(chamfer_dist_P2T_error_list) / len(chamfer_dist_P2T_error_list))
        print("Mean average Chamfer distance P2T error:", sum(mean_chamfer_dist_P2T_error_list) / len(mean_chamfer_dist_P2T_error_list))

        print("Mean Chamfer distance T2P target:", sum(chamfer_dist_T2P_target_list) / len(chamfer_dist_T2P_target_list))
        print("Mean average Chamfer distance T2P target:", sum(mean_chamfer_dist_T2P_target_list) / len(mean_chamfer_dist_T2P_target_list))
        print("Mean Chamfer distance T2P error:", sum(chamfer_dist_T2P_error_list) / len(chamfer_dist_T2P_error_list))
        print("Mean average Chamfer distance T2P error:", sum(mean_chamfer_dist_T2P_error_list) / len(mean_chamfer_dist_T2P_error_list))

        print("Mean Chamfer distance SYM target:", sum(chamfer_dist_SYM_target_list) / len(chamfer_dist_SYM_target_list))
        print("Mean Chamfer distance SYM error:", sum(chamfer_dist_SYM_error_list) / len(chamfer_dist_SYM_error_list))
    

    if stats[0]["mean_landmark_error"] is not None:
        landmark_error_list = [s["mean_landmark_error"] for s in stats]
        print("Mean landmark error:", sum(landmark_error_list) / len(landmark_error_list))


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, "stats.json")
    with open(output_path, "w") as f:
        json.dump(stats, f)
    print("Saved stats to:", output_path)
            


