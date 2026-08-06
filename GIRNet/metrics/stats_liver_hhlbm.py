import os
import json
import torch
import numpy as np
try:
    from metrics import metrics
    from data import vtk_utils
    from data import pc_utils
except:
    from . import metrics
    from . import vtk_utils
    from . import pc_utils



def surface_distance_calculator(
        preop,
        intraop,
        displ_pred,
        preop_mesh_filename=None,
        output_folder=None,
):
    """Calculate the surface distance between preop and intraop point clouds (downsampled)

    Args:
        preop (torch.tensor): preoperative point cloud, [N, 3]
        intraop (torch.tensor): intraoperative point cloud, [N, 3]
        displ_pred (torch.tensor): predicted displacement field for preoperative point cloud, [N, 3]
        output_folder (str, optional): folder where to save the meshes. Defaults to None.

    Returns:
        float: calculated surface distance
    """
    
    if preop_mesh_filename is not None and os.path.exists(preop_mesh_filename):
        preop_mesh = vtk_utils.load_mesh(preop_mesh_filename)

        valid_mask = preop.abs().max( dim = 1, keepdim = True )[0] < 1e4
        preop = preop[valid_mask.squeeze(-1)]
        displ_pred = displ_pred[valid_mask.squeeze(-1)]

        

        preop_mesh_deformed = vtk_utils.apply_deformation(
            mesh=preop_mesh,
            displacement_mesh=preop,
        )


    print("before masking:")
    print("preop", preop.shape)
    print("intraop", intraop.shape)
    n_points = preop.shape[-1]

    preop_deformed = preop + displ_pred

    points_internal, points_surface = pc_utils.split_surface_internal(
        point_cloud=preop_deformed,
        ratio=0.3
    )

    # mask out dummy points which are larger than 1e4
    valid_mask = points_surface.abs().max( dim = 1, keepdim = True )[0] < 1e4
    points_surface = points_surface[valid_mask.squeeze(-1)]

    valid_mask = points_internal.abs().max( dim = 1, keepdim = True )[0] < 1e4
    points_internal = points_internal[valid_mask.squeeze(-1)]

    valid_mask = intraop.abs().max( dim = 1, keepdim = True )[0] < 1e4
    intraop = intraop[valid_mask.squeeze(-1)]

    print("after mask:")
    print("points_internal", points_internal.shape)
    print("points_surface", points_surface.shape)
    print("intraop", intraop.shape)

    
    if output_folder:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        point_surface_vtk = vtk_utils.to_pointcloud(
            coords=points_surface,
        )

        point_internal_vtk = vtk_utils.to_pointcloud(
            coords=points_internal,
        )
        print("point_surface_vtk", point_surface_vtk.GetNumberOfPoints())
        print("point_internal_vtk", point_internal_vtk.GetNumberOfPoints())

        vtk_utils.write_mesh(
            mesh=point_surface_vtk,
            filename=os.path.join(output_folder, "test_split_surface.vtp"),
            verbose=True,
        )
        vtk_utils.write_mesh(
            mesh=point_internal_vtk,
            filename=os.path.join(output_folder, "test_split_internal.vtp"),
            verbose=True,
        )


    # Calculate Chamfer distance from intraop to preop
    chamfer_dist_intraop_to_preop = metrics.chamfer(
        prediction=points_surface.unsqueeze(0),
        target=intraop.unsqueeze(0),
        mode="T2P",
        reduction="mean",
    )
    print("Chamfer_dist_intraop_to_preop:", chamfer_dist_intraop_to_preop)

    return chamfer_dist_intraop_to_preop.item()
