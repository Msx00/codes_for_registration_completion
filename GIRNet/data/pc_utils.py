import numpy as np
import torch
import sys
sys.path.append("../")

from models.farthest_point_sampling import farthest_point_sampling
from data import vtk_utils
import random
import scipy
import open3d as o3d
from vtk.util import numpy_support

def pc_normalize(pc):
    """Normalize point clouds

    Args:
        pc (np array): point cloud to be normalized

    Returns:
        np array: normalized point cloud
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


## Utility function to ensure a filename has the expected ending:
# def check_extension(filename, expected_extension):
#     """ Utility function to ensure the filename has the expected extension """
#     assert filename.lower().endswith(expected_extension.lower()), \
#             f"Expected {filename} to have extension" +\
#             expected_extension + "!"

def set_npoints( tensors, n_target_points, mode="lin" ):
    """
        Select a subset of points from a point cloud if input point cloud has more points than n_target_points
        or add dummy points if input point cloud has less points than n_target_points.
        downsampling mode:
            "lin": linearly choose points
            "rand": randomly choose points
            "fps": choose points using farthest point sampling

    Args:
        tensors (tensor): input point cloud tensor
        n_target_points (int): number of taget points
        mode (str, optional): downsamling mode. Defaults to "lin".

    Returns:
        tensor: tensor with n_target_points points
    """

    resampled_tensors = []

    n_original_points = tensors[0].shape[0]
    # If we want less points than we currently have, choose a random subset:
    if n_original_points > n_target_points:
        if mode == "rand":
            choice = np.random.choice( n_original_points, n_target_points, replace=False )
        elif mode == "lin":
            choice = np.linspace(0, n_original_points, num=n_target_points, endpoint=False, dtype=int)
        elif mode == "fps":
            try:
                tensor_points = torch.FloatTensor(tensors[0]).transpose(1, 0).unsqueeze(0).cuda()
                # remove duplcates:

                choice, _ = farthest_point_sampling(tensor_points, n_target_points, random=False)
                choice = choice.squeeze(0).cpu().numpy()
                print("tensors[0].shape", tensors[0].shape, "choice", choice.shape, n_target_points)
            except:
                tensor_vtk = vtk_utils.to_pointcloud(
                    coords=numpy_support.numpy_to_vtk(tensors[0], ),
                )
                output_path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/others/debug/debug_fps/wrong_sample.vtp"
                vtk_utils.write_mesh(
                    mesh=tensor_vtk,
                    filename=output_path,
                    verbose=True,
                )
        # print("n_original_points", n_original_points, "--->", n_target_points)
        for i, tensor in enumerate(tensors):
            resampled = tensor[choice,:]
            resampled_tensors.append( resampled )
    # If we want more points, add random zeros:
    elif n_original_points < n_target_points:
        for i, tensor in enumerate(tensors):
            n_dummy_points = n_target_points - n_original_points
            f = tensor.shape[1]     # number of features
            dummy_points = np.full( (n_dummy_points, f), 999999 )
            resampled = np.concatenate( (tensor, dummy_points), axis = 0 )
            resampled_tensors.append( resampled )
    else:
        # Do nothing
        resampled_tensors = tensors

    return resampled_tensors

def set_point_ratio( internal, surface, ratio, total_points ):
    """ Ensure that the internal and surface point clouds have the correct number of points.

    Together, they should have total_points points. Of these, internal will have the 'ratio' fraction
    of points and surface will have '1-ratio' points.
    """
    # Calculate how many internal and how many surface points we want:
    npoints_internal = int( total_points*ratio )
    npoints_surface = total_points - npoints_internal
   
    assert npoints_internal + npoints_surface == total_points

    # print("---internal")
    resampled_internal = set_npoints( 
        tensors=internal, 
        n_target_points=npoints_internal, 
        # mode="fps",
        mode="lin",
    )
    # print("---surface")
    resampled_surface = set_npoints( 
        tensors=surface, 
        n_target_points=npoints_surface, 
        # mode="fps",
        mode="lin",
    )

    return resampled_internal, resampled_surface


def split_surface_internal( point_cloud, ratio=0.3 ):
    """ Split a point cloud into two point clouds, one representing the surface and the other the internal points.

    Args:
        point_cloud (np array): point cloud to be split, [N, D]
        ratio (float): ratio of surface points

    Returns:
        np array: surface point cloud
        np array: internal point cloud
    """
    npoints_total = point_cloud.shape[0]
    npoints_internal = int( npoints_total*ratio )
    npoints_surface = npoints_total - npoints_internal

    internal = point_cloud[0:npoints_internal, :]
    surface = point_cloud[npoints_internal:, :]

    return internal, surface




def generate_normals_o3d( point_cloud, show=False, output_path=False ):
    pcd = o3d.geometry.PointCloud()
    print("point_cloud", point_cloud.shape)
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=50))
    pcd.orient_normals_consistent_tangent_plane(k=10)
    normals = np.asarray(pcd.normals)
    if show:
        o3d.visualization.draw_geometries([pcd])
    if output_path:
        o3d.io.write_point_cloud(output_path, pcd)
    return normals



def preprocess_data( preop, intraop, pad_features_to=0 ):

    print("preop", preop.shape)
    print("intraop", intraop.shape)
    n_points = preop.shape[-1]
    batch_size = preop.shape[0]

    # Compute how many features we already have (everything is considered a feature except the first three
    # values, which are the point coordinates)
    n_preop_features_input = preop.shape[1] - 3
    n_intraop_features_input = intraop.shape[1] - 3

    # First 3 values are the coordinates:
    coords_preop = preop[:,0:3,:]
    coords_intraop = intraop[:,0:3,:]

    df_preop = preop[:,3,:]     # distance field
    df_preop = df_preop.unsqueeze(1)


    batch_size = preop.shape[0]

    # The preop features contain a "-1" for each point and a distance to the nearest surface
    point_type_preop = (-1)*torch.ones( (batch_size, 1, n_points) ).cuda()
    # The intraop features contain a "1" for each point. Since the preop features contain
    # the distance in the second feature, we'll also need to add a second feature, but
    # just keep it at zero.
    point_type_intraop = torch.ones( (batch_size, 1, n_points) ).cuda()

    dists = torch.cdist( coords_preop.permute(0,2,1), coords_intraop.permute(0,2,1) )
    dists_preop_to_intraop, _ = torch.min( dists, dim=2 )
    dists_intraop_to_preop, _ = torch.min( dists, dim=1 )

    dists_preop_to_intraop = dists_preop_to_intraop.unsqueeze(1)
    dists_intraop_to_preop = dists_intraop_to_preop.unsqueeze(1)

    features_preop = torch.cat( (point_type_preop, df_preop, dists_preop_to_intraop), dim=1 )
    features_intraop = torch.cat( (point_type_intraop, torch.zeros_like(df_preop), dists_intraop_to_preop), dim=1 )

    #print("features_preop", features_preop.shape)
    #print("features_intraop", features_intraop.shape)

    if n_preop_features_input > 1:
        features_preop_input = preop[:,4:,:]
        features_preop = torch.cat( (features_preop, features_preop_input), dim=1 )
    if n_intraop_features_input > 0:
        features_intraop_input = intraop[:,3:,:]
        features_intraop = torch.cat( (features_intraop, features_intraop_input), dim=1 )

    #print("features_preop final", features_preop.shape)
    #print("features_intraop final", features_intraop.shape)
 
    return coords_preop, features_preop, coords_intraop, features_intraop





def create_random_transform( max_ang_rad=0.25, max_translation=0.01 ):

    ang = random.uniform( -max_ang_rad, max_ang_rad )
    axis = np.random.rand( 3 )
    axis = axis / np.linalg.norm( axis )

    R = scipy.spatial.transform.Rotation.from_rotvec( ang*axis ).as_matrix()
    # print(R)

    t = np.random.rand(3) * max_translation*2 - max_translation

    H = np.eye(4)
    H[0:3,0:3] = R
    H[0:3,3] = t

    # print("Random transformation H:")
    # print(H)
    return torch.Tensor( H )



def apply_transform( points, H ):

    _, N = points.shape
    # print("N", N)
    H = torch.FloatTensor( H )
    points = torch.FloatTensor( points )
    points = torch.concatenate( (points, torch.ones(1,N)), dim = 0 )

    #transformed = (points.T @ H).T
    transformed = H @ points
    transformed = transformed.numpy()
    return transformed[0:3, :].T





# if __name__ == "__main__":
#     test_triangulate_point_cloud()
