import vtk
import os
import re
# import vtk_utils
try:
    from . import vtk_utils
    # from . import extract_surface
    #from . import statistics
    from . import pc_utils
except:
    try:
        import vtk_utils
        # import extract_surface
        #import statistics
        import pc_utils
    except:
        from data import vtk_utils
        # from data import extract_surface
        #from data import statistics
        from data import pc_utils

import torch
from torch.utils.data import Dataset, DataLoader
from vtk.util import numpy_support
import copy
import numpy as np
import random
# import pc_utils


def convert_vtp_to_pcd():
    vtp_path_list = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/LiverData/registration/cleaned/2021_04_19/laparoscopic_liver_segmentation_reduced_clean_downsampled.vtp"

    output_path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/LiverData/registration/cleaned/2021_04_19/laparoscopic_liver_segmentation_reduced_clean_downsampled.ply"

    mesh_vtp = vtk_utils.load_mesh(vtp_path_list)
    print("mesh_vtp.GetNumberOfPoints()", mesh_vtp.GetNumberOfPoints())

    # mesh_vtu = vtk_utils.poly_to_unstructured_grid(mesh_vtp)

    vtk_utils.write_mesh(
        mesh=mesh_vtp,
        filename=output_path,
    )
    print("Saved liver downsampled point cloud to", output_path)
    


def test_load_ply():
    path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/LiverData/registration/cleaned_v2s_formet/2021_04_19/laparoscopic_liver_segmentation_reduced_clean_downsampled_with_normals.ply"
    mesh = vtk_utils.load_mesh(path)
    normals = mesh.GetPointData().GetNormals()
    print("normals", normals)
    normals = numpy_support.vtk_to_numpy(normals)
    print("mesh.GetNumberOfPoints()", mesh.GetNumberOfPoints())
    print("normals.shape", normals.shape)


# convert_vtp_to_pcd()

def reset_seed(seed_num=1):
    """Set random seeds for reproducibility."""
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LaparoscopicDataset(Dataset):
    def __init__(self, 
        folder,
        scale=1,
        preop_file_name="preop.ply",
        intraop_file_name="intraop.ply",
        npoints=2500,
        center=True,
        random_pert = True,
        verbose = False,
    ):
        reset_seed(0)
        self.folder = folder
        self.scale = scale
        self.preop_file_name = preop_file_name
        self.npoints = npoints
        self.intraop_file_name = intraop_file_name
        self.center = center
        self.random_pert =random_pert
        self.valid_folder_list = self.retrieve_valid_folders()
        self.verbose = verbose
        print("Valid folders found:", len(self.valid_folder_list))

    def __len__(self):
        return len(self.valid_folder_list)

    def retrieve_valid_folders(self):
        folder_list = []
        for root, dirs, files in os.walk(self.folder):
            if os.path.exists(os.path.join(root, self.preop_file_name)) \
                and os.path.exists(os.path.join(root, self.intraop_file_name)):
                folder_list.append(root)

        return folder_list


    def __getitem__(self, index, ):
        
        folder_name = self.valid_folder_list[index]
        preop_path = os.path.join(folder_name, self.preop_file_name)
        intraop_path = os.path.join(folder_name, self.intraop_file_name)
        # Difference between self.folder and folder_name:
        relative_path = os.path.relpath(folder_name, self.folder)

        if self.verbose:
          print("folder_name:", folder_name)
          print("self.folder:", self.folder)
          print("relative_path:", relative_path)
        if self.verbose:
          print("Loading:", preop_path)
        preop = vtk_utils.load_mesh(preop_path)
        if self.verbose:
          print("Loading:", intraop_path)
        intraop = vtk_utils.load_mesh(intraop_path)
      
        if preop.GetNumberOfPoints() == 0:
            print("INVALIDIFY 1")
            self.valid = False
            raise IOError(f"Could not load {preop_path}")

        if self.scale != 1:
            t = vtk.vtkTransform()
            t.Scale( self.scale, self.scale, self.scale )
            tf = vtk.vtkTransformFilter()
            tf.SetTransform( t )
            tf.SetInputData( preop )
            tf.Update()
            preop = tf.GetOutput()

        # Dimensions check:
        xmin, xmax, ymin, ymax, zmin, zmax = preop.GetBounds()
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        assert max(dx,dy,dz) < 1 and min(dx,dy,dz) > 0.01, \
              f"Preoperative data loaded from {preop_path} has dimensions ({dx}, {dy}, {dz}).\n" +\
              f"Note: PIVOTS expects its data in meters, use the --scale parameter to adjust.\n" +\
              f"For example '--scale 1e-3' if your data is in millimeters."


        transform_center = None
        if self.center:
            bounds = [0]*6
            preop.GetBounds(bounds)
            x0 = bounds[0]
            x1 = bounds[1]
            y0 = bounds[2]
            y1 = bounds[3]
            z0 = bounds[4]
            z1 = bounds[5]
            transform_center =  [ - (x0 + x1 ) /2, - (y0 + y1 ) /2, - (z0 + z1 ) /2 ]
            # print(bounds)
            t = vtk.vtkTransform()
            t.Translate( transform_center )
            tf = vtk.vtkTransformFilter()
            tf.SetTransform( t )
            tf.SetInputData( preop )
            tf.Update()
            preop = tf.GetOutput()

        angle = 0
        axis = np.array([0, 0, 1])
        translation = np.array([0, 0, 0])
        if self.random_pert:
            # Generate random rotation and translation
            random_rotation = vtk.vtkTransform()
            random_translation = vtk.vtkTransform()

            # Random rotation between 0 and 20 degrees
            angle = np.random.uniform(0, 20)
            axis = np.random.uniform(-1, 1, size=3)
            axis = axis / np.linalg.norm(axis)  # Normalize the axis
            random_rotation.RotateWXYZ(angle, axis[0], axis[1], axis[2])

            # Random translation between 0 and 0.025 meters
            translation = np.random.uniform(-0.025, 0.025, size=3)
            random_translation.Translate(translation)

            # Combine rotation and translation
            combined_transform = vtk.vtkTransform()
            combined_transform.Concatenate(random_rotation)
            combined_transform.Concatenate(random_translation)
            print("rotation angle:", angle, "axis:", axis, "translation:", translation)

            # Apply the transformation to the preop mesh
            transform_filter = vtk.vtkTransformFilter()
            transform_filter.SetTransform(combined_transform)
            transform_filter.SetInputData(preop)
            transform_filter.Update()
            preop = transform_filter.GetOutput()


        # Load preoperative
        preop_surface = preop
        internal_points = vtk_utils.create_random_internal_points(
            surface_mesh=preop_surface,
            points_to_create = preop_surface.GetNumberOfPoints(),
        )

        # surface_normals = vtk_utils.compute_point_normals( preop_surface )
        # preop_surface.GetPointData().SetNormals( surface_normals )
        preop_surface_normals = pc_utils.generate_normals_o3d( numpy_support.vtk_to_numpy(preop_surface.GetPoints().GetData()), 
                                                              show=False, output_path=False )
        
        # debug
        # preop_surface_vtk_normals = vtk_utils.to_pointcloud(
        #     coords=numpy_support.vtk_to_numpy(preop_surface.GetPoints().GetData()),
        #     features=preop_surface_normals,
        #     features_name="Normals",
        # )
        # vtk_utils.write_mesh(
        #     mesh=preop_surface_vtk_normals,
        #     filename="/home/liupeng/ceph_home/others/debug/debug_laparoscopic_normals/preop_normals.vtp",
        # )
        preop_surface = vtk_utils.resample_polydata( preop_surface )

        # Update the normals array:
        surface_normals = preop_surface.GetPointData().GetNormals()

        distance_field = vtk_utils.df( mesh=internal_points, surface=preop_surface )

        preop_internal_xyz = numpy_support.vtk_to_numpy( internal_points.GetPoints().GetData() )
        preop_surface_xyz = numpy_support.vtk_to_numpy( preop_surface.GetPoints().GetData() )
        preop_internal_dists = np.expand_dims( numpy_support.vtk_to_numpy(distance_field), axis = 1 )
        # preop_surface_normals = numpy_support.vtk_to_numpy(surface_normals)

        n_internal_points = preop_internal_xyz.shape[0]
        n_surface_points = preop_surface_xyz.shape[0]

        #print("n_internal_points", n_internal_points)
        #print("n_surface_points", n_surface_points)
        #print("surface_normals", preop_surface_normals.shape)
        #exit()

        # The internal surface points can't have a "normal", so set the value to zero:
        preop_internal_normals = np.zeros( (n_internal_points, 3), dtype=float )
        preop_surface_dists = np.zeros( (n_surface_points, 1), dtype=float )

        if self.npoints > 0:

            # Ensure the number of points in each point cloud are correct. In total, they should equal
            # npoints, and be split into internal and surface points by the given ratio.
            # This will choose a random subset of points, or increase the number of points
            # by adding dummy points where necessary:
            r_internal, r_surface = \
                    pc_utils.set_point_ratio(
                        [preop_internal_xyz, preop_internal_dists, preop_internal_normals],
                        [preop_surface_xyz, preop_surface_dists, preop_surface_normals],
                        ratio = 0.3,
                        total_points = self.npoints ,
                )

            # Re-assign to original variables:
            preop_internal_xyz, preop_internal_dists, preop_internal_normals = r_internal
            preop_surface_xyz, preop_surface_dists, preop_surface_normals = r_surface
        
        preop_xyz = np.concatenate( (preop_internal_xyz, preop_surface_xyz), axis=0 )
        preop_dists = np.concatenate( (preop_internal_dists, preop_surface_dists), axis=0 )
        preop_normals = np.concatenate( (preop_internal_normals, preop_surface_normals), axis=0 )

        preop_np = np.concatenate( (preop_xyz, preop_dists, preop_normals), axis=1 )
        

        # Load intraoperative

        if self.scale != 1:
            t = vtk.vtkTransform()
            t.Scale( self.scale, self.scale, self.scale )
            tf = vtk.vtkTransformFilter()
            tf.SetTransform( t )
            tf.SetInputData( intraop )
            tf.Update()
            intraop = tf.GetOutput()

        # Dimensions check:
        xmin, xmax, ymin, ymax, zmin, zmax = intraop.GetBounds()
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        assert max(dx,dy,dz) < 1 and min(dx,dy,dz) > 0.01, \
              f"Intraoperative data loaded from {intraop_path} has dimensions ({dx}, {dy}, {dz}).\n" +\
              f"Note: PIVOTS expects its data in meters, use the --scale parameter to adjust.\n" +\
              f"For example '--scale 1e-3' if your data is in millimeters."
        
        if self.center:
            t = vtk.vtkTransform()
            t.Translate( transform_center )
            tf = vtk.vtkTransformFilter()
            tf.SetTransform( t )
            tf.SetInputData( intraop )
            tf.Update()
            intraop = tf.GetOutput()

        intraop_ply = vtk_utils.unstructured_grid_to_poly(intraop)
        intraop_ply = vtk_utils.resample_polydata( intraop_ply, max_num_points = 2500, clean=False )
        
        #print("NUM POINTS RESAMPLED:", partial_surface.GetNumberOfPoints())

        #volume_np = vtk_to_numpy(self.volume.GetPoints().GetData())
        intraop_np =  numpy_support.vtk_to_numpy(intraop_ply.GetPoints().GetData())
        # intraop_normals = numpy_support.vtk_to_numpy(intraop.GetPointData().GetNormals())
        intraop_normals = pc_utils.generate_normals_o3d( intraop_np, show=False, output_path=False )


        # debug
        # intraop_vtk_normals = vtk_utils.to_pointcloud(
        #     coords=numpy_support.numpy_to_vtk(intraop_np),
        #     features=intraop_normals,
        #     features_name="Normals",
        # )
        # vtk_utils.write_mesh(
        #     mesh=intraop_vtk_normals,
        #     filename="/home/liupeng/ceph_home/others/debug/debug_laparoscopic_normals/intraop_normals.vtp",
        # )
        # print("Saved intraop normals to /home/liupeng/ceph_home/others/debug/debug_laparoscopic_normals/intraop_normals.vtp")

        intraop_np = np.concatenate( (intraop_np, intraop_normals), axis=1 )

        

        #if scale != 1:
            #surface_np = surface_np * scale
            #control_points_np = control_points_np * scale

        if self.npoints > 0:
            r_surface = pc_utils.set_npoints( [intraop_np], self.npoints )
            intraop_np = r_surface[0]

        return { 
            "preop": torch.FloatTensor(preop_np).transpose(0,1), 
            "intraop": torch.FloatTensor(intraop_np).transpose(0,1),
            "preop_mesh_filepath": preop_path,
            "intraop_mesh_filepath": intraop_path,
            "geometry": {
                "preop": preop,
                "intraop": intraop,
            },
            "perturbation":{
                "rotation": {"angle": angle, "axis": axis},
                "translation": translation,
            },
            "relative_path": relative_path,
            "transform_center": transform_center,
        }
   
    def find_files_by_regex( self, index: int, pattern: str ):
        """
        Retrieve all file paths in a folder (non-recursive) whose filenames
        match the given regex pattern.

        Args:
            index (int): Index of the found sample folders
            pattern (str): Regex pattern to match filenames

        Returns:
            List[str]: List of matching file paths
        """
        folder = self.valid_folder_list[index]
        regex = re.compile(pattern)
        matches = []

        print(f"Looking for {regex} {pattern} in {folder}")
        for name in os.listdir(folder):
            print(name)
            path = os.path.join(folder, name)
            print( os.path.isfile(path), regex.search(name) )
            if os.path.isfile(path) and regex.search(name):
                print("is file, match")
                matches.append(path)

        return matches



if __name__ == "__main__":
    # dataset = LaparascopicDataset(
    #     folder="/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/LiverData/registration/cleaned_v2s_format/",
    #     preop_file_name="preop.vtp",
    #     intraop_file_name="intraop.ply",
    # )

    dataset = LaparoscopicDataset(
        folder="/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/LiverData/registration/depthreconstruction_v2s_format/",
        preop_file_name="output_source_cloud.pcd",
        # preop_file_name="source_test.stl",
        intraop_file_name="output_target_cloud.pcd",
    )

    # test_load_ply()    

    print("len(dataset)", len(dataset))
    # print("dataset[0]", dataset[0])
    print(dataset[0]["preop"].shape, dataset[0]["intraop"].shape)
