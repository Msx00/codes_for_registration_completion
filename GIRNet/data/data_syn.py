import os, sys
import scipy.spatial
import torch
import re
import random
import warnings
import math
import yaml
import numpy as np
import torch
import vtk

try:
    from . import vtk_utils
    from . import extract_surface
    #from . import statistics
    from . import pc_utils
except:
    try:
        import vtk_utils
        import extract_surface
        import pc_utils
        #import statistics
    except:
        from data import vtk_utils
        from data import extract_surface
        #from data import statistics
        from data import pc_utils


from tqdm import tqdm
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
import copy
warnings.filterwarnings('ignore')
import traceback
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import scipy
vtk.vtkObject.GlobalWarningDisplayOff() 


class LiverSampleCombined():

    def __init__(self, path, int_id, frame = None, check_for_files=True,
            scale=1, load_normals=True, calc_stats=False, ratio = 0.3, 
            append_curvature=False, center=False, **kwargs):
        self.path = path
        self.stats = None
        self.load_stats()
        # print(self.path)
        self.id = int_id
        self.valid = True
        self.scale = scale
        self.frame = frame
        self.filenames = {}
        self.geometry = edict()
        self.displ_array_name = "displacement"
        self.retrieve_filenames(frame=frame, **kwargs)
        if check_for_files:
            self.check_filenames()
        self.calc_stats = calc_stats
        self.ratio = ratio
        self.append_curvature = append_curvature
        self.center= center

    def load_stats( self ):
        stats_filename = os.path.join(self.path, "statistics.yaml")
        if os.path.exists(stats_filename):
            with open(stats_filename) as f:
                self.stats = yaml.safe_load(f)
        else:
            print("no stats file!")
            self.stats = None


    def retrieve_filenames (self, frame=None, **kwargs):
        # super().retrieve_filenames( frame, **kwargs )

        # If needed, add the frame number to the filename:
        # If frame is None this does nothing:
        intraop_filename = self.append_frame( kwargs["intraop_filename"], frame )
        # print("intraop_filename:", intraop_filename, self.valid)
        # intraop_full_filename used for intraoperative normal calculation
        intraop_full_filename = self.append_frame( kwargs["intraop_full_filename"], frame )
        # print("intraop_full_filename:", intraop_full_filename, self.valid)

        # print("updating filename list", kwargs)
        self.filenames.update({
            "preop": kwargs["preop_filename"],
            "intraop": intraop_filename,
            "intraop_full": intraop_full_filename,
        })


        
    def append_frame( self, filename, frame=None ):
        """ If needed, add the frame number to the filename:
        If frame is None this does nothing
        """
        print( filename, frame, self.valid )

        if frame == None:
            return filename

        if frame == math.inf:
            #print(yaml.dump(self.stats), self.valid)
            frame = int(self.stats["SimulationBlock_0"]["simulation_frames"])

        assert type(frame) == int
    
        basename, ext = os.path.splitext( filename )
        filename = f"{basename}_f{frame}{ext}"
        print("filename", filename, self.valid)
        return filename




    def load(self, npoints=0, flip_axes="", min_num_valid_points=-1):
        """Load sample from disk accorrding to filename list

        Args:
            npoints (int, optional): number of points of the sampled point cloud(s). Defaults to 0.

        Raises:
            IOError: File I/O error

        Returns:
            edict(): return a dictionary of the loaded point cloud(s)
                    including:
                        preop: preoperative points
                        displ: intraoperative points
        """
        res = edict()

        ##########################################################
        # print("loading data...")
        preop_path = os.path.join(self.path, self.filenames["preop"])
        preop_volume = vtk_utils.load_mesh(preop_path)
        if preop_volume.GetNumberOfPoints() == 0:
            print("INVALIDIFY 1")
            self.valid = False
            raise IOError(f"Could not load {preop_path}")
        
        intraop_path = os.path.join(self.path, self.filenames["intraop"])
        intraop_mesh = vtk_utils.load_mesh(intraop_path)
        if intraop_mesh.GetNumberOfPoints() == 0:
            print("INVALIDIFY 5")
            self.valid = False
            raise IOError(f"Could not load {intraop_path}")
        
        intraop_full_path = os.path.join(self.path, self.filenames["intraop_full"])
        # print(p)
        intraop_mesh_full = vtk_utils.load_mesh(intraop_full_path)
        if intraop_mesh_full.GetNumberOfPoints() == 0:
            print("INVALIDIFY 5")
            self.valid = False
            raise IOError(f"Could not load {intraop_full_path}")
        
        # if self.center:
        if True:
            preop_volume, offset = vtk_utils.center_mesh(mesh=preop_volume)
            intraop_mesh_full, offset = vtk_utils.center_mesh(mesh=intraop_mesh_full)
            intraop_mesh = vtk_utils.transform_mesh(mesh=intraop_mesh, trans=offset)

        # If the displacement field is not present, but pre- and intraoperative volumes
        # are provided, calculate it from the two volumes:
        if not preop_volume.GetPointData().HasArray(self.displ_array_name) and \
            "volume" in self.filenames["intraop_full"] and "volume" in self.filenames["preop"]:
                # assert preop_volume.GetNumberOfPoints() == intraop_mesh_full.GetNumberOfPoints(),\
                #       "Number of points in preop and intraop volume must be the same. {}".format(self.path)
                if not preop_volume.GetNumberOfPoints() == intraop_mesh_full.GetNumberOfPoints():
                    debug_output_path = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/wrong_samples.txt"
                    with open(debug_output_path, "a") as f:
                        f.write(self.path + "\n")
                        f.close()
                    raise AttributeError("Number of points in preop and intraop volume must be the same. {}".format(self.path))

                displ_vtk_array = vtk.vtkFloatArray()
                displ_vtk_array.SetNumberOfComponents(3)
                displ_vtk_array.SetNumberOfTuples(preop_volume.GetNumberOfPoints())
                displ_vtk_array.SetName(self.displ_array_name)
                for i in range(preop_volume.GetNumberOfPoints()):
                    d = [intraop_mesh_full.GetPoint(i)[c] - preop_volume.GetPoint(i)[c] for c in range(3)]
                    # print(d)
                    displ_vtk_array.SetTuple(i, d)
                preop_volume.GetPointData().AddArray(displ_vtk_array)

                # output_path = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/test_preop_volume_with_displ.vtu"
                # if not os.path.exists(output_path):
                #     vtk_utils.write_mesh(
                #         mesh=preop_volume,
                #         filename=output_path,
                #         verbose=True,
                #     )
                # output_path = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/test_intraop_volume_with_displ.vtu"
                # if not os.path.exists(output_path):
                #     vtk_utils.write_mesh(
                #         mesh=intraop_mesh_full,
                #         filename=output_path,
                #         verbose=True,
                #     )

        # 1. If preop is a surface, randomly create internal points, normals and displacement of such 
        # internal points will be generated later
        # 2. If preop is a volume, split the volume into surface and internal points, and compute the
        if preop_path.endswith(".stl"):
            # preop_surface = copy.deepcopy(preop_volume)
            preop_surface = vtk.vtkPolyData()
            preop_surface.DeepCopy(preop_volume)
            internal_points = vtk_utils.create_random_internal_points(preop_volume,
                    points_to_create = preop_surface.GetNumberOfPoints(), append_surface=False )
        else:
            preop_surface, internal_points = vtk_utils.split_surface_and_internal_points( preop_volume )
            # print("\n")
            # print("******************!split_surface_and_internal_points!!!!!")
            # print(preop_path)
            # print(preop_surface.GetNumberOfPoints(), internal_points.GetNumberOfPoints())


        # Curvature:
        _, preop_surface = vtk_utils.compute_curvature(preop_surface)

        # For the surface, compute a normal for each point:
        surface_normals = vtk_utils.compute_point_normals( preop_surface )
        preop_surface.GetPointData().SetNormals( surface_normals )

        # NOTE: If scale != 1, the resolution of resample_polydata should probably be adjusted first:
        preop_surface = vtk_utils.resample_polydata( preop_surface )
        # preop_surface = vtk_utils.remove_duplicates(preop_surface)
        # Update the normals array:
        surface_normals = preop_surface.GetPointData().GetNormals()
        preop_surface_curvature = preop_surface.GetPointData().GetArray("Curvature")
        preop_surface_curvature = vtk_to_numpy(preop_surface_curvature)
        preop_surface_curvature = np.expand_dims(preop_surface_curvature, axis=1)

        # For the internal points, compute the distance to the surface
        distance_field = vtk_utils.df( mesh=internal_points, surface=preop_surface )
        preop_internal_xyz = vtk_to_numpy( internal_points.GetPoints().GetData() )
        preop_surface_xyz = vtk_to_numpy( preop_surface.GetPoints().GetData() )
        preop_internal_dists = np.expand_dims( vtk_to_numpy(distance_field), axis = 1 )

        # Find duplicated points in internal points using the calculated distance field.
        # If the distance is smaller than threshold, the point is on the surface and should be removed from the internal points.
        # duplicate_index_mask = preop_internal_dists < 1e-6
        is_not_duplicate_preop_internal = preop_internal_dists[:, 0] >= 1e-7
        # print("is_not_duplicate", is_not_duplicate_preop_internal.shape, "preop_internal_xyz", preop_internal_xyz.shape)
        # print("before:", preop_internal_xyz.shape)
        preop_internal_xyz = preop_internal_xyz[is_not_duplicate_preop_internal]
        # print("after:", preop_internal_xyz.shape)
        preop_internal_dists = preop_internal_dists[is_not_duplicate_preop_internal]

        preop_surface_normals = vtk_to_numpy(surface_normals)

        n_internal_points = preop_internal_xyz.shape[0]
        n_surface_points = preop_surface_xyz.shape[0]

        # The internal surface points can't have a "normal", so set the value to zero:
        preop_internal_normals = np.zeros( (n_internal_points, 3), dtype=float )
        # The distance of surface to surface remains zero:
        preop_surface_dists = np.zeros( (n_surface_points, 1), dtype=float )

        preop_internal_curvature = np.zeros( (n_internal_points, 1), dtype=float )

        # print(preop_surface)
        # output_path = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/test_preop_surface_with_displ.vtp"
        # if not os.path.exists(output_path):
        #     vtk_utils.write_mesh(
        #         mesh=preop_surface,
        #         filename=output_path,
        #         verbose=True,
        #     )
        # output_path = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/test_preop_internal_with_displ.vtp"
        # if not os.path.exists(output_path):
        #     vtk_utils.write_mesh(
        #         mesh=internal_points,
        #         filename=output_path,
        #         verbose=True,
        #     )

        DEBUG_DUPLICATES = False
        if DEBUG_DUPLICATES:
            mask = np.absolute(preop_surface_xyz[:, 0]) < 1e3
            preop_xyz_debug = preop_surface_xyz[mask]
            preop_xyz_unique = np.unique(preop_xyz_debug, axis=0)
            # print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)

            has_duplicate = preop_xyz_debug.shape[0] != preop_xyz_unique.shape[0]
            if has_duplicate:
                print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)
                assert False, "There are duplicates in the preop point cloud! {}".format(preop_path)

            mask = np.absolute(preop_internal_xyz[:, 0]) < 1e3
            preop_xyz_debug = preop_internal_xyz[mask]
            preop_xyz_unique = np.unique(preop_xyz_debug, axis=0)
            # print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)

            has_duplicate = preop_xyz_debug.shape[0] != preop_xyz_unique.shape[0]
            if has_duplicate:
                print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)
                assert False, "There are duplicates in the preop point cloud!"


        ##########################################################
        ##################   intraoperative data ################
        ##########################################################
        # Ensure the full mesh has normals:
        intraop_mesh_full_surface = vtk_utils.extract_surface(intraop_mesh_full)
        normals = vtk_utils.compute_point_normals( intraop_mesh_full_surface ) 
        intraop_mesh_full_surface.GetPointData().SetNormals( normals )

        # Copy the normals from the full mesh to the partial mesh:
        vtk_utils.copy_normals( intraop_mesh_full_surface, intraop_mesh )

        # NOTE: If scale != 1, the resolution of resample_polydata should probably be adjusted first:
        # print("from stats:", self.stats["AddSurfaceNoiseBlock_number_of_remaining_points_1"])
        # print("before resample_polydata intraop_mesh.GetNumberOfPoints()", intraop_mesh.GetNumberOfPoints())
        # intraop_mesh = vtk_utils.resample_polydata( intraop_mesh, max_num_points=2500, clean=False )
        intraop_target_distance = 0.005
        intraop_mesh_resampled = vtk_utils.resample_polydata( intraop_mesh, clean=False, target_distance=intraop_target_distance )
        # print("after resample_polydata intraop_mesh.GetNumberOfPoints()", intraop_mesh.GetNumberOfPoints())
        idx_resample = 0
        while min_num_valid_points > 0 and intraop_mesh_resampled.GetNumberOfPoints() < min_num_valid_points * 0.8 and idx_resample < 10:
            intraop_target_distance *= 0.9
            intraop_mesh_resampled = vtk_utils.resample_polydata( intraop_mesh, clean=False, target_distance=intraop_target_distance )
            # print("sample idx:", self.id, "resampling intraop_mesh with target_distance:", intraop_target_distance, "min_num_valid_points * 0.8:", min_num_valid_points * 0.8, "intraop_mesh_resampled.GetNumberOfPoints()", intraop_mesh_resampled.GetNumberOfPoints())
            idx_resample += 1
        # if idx_resample > 0:
        #     print("sample idx", self.id, "resampled {} times".format(idx_resample + 1),  "after resample: intraop_mesh_resampled.GetNumberOfPoints()", intraop_mesh_resampled.GetNumberOfPoints())
        intraop_mesh = intraop_mesh_resampled
        # print("after resample_polydata intraop_mesh.GetNumberOfPoints()", intraop_mesh.GetNumberOfPoints())
        intraop_xyz = vtk_to_numpy(intraop_mesh.GetPoints().GetData())

        # remove duplicates in intraop_xyz
        # print("before:", intraop_xyz.shape)
        intraop_xyz = intraop_xyz.astype(np.float32)
        intraop_xyz, intraop_xyz_unique_index = np.unique(intraop_xyz.round(decimals=8), axis=0, return_index=True)
        # print("after:", intraop_xyz.shape)

        intraop_normals = vtk_to_numpy(intraop_mesh.GetPointData().GetNormals())
        intraop_normals = intraop_normals[intraop_xyz_unique_index]

        _, intraop_mesh_full_surface = vtk_utils.compute_curvature(mesh=intraop_mesh_full_surface)
        vtk_utils.copy_curvature(full_source_mesh=intraop_mesh_full_surface, partial_target_mesh=intraop_mesh)

        intraop_curvature = vtk_to_numpy(intraop_mesh.GetPointData().GetArray("Curvature"))
        intraop_curvature = intraop_curvature[intraop_xyz_unique_index]
        intraop_curvature = np.expand_dims(intraop_curvature, axis=1)
        if self.append_curvature:
            intraop = np.concatenate( (intraop_xyz, intraop_normals, intraop_curvature), axis=1 )
        else:
            intraop = np.concatenate( (intraop_xyz, intraop_normals), axis=1 )

        # TODO what happend if the preoperative mesh is filled with new random internal points?
        self.displ_array_name = "displacement"
        if self.frame:
            frame = self.frame
            if frame == math.inf:
                #print(yaml.dump(self.stats), self.valid)
                frame = int(self.stats["SimulationBlock_0"]["simulation_frames"])
            self.displ_array_name = f"displacement_f{frame}"

        displ_internal = internal_points.GetPointData().GetArray( self.displ_array_name )
        displ_surface = preop_surface.GetPointData().GetArray( self.displ_array_name )

        if not displ_internal or not displ_surface:
            # print("INVALIDIFY 2")
            self.valid = False
            print("WARNING: Ground-truth displacment field not found! Setting to zero.")
            #displ = np.zeros( ( preop.shape[0], 3) )
            displ_internal_np = np.zeros( ( preop_internal_xyz.shape[0], 3) )
            displ_surface_np = np.zeros( ( preop_surface_xyz.shape[0], 3) )
        else:
            displ_internal_np = vtk_to_numpy( displ_internal )
            displ_surface_np = vtk_to_numpy( displ_surface )

            displ_internal_np = displ_internal_np[is_not_duplicate_preop_internal]
            #displ = vtk_to_numpy(displ)

        # print("preop_internal_xyz", preop_internal_xyz.shape)
        # print("preop_surface_xyz", preop_surface_xyz.shape)
        # print("intraop_xyz.shape", intraop_xyz.shape)

        # Subsampling and ratio adjustment:
        # downsample the all preoperative point clouds so that they have the same amount of points 
        if npoints > 0:
            # Ensure the number of points in each point cloud are correct. In total, they should equal
            # npoints, and be split into internal and surface points by the given ratio.
            # This will choose a random subset of points, or increase the number of points
            # by adding dummy points where necessary:
            r_internal, r_surface = \
                    pc_utils.set_point_ratio(
                        internal=[preop_internal_xyz, preop_internal_dists, preop_internal_normals, displ_internal_np, preop_internal_curvature],
                        surface=[preop_surface_xyz, preop_surface_dists, preop_surface_normals, displ_surface_np, preop_surface_curvature],
                        ratio = self.ratio,
                        total_points = npoints,
            )
            # Re-assign to original variables:
            preop_internal_xyz, preop_internal_dists, preop_internal_normals, displ_internal_np, preop_internal_curvature = r_internal
            preop_surface_xyz, preop_surface_dists, preop_surface_normals, displ_surface_np, preop_surface_curvature = r_surface
            # The set_point_ratio function sets some values to 99999 for the dummy points. For the displacement
            # field, this is not good, because the network will output 0 for dummy points. To not influence
            # the displacement error calculation, set the displacement for these points to 0:
            displ_internal_np[displ_internal_np > 5000] = 0
            displ_surface_np[displ_surface_np > 5000] = 0

            # print("preop_internal_xyz", preop_internal_xyz.shape)
            # print("preop_surface_xyz", preop_surface_xyz.shape)

            # Same for intraoperative points:

            r_surface = pc_utils.set_npoints( [intraop], npoints )
            intraop = r_surface[0]

        # DEBUG_DUPLICATES = True
        if DEBUG_DUPLICATES:
            mask = np.absolute(preop_internal_xyz[:, 0]) < 1e3
            preop_xyz_debug = preop_internal_xyz[mask]
            preop_xyz_unique = np.unique(preop_xyz_debug, axis=0)
            # print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)

            has_duplicate = preop_xyz_debug.shape[0] != preop_xyz_unique.shape[0]
            if has_duplicate:
                print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)
                assert False, "There are duplicates in the preop point cloud!"

            mask = np.absolute(preop_surface_xyz[:, 0]) < 1e3
            preop_xyz_debug = preop_surface_xyz[mask]
            preop_xyz_unique = np.unique(preop_xyz_debug, axis=0)
            # print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)

            has_duplicate = preop_xyz_debug.shape[0] != preop_xyz_unique.shape[0]
            if has_duplicate:
                print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)
                assert False, "There are duplicates in the preop point cloud!"


        # Combine internal points and surface points along point axis (0):
        preop_xyz = np.concatenate( (preop_internal_xyz, preop_surface_xyz), axis=0 )
        preop_dists = np.concatenate( (preop_internal_dists, preop_surface_dists), axis=0 )
        preop_normals = np.concatenate( (preop_internal_normals, preop_surface_normals), axis=0 )
        preop_curvature = np.concatenate( (preop_internal_curvature, preop_surface_curvature), axis=0 )


        displ = np.concatenate( (displ_internal_np, displ_surface_np), axis=0 )
        # print("displ_internal_np", displ_internal_np.shape, "displ_surface_np", displ_surface_np.shape)
        # print("displ", displ.shape)
        if self.append_curvature:
            preop = np.concatenate( (preop_xyz, preop_dists, preop_normals, preop_curvature), axis=1 )
        else:
            preop = np.concatenate( (preop_xyz, preop_dists, preop_normals), axis=1 )

        if npoints > 0:
            assert preop.shape[0] == npoints
            assert displ.shape[0] == npoints
            assert intraop.shape[0] == npoints

        # # save debug:
        # mask = np.absolute(preop_xyz[:, 0]) < 1e3
        # preop_xyz_debug = preop_xyz[mask]
        # displ_debug = displ[mask]
        # print("preop_xyz_debug", preop_xyz_debug.shape)
        # print("displ_debug", displ_debug.shape)
        # preop_vtk_debug = vtk_utils.to_pointcloud(
        #     coords=numpy_to_vtk(preop_xyz_debug),
        #     features=displ_debug,
        #     features_name="displ",
        # )
        # output_path = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/test_preop_sampled_w_displ.vtp"
        # if not os.path.exists(output_path):
        #     vtk_utils.write_mesh(
        #         mesh=preop_vtk_debug,
        #         filename=output_path,
        #         verbose=True,
        #     )

        preop = preop * self.scale
        intraop = intraop * self.scale 
        displ = displ * self.scale

        if "X" in flip_axes:
            preop[:, 0] = -preop[:, 0]      # flip pos.x
            preop[:, 4] = -preop[:, 4]      # flip normal.x
            displ[:, 0] = -displ[:, 0]
            intraop[:, 0] = -intraop[:, 0]
            intraop[:, 3] = -intraop[:, 3]
        if "Y" in flip_axes:
            preop[:, 1] = -preop[:, 1]      # flip pos.y
            preop[:, 5] = -preop[:, 5]      # flip normal.y
            displ[:, 1] = -displ[:, 1]
            intraop[:, 1] = -intraop[:, 1]
            intraop[:, 4] = -intraop[:, 4]
        if "Z" in flip_axes:
            preop[:, 2] = -preop[:, 2]
            preop[:, 6] = -preop[:, 6]
            displ[:, 2] = -displ[:, 2]
            intraop[:, 2] = -intraop[:, 2]
            intraop[:, 5] = -intraop[:, 5]

        # save the original VTK meshes of preoperative volume and surface to a dict
        self.geometry.update({
            "preop_volume": preop_volume,
            "preop_surface": preop_surface,
            "preop_internal": internal_points,
            "intraop_surface": intraop_mesh,
            "intraop_volume": intraop_mesh_full,
        })

        # print("preop", preop.shape)
        # print("displ", displ.shape)
        # print("intraop", intraop.shape)

        # mask = np.absolute(preop[:, 0]) < 1e3
        # preop_xyz_debug = preop[mask]
        # preop_xyz_debug = preop_xyz_debug[:, 0:3]0ß
        # displ_debug = displ[mask]
        # print("preop_xyz_debug", preop_xyz_debug.shape)
        # print("displ_debug", displ_debug.shape)
        # preop_vtk_debug = vtk_utils.to_pointcloud(
        #     coords=numpy_to_vtk(preop_xyz_debug),
        #     features=displ_debug,
        #     features_name="displ",
        # )
        # output_path = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/test_preop_sampled_w_displ.vtp"
        # if not os.path.exists(output_path):
        #     vtk_utils.write_mesh(
        #         mesh=preop_vtk_debug,
        #         filename=output_path,
        #         verbose=True,
        #     )

        # debug duplicates:
        if DEBUG_DUPLICATES:
            preop_xyz = preop[:, 0:3]
            mask = np.absolute(preop_xyz[:, 0]) < 1e3
            preop_xyz_debug = preop_xyz[mask]
            preop_xyz_unique = np.unique(preop_xyz_debug, axis=0)
            print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)

            has_duplicate = preop_xyz_debug.shape[0] != preop_xyz_unique.shape[0]
            if has_duplicate:
                print("preop_xyz_debug.shape", preop_xyz_debug.shape, " preop_xyz_unique.shape", preop_xyz_unique.shape)
                assert False, "There are duplicates in the preop point cloud!"
            
            print("preop_xyz[0, :3]", preop_xyz[0, :])
            print("preop_xyz[750, :3]", preop_xyz[750, :])

            preop_xyz_torch = torch.from_numpy(preop_xyz)
            preop_xyz_torch = preop_xyz_torch.float()
            mask = torch.all(torch.abs(preop_xyz_torch) < 100, dim=1)
            print("mash.shape", mask.shape)
            preop_xyz_torch_filtered = preop_xyz_torch[mask]
            preop_xyz_torch_unique = torch.unique(preop_xyz_torch_filtered, dim=0)
            has_duplicate = preop_xyz_torch_filtered.shape[0] != preop_xyz_torch_unique.shape[0]
            print("preop_xyz_torch_filtered.shape", preop_xyz_torch_filtered.shape, " preop_xyz_torch_unique.shape", preop_xyz_torch_unique.shape)
            if has_duplicate:
                assert False, "Duplicate points in preop, preop_unique.shape:{}, preop_filtered.shape:{}".format(preop_xyz_torch_unique.shape, preop_xyz_torch_filtered.shape)

            # calculate distance between surface and internal points, and find distances that are zero
            preop_surface_xyz_torch = torch.from_numpy(preop_surface_xyz)
            preop_internal_xyz_torch = torch.from_numpy(preop_internal_xyz)
            # change to float32
            preop_surface_xyz_torch = preop_surface_xyz_torch.float()
            preop_internal_xyz_torch = preop_internal_xyz_torch.float()

            distances = torch.cdist(preop_surface_xyz_torch, preop_internal_xyz_torch, p=2)

            duplicate_pairs = torch.nonzero(distances < 1e-6, as_tuple=False)  # (i, j)

            # 获取重复点的坐标
            duplicate_coords = torch.cat((preop_surface_xyz_torch[duplicate_pairs[:, 0]], preop_surface_xyz_torch[duplicate_pairs[:, 1]]), dim=0).unique(dim=0)

            # 转换索引对为列表形式
            duplicate_indices = [(pair[0].item(), pair[1].item()) for pair in duplicate_pairs]
            # print("duplicate_pairs", duplicate_pairs.shape, duplicate_pairs)
            print("duplicate_indices", len(duplicate_indices), duplicate_indices)
            print("duplicate_coords", duplicate_coords.shape, duplicate_coords)

    


        # print(preop[:10, :3])
        # print("preop[0, :3]", preop[0, :3])
        # print("preop[750, :3]", preop[750, :3])

        # return preop and displ in shape of [F, n_points]

        youngs_modulus = 0.0
        poisson_ratio = 0.0
        if self.stats is not None and "SimulationBlock_0" in self.stats.keys():
            youngs_modulus = self.stats["SimulationBlock_0"]["young_modulus"]
            poisson_ratio = self.stats["SimulationBlock_0"]["poisson_ratio"]
        elif  self.stats is not None and  "SofaSimulationBlock_young_modulus" in self.stats.keys():
            youngs_modulus = self.stats["SofaSimulationBlock_young_modulus"]
            poisson_ratio = self.stats["SofaSimulationBlock_poisson_ratio"]

        res.update({
            "preop": preop.transpose( (1,0) ),
            "displ": displ.transpose( (1,0) ),
            "intraop": intraop.transpose( (1,0) ),
            "idx": [self.id,],
            "youngs_modulus": [youngs_modulus],
            "poissons_ratio": [poisson_ratio],
        })

        return res


    def check_filenames(self):
        """Check if the file path is correct
        """
        for k, f in self.filenames.items():
            if isinstance(f, list):
                for ff in f:
                    print("checking:", k, ff)
                    if not os.path.exists(os.path.join(self.path, ff)):
                        print("no such file", os.path.join(self.path, ff))
                        print("INVALIDIFY 3")
                        self.valid = False
            else:
                print("checking:", k, f)
                if not os.path.exists(os.path.join(self.path, f)):
                    print("INVALIDIFY 4")
                    print("no such file", os.path.join(self.path, f))
                    self.valid = False


    def write(self, coords, features=None, filename="preop_volume_with_est_displacement.vtp", log_dir=None ):
        """Write point clouds to disk as mesh

        Args:
            coords (torch.Tensor): coordinates of the point cloud, shape: [3, N]
            filename (str, optional): filename of the output file. Defaults to "preop_volume_with_est_displacement.vtp".
            log_dir (str, optional): folder where to save the output file. Defaults to None.
        """
        if isinstance(coords, torch.Tensor):
            coords = coords.permute(1,0).cpu().numpy()
        if features != None and isinstance(features, torch.Tensor):
            features = features.permute(1,0).cpu().numpy()

        #print("Writing results point cloud")
        #print("Coords:", coords.shape)
        #if features is not None:
        #    print("Features:", features.shape)

        pd = vtk_utils.to_pointcloud(
            coords=numpy_to_vtk(coords/self.scale),
            features=(features/self.scale) if (features is not None) else None
        )
        if log_dir is None:
            p = os.path.join(self.path, filename)
        else:
            #int_str = DisplDataset.int_id_to_str( self.id )
            sample_folder_name = os.path.basename( self.path )
            study_folder_name = os.path.basename( os.path.dirname( self.path ) )
            folder_path = os.path.join(log_dir, study_folder_name, sample_folder_name)
            print(self.path, sample_folder_name, study_folder_name, folder_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            p = os.path.join(folder_path, filename)
        # DataSample.write_vtp(p, pd)
        vtk_utils.write_mesh(mesh=pd, filename=p, verbose=True)






if __name__ == "__main__":
    pass
