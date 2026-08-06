import os
import numpy as np
import torch
import vtk
import math
import pickle as pkl

from torch.utils.data import Dataset
from glob import glob
from easydict import EasyDict as edict
from data import pc_utils, vtk_utils
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from data.data_syn import LiverSampleCombined
from data.data import DisplDataset, LiverSample


class LiverSampleAMOS(LiverSampleCombined):
    def __init__(self, 
        path, 
        int_id, 
        frame=None, 
        check_for_files=True, 
        scale=1, 
        load_normals=True, 
        return_all_intraop=True,
        **kwargs
    ):
        super().__init__(path, int_id, frame, check_for_files, scale, load_normals, **kwargs)
        self.intraop_perlin_noise_list, self.intraop_gaussian_noise_list = self.retrieve_noise_stats(self.filenames["intraop"])
        self.stats = kwargs.get("stats", None)
        self.return_all_intraop = return_all_intraop

    def retrieve_filenames(self, frame=None, **kwargs):
        # rewrite retrieve_filenames to get the intraoperative names for AMOS dataset

        # If a single exact intraop filename is provided, use it directly and skip directory scan.
        if "intraop_filename" in kwargs:
            intraop_surface_filename_list = [kwargs["intraop_filename"]]
        else:
            intraop_surface_filename_prefix = kwargs.get('intraop_surface_filename_prefix', 'intraop') # second argument is default value
            print("intraop_surface_filename_prefix", intraop_surface_filename_prefix)
            frame = kwargs["intraop_full_filename"].split("_")[2][:2]
            intraop_surface_filename_list = sorted(filter(lambda x: intraop_surface_filename_prefix in x and frame in x, os.listdir(self.path)))

        self.filenames.update({
            "preop": kwargs["preop_filename"],
            "intraop": intraop_surface_filename_list,
            "intraop_full": kwargs["intraop_full_filename"],
        })
        

    def retrieve_noise_stats(self, intraop_surface_filename_list):
        # liver_surface_partial_noisy_pl_0_gs_0_f0.vtp
        perlin_noise_list = []
        gaussian_noise_list = []
        try:
            for filename in intraop_surface_filename_list:
                # print(filename)
                perlin_noise = int(filename.split('_')[5])
                gaussian_noise = float(filename.split('_')[7].replace("-", "."))
                perlin_noise_list.append(perlin_noise)
                gaussian_noise_list.append(gaussian_noise)
        except:
            print("ERROR: noise stats not found!")
            perlin_noise_list = [0] * len(intraop_surface_filename_list)
            gaussian_noise_list = [0.0] * len(intraop_surface_filename_list)

        return perlin_noise_list, gaussian_noise_list


    def load(self, npoints=0, flip_axes="", min_num_valid_points=-1, preop_volume=None, intraop_mesh=None, intraop_mesh_full=None):
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
        

        intraop_full_path = os.path.join(self.path, self.filenames["intraop_full"])
        # print(p)
        intraop_mesh_full = vtk_utils.load_mesh(intraop_full_path)
        if intraop_mesh_full.GetNumberOfPoints() == 0:
            print("INVALIDIFY 5")
            self.valid = False
            raise IOError(f"Could not load {intraop_full_path}")
        
        ### test translate all meshes to further away from the origin:
        # Get the center of preop_volume:
        # bounds = [0]*6
        # preop_volume.GetBounds(bounds)
        # dx = (bounds[1]+bounds[0])*0.5
        # dy = (bounds[3]+bounds[2])*0.5
        # dz = (bounds[5]+bounds[4])*0.5

        # # Translate the preop_volume by three times of the origin to make it further away from the origin:
        # offset = [dx*3, dy*3, dz*3]
        # preop_volume = vtk_utils.transform_mesh(preop_volume, offset)
        # # same for the itnraop_mesh_full
        # intraop_mesh_full = vtk_utils.transform_mesh(intraop_mesh_full, offset)
        ### end test (finished, rigid offset plays important role in the data)

        ### Center all meshes according to the bounding box center of preop_volume:
        preop_volume, offset = vtk_utils.center_mesh(mesh=preop_volume)
        intraop_mesh_full = vtk_utils.transform_mesh(mesh=intraop_mesh_full, trans=offset)

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

        # preop_volume = vtk_utils.remove_duplicates(preop_volume)
        
        # 1. If preop is a surface, randomly create internal points, normals and displacement of such 
        # internal points will be generated later
        # 2. If preop is a volume, split the volume into surface and internal points, and compute the
        if preop_path.endswith(".stl"):
            # preop_surface = copy.deepcopy(preop_volume)
            preop_surface = vtk.vtkPolyData()
            preop_surface.DeepCopy(preop_volume)
            internal_points = vtk_utils.create_random_internal_points(
                surface_mesh =preop_volume,
                points_to_create = preop_surface.GetNumberOfPoints(), 
                append_surface =False,
            )
        else:
            preop_surface, internal_points = vtk_utils.split_surface_and_internal_points( preop_volume )

        # For the surface, compute a normal for each point:
        surface_normals = vtk_utils.compute_point_normals( preop_surface )
        preop_surface.GetPointData().SetNormals( surface_normals )

        # NOTE: If scale != 1, the resolution of resample_polydata should probably be adjusted first:
        preop_surface = vtk_utils.resample_polydata( preop_surface )
        # Update the normals array:
        surface_normals = preop_surface.GetPointData().GetNormals()
        
        # For the internal points, compute the distance to the surface

        distance_field = vtk_utils.df( mesh=internal_points, surface=preop_surface )
        preop_internal_xyz = vtk_to_numpy( internal_points.GetPoints().GetData() )
        preop_surface_xyz = vtk_to_numpy( preop_surface.GetPoints().GetData() )
        preop_internal_dists = np.expand_dims( vtk_to_numpy(distance_field), axis = 1 )
        preop_surface_normals = vtk_to_numpy(surface_normals)

        n_internal_points = preop_internal_xyz.shape[0]
        n_surface_points = preop_surface_xyz.shape[0]

        # The internal surface points can't have a "normal", so set the value to zero:
        preop_internal_normals = np.zeros( (n_internal_points, 3), dtype=float )
        # The distance of surface to surface remains zero:
        preop_surface_dists = np.zeros( (n_surface_points, 1), dtype=float )



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
            #displ = vtk_to_numpy(displ)

        # Subsampling and ratio adjustment:
        # downsample the all preoperative point clouds so that they have the same amount of points 
        if npoints > 0:
            # Ensure the number of points in each point cloud are correct. In total, they should equal
            # npoints, and be split into internal and surface points by the given ratio.
            # This will choose a random subset of points, or increase the number of points
            # by adding dummy points where necessary:
            r_internal, r_surface = \
                    pc_utils.set_point_ratio(
                        internal=[preop_internal_xyz, preop_internal_dists, preop_internal_normals, displ_internal_np],
                        surface=[preop_surface_xyz, preop_surface_dists, preop_surface_normals, displ_surface_np],
                        ratio = 0.3,
                        total_points = npoints,
            )
            # Re-assign to original variables:
            preop_internal_xyz, preop_internal_dists, preop_internal_normals, displ_internal_np = r_internal
            preop_surface_xyz, preop_surface_dists, preop_surface_normals, displ_surface_np = r_surface
            # The set_point_ratio function sets some values to 99999 for the dummy points. For the displacement
            # field, this is not good, because the network will output 0 for dummy points. To not influence
            # the displacement error calculation, set the displacement for these points to 0:
            displ_internal_np[displ_internal_np > 5000] = 0
            displ_surface_np[displ_surface_np > 5000] = 0



        # Combine internal points and surface points along point axis (0):
        preop_xyz = np.concatenate( (preop_internal_xyz, preop_surface_xyz), axis=0 )
        preop_dists = np.concatenate( (preop_internal_dists, preop_surface_dists), axis=0 )
        preop_normals = np.concatenate( (preop_internal_normals, preop_surface_normals), axis=0 )
        displ = np.concatenate( (displ_internal_np, displ_surface_np), axis=0 )
        preop = np.concatenate( (preop_xyz, preop_dists, preop_normals), axis=1 )

        if npoints > 0:
            assert preop.shape[0] == npoints
            assert displ.shape[0] == npoints
            # assert intraop.shape[0] == npoints

        preop = preop * self.scale
        displ = displ * self.scale

        if "X" in flip_axes:
            preop[:, 0] = -preop[:, 0]      # flip pos.x
            preop[:, 4] = -preop[:, 4]      # flip normal.x
            displ[:, 0] = -displ[:, 0]
        if "Y" in flip_axes:
            preop[:, 1] = -preop[:, 1]      # flip pos.y
            preop[:, 5] = -preop[:, 5]      # flip normal.y
            displ[:, 1] = -displ[:, 1]
        if "Z" in flip_axes:
            preop[:, 2] = -preop[:, 2]
            preop[:, 6] = -preop[:, 6]
            displ[:, 2] = -displ[:, 2]

        preop = preop.transpose( (1,0) )
        displ = displ.transpose( (1,0) )



        ##########################################################
        ##################   intraoperative data ################
        ##########################################################
        intraop_list = []
        intraop_mesh_list = []

        # Ensure the full mesh has normals:
        intraop_mesh_full_surface = vtk_utils.extract_surface(intraop_mesh_full)
        normals = vtk_utils.compute_point_normals( intraop_mesh_full_surface ) 
        intraop_mesh_full_surface.GetPointData().SetNormals( normals )

        for intraop_filename in self.filenames["intraop"]:
            intraop_path = os.path.join(self.path,intraop_filename)
            intraop_mesh = vtk_utils.load_mesh(intraop_path)
            if intraop_mesh.GetNumberOfPoints() == 0:
                print("INVALIDIFY 5")
                self.valid = False
                raise IOError(f"Could not load {intraop_path}")
            # center the intraop_mesh according to the preop_volume:
            intraop_mesh = vtk_utils.transform_mesh(intraop_mesh, offset)
            intraop_mesh_list.append(intraop_mesh)

            # Copy the normals from the full mesh to the partial mesh:
            vtk_utils.copy_normals( intraop_mesh_full_surface, intraop_mesh )

            # NOTE: If scale != 1, the resolution of resample_polydata should probably be adjusted first:
            # print("intraop_mesh before resample:", intraop_mesh.GetNumberOfPoints())
            # intraop_mesh = vtk_utils.resample_polydata( intraop_mesh, max_num_points=2500, clean=False)
            intraop_target_distance = 0.005
            intraop_mesh_resampled = vtk_utils.resample_polydata( intraop_mesh, clean=False, target_distance=intraop_target_distance )
            # print("after resample_polydata intraop_mesh.GetNumberOfPoints()", intraop_mesh.GetNumberOfPoints())
            idx_resample = 0
            while min_num_valid_points > 0 and intraop_mesh_resampled.GetNumberOfPoints() < min_num_valid_points * 0.8 and idx_resample < 10:
                intraop_target_distance *= 0.9
                intraop_mesh_resampled = vtk_utils.resample_polydata( intraop_mesh, clean=False, target_distance=intraop_target_distance )
                print("sample idx:", self.id, "resampling intraop_mesh with target_distance:", intraop_target_distance, "min_num_valid_points * 0.8:", min_num_valid_points * 0.8, "intraop_mesh_resampled.GetNumberOfPoints()", intraop_mesh_resampled.GetNumberOfPoints())
                idx_resample += 1
            intraop_mesh = intraop_mesh_resampled

            # print("intraop_mesh after resample:", intraop_mesh.GetNumberOfPoints())



            intraop_xyz = vtk_to_numpy(intraop_mesh.GetPoints().GetData())
            intraop_normals = vtk_to_numpy(intraop_mesh.GetPointData().GetNormals())
            intraop = np.concatenate( (intraop_xyz, intraop_normals), axis=1 )

            if npoints > 0:
                r_surface = pc_utils.set_npoints( [intraop], npoints )
                intraop = r_surface[0]

                assert intraop.shape[0] == npoints
            
            intraop = intraop * self.scale

            if "X" in flip_axes:
                intraop[:, 0] = -intraop[:, 0]
                intraop[:, 3] = -intraop[:, 3]
            if "Y" in flip_axes:
                intraop[:, 1] = -intraop[:, 1]
                intraop[:, 4] = -intraop[:, 4]
            if "Z" in flip_axes:
                intraop[:, 2] = -intraop[:, 2]
                intraop[:, 5] = -intraop[:, 5]

            intraop = intraop.transpose( (1,0) )
            # print("intraop.shape", intraop.shape)
            intraop_list.append(np.expand_dims(intraop, axis=0))


        # save the original VTK meshes of preoperative volume and surface to a dict
        self.geometry.update({
            "preop_volume": preop_volume,
            "preop_surface": preop_surface,
            "preop_internal": internal_points,
            "intraop_surface": intraop_mesh_list,
            "intraop_volume": intraop_mesh_full,
        })

        # print("preop", preop.shape)
        # print("displ", displ.shape)
        # print("intraop", intraop.shape)

        # intraop_combined = np.concatenate([np.expand_dims(intraop_list, axis=0),], axis=0)
        # intraop_combined = np.concatenate([np.expand_dims(intraop_list, axis=0),], axis=0)
        intraop_combined = np.concatenate(intraop_list, axis=0)
        # print("intraop_combined.shape", intraop_combined.shape)
        # return preop and displ in shape of [F, n_points]
        # print("preop.shape", preop.shape)
        # print("displ.shape", displ.shape)
        # print("intraop_combined.shape", intraop_combined.shape)

        _stats_key = "AddSurfaceNoiseBlock_estimated_noisy_surface_fraction_{}".format(self.filenames["intraop_full"].split("_")[2][:2])
        _surface_area = self.stats[_stats_key] if self.stats is not None and _stats_key in self.stats else 0.0

        if self.return_all_intraop:
            res.update({
                "preop": preop,
                "displ": displ,
                "intraop": intraop_combined,
                "idx": [self.id,],
                "perlin_noise": self.intraop_perlin_noise_list,
                "gaussian_noise": self.intraop_gaussian_noise_list,
                "intraop_surface_area": [_surface_area],
            })
        else:
            res.update({
                "preop": preop,
                "displ": displ,
                "intraop": np.squeeze(intraop_list[0]),
                "idx": [self.id,],
                "perlin_noise": [self.intraop_perlin_noise_list[0], ],
                "gaussian_noise": [self.intraop_gaussian_noise_list[0],],
                "intraop_surface_area": [_surface_area],
            })

        return res



    def reload(self, npoints, preop, intraop, displ, idx_iter, debug_folder, preop_volume):
        """Reload sample based on given displacement field, the sample will interpolate the displacement field
        onto the preoperative mesh and deform it to get a new preoperative point cloud.
        As the preoperative mesh is changed, the preoperative features will be recalculated, e.g. normals, distance field, etc.
        The intraoperative point clouds will be kept the same.

        Args:
            displ (np.array): displacement field of shape [3, n_points]
        """
    
        mask_pre= np.absolute(preop[:, 0]) < 1e3
        preop = preop[mask_pre]
        displ = displ[mask_pre]
        preop = numpy_to_vtk(preop)

        preop_vtk = vtk_utils.to_pointcloud(
            coords=preop, 
            features=displ,
            features_name="displacement_predicted",
        )

        if preop_volume and preop_volume.GetPointData().HasArray("displacement_predicted"):
            preop_volume.GetPointData().RemoveArray("displacement_predicted")
        preop_volume_deformed = vtk_utils.apply_deformation(
            # mesh=self.geometry["preop_volume"] if not "preop_volume_deformed" in self.geometry else self.geometry["preop_volume_deformed"],
            # mesh=self.geometry["preop_volume"],
            mesh = preop_volume if idx_iter > 0 else self.geometry["preop_volume"],
            displacement_mesh=preop_vtk,
            displacement_array_name="displacement_predicted",
        )

        preop_volume = preop_volume_deformed

        DEBUG = True
        if DEBUG and debug_folder is not None:
            # Save the deformed preoperative volume to a file:
            # debug_path = "/mnt/cluster/workspaces/liupeng/debug"
            output_path = os.path.join(debug_folder, "preop_deformed_iter_{}.vtu".format(idx_iter))
            vtk_utils.write_mesh(
                mesh=preop_volume_deformed,
                filename=output_path,
            )
            print("saved deformed preop to", output_path)
        
        res = edict()

        ##########################################################
        # print("loading data...")
        # preop_path = os.path.join(self.path, self.filenames["preop"])
        # preop_volume = vtk_utils.load_mesh(preop_path)
        if preop_volume.GetNumberOfPoints() == 0:
            print("INVALIDIFY 1")
            self.valid = False
            raise IOError(f"Preoperative mesh is invalid")
        

        # intraop_full_path = os.path.join(self.path, self.filenames["intraop_full"])
        # print(p)
        # intraop_mesh_full = vtk_utils.load_mesh(intraop_full_path)
        intraop_mesh_full = self.geometry["intraop_volume"]
        if intraop_mesh_full.GetNumberOfPoints() == 0:
            print("INVALIDIFY 5")
            self.valid = False
            raise IOError(f"Intraoperative mesh is invalid")
        
        # If the displacement field is not present, but pre- and intraoperative volumes
        # are provided, calculate it from the two volumes:
        # if not preop_volume.GetPointData().HasArray(self.displ_array_name) and \
        #     "volume" in self.filenames["intraop_full"] and "volume" in self.filenames["preop"]:
                # assert preop_volume.GetNumberOfPoints() == intraop_mesh_full.GetNumberOfPoints(),\
                #       "Number of points in preop and intraop volume must be the same. {}".format(self.path)

        if preop_volume.GetPointData().HasArray(self.displ_array_name):
            preop_volume.GetPointData().RemoveArray(self.displ_array_name)
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

        # 1. If preop is a surface, randomly create internal points, normals and displacement of such 
        # internal points will be generated later
        # 2. If preop is a volume, split the volume into surface and internal points, and compute the
        # if preop_path.endswith(".stl"):
        #     # preop_surface = copy.deepcopy(preop_volume)
        #     preop_surface = vtk.vtkPolyData()
        #     preop_surface.DeepCopy(preop_volume)
        #     internal_points = vtk_utils.create_random_internal_points(
        #         surface_mesh =preop_volume,
        #         points_to_create = preop_surface.GetNumberOfPoints(), 
        #         append_surface =False,
        #     )
        # else:
        preop_surface, internal_points = vtk_utils.split_surface_and_internal_points( preop_volume )

        # For the surface, compute a normal for each point:
        surface_normals = vtk_utils.compute_point_normals( preop_surface )
        preop_surface.GetPointData().SetNormals( surface_normals )

        # NOTE: If scale != 1, the resolution of resample_polydata should probably be adjusted first:
        preop_surface = vtk_utils.resample_polydata( preop_surface )
        # Update the normals array:
        surface_normals = preop_surface.GetPointData().GetNormals()
        
        # For the internal points, compute the distance to the surface

        distance_field = vtk_utils.df( mesh=internal_points, surface=preop_surface )
        preop_internal_xyz = vtk_to_numpy( internal_points.GetPoints().GetData() )
        preop_surface_xyz = vtk_to_numpy( preop_surface.GetPoints().GetData() )
        preop_internal_dists = np.expand_dims( vtk_to_numpy(distance_field), axis = 1 )
        preop_surface_normals = vtk_to_numpy(surface_normals)

        n_internal_points = preop_internal_xyz.shape[0]
        n_surface_points = preop_surface_xyz.shape[0]

        # The internal surface points can't have a "normal", so set the value to zero:
        preop_internal_normals = np.zeros( (n_internal_points, 3), dtype=float )
        # The distance of surface to surface remains zero:
        preop_surface_dists = np.zeros( (n_surface_points, 1), dtype=float )


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
            #displ = vtk_to_numpy(displ)

        # Subsampling and ratio adjustment:
        # downsample the all preoperative point clouds so that they have the same amount of points 
        if npoints > 0:
            # Ensure the number of points in each point cloud are correct. In total, they should equal
            # npoints, and be split into internal and surface points by the given ratio.
            # This will choose a random subset of points, or increase the number of points
            # by adding dummy points where necessary:
            r_internal, r_surface = \
                    pc_utils.set_point_ratio(
                        internal=[preop_internal_xyz, preop_internal_dists, preop_internal_normals, displ_internal_np],
                        surface=[preop_surface_xyz, preop_surface_dists, preop_surface_normals, displ_surface_np],
                        ratio = 0.3,
                        total_points = npoints,
            )
            # Re-assign to original variables:
            preop_internal_xyz, preop_internal_dists, preop_internal_normals, displ_internal_np = r_internal
            preop_surface_xyz, preop_surface_dists, preop_surface_normals, displ_surface_np = r_surface
            # The set_point_ratio function sets some values to 99999 for the dummy points. For the displacement
            # field, this is not good, because the network will output 0 for dummy points. To not influence
            # the displacement error calculation, set the displacement for these points to 0:
            displ_internal_np[displ_internal_np > 5000] = 0
            displ_surface_np[displ_surface_np > 5000] = 0



        # Combine internal points and surface points along point axis (0):
        preop_xyz = np.concatenate( (preop_internal_xyz, preop_surface_xyz), axis=0 )
        preop_dists = np.concatenate( (preop_internal_dists, preop_surface_dists), axis=0 )
        preop_normals = np.concatenate( (preop_internal_normals, preop_surface_normals), axis=0 )
        displ = np.concatenate( (displ_internal_np, displ_surface_np), axis=0 )
        preop = np.concatenate( (preop_xyz, preop_dists, preop_normals), axis=1 )

        if npoints > 0:
            assert preop.shape[0] == npoints
            assert displ.shape[0] == npoints
            # assert intraop.shape[0] == npoints

        preop = preop * self.scale
        # intraop = intraop * self.scale 
        displ = displ * self.scale

        # if "X" in flip_axes:
        #     preop[:, 0] = -preop[:, 0]      # flip pos.x
        #     preop[:, 4] = -preop[:, 4]      # flip normal.x
        #     displ[:, 0] = -displ[:, 0]
        # if "Y" in flip_axes:
        #     preop[:, 1] = -preop[:, 1]      # flip pos.y
        #     preop[:, 5] = -preop[:, 5]      # flip normal.y
        #     displ[:, 1] = -displ[:, 1]
        # if "Z" in flip_axes:
        #     preop[:, 2] = -preop[:, 2]
        #     preop[:, 6] = -preop[:, 6]
        #     displ[:, 2] = -displ[:, 2]

        preop = preop.transpose( (1,0) )
        displ = displ.transpose( (1,0) )


        ##########################################################
        ##################   intraoperative data ################
        ##########################################################
        # intraop_list = []
        # intraop_mesh_list = []

        # # Ensure the full mesh has normals:
        # intraop_mesh_full_surface = vtk_utils.extract_surface(intraop_mesh_full)
        # normals = vtk_utils.compute_point_normals( intraop_mesh_full_surface ) 
        # intraop_mesh_full_surface.GetPointData().SetNormals( normals )

        # for intraop_filename in self.filenames["intraop"]:
        #     intraop_path = os.path.join(self.path,intraop_filename)
        #     intraop_mesh = vtk_utils.load_mesh(intraop_path)
        #     if intraop_mesh.GetNumberOfPoints() == 0:
        #         print("INVALIDIFY 5")
        #         self.valid = False
        #         raise IOError(f"Could not load {intraop_path}")
        #     intraop_mesh_list.append(intraop_mesh)

        #     # Copy the normals from the full mesh to the partial mesh:
        #     vtk_utils.copy_normals( intraop_mesh_full_surface, intraop_mesh )

        #     # NOTE: If scale != 1, the resolution of resample_polydata should probably be adjusted first:
        #     intraop_mesh = vtk_utils.resample_polydata( intraop_mesh, max_num_points=2500 )

        #     intraop_xyz = vtk_to_numpy(intraop_mesh.GetPoints().GetData())
        #     intraop_normals = vtk_to_numpy(intraop_mesh.GetPointData().GetNormals())
        #     intraop = np.concatenate( (intraop_xyz, intraop_normals), axis=1 )

        #     if npoints > 0:
        #         r_surface = pc_utils.set_npoints( [intraop], npoints )
        #         intraop = r_surface[0]

        #         assert intraop.shape[0] == npoints
            
        #     # intraop = intraop * self.scale

        #     # if "X" in flip_axes:
        #     #     intraop[:, 0] = -intraop[:, 0]
        #     #     intraop[:, 3] = -intraop[:, 3]
        #     # if "Y" in flip_axes:
        #     #     intraop[:, 1] = -intraop[:, 1]
        #     #     intraop[:, 4] = -intraop[:, 4]
        #     # if "Z" in flip_axes:
        #     #     intraop[:, 2] = -intraop[:, 2]
        #     #     intraop[:, 5] = -intraop[:, 5]

        #     intraop = intraop.transpose( (1,0) )
        #     # print("intraop.shape", intraop.shape)
        #     intraop_list.append(np.expand_dims(intraop, axis=0))


        # save the original VTK meshes of preoperative volume and surface to a dict
        # self.geometry.update({
        #     "preop_volume": preop_volume,
        #     "preop_surface": preop_surface,
        #     "preop_internal": internal_points,
        #     "intraop_surface": intraop_mesh_list,
        #     "intraop_volume": intraop_mesh_full,
        # })

        self.geometry["preop_volume_deformed"] = preop_volume

        # print("preop", preop.shape)
        # print("displ", displ.shape)
        # print("intraop", intraop.shape)

        # intraop_combined = np.concatenate([np.expand_dims(intraop_list, axis=0),], axis=0)
        # intraop_combined = np.concatenate([np.expand_dims(intraop_list, axis=0),], axis=0)
        # intraop_combined = np.concatenate(intraop_list, axis=0)
        # print("intraop_combined.shape", intraop_combined.shape)
        # return preop and displ in shape of [F, n_points]
        # print("preop.shape", preop.shape)
        # print("displ.shape", displ.shape)
        # print("intraop_combined.shape", intraop_combined.shape)

        res.update({
            "preop": torch.tensor(preop, dtype=torch.float32),
            "displ": torch.tensor(displ, dtype=torch.float32),
            # "intraop": torch.tensor(intraop_combined, dtype=torch.float32),
            "idx": torch.tensor( [self.id,], dtype=torch.int16),
            # "perlin_noise": self.intraop_perlin_noise_list,
            # "gaussian_noise": self.intraop_gaussian_noise_list,
            # "intraop_surface_area": [self.stats["AddSurfaceNoiseBlock_estimated_noisy_surface_fraction_{}".format(self.filenames["intraop_full"].split("_")[2][:2])], ],
        })

        return res, preop_volume




class DisplDatasetAMOS(DisplDataset):
    def __init__(self, 
            path, 
            npoints=2500, 
            nsamples=1000, 
            start_sample=0, 
            verbose=True, 
            sample_class=LiverSampleAMOS, 
            criteria=None, 
            scale=1, 
            augmentation=True, 
            quick_preload=False, 
            stats=None, 
            frame="NONE", 
            return_all_intraop=True,
            white_list=[],
            **kwargs
        ):
        self.white_list = white_list
        self.return_all_intraop = return_all_intraop    
        super().__init__(path, npoints, nsamples, start_sample, verbose, sample_class, criteria, scale, augmentation, quick_preload, stats, frame, **kwargs)
        
        

    def _find_potential_samples(self, start_sample, nsamples, criteria=None):
        print("intraop_type:", self.intraop_type)
        last_sample = start_sample + nsamples - 1

        stats = self.stats
        # stats_filename = os.path.join(self.storage_path, "statistics.yaml")
        stats_filename = os.path.join(self.storage_path, "statistics.pkl")

        if stats is None and os.path.exists(stats_filename):
            # If stats file exists, looking up the statistics.yaml file to find the valid samples and their corresponding statistics
            # print("Found statistics file:", stats_filename)
            # with open(stats_filename) as f:
            #     stats = yaml.safe_load(f)
            #     # print(stats)
            #     print(f"Loaded statistics.yaml. Contains {len(stats)} entries.")
            print("Loading statistics from", stats_filename)
            with open(stats_filename, 'rb') as f:
                stats = pkl.load(f)
                f.close()


        if len(self.white_list) > 0:
            print("Loading from white list!!!")
            for int_id in self.white_list:
                folder = f"{int_id:06}"
                print("---------------------")
                print("folder", folder)
                path = os.path.join( self.storage_path, folder )
                sample = self.sample_class( 
                    path = path, 
                    int_id = int_id,
                    check_for_files = False, 
                    scale = self.scale,
                    frame = self.frame,
                    stats = stats[int_id],
                    **self.filename_list,
                )
                self._samples_list.append( sample )
        else:
            # wrong_sample_list = [12311, 6012, 15498]
            # wrong_sample_list = [69298,] # this one has degenerated faces
            wrong_sample_list = []
            if stats is not None:
                # if we successfully loaded the statistics file, we can use it to filter the samples
                print("Checking sample folders:")
                #for int_id, s in tqdm(stats.items(), total=len(stats)):
                for int_id in range(start_sample, last_sample):
                    print("---------------------")

                    if not int_id in stats or int_id in wrong_sample_list:
                        print("\tinvalid, doesn't exist")
                        continue

                    # filename_list_this_sample = self.filename_list
                    # filename_list_this_sample["intraop_filename"] = intraop_filename_this_sample
                    # print("filename_list_this_sample", filename_list_this_sample)
                    s = stats[int_id]
                    # print("criteria", criteria, criteria(s))
                    if criteria is not None:
                        # Check if the statistics of this sample meet the criteria, otherwise
                        # skip this sample
                        if criteria(s, self.intraop_type, self.cleanliness) == False:
                            print("\tinvalid, criteria")
                            continue
                        print("\tvalid")
                    print("id", int_id, start_sample, last_sample)

                    if int_id >= start_sample and int_id <= last_sample:
                        folder = f"{int_id:06}"
                        print("folder", folder)

                        path = os.path.join( self.storage_path, folder )
                        # print(self.sample_class)
                        sample = self.sample_class( 
                                path = path, 
                                int_id = int_id,
                                check_for_files = True, 
                                scale = self.scale,
                                frame = self.frame,
                                stats = s,
                                return_all_intraop=self.return_all_intraop,
                                **self.filename_list,
                        )
                        # print(sample)
                        if sample.valid:
                            self._samples_list.append( sample )
                            # retrieve the statistics of the valid samples
                            stats_sample = {}
                            for k in self.resample_stats_key_list:
                                if k in s:
                                    stats_sample[k] = s[k]
                            self._samples_stats_list.append(stats_sample)
                            print("appended", len(self._samples_list))

                        print("after", len(self._samples_list))


    def reload_sample(self, idx, npoints, preop, intraop, displ, idx_iter, preop_volume_deformed, debug_folder=None):
        ### Reload sample based on given displacement field, the sample will interpolate the displacement field
        ### onto the preoperative mesh and deform it to get a new preoperative point cloud.
        ### As the preoperative mesh is changed, the preoperative features will be recalculated, e.g. normals, distance field, etc.
        ### The intraoperative point clouds will be kept the same.

        # print("reloading sample...")
        print("debug_folder", debug_folder)
        res =  self._samples_list[idx].reload(
            npoints=npoints, 
            preop=preop, 
            intraop=intraop, 
            displ=displ, 
            idx_iter=idx_iter, 
            preop_volume=preop_volume_deformed,
            debug_folder=debug_folder,
        )
        return res


