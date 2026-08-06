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
import pickle as pkl

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
        import data_syn
    except:
        from data import vtk_utils
        from data import extract_surface
        #from data import statistics
        from data import pc_utils
        from data import data_syn


from tqdm import tqdm
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
import copy
warnings.filterwarnings('ignore')
import traceback
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import scipy
vtk.vtkObject.GlobalWarningDisplayOff() 


np.random.seed(1234)

class LiverPreopSample():
    """LiverPreopSample class, used to load preoperative liver-related data, including
    preoperative point cloud, displacement field, and surface normal field.

    This class is created for the tests that only need preoperative information, and 
    considered as parent for other liver dataset classes with intraoperative information.
    """
    def __init__(self, path, int_id, frame = None, check_for_files=True,
            scale=1, **kwargs):
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
        self.retrieve_filenames(frame=frame, **kwargs)
        if check_for_files:
            self.check_filenames()

    def load_stats( self ):
        stats_filename = os.path.join(self.path, "statistics.yaml")
        if os.path.exists(stats_filename):
            with open(stats_filename) as f:
                self.stats = yaml.safe_load(f)
        else:
            print("no stats file!")
            self.stats = None


    def retrieve_filenames (self, frame=None, **kwargs):
        """Update filename list
        """
        self.filenames.update({
            "preop": kwargs["preop_filename"],
        })
        

    def load(self, npoints=0, flip_axes=""):
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
        p = os.path.join(self.path, self.filenames["preop"])
        preop_volume = vtk_utils.load_mesh(p)
        if preop_volume.GetNumberOfPoints() == 0:
            print("INVALIDIFY 1")
            self.valid = False
            raise IOError(f"Could not load {p}")

        # 1. If preop is a surface, randomly create internal points, normals and displacement of such 
        # internal points will be generated later
        # 2. If preop is a volume, split the volume into surface and internal points, and compute the
        if p.endswith(".stl"):
            # preop_surface = copy.deepcopy(preop_volume)
            preop_surface = vtk.vtkPolyData()
            preop_surface.DeepCopy(preop_volume)
            internal_points = vtk_utils.create_random_internal_points(preop_volume,
                    points_to_create = preop_surface.GetNumberOfPoints(), append_surface=False )
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
            print("INVALIDIFY 2")
            self.valid = False
            #raise IOError("Displacement field not found!")
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

            # print("preop_internal_xyz", preop_internal_xyz.shape)
            # print("preop_surface_xyz", preop_surface_xyz.shape)


        # Combine internal points and surface points along point axis (0):
        preop_xyz = np.concatenate( (preop_internal_xyz, preop_surface_xyz), axis=0 )
        preop_dists = np.concatenate( (preop_internal_dists, preop_surface_dists), axis=0 )
        preop_normals = np.concatenate( (preop_internal_normals, preop_surface_normals), axis=0 )
        displ = np.concatenate( (displ_internal_np, displ_surface_np), axis=0 )
        preop = np.concatenate( (preop_xyz, preop_dists, preop_normals), axis=1 )

        if npoints > 0:
            assert preop.shape[0] == npoints

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

        # save the original VTK meshes of preoperative volume and surface to a dict
        self.geometry.update({
            "preop_volume": preop_volume,
            "preop_surface": preop_surface,
        })

        # return preop and displ in shape of [F, n_points]
        res.update({
            "preop": preop.transpose( (1,0) ),
            "displ": displ.transpose( (1,0) ),
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




class LiverSample(LiverPreopSample):
    """Liver sample, inherented from LiverPreopSample.
      Each consists of three point cloud data: preop, intraop, displ
        
    Args:
        frame: Frame number to load, will modify the end of the filenames to append frame number. Valid values:
            None: no dynamic data is present, i.e. no files don't contain frame numbers (default) and no frame
                number will be appended to file names
            Integer, telling the sample which frame to load
            math.inf to load whatever the last frame is that is found for this sample
    """
    def __init__(self, path, int_id, 
                check_for_files = True, 
                scale = 1,
                frame = None,
                **kwargs):
        super().__init__(
                path,
                int_id,
                scale=scale,
                frame=frame,
                check_for_files=False,
                **kwargs,
        )
        if check_for_files:
            self.check_filenames()

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


    def retrieve_filenames (self, frame=None, **kwargs):
        super().retrieve_filenames( frame, **kwargs )

        # If needed, add the frame number to the filename:
        # If frame is None this does nothing:
        intraop_filename = self.append_frame( kwargs["intraop_filename"], frame )
        print("intraop_filename:", intraop_filename, self.valid)
        # intraop_full_filename used for intraoperative normal calculation
        intraop_full_filename = self.append_frame( kwargs["intraop_full_filename"], frame )
        print("intraop_full_filename:", intraop_full_filename, self.valid)

        # print("updating filename list", kwargs)
        self.filenames.update({
            #"preop": kwargs["preop_filename"],
            "intraop": intraop_filename,
            "intraop_full": intraop_full_filename,
        })


    def load(self, npoints=0, flip_axes=""):
        """Overwrite the load function of the parent class LiverPreopSample
        Load intraoperative data, including intraoperative surface, normals

        Args:
            npoints (int, optional): number of points. Defaults to 0.
            flip_axes (str, optional): axis to flip around. Defaults to "".

        Returns:
            dict: dictionary of the loaded point cloud data
        """
        res = super().load( npoints=npoints, flip_axes=flip_axes )
        p = os.path.join(self.path, self.filenames["intraop"])
        intraop_mesh = vtk_utils.load_mesh(p)
        if intraop_mesh.GetNumberOfPoints() == 0:
            print("INVALIDIFY 5")
            self.valid = False
            raise IOError(f"Could not load {p}")
        p = os.path.join(self.path, self.filenames["intraop_full"])
        # print(p)
        intraop_mesh_full = vtk_utils.load_mesh(p)
        if intraop_mesh_full.GetNumberOfPoints() == 0:
            print("INVALIDIFY 5")
            self.valid = False
            raise IOError(f"Could not load {p}")

        # Ensure the full mesh has normals:
        intraop_mesh_full_surface = vtk_utils.extract_surface(intraop_mesh_full)
        normals = vtk_utils.compute_point_normals( intraop_mesh_full_surface ) 
        intraop_mesh_full_surface.GetPointData().SetNormals( normals )

        # Copy the normals from the full mesh to the partial mesh:
        vtk_utils.copy_normals( intraop_mesh_full_surface, intraop_mesh )

        # NOTE: If scale != 1, the resolution of resample_polydata should probably be adjusted first:
        intraop_mesh = vtk_utils.resample_polydata( intraop_mesh, max_num_points=2500 )

        intraop_xyz = vtk_to_numpy(intraop_mesh.GetPoints().GetData())
        intraop_normals = vtk_to_numpy(intraop_mesh.GetPointData().GetNormals())
        intraop = np.concatenate( (intraop_xyz, intraop_normals), axis=1 )

        intraop = intraop * self.scale 

        # If preop_mesh_full and intraop_mesh_full is available, and the displacement is still empty,
        # we can caluclate displacement field manually from the two meshes
        if "volume" in self.filenames["intraop_full"] and "volume" in self.filenames["preop"] and not np.any(res["displ"]):
            print("Calculating displacement field from preop and intraop meshes.")
            self.valid = True
            preop_volume = self.geometry["preop_volume"]
            # preop_volume_coords = vtk_to_numpy(preop_volume.GetPoints().GetData())
            # intraop_volume_coords = vtk_to_numpy(intraop_mesh_full.GetPoints().GetData())
            # displ = intraop_volume_coords - preop_volume_coords

            assert preop_volume.GetNumberOfPoints() == intraop_mesh_full.GetNumberOfPoints(), "Number of points in preop and intraop volume must be the same."

            displ_vtk_array = vtk.vtkFloatArray()
            displ_vtk_array.SetNumberOfComponents(3)
            displ_vtk_array.SetNumberOfTuples(preop_volume.GetNumberOfPoints())
            displ_vtk_array.SetName(self.displ_array_name)
            for i in range(preop_volume.GetNumberOfPoints()):
                d = [preop_volume.GetPoint(i)[c] - intraop_mesh_full.GetPoint(i)[c] for c in range(3)]
                # print(d)
                displ_vtk_array.SetTuple(i, d)
            preop_volume.GetPointData().AddArray(displ_vtk_array)
            
            preop_surface, internal_points = vtk_utils.split_surface_and_internal_points( preop_volume )
            # print("preop_surface.GetNumberOfPoints()", preop_surface.GetNumberOfPoints(), "internal_points.GetNumberOfPoints()", internal_points.GetNumberOfPoints())
            displ_internal = internal_points.GetPointData().GetArray( self.displ_array_name )
            displ_surface = preop_surface.GetPointData().GetArray( self.displ_array_name )

            displ_internal_np = vtk_to_numpy( displ_internal )
            displ_surface_np = vtk_to_numpy( displ_surface )

            if npoints > 0:
                # Ensure the number of points in each point cloud are correct. In total, they should equal
                # npoints, and be split into internal and surface points by the given ratio.
                # This will choose a random subset of points, or increase the number of points
                # by adding dummy points where necessary:
                r_internal, r_surface = \
                        pc_utils.set_point_ratio(
                            internal=[displ_internal_np],
                            surface=[displ_surface_np],
                            ratio = 0.3,
                            total_points = npoints,
                )
                # Re-assign to original variables:
                displ_internal_np = r_internal[0]
                displ_surface_np = r_surface[0]
                # The set_point_ratio function sets some values to 99999 for the dummy points. For the displacement
                # field, this is not good, because the network will output 0 for dummy points. To not influence
                # the displacement error calculation, set the displacement for these points to 0:
                displ_internal_np[displ_internal_np > 5000] = 0
                displ_surface_np[displ_surface_np > 5000] = 0

            print("displ_internal_np.shape", displ_internal_np.shape)
            print("displ_surface_np.shape", displ_surface_np.shape)

            # save preoperative mesh for debugging
            output_path = "/home/liupeng/ceph_home/NeuralNetworks/OrganDeformNet/datasets/tests/preop_volume_with_gt_displacement.vtk"
            if not os.path.exists(output_path):
                vtk_utils.write_mesh(mesh=preop_volume, filename=output_path, verbose=True)
            
            output_path = "/home/liupeng/ceph_home/NeuralNetworks/OrganDeformNet/datasets/tests/preop_surface_with_gt.vtp"
            if not os.path.exists(output_path):
                print("preop_surface.GetNumberOfPoints()", preop_surface.GetNumberOfPoints())
                vtk_utils.write_mesh(mesh=preop_surface, filename=output_path, verbose=True)
            
            output_path = "/home/liupeng/ceph_home/NeuralNetworks/OrganDeformNet/datasets/tests/internal_points_with_gt.vtp"
            if not os.path.exists(output_path):
                print("internal_points.GetNumberOfPoints()", internal_points.GetNumberOfPoints())
                vtk_utils.write_mesh(mesh=internal_points, filename=output_path, verbose=True)

            displ = np.concatenate( (displ_internal_np, displ_surface_np), axis=0 )

            displ = displ * self.scale

            if "X" in flip_axes:
                displ[:, 0] = -displ[:, 0]
            if "Y" in flip_axes:
                displ[:, 1] = -displ[:, 1]
            if "Z" in flip_axes:
                displ[:, 2] = -displ[:, 2]

            res.update({
                "displ": displ.transpose( (1,0) ),
            })


        if npoints > 0:
            r_surface = pc_utils.set_npoints( [intraop], npoints )
            intraop = r_surface[0]

        if "X" in flip_axes:
            intraop[:, 0] = -intraop[:, 0]  # flip pos.x
            intraop[:, 3] = -intraop[:, 3]  # flip normal.x
        if "Y" in flip_axes:
            intraop[:, 1] = -intraop[:, 1]  # pos
            intraop[:, 4] = -intraop[:, 4]  # normal
        if "Z" in flip_axes:
            intraop[:, 2] = -intraop[:, 2]
            intraop[:, 5] = -intraop[:, 5]

        # save the original VTK mesh of intraoperative surface to dict
        self.geometry.update({
            "intraop_surface": intraop_mesh,
            "intraop_surface_full": intraop_mesh_full_surface,
            "intraop_volume": intraop_mesh_full, # this can also be the full surface 
        })

        res.update({
            "intraop": intraop.transpose( (1, 0) )
        })
        return res


class LiverSampleInternals(LiverSample):
    """Inherited from LiverSample, used to load internal data of the liver
    """
    def __init__(self, path, int_id, 
                check_for_files = True, 
                scale = 1,
                frame = None,
                **kwargs):
        super().__init__(
                path,
                int_id,
                scale=scale,
                frame=frame,
                check_for_files=False,
                **kwargs,
        )
        if check_for_files:
            self.check_filenames()

    def retrieve_filenames (self, frame=None, **kwargs):
        super().retrieve_filenames( frame, **kwargs )

        # If needed, add the frame number to the filename:
        # If frame is None this does nothing:
        intraop_internal_filenames = []
        for f in kwargs["intraop_internal_filenames"]:
            f_with_frame = self.append_frame( f, frame )
            intraop_internal_filenames.append( f_with_frame )

        self.filenames.update({
            "preop_internal": kwargs["preop_internal_filename"],
            "intraop_internal": intraop_internal_filenames,
        })


    def load(self, npoints=0, flip_axes=""):
        res = super().load( npoints=npoints, flip_axes=flip_axes )

        loadable_meshes = []
        for f in self.filenames["intraop_internal"]:
            p = os.path.join(self.path, f)
            if os.path.exists(p):
                loadable_meshes.append(p)

        num_meshes = random.randint(1, len(loadable_meshes))
        meshes_to_load = random.sample( loadable_meshes, k=num_meshes )
        append_filter = vtk.vtkAppendPolyData()
        #print(f"Loading {len(meshes_to_load)} internal intraop meshes.")
        for p in meshes_to_load:
            mesh = vtk_utils.load_mesh(p)
            append_filter.AddInputData( mesh )
        append_filter.Update()
        intraop_internal_mesh = append_filter.GetOutput()
        if intraop_internal_mesh.GetNumberOfPoints() == 0:
            self.valid = False
            raise IOError(f"Sampled internal meshes empty.")
        intraop_internal = vtk_to_numpy(intraop_internal_mesh.GetPoints().GetData())
        intraop_internal = intraop_internal * self.scale 

        p = os.path.join(self.path, self.filenames["preop_internal"])
        preop_internal_mesh = vtk_utils.load_mesh(p)
        if preop_internal_mesh.GetNumberOfPoints() == 0:
            self.valid = False
            raise IOError(f"Could not load {p}")

        # Calculate the distance from each internal point to the nearest surface point:
        preop_surface = self.geometry["preop_surface"]
        distance_field = vtk_utils.df( mesh=preop_internal_mesh, surface=preop_surface)
        # Convert to numpy and concatenate
        preop_internal = vtk_to_numpy(preop_internal_mesh.GetPoints().GetData())
        preop_internal_dists = vtk_to_numpy(distance_field)
        preop_internal = np.concatenate( (preop_internal, np.expand_dims(preop_internal_dists,axis=1)), axis=1 )
        preop_internal = preop_internal * self.scale 


        # Interpolate the displacemnt field from the preoperative volume to the preoperative internal points:
        preop_volume = self.geometry["preop_volume"]
        interpolated = vtk_utils.interpolate_deformation( preop_internal_mesh,
                                                            preop_volume, self.displ_array_name )
        displ_internal = vtk_to_numpy( interpolated.GetPointData().GetArray( self.displ_array_name ) )
        #print("displ_internal max", displ_internal.max(), displ_internal.min())

        # Subsample internal points
        if npoints > 0:
            r = pc_utils.set_npoints( [intraop_internal], npoints )
            intraop_internal = r[0]

            r = pc_utils.set_npoints( [preop_internal, displ_internal], npoints )
            preop_internal = r[0]
            displ_internal = r[1]

            # The set_npoints sets some function to 99999. Those should get a displacement of zero:
            displ_internal[displ_internal > 5000] = 0

        if "X" in flip_axes:
            intraop_internal[:, 0] = -intraop_internal[:, 0]
            preop_internal[:, 0] = -preop_internal[:, 0]
            displ_internal[:, 0] = -displ_internal[:, 0]
        if "Y" in flip_axes:
            intraop_internal[:, 1] = -intraop_internal[:, 1]
            preop_internal[:, 1] = -preop_internal[:, 1]
            displ_internal[:, 1] = -displ_internal[:, 1]
        if "Z" in flip_axes:
            intraop_internal[:, 2] = -intraop_internal[:, 2]
            preop_internal[:, 2] = -preop_internal[:, 2]
            displ_internal[:, 2] = -displ_internal[:, 2]

        # return internal information in shape of [F, n_points]
        res.update({
            "preop_internal": preop_internal.transpose( (1, 0) ),
            "intraop_internal": intraop_internal.transpose( (1, 0) ),
            "displ_internal": displ_internal.transpose( (1, 0) )
        })
        return res



# class LiverMatchingCuesSample(LiverSample):
#     def __init__(self, path, int_id, 
#                  check_for_files=True, 
#                  scale=1, 
#                 #  frame=None, 
#                  cue_array_name_prefix="cue",
#                 #  cue_filename_suffix="",
#                  **kwargs,
#         ):
#         super().__init__(
#             path, 
#             int_id, 
#             check_for_files, 
#             scale, 
#             # frame=None, 
#             **kwargs,
#         )
#         self.cue_array_name_prefix = cue_array_name_prefix
#         if check_for_files:
#             self.check_filenames()


#     def load(self, npoints=0, flip_axes=""):
#         res = super().load( npoints=npoints, flip_axes=flip_axes )
#         preop_volume = self.geometry["preop_volume"]
#         print("preop_volume.GetNumberOfPoints()", preop_volume.GetNumberOfPoints())
#         print(preop_volume.GetPointData().HasArray("cue_0"))


#         res.update({
#             "preop_cues": [0],
#             "intraop_cues": [0],
#         })

#         return res
        



class LiverGeometrySample(LiverSample):
    """Liver sample, inherited from LiverSample.
      Each consists of three point cloud data: preop, intraop and displ, also the 
        
    Args:
        LiverPreopSample (_type_): Parent class
    """
    def __init__(self, path, int_id, 
                check_for_files = True, 
                scale = 1,
                **kwargs):
        super().__init__(
                path,
                int_id,
                scale=scale,
                check_for_files=check_for_files,
                **kwargs)


    def load(self, npoints=0, flip_axes=""):
        res = super().load( npoints=npoints, flip_axes=flip_axes )

        preop_subsampled = res.preop[0:3, ...]
        preop_subsampled = vtk_utils.to_pointcloud(coords=numpy_to_vtk(preop_subsampled.transpose([1, 0])), )
        preop_subsampled_idx_on_original = vtk_utils.closest_vetices(
            source=preop_subsampled, 
            target=self.geometry["preop_volume"], 
            )
        # print("len(preop_subsampled_idx_on_original)", len(preop_subsampled_idx_on_original))

        intraop_subsampled = res.intraop
        intraop_subsampled = vtk_utils.to_pointcloud(coords=numpy_to_vtk(intraop_subsampled.transpose([1, 0])), )
        intraop_subsampled_idx_on_original = vtk_utils.closest_vetices(
            source=intraop_subsampled, 
            target=self.geometry["intraop_surface"], 
            )
        # print("len(intraop_subsampled_idx_on_original)", len(intraop_subsampled_idx_on_original))

        # print(self.geometry["preop_surface"].GetNumberOfPoints(), self.geometry["preop_volume"].GetNumberOfPoints())
        preop_surface_idx_on_original = vtk_utils.closest_vetices(
            source=self.geometry["preop_surface"], 
            target=self.geometry["preop_volume"],
            )
        preop_surface_idx_on_original = np.asarray(preop_surface_idx_on_original)
        # print("preop_surface_idx_on_original.shape", preop_surface_idx_on_original.shape)

        preop_surface_verts = vtk_to_numpy(self.geometry["preop_surface"].GetPoints().GetData())
        preop_surface_faces = [
            (
                self.geometry["preop_surface"].GetCell(idx_c).GetPointId(0), 
                self.geometry["preop_surface"].GetCell(idx_c).GetPointId(1), 
                self.geometry["preop_surface"].GetCell(idx_c).GetPointId(2), 
            )
            for idx_c in range(self.geometry["preop_surface"].GetNumberOfCells())]
        preop_surface_faces = np.asarray(preop_surface_faces)

        res.update({
            "preop_subsampled_idx_on_original" : preop_subsampled_idx_on_original,
            "intraop_subsampled_idx_on_original" : intraop_subsampled_idx_on_original,
            "preop_surface_idx_on_original" : preop_surface_idx_on_original,
            "preop_surface_verts" : preop_surface_verts.transpose([1, 0]),
            "preop_surface_faces" : preop_surface_faces.transpose([1, 0]),
        })

        return res


class LiverSampleMatchingCues(LiverSample):
    def __init__(self, path, int_id, 
            check_for_files = True, 
            scale = 1,
            cue_array_name_prefix="",
            num_cues=4, # number of cues to be loaded in total
            random_cue_num=-1, # number of cues to be randomly selected and returned 
            shuffle_cue=False,
            **kwargs,
        ):
        self.cue_array_name_prefix = cue_array_name_prefix
        self.load_cue_from_files = False
        self.num_cues=num_cues
        self.shuffle_cue = shuffle_cue
        super().__init__(
                path,
                int_id,
                scale=scale,
                check_for_files=check_for_files,
                **kwargs
        )
        # print(self.cue_array_name_prefix)
        self.cue_valid_bits = np.ones(self.num_cues)
        if random_cue_num == -1:
            # -1 means the number of returned cues are totally random
            self.cue_valid_bits = np.random.randint(0, 2, self.num_cues)
        # elif random_cue_num == 0:
        #     # 0 means no cues are returned
        #     self.cue_valid_bits = np.zeros(self.num_cues)
        elif random_cue_num >= 0:
            # >0 means |random_cue_num| cues will be randomly selected 
            self.cue_valid_bits = np.zeros(self.num_cues)
            self.cue_valid_bits[:random_cue_num] = 1
            if self.shuffle_cue:
                np.random.shuffle(self.cue_valid_bits)
        print(self.id, "self.cue_valid_bits", self.cue_valid_bits)


    def retrieve_filenames (self, **kwargs):
        # print("updating filename list", kwargs)
        super().retrieve_filenames( **kwargs )
        if self.cue_array_name_prefix == "":
            # If cue array name prefix is not provided, check the cue file names
            if "preop_cue_filename_list" in kwargs and "intraop_cue_filename_list" in kwargs:   
                self.filenames.update({
                    "preop_cues": kwargs["preop_cues_filename_list"],
                    "intraop_cues": kwargs["intraop_cues_filename_list"],
                    # "preop_cue": kwargs["preop_cue_filename"],
                    # "intraop_cue": kwargs["intraop_cue_filename"],
                })
                self.load_cue_from_files = True
                self.num_cues = len(self.filenames["preop_cues"])
            else:
                raise ValueError("Please provide either cue array prefix name or cue file names.")


    def load(self, npoints=0, flip_axes="", save_cues=False, ):
        res = super().load( npoints=npoints, flip_axes=flip_axes )
        # if self.load_cue_from_files:
        #     return self.load_from_files(res, npoints, flip_axes, save_cues)
        # else:
        return self.load_from_arrays(res, npoints, flip_axes, save_cues, )

    
    def load_from_arrays(self, res,  npoints=0, flip_axes="", save_cues=False, ):
        # print("self.id", self.id)
        preop = res.preop[0:3, ...].transpose(1, 0)
        intraop = res.intraop[0:3, ...].transpose(1, 0)
        # print("before removing dummy points", preop.shape, intraop.shape)
        # Remove dummy points
        # mask = np.absolute(preop[:, 0]) < 1e3
        # preop = preop[mask]
        # mask = np.absolute(intraop[:, 0]) < 1e3
        # intraop = intraop[mask]
        # print("after removing dummy points", preop.shape, intraop.shape)
        preop_coords_full = vtk_to_numpy(self.geometry["preop_volume"].GetPoints().GetData())
        intraop_coords_full = vtk_to_numpy(self.geometry["intraop_surface"].GetPoints().GetData())
        # Get the flipped preop and intraop coordinates, used to 
        # explore original point indices for downsampled points
        preop_coords_full_flipped = np.copy(preop_coords_full)
        intraop_coords_full_flipped = np.copy(intraop_coords_full)
        if "X" in flip_axes:
            preop_coords_full_flipped[:, 0] = -preop_coords_full_flipped[:, 0]
            intraop_coords_full_flipped[:, 0] = -intraop_coords_full_flipped[:, 0]
        if "Y" in flip_axes:
            preop_coords_full_flipped[:, 1] = -preop_coords_full_flipped[:, 1]
            intraop_coords_full_flipped[:, 1] = -intraop_coords_full_flipped[:, 1]
        if "Z" in flip_axes:
            preop_coords_full_flipped[:, 2] = -preop_coords_full_flipped[:, 2]
            intraop_coords_full_flipped[:, 2] = -intraop_coords_full_flipped[:, 2]


        # cue_valid_bits = [1,] * self.num_cues
        # cue_valid_bits = np.ones(self.num_cues)
        # if random_cue_num:
        #     cue_valid_bits = np.random.randint(0, 2, self.num_cues)
        # print(self.id, "self.cue_valid_bits", self.cue_valid_bits)

        preop_cue_list = []
        intraop_cue_list = []
        preop_cue_mask_list = []
        intraop_cue_mask_list = []
        
        for idx_cue in range(self.num_cues):
            # print("========loading cue:", idx_cue)
            #  Load cues from arrays in VTK meshes:
            cue_name = f"{self.cue_array_name_prefix}_{idx_cue}"
            preop_cue_array = vtk_to_numpy(self.geometry["preop_volume"].GetPointData().GetArray(cue_name))
            intraop_cue_array = vtk_to_numpy(self.geometry["intraop_surface"].GetPointData().GetArray(cue_name))

            # # preop_cue_index = np.where(preop_cue_array > 0)[0]
            # # intraop_cue_index = np.where(intraop_cue_array > 0)[0]
            # preop_cue_coords = preop_coords_full[preop_cue_array > 0, :]
            # intraop_cue_coords = intraop_coords_full[intraop_cue_array > 0, :]
            # print("preop_cue_coords.shape", preop_cue_coords.shape, "intraop_cue_coords.shape", intraop_cue_coords.shape)

            # thresh = vtk.vtkThreshold()
            # thresh.SetInputData( self.geometry["preop_volume"] )
            # thresh.SetInputArrayToProcess( 0,0,0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, cue_name )
            # thresh.ThresholdByUpper( 0.5 )
            # thresh.Update()

            thresh = vtk.vtkThresholdPoints()
            thresh.SetInputData( self.geometry["preop_volume"] )
            thresh.SetInputArrayToProcess(0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes.SCALARS, cue_name )
            thresh.ThresholdByUpper(0.9)
            thresh.Update()
            preop_cue_mesh = thresh.GetOutput()

            # thresh.SetInputData( self.geometry["intraop_surface"] )
            # thresh.SetInputArrayToProcess( 0,0,0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, cue_name )
            # thresh.ThresholdByUpper( 0.5 )
            # thresh.Update()
            # intraop_cue_mesh = thresh.GetOutput()
            thresh = vtk.vtkThresholdPoints()
            thresh.SetInputData( self.geometry["intraop_surface"] )
            thresh.SetInputArrayToProcess(0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes.SCALARS, cue_name )
            thresh.ThresholdByUpper(0.9)
            thresh.Update()
            intraop_cue_mesh = thresh.GetOutput()
            # print("preop_cue_mesh.GetNumberOfPoints()", preop_cue_mesh.GetNumberOfPoints(), "intraop_cue_mesh.GetNumberOfPoints()", intraop_cue_mesh.GetNumberOfPoints())

            # dist_array, closest_id_list = vtk_utils.df(
            #     mesh=self.geometry["preop_volume"],
            #     surface=preop_cue,
            #     return_idx=True,
            # )
            
            preop_cue_coords = vtk_to_numpy(preop_cue_mesh.GetPoints().GetData())
            intraop_cue_coords = vtk_to_numpy(intraop_cue_mesh.GetPoints().GetData())

            # distance from preop volume to preop cue
            dist_preop_cue = scipy.spatial.distance.cdist(preop_coords_full, preop_cue_coords)
            dist_intraop_cue = scipy.spatial.distance.cdist(intraop_coords_full, intraop_cue_coords)
            # print("dist_preop_cue.shape", dist_preop_cue.shape, "dist_intraop_cue.shape", dist_intraop_cue.shape)
            dist_preop_cue = dist_preop_cue.min(axis=1)
            dist_intraop_cue = dist_intraop_cue.min(axis=1)
            # print("dist_preop_cue.shape", dist_preop_cue.shape, "dist_intraop_cue.shape", dist_intraop_cue.shape)

            # Get the index of downsampled preop and intraop on full coordinates:
            dist_downsampled_preop_to_full = scipy.spatial.distance.cdist(preop, preop_coords_full_flipped)
            dist_downsampled_intraop_to_full = scipy.spatial.distance.cdist(intraop, intraop_coords_full_flipped)
            min_indices_preop = dist_downsampled_preop_to_full.argmin(axis=1)
            min_indices_intraop = dist_downsampled_intraop_to_full.argmin(axis=1)
            # print("min_indices_preop.shape", min_indices_preop.shape, "min_indices_intraop.shape", min_indices_intraop.shape)

            # Get the distance field of cues for downsampled preop and intraop coordinates
            df_preop_cue = dist_preop_cue[min_indices_preop]
            df_intraop_cue = dist_intraop_cue[min_indices_intraop]
            # print("df_preop_cue.shape", df_preop_cue.shape, "df_intraop_cue.shape", df_intraop_cue.shape)

            # Get the mask of cues for downsampled preop and intraop coordinates
            mask_preop_cue = preop_cue_array[min_indices_preop, ]
            mask_intraop_cue = intraop_cue_array[min_indices_intraop, ]
            # print("mask_preop_cue.shape", mask_preop_cue.shape, "mask_intraop_cue.shape", mask_intraop_cue.shape)

            preop_cue = np.concatenate( (np.expand_dims(df_preop_cue, axis=1), np.expand_dims(mask_preop_cue, axis=1)), axis=1 )
            intraop_cue = np.concatenate( ( np.expand_dims(df_intraop_cue, axis=1), np.expand_dims(mask_intraop_cue, axis=1)), axis=1 )
            # print("preop_cue.shape", preop_cue.shape, "intraop_cue.shape", intraop_cue.shape)

            if self.cue_valid_bits[idx_cue] == 1:
                preop_cue_list.append(preop_cue.transpose(1, 0))
                intraop_cue_list.append(intraop_cue.transpose(1, 0))
            elif self.cue_valid_bits[idx_cue] == 0:
                # print(self.id, "idx_cue", idx_cue, "is masked out")
                preop_cue_list.append(np.zeros(preop_cue.shape).transpose(1, 0))
                intraop_cue_list.append(np.zeros(intraop_cue.shape).transpose(1, 0))

            preop_cue_mask_list.append(mask_preop_cue)
            intraop_cue_mask_list.append(mask_intraop_cue)

            debug = False
            if debug:
                debug_folder = "/home/liupeng/ceph_home/others/debug/"
                output_path = os.path.join(debug_folder, f"preop_cue_{idx_cue}.vtp")
                vtk_utils.write_mesh(preop_cue_mesh, output_path, verbose=True)
                output_path = os.path.join(debug_folder, f"intraop_cue_{idx_cue}.vtp")
                vtk_utils.write_mesh(intraop_cue_mesh, output_path, verbose=True)

        preop_cues = np.concatenate(preop_cue_list, axis=0)
        intraop_cues = np.concatenate(intraop_cue_list, axis=0)

        # print("preop_cues.shape", preop_cues.shape, "intraop_cues.shape", intraop_cues.shape)

        preop = np.concatenate( (res.preop, preop_cues), axis=0 )
        intraop = np.concatenate( (res.intraop, intraop_cues), axis=0 )
        # print("preop.shape", preop.shape, "intraop.shape", intraop.shape)
        # print("self.id", self.id, "preop.shape", preop.shape, "intraop.shape", intraop.shape)
        # print("res.preop.shape", res.preop.shape, "res.intraop.shape", res.intraop.shape, "preop_cues.shape", preop_cues.shape, "intraop_cues.shape", intraop_cues.shape)

        res.update({
            # "preop_cue_list": preop_cue_list,
            # "intraop_cue_list": intraop_cue_list,
            # "preop_cues" :  preop_cues,
            # "intraop_cues" :  intraop_cues,
            "preop": preop,
            "intraop": intraop,
            "preop_cue_mask_list": preop_cue_mask_list,
            "intraop_cue_mask_list": intraop_cue_mask_list,
            "cue_valid_bits": self.cue_valid_bits,
        })

        return res


    # def load_from_files(self, res, npoints=0, flip_axes="", save_cues=False):
        
    #     # assert len(self.filenames["preop_cues"]) == len(self.filenames["intraop_cues"])
    #     # res.update({
    #     #         "preop_cues": [],
    #     #         "intraop_cues": [],
    #     #     })

    #     preop = res.preop[0:3, ...]
    #     intraop = res.intraop
    #     preop = torch.Tensor(preop).to("cuda").permute(1, 0).unsqueeze(0)
    #     intraop = torch.Tensor(intraop).to("cuda").permute(1, 0).unsqueeze(0)


    #     preop_cues_list, intraop_cues_list = [], []
    #     for idx_cue in range(self.num_cues):
    #         # if self.load_cue_from_files:
    #         # Load cues from mesh files:
    #         cue_path_pre = os.path.join(self.path, self.filenames["preop_cues"][idx_cue])
    #         cue_path_intra = os.path.join(self.path, self.filenames["intraop_cues"][idx_cue])
    #         preop_cue = vtk_utils.load_mesh(cue_path_pre)
    #         intra_cue = vtk_utils.load_mesh(cue_path_intra)
    #         if preop_cue.GetNumberOfPoints() == 0 or intra_cue.GetNumberOfPoints() == 0:
    #             print("INVALIDIFY 6")
    #             self.valid = False
    #             raise IOError(f"Could not load {cue_path_pre} or {cue_path_intra}")
    #         preop_cue = vtk_to_numpy(preop_cue.GetPoints().GetData())
    #         intraop_cue = vtk_to_numpy(intra_cue.GetPoints().GetData())
        
    #         preop_cue = preop_cue * self.scale 
    #         intraop_cue = intraop_cue * self.scale
    #         # if npoints > 0:
    #         #     #choice = np.random.choice( preop_cue.shape[0], npoints, replace=True)
    #         #     #intraop = intraop[choice, :]
    #         #     if preop_cue.shape[0] < npoints:
    #         #         # Create dummy points and add them to the intraop data:
    #         #         n_dummy_points = npoints - preop_cue.shape[0]
    #         #         dummy_points = np.full( (n_dummy_points, 3), 999999 )
    #         #         preop_cue = np.concatenate( (preop_cue, dummy_points), axis = 0 )
    #         if "X" in flip_axes:
    #             preop_cue[:, 0] = -preop_cue[:, 0]
    #             intraop_cue[:, 0] = -intraop_cue[:, 0]
    #         if "Y" in flip_axes:
    #             preop_cue[:, 1] = -preop_cue[:, 1]
    #             intraop_cue[:, 1] = -intraop_cue[:, 1]
    #         if "Z" in flip_axes:
    #             preop_cue[:, 2] = -preop_cue[:, 2]
    #             intraop_cue[:, 2] = -intraop_cue[:, 2]

    #         preop_cue = torch.Tensor(preop_cue).to("cuda").unsqueeze(0)
    #         intraop_cue = torch.Tensor(intraop_cue).to("cuda").unsqueeze(0)
    #         # print(preop.shape, intraop.shape, preop_cue.shape)

    #         # for the points in preop/intra_cue, find the closest point on 
    #         # the downsampled preop/intra
    #         dist_pre_cue = torch.cdist(preop_cue, preop, p=2)
    #         dist_intra_cue = torch.cdist(intraop_cue, intraop, p=2)
    #         # print(dist.shape)
    #         _, min_indices_pre_cue = torch.topk(dist_pre_cue, k=1, dim=2, largest=False, sorted=False)
    #         _, min_indices_intra_cue = torch.topk(dist_intra_cue, k=1, dim=2, largest=False, sorted=False)
    #         # print(min_dists.shape, min_indices.shape)
    #         # min_indices = min_indices.squeeze(-1).squeeze(0)
    #         # print(min_indices)

    #         preop_cue_mask = torch.zeros_like(preop)
    #         intraop_cue_mask = torch.zeros_like(intraop)
    #         preop_cue_mask[0, min_indices_pre_cue, :] = 1
    #         intraop_cue_mask[0, min_indices_intra_cue, :] = 1
    #         # print("preop_cue_mask point number:", torch.count_nonzero(preop_cue_mask), "intraop_cue_mask point number:", torch.count_nonzero(intraop_cue_mask))
    #         if save_cues:
    #             # Preop and intraop cues on downsampled because preop and intraop coordinates
    #             # are downsampled to npoints
    #             preop_cue_downsampled = torch.gather(preop, dim=1, index=min_indices_pre_cue.repeat(1, 1, 3)).squeeze(0)
    #             intraop_cue_downsampled = torch.gather(intraop, dim=1, index=min_indices_intra_cue.repeat(1, 1, 3)).squeeze(0)
    #             # print("preop_cue_downsampled.shape", preop_cue_downsampled.shape, "intraop_cue_downsampled.shape", intraop_cue_downsampled.shape)
    #             # print(intraop_cue_downsampled)
    #             self.write(
    #                 coords= preop_cue_downsampled.permute(1, 0),
    #                 filename="test_closest_preop_cue_on_downsampled_pc_{}.vtp".format(idx_cue),
    #                 # log_dir="/home/haoyum/download/3D-PhysNet/data/processed_data/2021-07-01_14-00-00",
    #             )
    #             self.write(
    #                 coords= intraop_cue_downsampled.permute(1, 0),
    #                 filename="test_closest_intraop_cue_on_downsampled_pc_{}.vtp".format(idx_cue),
    #             )
    #         # res["preop_cues"].append(preop_cue_mask)
    #         # res["intraop_cues"].append(intraop_cue_mask)
    #         preop_cues_list.append(preop_cue_mask)
    #         intraop_cues_list.append(intraop_cue_mask)
    #     res.update({
    #         "preop_cues": torch.cat(preop_cues_list, dim=0).to("cpu"),
    #         "intraop_cues": torch.cat(intraop_cues_list, dim=0).to("cpu"),
    #     })

    #     # for k in res.keys():
    #     #     print(k, res[k].shape)
    #     return res



class DisplDataset(Dataset):
    def __init__(self, path, npoints=2500, nsamples=1000, start_sample=0, verbose=True,
            sample_class=LiverSample,
            criteria=None,
            scale=1,
            augmentation = True,
            quick_preload = False,
            stats=None,
            # resample_stats_key = None,
            append_curvature = False,
            frame = "NONE",
            **kwargs):
        """
        Args:
            quick_preload: Set to True to avoid going through all folders if only a small number
                of samples is needed, i.e. nsamples < 100
            frame: NONE, or RANDOM or LAST (TODO: Implement "RANDOM" mode)
            resample_criteria: options to resample/reorder the valid samples based on the criteria of difficulties recorded in stats:
                    1. max_displ: max displacement
                    2. mean_displ: mean displacement
                    3. vis_surface_area: visible surface area
                    4. vis_surface_point: visible point numbers 
                    5. amount of noise
        """

        self.sample_class = sample_class
        self.npoints = npoints
        self.storage_path = path
        self.verbose = verbose
        self.scale = scale
        self.augmentation = augmentation
        self.quick_preload = quick_preload
        self.stats = stats

        if frame != "NONE" and frame != "LAST":
            raise NotImplementedError( f"Mode '{frame}' for argument 'frame' not implemented!" )
        if frame == "NONE":
            self.frame = None
        else:
            self.frame = math.inf

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

        self.filename_list = kwargs
        # self.intraop_type = "camera" if "cam" in self.filename_list["intraop_filename"] else "random"
        # if the provided intraop_filename is a list, then it is a mixed type
        if "intraop_filename" in self.filename_list.keys():
            if isinstance(self.filename_list["intraop_filename"], list):
                self.intraop_type = "mixed"
                self.intraop_filename_list = self.filename_list["intraop_filename"]
                # hard-coded here, as we only use noisy surfaces for training right now...
                self.cleanliness = "noisy"
            else:
                if "cam" in self.filename_list["intraop_filename"]:
                    self.intraop_type = "camera"
                else:
                    self.intraop_type = "random"

                if "noisy" in self.filename_list["intraop_filename"]:
                    self.cleanliness = "noisy"
                else:
                    self.cleanliness = "clean"
        elif "intraop_surface_filename_prefix" in self.filename_list.keys():
            if "cam" in self.filename_list["intraop_surface_filename_prefix"]:
                self.intraop_type = "camera"
            else:
                self.intraop_type = "random"


        self.min_num_valid_points = -1

        self._samples_path_list = []
        self._samples_list = []

        if nsamples > 500 and quick_preload:
            print("WARNING: quick_preload is enabled for DisplDataset, but nsamples is large. Consider disabling.")

        self.resample_stats_key_list = [
            "DisplacementStatisticsWithRigidDisplacementBlock_max_displacement_f0_to_f1",
            "DisplacementStatisticsWithRigidDisplacementBlock_mean_displacement_f0_to_f1",
            "DisplacementStatisticsBlock_max_displacement_f0_to_f1", 
            "DisplacementStatisticsBlock_mean_displacement_f0_to_f1", 
            "AddSurfaceNoiseBlock_estimated_noisy_surface_fraction_f1",                
            "AddSurfaceNoiseFromCameraBlock_estimated_noisy_surface_fraction_f1",
            "AddSurfaceNoiseBlock_number_of_remaining_points_1",
            "AddSurfaceNoiseFromCameraBlock_number_of_remaining_points_1",
        ]

        self._samples_stats_list = []
        self.append_curvature = append_curvature
        print("Analyzing data folder...")
        self._find_potential_samples( start_sample, nsamples, criteria=criteria )

        self.nsamples = len(self._samples_list)
        print("found {} samples".format(self.nsamples))
        if self.nsamples <= 0:
            raise IOError( f"Found no samples with indices {start_sample} - {start_sample+nsamples} in {path}" )

        # Keep track of which samples are invalid so we can instead load others:
        # TODO: This may not work due to threading - double check!
        self._valid_samples = [i for i in range(self.nsamples)]

        # flag to indicate if the samples are sorted according to the statistics
        self.sorted = False

        


    # def resample_valid_sample_list(self, stats, criteria=None):
    #     """Resample (reorder) the list of valid samples based on criteria of difficulties recorded in stats:
    #         1. max displacement 
    #         2. visiable surface amount 
    #         3. amount of noise
    #     """

    #     if criteria == "max_displ":
    #         p = 

    #     for i in range(self.nsamples):
    #         if criteria is None or criteria( self._samples_list[i] ):
    #             self._valid_samples.append( i )

    def _find_potential_samples(self, start_sample, nsamples, criteria=None):
        print("intraop_type:", self.intraop_type)
        last_sample = start_sample + nsamples - 1

        if self.quick_preload:   # For a small number of samples
            for int_id in range(start_sample, start_sample+nsamples):
                folder = f"{int_id:06}"
                path = os.path.join( self.storage_path, folder )
                if os.path.exists( path ):
                    sample = self.sample_class( 
                        path = path, 
                        int_id = int_id,
                        check_for_files = True, 
                        scale = self.scale,
                        frame = self.frame,
                        **self.filename_list,
                    )
                    if sample.valid:
                        self._samples_list.append( sample )

        else:
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

            # wrong_sample_list = [12311, 6012, 15498]
            # wrong_sample_list = [69298,] # this one has degenerated faces
            wrong_sample_list = [47097, 47098, 47099, 47101, 5933, 5934, 5935, 5937,]
            if stats is not None or criteria is None:
                # if we successfully loaded the statistics file, we can use it to filter the samples
                print("Checking sample folders:")
                #for int_id, s in tqdm(stats.items(), total=len(stats)):
                for int_id in range(start_sample, last_sample):
                    print("---------------------")
                    # print(int_id)

                    # Assume they are ordered:
                    #if int_id > last_sample:
                    #    break

                    if not int_id in stats or int_id in wrong_sample_list:
                        print("\tinvalid, doesn't exist")
                        continue
                    
                    if self.intraop_type == "mixed":
                        # select a random intraop type
                        intraop_filename_this_sample = self.intraop_filename_list[np.random.randint(0, len(self.intraop_filename_list))]
                        if "cam" in intraop_filename_this_sample:
                            intraop_type_this_sample = "camera"
                        else:
                            intraop_type_this_sample = "random"
                    else:
                        intraop_filename_this_sample = self.filename_list["intraop_filename"]
                        intraop_type_this_sample = self.intraop_type

                    filename_list_this_sample = self.filename_list
                    filename_list_this_sample["intraop_filename"] = intraop_filename_this_sample
                    print("filename_list_this_sample", filename_list_this_sample)
                    s = None
                    # print("criteria", criteria, criteria(s))
                    if criteria is not None:
                        s = stats[int_id]
                        # Check if the statistics of this sample meet the criteria, otherwise
                        # skip this sample
                        if criteria(s, intraop_type_this_sample, self.cleanliness) == False:
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
                                # **self.filename_list,
                                append_curvature=self.append_curvature,
                                **filename_list_this_sample,
                        )
                        # print(sample)
                        if sample.valid:
                            self._samples_list.append( sample )
                            # retrieve the statistics of the valid samples
                            stats_sample = {}
                            # for k in self.resample_stats_key_list:
                            #     if k in s:
                            #         stats_sample[k] = s[k]
                            self._samples_stats_list.append(stats_sample)
                            print("appended", len(self._samples_list))

                        print("after", len(self._samples_list))

            else:
                # Now force to provide the statistics or filepath
                raise ValueError("No statistics file provided or found!!!")

            # else:
            #     # If no statistics file / stats provided, just load all samples in the range
            #     # Find all (direct) subdirectories:
            #     folders = [d for d in os.listdir(self.storage_path)
            #                 if os.path.isdir(os.path.join(self.storage_path, d))]
            #     for folder in folders:
            #         # Use only directories where the folder name is a positive integer (possibly
            #         # with leading zeros, which will be ignored):
            #         if re.fullmatch( "[0-9]+", folder ):
            #             int_id = int(folder)
            #             if int_id >= start_sample and int_id <= last_sample:
            #                 int_str = DisplDataset.int_id_to_str( int_id )
            #                 path = os.path.join( self.storage_path, folder )

            #                 sample = self.sample_class( 
            #                         path = path, 
            #                         int_id = int_id,
            #                         check_for_files = True, 
            #                         scale = self.scale,
            #                         frame = self.frame,
            #                         **self.filename_list,
            #                 )

            #                 if sample.valid:
            #                     self._samples_list.append( sample )

        # Sort the samples list by the used ID:
        # TODO: do we need this? 
        # self._samples_list.sort( key=lambda s: s.id )


    def __len__(self):
        return self.nsamples


    def get_sample(self, index):
        cur_index = self._valid_samples[index]
        sample = self._samples_list[cur_index]
        return sample
    
    
    def get_meshes(self, index):
        cur_index = self._valid_samples[index]
        sample = self._samples_list[cur_index]
        meshes = sample.geometry
        return meshes


    def resample(self, resample_key, difficulty=None, type="sort"):
        """Reorder / resample the valid samples based on the difficulty of the samples.
        This function is usually executed once for all training samples

        Args:
            resample_key_list (list): list of key(s) in statistics
            difficulty (list, optional): a list of difficulty if resample key is not provided. Defaults to None.
            type (str, optional): sort or resample the list. Defaults to "sort".
        """
        print("Resampling dataset for curriculum learning...")
        print("\tdifficulty:", resample_key if difficulty is None else "user defined")
        print("\ttype:", type)


        if difficulty is not None:
            # if difficulty is given by user:
            assert len(difficulty) == len(self._samples_list) == len(self._samples_stats_list), "difficulty list should have the same length as the samples list"

        else:
            # if not, we use the stats collected from statistics.yaml
            assert resample_key in self.resample_stats_key_list, "invalid resample key {}".format(resample_key)
            resample_stats_list = [ s[resample_key] for s in self._samples_stats_list ]
            # for r in resample_stats_list:
            #     print(r)
            resample_stats_list = np.array(resample_stats_list, dtype=np.float64)
            resample_stats_list /= np.sum(resample_stats_list).astype(np.float64)
            # if "Displ" in resample_key:
            #     resample_stats_list = 1 - resample_stats_list
            #     resample_stats_list /= resample_stats_list.sum()
            difficulty = resample_stats_list

            # resample_stats_list_total = []
            # for resample_key in resample_key_list:
            #     assert resample_key in self.resample_stats_key_list, "invalid resample key {}".format(resample_key)
            #     resample_stats_list = [ s[resample_key] for s in self._samples_stats_list ]
            #     # for r in resample_stats_list:
            #     #     print(r)
            #     resample_stats_list = np.array(resample_stats_list, dtype=np.float64)
            #     resample_stats_list /= np.sum(resample_stats_list).astype(np.float64)
            #     # if "Displ" in resample_key:
            #     #     resample_stats_list = 1 - resample_stats_list
            #     #     resample_stats_list /= resample_stats_list.sum()
            #     if len(resample_stats_list_total) == 0:
            #         resample_stats_list_total = resample_stats_list
            #     else:
            #         resample_stats_list_total *= resample_stats_list
            # difficulty = resample_stats_list


        if type == "sort":
            # combine sample and difficulty, then sort by d
            if "vis" in resample_key:
                # less visible area/points means more difficult, we have to reverse the order
                difficulty = 1 - difficulty
                difficulty /= difficulty.sum()
            s_d = zip(self._samples_list, difficulty)
            s_d = sorted(s_d, key=lambda x: x[1])
            self._samples_list = [s for s, d in s_d]
        elif type == "resample":
            # resample the samples based on the difficulty (probability)
            # 
            difficulty = np.array(difficulty)
            difficulty /= difficulty.sum()
            if "Displ" in resample_key:
                difficulty = 1 - difficulty
                difficulty /= difficulty.sum()
            self._samples_list = np.random.choice(
                a=self._samples_list, 
                size=len(self._samples_list), 
                p=difficulty, 
                replace=False,
            )
        else:
            raise ValueError("Not supported type:{}".format(type))



    def resample_dynamic(self, resample_key_list, epoch, epoch_total, stages=1, return_prob_only=False):
        """Resample the valid samples based on the difficulty of the samples and the current epoch.
        The difficulty comes from the selected resample key(s). If multiple keys are provided, the difficulty is the 
        combination of the statistics. 
        'Dynamic' refers to the flexible difficulty based on the current epoch number (different purpose as the resample
        fuction), we want the distribution of the dataset to gradually change from distribution fully created by the 
        statistics to a uniform distribution.
        For example:
                diff_final = (1 - mean_displ) * w_epoch + uniform * (1 - w_epoch)

        Args:
            resample_key_list (_type_): _description_
            epoch (_type_): _description_
        """
        print("Resampling dataset for curriculum learning...")
        print("\tkeys:", resample_key_list)
        print("\tcurrent epoch:", epoch, "total epoch", epoch_total)
        print("\tcurrent stage:", math.floor( (epoch / epoch_total) * stages ), "total stages:", stages)

        assert epoch >= 0, "epoch should be a positive integer"

        # we use the stats collected from statistics.yaml
        resample_stats_list_total = []
        for resample_key in resample_key_list:
            assert resample_key in self.resample_stats_key_list, "invalid resample key {}".format(resample_key)
            stats_list = np.array([ s[resample_key] for s in self._samples_stats_list ], dtype=np.float64)
            if "Displ" in resample_key:
                stats_list = 1 - stats_list
            # stats_list /= np.sum(stats_list).astype(np.float64)
            stats_list /= np.max(stats_list)
            if len(resample_stats_list_total) == 0:
                resample_stats_list_total = stats_list
            else:
                resample_stats_list_total *= stats_list

        # prob_difficulty = np.asarray(resample_stats_list_total) / np.sum(resample_stats_list_total)
        prob_difficulty = resample_stats_list_total / np.max(resample_stats_list_total)
        prob_uniform = np.ones(len(self._samples_list)) / len(self._samples_list)

        if not self.sorted:
            # sort the samples based on prob_difficulty at the beginning
            print("sort the dataset based on the difficulty")
            s_d = zip(self._samples_list, prob_difficulty)
            s_d = sorted(s_d, key=lambda x: x[1])
            self._samples_list = [s for s, d in s_d]
            self.sorted = True

        # create linear prob_difficulty
        # as the distribution of the calculated prob_difficulty from the statistics is not even, only little samples has
        # high probability, most samples have very low probability, using such probabilites the sampler will tend to sample
        # more easy samples, so we want to make the distribution more linear
        prob_difficulty = np.linspace(1e-6, 1, len(self._samples_list))

        # dynamic probability, at the beginning of the training, we want the distribution to be fully created by the statistics
        # as the training goes on, we want the distribution to be more uniform, so the network will see all samples
        # prob_dynamic = prob_difficulty * (stages - math.floor( (epoch / 100) * stages ) ) + prob_uniform * ( math.ceil( (epoch / 100) * stages ))
        if stages == 1:
            # distribution will change every epoch
            prob_dynamic = prob_difficulty * (1 - epoch / epoch_total) + prob_uniform * (epoch / epoch_total)
        else:
            # same stage will have the same distribution
            prob_dynamic = prob_difficulty * (stages - 1 - math.floor( (epoch / epoch_total) * stages ) ) + prob_uniform * ( math.floor( (epoch / epoch_total) * stages ))
        
        if return_prob_only:
            print("only return the probability")
            return prob_dynamic
        else:
            print("resample the dataset based on the dynamic probability")
            prob_dynamic /= np.sum(prob_dynamic)
            self._samples_list = np.random.choice(
                a=self._samples_list, 
                size=len(self._samples_list), 
                p=prob_dynamic, 
                replace=True,
            )
            
    def get_statistics(self):
        return self._samples_stats_list


    def set_min_num_valid_points(self, min_num_valid_points):
        self.min_num_valid_points = min_num_valid_points


    def __getitem__(self, index):
        # print("loading...", index)
        values_per_point = 3
        # TODO: This may not work due to threading - double check!
        cur_index = self._valid_samples[index]

        flip_axes = ""
        if self.augmentation:
            if random.random() > 0.5:
                flip_axes += "X"
            if random.random() > 0.5:
                flip_axes += "Y"
            if random.random() > 0.5:
                flip_axes += "Z"

        num_tries = 0
        while num_tries < 100:
            try:
                sample = self._samples_list[cur_index]
                # print(sample.path)
                # preop, intraop, displ = sample.load()
                res = sample.load(
                    npoints=self.npoints, 
                    min_num_valid_points=self.min_num_valid_points,
                    flip_axes = flip_axes,
                )

                for k in res.keys():
                    # print("k", k, res[k])
                    res[k] =  torch.Tensor( res[k])


                # If we managed to successfully load the sample:
                self._valid_samples[index] = cur_index
                # return preop, intraop, displ
                return res
            except IOError as e:
                if self.verbose:
                    print(f"IOError loading sample {cur_index}:\n\t{e}")
                    sample = self._samples_list[cur_index]
                    print("\tfrom path:", sample.path)
                    traceback.print_exc()
                    print("Try next sample")
                num_tries += 1
                cur_index += 1                  # TODO: Note: incrementing by 1 will mean that those samples that  follow invalid samples will be over-represented in the dataet. Maybe better to randomly sample?
                if cur_index >= len(self._samples_list):
                    cur_index = 0
            except AttributeError as e:
                if self.verbose:
                    print("AttributeError while loading sample", cur_index)
                    print("\t", e)
                    sample = self._samples_list[cur_index]
                    print("\tfrom path:", sample.path)
                    traceback.print_exc()
                    print("Try next sample")
                num_tries += 1
                cur_index += 1                  # TODO: Note: incrementing by 1 will mean that those samples that  follow invalid samples will be over-represented in the dataet. Maybe better to randomly sample?
                if cur_index >= len(self._samples_list):
                    cur_index = 0

    @classmethod
    def int_id_to_str(cls, id_as_int, length=6):
        """
        Converts an integer id to the corresponding sample folder name.
        Note that this conversion does not guarantee that this folder exists!
        Arguments:
        - length: Integer, gives the number of digits to use. String will be zero-padded if
            necessary.
        """
        assert type(id_as_int) == int, "int_id_to_str argument must be integer!"
        id_as_str = str( id_as_int )

        return id_as_str.zfill( length )



def save_point_cloud( tensor, filename ):
    t = tensor.permute( 1, 0 ).cpu().numpy()
    np.savetxt( filename, t, delimiter="," )



def resample_by_given_stats(
        stats,
        resample_key_list,
        epoch,
        epoch_total,
):
    """This function is used to calculate difficulties for sub datasets based on the given statistics.

    Args:
        stats (list): list of statistics of the samples
        resample_key_list (list): list of statistics keys to calculate the difficulties
    """


    print("Resampling dataset for curriculum learning...")
    print("\tkeys:", resample_key_list)


    # assert epoch >= 0, "epoch should be a positive integer"

    # we use the stats collected from statistics.yaml
    resample_stats_list_total = []
    for resample_key in resample_key_list:
        # assert resample_key in resample_stats_key_list, "invalid resample key {}".format(resample_key)
        stats_list = np.array([ s[resample_key] for s in stats ], dtype=np.float64)
        if "Displ" in resample_key:
            stats_list = 1 - stats_list
        # stats_list /= np.sum(stats_list).astype(np.float64)
        stats_list /= np.max(stats_list)
        if len(resample_stats_list_total) == 0:
            resample_stats_list_total = stats_list
        else:
            resample_stats_list_total *= stats_list

    # prob_difficulty = np.asarray(resample_stats_list_total) / np.sum(resample_stats_list_total)
    prob_difficulty = resample_stats_list_total / np.max(resample_stats_list_total)
    prob_uniform = np.ones(len(resample_stats_list_total))

    # create linear prob_difficulty
    # as the distribution of the calculated prob_difficulty from the statistics is not even, only little samples has
    # high probability, most samples have very low probability, using such probabilites the sampler will tend to sample
    # more easy samples, so we want to make the distribution more linear
    # prob_difficulty = np.linspace(1e-6, 1, len(self._samples_list))

    # dynamic probability, at the beginning of the training, we want the distribution to be fully created by the statistics
    # as the training goes on, we want the distribution to be more uniform, so the network will see all samples
    # prob_dynamic = prob_difficulty * (stages - math.floor( (epoch / 100) * stages ) ) + prob_uniform * ( math.ceil( (epoch / 100) * stages ))
    # if stages == 1:
        # distribution will change every epoch
    prob_dynamic = prob_difficulty * (1 - epoch / (epoch_total - 1)) + prob_uniform * (epoch / (epoch_total - 1 ))
    # else:
    #     # same stage will have the same distribution
    #     prob_dynamic = prob_difficulty * (stages - 1 - math.floor( (epoch / epoch_total) * stages ) ) + prob_uniform * ( math.floor( (epoch / epoch_total) * stages ))
    

    return prob_dynamic



