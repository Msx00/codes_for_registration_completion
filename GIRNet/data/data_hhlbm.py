import os, sys
import json
import glob
import random
import scipy

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle as pkl

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
vtk.vtkObject.GlobalWarningDisplayOff() 

try:
    from . import vtk_utils
    from . import extract_surface
    #from . import statistics
    from . import pc_utils
except:
    try:
        import vtk_utils
        import extract_surface
        #import statistics
        import pc_utils
    except:
        from data import vtk_utils
        from data import extract_surface
        #from data import statistics
        from data import pc_utils


# set random seeds
def reset_seed(seed_num=1):
    """Set random seeds for reproducibility."""
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Scan():

    def __init__( 
            self, 
            folder_preprocessed, 
            folder_landmarks, 
            scan_name,
            npoints=0, 
            scale = 1, 
            filename_liversurface="liver.vtp", 
            filename_partial_surface="liver_focused.vtp",
            filenames_internal_slices=["internals_contour.vtp"],
            filenames_internal_full=["artery.vtp","vein.vtp"],
            folder_name_partial_surface=None,
            load_internals = True
        ):

        self.folder_preprocessed = folder_preprocessed
        self.folder_landmarks = folder_landmarks
        self.scan_name = scan_name
        self.npoints = npoints
        self.scale = scale
        self.filename_liversurface = filename_liversurface
        self.filename_partial_surface = filename_partial_surface
        self.filenames_internal_slices = filenames_internal_slices
        self.filenames_internal_full = filenames_internal_full
        self.folder_name_partial_surface = folder_name_partial_surface
        self.load_internals = load_internals
        # self.geomerty = {}
        self.original_surface = None

        # load once to check whether the scan is suited as preop or intraop
        # real meshes will be loaded by calling load() function in Patient class
        self.load()

    def load( self, center=False, center_offset=[] ):
        """Load all meshes for this scan including preop, intraop, and internal structures.

        Args:
            center (bool, optional): if this scan is chosen to be preoperative scan, the preop_volume needs to be centered. Defaults to False.
            center_offset (list, optional): if this scan is chosen to be intraoperative scan, the intraop_surfaces need to be centered by the given center offset. Defaults to [].

        """
        self.mesh_filenames = []

        #####################################################
        ## Load full surface mesh (usually used as 'preoperative' mesh

        # Find a liver.vtp file in folder_preprocessed:
        files = glob.glob( os.path.join(self.folder_preprocessed, "**", self.filename_liversurface), recursive=True )
        assert len(files) == 1, f"Expected exactly one '{self.filename_liversurface}' file in {self.folder_preprocessed}, " +\
                f"but found {len(files)}"
        vtp_filename = files[0]
        print("loading vtp file:", vtp_filename)
        folder_name, file_name = os.path.split( vtp_filename )

        files = glob.glob( os.path.join( folder_name, "segmentation_error*"), recursive=True )
        files_r = [os.path.split(f)[1] for f in files]
        #print("\tsegmentation errors found:", files_r)
        #if os.path.exists( os.path.join( folder_name, "segmentation_errors_internals" ) ):
        #    raise IOError( f"Will not load segmentation {vtp_filename}, segmentation_errors_internals file detected." )
        self.suited_as_preop = True
        self.suited_as_intraop = True
        if os.path.exists( os.path.join( folder_name, "segmentation_error_surface" ) ):
            self.suited_as_preop = False
            self.suited_as_intraop = False
            print( "Surface errors detected. Not using as preop or intraop." )
        if os.path.exists( os.path.join( folder_name, "segmentation_error_internals" ) ):
            self.suited_as_intraop = False
            print( "Internal errors detected. Not using as intraop." )
        if os.path.exists( os.path.join( folder_name, "segmentation_error_surface_piece_missing" ) ):
            self.suited_as_preop = False
            print( "Surface errors detected (piece missing). Not using as preop." )

        if not self.suited_as_preop and not self.suited_as_intraop:
            raise IOError( f"Will not load segmentation {vtp_filename}, segmentation_errors found" )

        preop_surface = vtk_utils.load_mesh( vtp_filename )
        print("preop_surface:", preop_surface.GetNumberOfPoints())

        self.mesh_filenames.append( vtp_filename )

        # First, scale point cloud from mm to m:
        t = vtk.vtkTransform()
        t.Scale( self.scale, self.scale, self.scale )
        tf = vtk.vtkTransformFilter()
        tf.SetTransform( t )
        tf.SetInputData( preop_surface )
        tf.Update()
        preop_surface = tf.GetOutput()

        print("center_offset:", center_offset)
        print("center:", center)
        self.center_offset = []
        if len(center_offset) > 0:
            # if center_offset is provided, meaning this scan is intraoperative scan and 
            # center_offset is from the preoperative scan, so center all meshes by the 
            # given center offset:
            self.center_offset = center_offset
            preop_surface = vtk_utils.transform_mesh(
                mesh=preop_surface,
                trans=self.center_offset,
            )
        elif center == True:
            # center is True, meaning this scan is preoperative scan, so center the mesh:
            preop_surface, preop_center_offset = vtk_utils.center_mesh(mesh=preop_surface)
            # all the following meshes will be transformed by this center offset:
            self.center_offset = preop_center_offset

        self.estimated_volume_size = vtk_utils.estimate_volume_poly( preop_surface )
        self.surface_area = vtk_utils.calc_surface_area( preop_surface )

        #preop_surface = self.surface

        internal_points = vtk_utils.create_random_internal_points(
                preop_surface, points_to_create=preop_surface.GetNumberOfPoints(), append_surface=False )


        if internal_points.GetNumberOfPoints() == 0:
            print( f"Could not create random internal points for mesh. Will not use as preop. volume." )
            self.suited_as_preop = False
            internal_points = preop_surface # just so we can keep working, won't be used.
        
        #print("Number of points:", preop_surface.GetNumberOfPoints(), internal_points.GetNumberOfPoints())
        #print("\tRatio:", preop_surface.GetNumberOfPoints()/internal_points.GetNumberOfPoints())

        if not preop_surface.GetPointData().GetNormals():
            print("No normals found in preop_surface, computing them...")

            # For the surface, compute a normal for each point:
            surface_normals = vtk_utils.compute_point_normals( preop_surface )

            preop_surface.GetPointData().SetNormals( surface_normals )

        self.original_surface = preop_surface

        # Resample to a certain distance between points:
        preop_surface = vtk_utils.resample_polydata( preop_surface )
        # Update the normals array:
        surface_normals = preop_surface.GetPointData().GetNormals()

        # Subsample points:
        #mask = vtk.vtkMaskPoints()
        #mask.SetInputData( preop_surface )
        #mask.SetRandomMode( True )
        #mask.SetRandomModeType(1)   # random sample
        #mask.SetMaximumNumberOfPoints( int(internal_points.GetNumberOfPoints()*0.3) )
        #mask.Update()
        #preop_surface = mask.GetOutput()

        #print("Number of points:", preop_surface.GetNumberOfPoints(), internal_points.GetNumberOfPoints())
        #print("\tRatio:", preop_surface.GetNumberOfPoints()/internal_points.GetNumberOfPoints())

        ## Update to only keep the normals of the selected points:
        #surface_normals = preop_surface.GetPointData().GetNormals()

        #writer = vtk.vtkXMLPolyDataWriter()
        #writer.SetFileName("/tmp/surface_with_computed_normals.vtp")
        #writer.SetInputData( preop_surface )
        #writer.Update()

        # For the internal points, compute the distance to the surface
        print("Computing distance field")
        distance_field = vtk_utils.df( mesh=internal_points, surface=preop_surface )
        print("...done")

        preop_internal_xyz = vtk_to_numpy( internal_points.GetPoints().GetData() )
        preop_surface_xyz = vtk_to_numpy( preop_surface.GetPoints().GetData() )
        preop_internal_dists = np.expand_dims( vtk_to_numpy(distance_field), axis = 1 )
        preop_surface_normals = vtk_to_numpy(surface_normals)

        # TODO: Is this always necessary?!
        #preop_surface_normals = -1 * preop_surface_normals

        # HHLBM has inverted faces!
        # preop_surface_normals = -preop_surface_normals

        n_internal_points = preop_internal_xyz.shape[0]
        n_surface_points = preop_surface_xyz.shape[0]

        #print("n_internal_points", n_internal_points)
        #print("n_surface_points", n_surface_points)
        #print("surface_normals", preop_surface_normals.shape)

        # The internal surface points can't have a "normal", so set the value to zero:
        preop_internal_normals = np.zeros( (n_internal_points, 3), dtype=float )
        preop_surface_dists = np.zeros( (n_surface_points, 1), dtype=float )


        # downsample or add dummy points so that all point clouds have the same amount of points 
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
                        total_points = self.npoints )

            # Re-assign to original variables:
            preop_internal_xyz, preop_internal_dists, preop_internal_normals = r_internal
            preop_surface_xyz, preop_surface_dists, preop_surface_normals = r_surface

        #if volume_np.shape[0] > npoints:
        #    choice = np.random.choice( volume_np.shape[0], npoints, replace=True)
        #    self.volume_np = volume_np[choice, :]
        #if volume_np.shape[0] < npoints:
        #    # Create dummy points and add them to the intraop data:
        #    n_dummy_points = npoints - volume_np.shape[0]
        #    dummy_points = np.full( (n_dummy_points, 3), 999999 )
        #    self.volume_np = np.concatenate( (volume_np, dummy_points), axis = 0 )

        #print("internal", preop_internal_xyz.shape )
        #print("internal normals", preop_internal_normals.shape )
        #print("internal dists", preop_internal_dists.shape )
        #print("surface", preop_surface_xyz.shape )
        #print("surface normals", preop_surface_normals.shape )
        #print("surface dists", preop_surface_dists.shape )

        # Combine internal points and surface points along point axis (0):
        preop_xyz = np.concatenate( (preop_internal_xyz, preop_surface_xyz), axis=0 )
        preop_dists = np.concatenate( (preop_internal_dists, preop_surface_dists), axis=0 )
        preop_normals = np.concatenate( (preop_internal_normals, preop_surface_normals), axis=0 )
        #print("preop_xyz:", preop_xyz.shape)
        #print("preop_dists:", preop_dists.shape)
        #print("preop_normals:", preop_normals.shape)
        #if scale != 1:
        #    preop_xyz = preop_xyz * scale
        #    preop_dists = preop_dists * scale
        # Combine point coordinates and features along the feature axis:
        preop_np = np.concatenate( (preop_xyz, preop_dists, preop_normals), axis=1 )

        #max_dimensions_of_volume = preop_xyz.max( axis=0 )
        #min_dimensions_of_volume = preop_xyz.min( axis=0 )

        #print("preop:", preop.shape)
        
        # coor_points_idx_list = vtk_utils.closest_vetices(source=mesh_surface, target=mesh)
        # print(coor_points_idx_list)
        # mesh_surface_data = vtk_to_numpy(mesh_surface)

        #print("displ:", displ.shape)
        #preop = preop * scale

        # downsample the all preoperative point clouds so that they have the same amount of points 
        #if npoints > 0:
        #    #choice = np.random.choice( preop.shape[0], npoints, replace=True)
        #    # Non-random, deterministic loading:
        #    n_total_points = preop.shape[0]
        #    choice = np.linspace(0, n_total_points, num=npoints, endpoint=False, dtype=int)
        #    #print(choice, choice.dtype)
        #    preop = preop[choice, :]

        #####################################################
        ## Apply scale if necessary:
        #if scale != 1:
        #    tf = vtk.vtkTransform()
        #    tf.Scale( scale, scale, scale )
        #    tf_filter = vtk.vtkTransformFilter()
        #    tf_filter.SetTransform( tf )

        #    tf_filter.SetInputData( self.surface )
        #    tf_filter.Update()
        #    self.surface = tf_filter.GetOutput()

        #DEBUG_WRITE = True
        #if DEBUG_WRITE:
        #    self.volume.GetPointData().AddArray(distance_field)

        #    writer = vtk.vtkXMLPolyDataWriter()
        #    writer.SetFileName( "/tmp/volume.vtp" )
        #    writer.SetInputData( self.volume )
        #    writer.Update()

        #####################################################
        ## Load landmarks (use to calculate a sparse ground-truth displacement)
        ## Landmarks may be given in .json files (exported from 3D Slicer) or as .vtp files
        # annotator = "pl"
        annotator = "bg"
        try:
            files = sorted(glob.glob( os.path.join(self.folder_landmarks, "**", "*landmarks*.json"), recursive=True ))
            # print(os.path.join(self.folder_landmarks, "**", "*landmarks*.json"))
            # files = glob.glob( os.path.join(self.folder_landmarks, "**" "*landmarks*.json"), recursive=True )
            print("files:", files)
            # this is for only one annotator:
            # assert len(files) == 1, f"Expected exactly one '*landmarks*.json' file in {self.folder_landmarks}, " +\
            #         f"but found {len(files)}"
            # for multiple annotator:
            assert len(files) > 0, f"Expected at least one '*landmarks*.json' file in {self.folder_landmarks}"
        except AssertionError as e:
            print("Could not find landmarks*.json file. Trying 'landmarks.vtp'")
            files = glob.glob( os.path.join(self.folder_landmarks, "**", "landmarks.vtp"), recursive=True )
            assert len(files) == 1, f"Expected exactly one '*landmarks.json' or one '*landmarks*.vtp' file in {self.folder_landmarks}, " +\
                    f"but found more or less"
        
        if len(files) == 1:
            landmarks_file = files[0]
        else:
            # landmarks_file = list(filter(lambda x: annotator in x.split("/")[-1], files))[0]
            landmarks_file = list(filter(lambda x: annotator in x.split("/")[-1], files))
            if len(landmarks_file) == 0:
                # there can be multiple other annotators, so just take the first one
                landmarks_file = files[0]
            else:
                landmarks_file = landmarks_file[0]

        print("selected landmarks file:", landmarks_file)

        control_points_vtk = vtk.vtkPoints()
        if landmarks_file.endswith("json"):
            with open( landmarks_file, "r", encoding="utf-8",  ) as f:
                landmark_data = json.load( f )
                # print(landmark_data)
                control_points = landmark_data["markups"][0]["controlPoints"]
                for p in control_points:
                    control_points_vtk.InsertNextPoint( *p["position"] )
        else:
            # control_points_vtk = vtk_utils.load_mesh( landmarks_file ).GetPoints()
            control_points_vtk = vtk_utils.load_mesh( landmarks_file )
            #tr = vtk.vtkTransform()
            #tr.PreMultiply()
            #tr.RotateZ(180)
    
            #pd = vtk.vtkPolyData()
            #pd.SetPoints( control_points_vtk )
            #tr_f = vtk.vtkTransformFilter()
            #tr_f.SetTransform( tr )
            #tr_f.SetInputData( pd )
            #tr_f.Update()
            #control_points_vtk = tr_f.GetOutput().GetPoints()

        print("control_points_vtk.GetNumberOfPoints() ", control_points_vtk.GetNumberOfPoints() )
        assert control_points_vtk.GetNumberOfPoints() > 0



        if self.scale != 1:
            t = vtk.vtkTransform()
            t.Scale( self.scale, self.scale, self.scale )
            tf = vtk.vtkTransformFilter()
            tf.SetTransform( t )

            pd = vtk.vtkPolyData()
            pd.SetPoints( control_points_vtk )
            tf.SetInputData( pd )
            tf.Update()
            # control_points_vtk = tf.GetOutput().GetPoints()
            control_points_vtk = tf.GetOutput()

        control_points_np = vtk_to_numpy(control_points_vtk.GetPoints().GetData())
        control_points_mask = np.any(control_points_np, axis=1)

        # center the control points by offset:
        if len(self.center_offset) > 0:
            control_points_vtk = vtk_utils.transform_mesh(
                mesh=control_points_vtk,
                trans=self.center_offset
            )

        control_points_np = vtk_to_numpy(control_points_vtk.GetPoints().GetData())
        control_points_np[~control_points_mask] = 0.0

        print( f"Found {control_points_vtk.GetNumberOfPoints()} landmarks (valid: {control_points_mask.sum()} )")

        #####################################################
        # SELECT ONLY PART OF SURFACE:

        ### Multiple partial surfaces
        # if the partial surface folder is give, try to load all the partial surfaces 
        # from there into a list 
        # this list is used for experiments where we want to use multiple partial surfaces
        # with different amounts of points
        self.partial_surface_list = []
        self.partial_surface_filenames = []
        self.partial_surface_list_np = []
        self.partial_surface_area_list = []
        self.partial_surface_area_list_percentage = []
        if self.folder_name_partial_surface:
            self.partial_surface_list = []
            self.partial_surface_list_np = []
            partial_surface_files = sorted(glob.glob( os.path.join(self.folder_preprocessed, self.folder_name_partial_surface, "liver_focused*"), recursive=True,  ))
            print("partial_surface_files:", partial_surface_files)
            self.partial_surface_filenames = partial_surface_files
            for idx_p, partial_surface_file in enumerate(partial_surface_files):
                print("loading partial surface {} of {}".format(idx_p, len(partial_surface_files)))
                partial_surface = vtk_utils.load_mesh( partial_surface_file )
                if self.scale != 1:
                    t = vtk.vtkTransform()
                    t.Scale( self.scale, self.scale, self.scale )
                    tf = vtk.vtkTransformFilter()
                    tf.SetTransform( t )
                    tf.SetInputData( partial_surface )
                    tf.Update()
                    partial_surface = tf.GetOutput()
                
                # center intraoperative surfaces by center offset:
                if len(self.center_offset) > 0:
                    partial_surface = vtk_utils.transform_mesh(
                        mesh=partial_surface,
                        trans=self.center_offset,
                    )

                # print("NUM POINTS:", partial_surface.GetNumberOfPoints())
                partial_surface = vtk_utils.resample_polydata( partial_surface, max_num_points = 2500, clean=False )
                # print("NUM POINTS RESAMPLED:", partial_surface.GetNumberOfPoints())

                #volume_np = vtk_to_numpy(self.volume.GetPoints().GetData())
                surface_np = vtk_to_numpy(partial_surface.GetPoints().GetData())
                surface_normals = vtk_to_numpy(partial_surface.GetPointData().GetNormals())
                # surface_normals = -1 * surface_normals
                # print("surface_np.shape", surface_np.shape)
                # print(surface_np[:3, ])
                surface_np = np.concatenate( (surface_np, surface_normals), axis=1 )

                if self.npoints > 0:
                    r_surface = pc_utils.set_npoints( [surface_np], self.npoints )
                    #print("Adjust number of points to:", npoints, "from", surface_np.shape[0], "to", r_surface[0].shape[0])
                    surface_np = r_surface[0]

                self.partial_surface_list.append(partial_surface)
                self.partial_surface_list_np.append(surface_np)

            stats_path = glob.glob(os.path.join(self.folder_preprocessed, self.folder_name_partial_surface, "*.pkl"))[0]
            with open(stats_path, "rb") as f:
                stats = pkl.load(f)
                f.close()
            for idx_stat in range(len(stats)):
                cur_partial_surface_area = stats[idx_stat]["area"] * self.scale * self.scale
                self.partial_surface_area_list.append(cur_partial_surface_area)
                print("cur_partial_surface_area  / self.surface_area", cur_partial_surface_area, self.surface_area, cur_partial_surface_area  / self.surface_area)
                # print("stats[idx_stat]['area'] / self.surface_area:",  stats[idx_stat]["area"], self.surface_area,  stats[idx_stat]["area"] / self.surface_area)
                self.partial_surface_area_list_percentage.append(cur_partial_surface_area  / self.surface_area)
            
        else:
            ### Single partial surface
            # if no partial surface folder is given, try to load the single partial surface use the give
            # filename and if it does not exist, create a new partial surface
            # 
            # Check if a partial surface is available, if not, create one:
            files = glob.glob( os.path.join(self.folder_preprocessed, "**", self.filename_partial_surface), recursive=True )
            if len(files) > 0:  
                vtp_filename = files[0]
                #print("loading vtp file:", vtp_filename)
                partial_surface = vtk_utils.load_mesh( vtp_filename )

                if self.scale != 1:
                    t = vtk.vtkTransform()
                    t.Scale( self.scale, self.scale, self.scale )
                    tf = vtk.vtkTransformFilter()
                    tf.SetTransform( t )
                    tf.SetInputData( partial_surface )
                    tf.Update()
                    partial_surface = tf.GetOutput()
                
                # center intraoperative surfaces by center offset:
                if len(self.center_offset) > 0:
                    partial_surface = vtk_utils.transform_mesh(
                        mesh=partial_surface,
                        trans=self.center_offset,
                    )
                print("partial_surface.GetNumberOfPoints():", partial_surface.GetNumberOfPoints())
            else:
                select_random_piece = False
                if select_random_piece:
                    partial_surface = extract_surface.random_surface(
                        self.original_surface,
                        #surface_amount = random.random()*0.1+0.2,
                        #surface_amount = random.random()*0.1+0.7,
                        #surface_amount = random.random()*0.4+0.2,
                        surface_amount = random.random()*0.1+0.4,
                        w_distance = 15.875,
                        w_normal = 0.625,
                        w_noise = 0,#0.15,
                    )

                    # center intraoperative surfaces by center offset:
                    if len(self.center_offset) > 0:
                        partial_surface = vtk_utils.transform_mesh(
                            mesh=partial_surface,
                            trans=self.center_offset,
                        )

                else:
                    partial_surface = self.original_surface

            print("before resample: partial_surface:", partial_surface.GetNumberOfPoints())
            partial_surface = vtk_utils.resample_polydata( partial_surface, max_num_points = 2500, clean=False )
            print("after resample: partial_surface:", partial_surface.GetNumberOfPoints())

            #volume_np = vtk_to_numpy(self.volume.GetPoints().GetData())
            surface_np = vtk_to_numpy(partial_surface.GetPoints().GetData())
            surface_normals = vtk_to_numpy(partial_surface.GetPointData().GetNormals())

            # TODO: why is this necessary?
            #surface_normals = -1 * surface_normals

            # HHLBM has inverted faces!
            # surface_normals = -surface_normals
            #if scale != 1:
            #    surface_np = surface_np * scale

            surface_np = np.concatenate( (surface_np, surface_normals), axis=1 )

        
            #if scale != 1:
                #surface_np = surface_np * scale
                #control_points_np = control_points_np * scale

            if self.npoints > 0:
                r_surface = pc_utils.set_npoints( [surface_np], self.npoints )
                surface_np = r_surface[0]



        #############################
        
    
        # Perform a sanity check to catch if the landmarks are _outside_ the volume.
        # This sometimes happend because Slicer adds a transform to the landmarks that we didn't apply.
        max_dimensions_of_volume = preop_xyz.max( axis=0 )
        min_dimensions_of_volume = preop_xyz.min( axis=0 )
        max_dimensions_of_control_points = control_points_np.max( axis=0 )
        min_dimensions_of_control_points = control_points_np.min( axis=0 )

        #print("Volume dimensions:")
        #print("\tmin", min_dimensions_of_volume)
        #print("\tmax", max_dimensions_of_volume)
        #print("Landmark dimensions:")
        #print("\tmin", min_dimensions_of_control_points)
        #print("\tmax", max_dimensions_of_control_points)

        assert np.greater( max_dimensions_of_volume, max_dimensions_of_control_points ).all() and \
            np.less( min_dimensions_of_volume, min_dimensions_of_control_points ).all(), \
            "Not all landmarks lie within the organ's bounding box!"


        #self.preop_np = preop_np
        #self.surface_np = surface_np
        #self.landmarks_np = control_points_np

        self.volume = torch.Tensor( preop_np ).transpose(0,1)
        if len(self.partial_surface_list_np) > 0:
            self.surface = []
            for partial_surface_np in self.partial_surface_list_np:
                self.surface.append(torch.Tensor( partial_surface_np ).transpose(0,1))
        else:
            self.surface = torch.Tensor( surface_np ).transpose(0,1)
        self.landmarks = torch.Tensor( control_points_np ).transpose(0,1)

        #if control_points_np.shape[0] > npoints:
        #    choice = np.random.choice( control_points_np.shape[0], npoints, replace=True)
        #    self.landmarks_np = control_points_np[choice, :]
        #if control_points_np.shape[0] < npoints:
        #    # Create dummy points and add them to the intraop data:
        #    n_dummy_points = npoints - control_points_np.shape[0]
        #    dummy_points = np.full( (n_dummy_points, 3), 999999 )
        #    self.landmarks_np = np.concatenate( (control_points_np, dummy_points), axis = 0 )
        
        #############################
        # Check for other mesh files and append them if found:

        for filename in ["artery.vtp", "vein.vtp", "tumor.vtp"]:
            files = glob.glob( os.path.join(self.folder_preprocessed, "**", filename), recursive=True )
            for f in files:
                self.mesh_filenames.append( f )


        #############################
        # Optionally load internal structures, i.e. preoperative full meshes of vessels etc. and
        # intraoperative (segmented) ultrasound scans.

        if self.load_internals:
            internal_slices_vtk = self.load_multiple_meshes( self.folder_preprocessed, self.filenames_internal_slices )
            if not internal_slices_vtk or internal_slices_vtk.GetNumberOfPoints() == 0:
                self.suited_as_intraop = False
                print( "No partial internals found, but --no_internals not set. Not using this as intraop." )
            else:
                # center internal slices:
                if len(self.center_offset) > 0:
                    internal_slices_vtk = vtk_utils.transform_mesh(
                        mesh=internal_slices_vtk,
                        trans=center_offset,
                    )
                internal_slices = vtk_to_numpy(internal_slices_vtk.GetPoints().GetData())
                internal_slices = internal_slices * self.scale 

                if self.npoints > 0:
                    r = pc_utils.set_npoints( [internal_slices], self.npoints )
                    internal_slices = r[0]

                self.internal_slices = torch.Tensor( internal_slices ).transpose(0,1)

            internal_full_vtk = self.load_multiple_meshes( self.folder_preprocessed, self.filenames_internal_full )
            if not internal_full_vtk or internal_full_vtk.GetNumberOfPoints() == 0:
                self.suited_as_preop = False
                print( "No full internals found, but --no_internals not set. Not using this as preop." )
            else:
                # center internal full:
                if len(self.center_offset) > 0:
                    internal_full_vtk = vtk_utils.transform_mesh(
                        mesh=internal_full_vtk,
                        trans=self.center_offset,
                    )
                internal_full = vtk_to_numpy(internal_full_vtk.GetPoints().GetData())
                #self.internal_full = internal_full * self.scale 

                # Calculate the distance from each internal point to the nearest surface point:
                distance_field = vtk_utils.df( mesh=internal_full_vtk, surface=preop_surface )
                # Convert to numpy and concatenate
                internal_dists = vtk_to_numpy(distance_field)
                internal_full = np.concatenate( (internal_full, np.expand_dims(internal_dists,axis=1)), axis=1 )
                internal_full = internal_full * self.scale 

                if self.npoints > 0:
                    r = pc_utils.set_npoints( [internal_full], self.npoints )
                    internal_full = r[0]

                self.internal_full = torch.Tensor( internal_full ).transpose(0,1)



    def load_multiple_meshes( self, root_folder, filenames ):

        append = vtk.vtkAppendPolyData()
    
        loaded = 0
        for filename in filenames:
            files = glob.glob( os.path.join( root_folder, "**", filename ), recursive=True )
            for f in files:
                mesh = vtk_utils.load_mesh( f )
                if mesh.GetNumberOfPoints() > 0:
                    loaded += 1
                    append.AddInputData( mesh )

        if loaded > 0:
            append.Update()
            return append.GetOutput()

        return None


# class ScanManualIntraop:
#     def __init__(self) -> None:
#         pass


class Patient():

    def __init__( self, 
                 folder_preprocessed, 
                 folder_landmarks, 
                 filename_liversurface, 
                filename_partial_surface,  
                folder_name_partial_surface = None,
                npoints = 0, scale = 1,
                preop_folder_name = None, 
                preop_folder_idx = None,
                load_internals = True,
                pre_align_surfaces = False,
                perturb_preop_randomly = False ):

        print("======================")
        print("Loading patient from:")
        print("\tPreprocessed:", folder_preprocessed )
        print("\tLandmarks:", folder_landmarks )

        #if not "02" in folder_preprocessed:
        #    raise IOError()

        self.load_internals = load_internals
        self.pre_align_surfaces = pre_align_surfaces
        self.perturb_preop_randomly = perturb_preop_randomly

        self.all_scans = [scan for scan in os.listdir( folder_preprocessed ) \
                if "vibe_dixon" in scan.lower() \
                # or "Exh" in scan \
                # or "Inh" in scan
                or "inhale" in scan.lower() \
                or "exhale" in scan.lower()
        ]

        self.all_scans.sort()

        self.loaded_scans = []


        for i in range(len(self.all_scans)):
            scan_name = self.all_scans[i]
            print("-----------")
            print("loading scan:", scan_name)
            try:
                s = Scan( os.path.join(folder_preprocessed, scan_name),
                        os.path.join(folder_landmarks, scan_name),
                        scan_name = scan_name,
                        npoints = npoints,
                        scale = scale,
                        # filename_liversurface="aligned_liver_gt.stl", # for HHLBM
                        # filename_liversurface="aligned_liver_gt_smoothed.stl",
                        # filename_liversurface="liver.vtp",
                        filename_liversurface=filename_liversurface if filename_liversurface is not None else "aligned_liver_gt.stl" , 
                        filename_partial_surface=filename_partial_surface if filename_partial_surface is not None else "liver_focused.vtp",
                        folder_name_partial_surface=folder_name_partial_surface, #"intraop_surfaces",
                        load_internals = load_internals,
                    )

                if preop_folder_name:
                    if preop_folder_name in scan_name:
                        s.suited_as_intraop = False
                        print(f"Scan {scan_name} matches preop_folder_name {preop_folder_name}")
                    else:
                        s.suited_as_preop = False
                        print(f"Scan {scan_name} does not match preop_folder_name {preop_folder_name}")
                else:
                    if preop_folder_idx is not None:
                        if i == preop_folder_idx:
                            s.suited_as_intraop = False
                            print(f"Scan {scan_name} matches preop_folder_idx {preop_folder_idx}")
                        else:
                            s.suited_as_preop = False
                            print(f"Scan {scan_name} does not match preop_folder_idx {preop_folder_idx}")


                self.loaded_scans.append( s )

                #if len(loaded_scans) > 2:
                #    break   # DEBUG!! TODO REMOVE!

            except IOError as err:
                print(err)
            except AssertionError as err:
                print(err)
        print("loaded_scans:", len(self.loaded_scans))
        for s in self.loaded_scans:
            print("\t", "suited as preop:", s.suited_as_preop, "suited as intraop:", s.suited_as_intraop)

        # Find a scan that has the full volume segmented correctly without any pieces of the surface missing
        # Only such a scan is considered as a valid preoperative scan.
        # Other scans where pieces _are_ missing can still be considered for the intraoperative surface,
        # where we only need partial information, but not for the preoperative volume.
        self.scan_pairs = []
        for i, preop_scan in enumerate(self.loaded_scans):
            if preop_scan.suited_as_preop:
                for intraop_scan in self.loaded_scans:
                    if not intraop_scan == preop_scan:
                        if intraop_scan.suited_as_intraop:
                            self.scan_pairs.append( (preop_scan, intraop_scan) )

        if len(self.scan_pairs) == 0:
            raise IOError("No pair of valid scans found for patient! Maybe the segmentations were bad?")

        self.folder_preprocessed = folder_preprocessed
        self.folder_landmarks = folder_landmarks

    def __getitem__( self, i ):

        preop_scan, intraop_scan = self.scan_pairs[i]

        # RELOAD:
        print("reloading")
        preop_scan.load(
            center=True,
            center_offset=[]
        )
        center_offset = preop_scan.center_offset
        print("moving intraop scan by center offset:", center_offset)
        intraop_scan.load(
            center=False,
            center_offset = center_offset
        )

        num_preop_landmarks = preop_scan.landmarks.shape[1]
        num_intraop_landmarks = intraop_scan.landmarks.shape[1]

        assert num_preop_landmarks == num_intraop_landmarks, "The HHLBM dataloader assumes that scans have" +\
                " the same amount of landmark points marked on them, and that the i'th landmark in a scan " +\
                "corresponds to the i'th landmark in the other scans (for the same patient). However, in " +\
                f"folder {self.folder_landmarks} we found scans with {num_preop_landmarks} and " +\
                f"{num_intraop_landmarks} landmarks."

        print("Scan volumes:", preop_scan.estimated_volume_size, intraop_scan.estimated_volume_size)

        # Search for points which are close to 0,0,0 (invalid!) and discard them:
        preop_lengths = np.linalg.norm( preop_scan.landmarks, axis = 0 )
        intraop_lengths = np.linalg.norm( intraop_scan.landmarks, axis = 0 )
        # If EITHER the preop OR the intraop landmark is close to zero, we'll discard it
        mask = (preop_lengths > 1e-6) * (intraop_lengths > 1e-6)

        #print("_--------------------------")
        #print(preop_scan.landmarks.shape)
        #print(preop_scan.landmarks)
        #print(preop_lengths)
        #print("----")
        #print(intraop_scan.landmarks.shape)
        #print(intraop_scan.landmarks)
        #print(intraop_lengths)
        #print(mask)

        if sum(mask) != num_preop_landmarks:
            diff = num_preop_landmarks - sum(mask)
            print(f"WARNING: Discarding {diff} landmarks, because position was 0,0,0.")
            preop_landmarks = preop_scan.landmarks[:, mask]
            intraop_landmarks = intraop_scan.landmarks[:, mask]
        else:
            preop_landmarks = preop_scan.landmarks
            intraop_landmarks = intraop_scan.landmarks

        # These might get overwritten/modified, so clone!
        #preop_scan_volume = preop_scan.volume.clone()
        #preop_landmarks = preop_landmarks.clone()
        #if self.load_internals:
        #    preop_scan_internal_full = preop_scan.internal_full.clone()

        if self.pre_align_surfaces:
            # TODO: This fails if there are dummy points in the data!!!
            # A simple filtering of source_np and target_np should be enough to fix this.

            from simpleicp import SimpleICP, PointCloud

            print(preop_scan.volume.shape)
            print(intraop_scan.volume.shape)
            source_np = preop_scan.volume
            x = source_np[0, :]
            y = source_np[1, :]
            z = source_np[2, :]
            source = PointCloud( {"x":x, "y":y, "z":z} )
            target_np = intraop_scan.surface
            x = target_np[0, :]
            y = target_np[1, :]
            z = target_np[2, :]
            target = PointCloud( {"x":x, "y":y, "z":z} )


            icp = SimpleICP()
            icp.add_point_clouds( target, source )
            H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = \
                    icp.run(max_overlap_distance=1)

            H = torch.Tensor( H )

            #rot = torch.Tensor(H[:3,:3])
            #transl = torch.Tensor(H[0:3,3]).unsqueeze(1)

            # align the preop data:

            preop_scan.volume[0:3, :] = apply_transform( preop_scan.volume[0:3, :], H )
            #preop_scan_volume[0:3, :] = ((preop_scan_volume[0:3, :]+transl).T @ rot).T
            preop_landmarks = apply_transform( preop_landmarks, H )
            #preop_landmarks = preop_landmarks @ H

            if self.load_internals:
                preop_scan.internal_full[0:3, :] = apply_transform( preop_scan.internal_full[0:3, :], H )

            print("preop_scan_volume", preop_scan.volume.shape)
            print("preop_landmarks", preop_landmarks.shape)

        if self.perturb_preop_randomly:
            H = create_random_transform(
                # max_ang_rad=0.30,
                # max_translation=0.02,
                max_ang_rad=0.20,
                max_translation=0.025
            )

            # save H to file:
            output_path_H = os.path.join(preop_scan.folder_preprocessed, "pertubation_H.pkl")
            with open(output_path_H, "wb") as f:
                pkl.dump(H, f)
                f.close()
            print("Saved pertubation H to", output_path_H)

            preop_scan.volume[0:3, :] = apply_transform( preop_scan.volume[0:3, :], H )
            #preop_scan_volume[0:3, :] = ((preop_scan_volume[0:3, :]+transl).T @ rot).T
            preop_landmarks = apply_transform( preop_landmarks, H )
            #preop_landmarks = preop_landmarks @ H

            if self.load_internals:
                preop_scan.internal_full[0:3, :] = apply_transform( preop_scan.internal_full[0:3, :], H )



        # return preop_scan.volume, preop_scan.surface, preop_landmarks, preop_scan.mesh_filenames, \
        #         intraop_scan.volume, intraop_scan.surface, intraop_landmarks, intraop_scan.mesh_filenames, intraop_scan.partial_surface_list_np
        #return preop_scan.volume, preop_scan.surface, preop_landmarks, preop_scan.mesh_filenames, \
        #        intraop_scan.volume, intraop_scan.surface, intraop_landmarks, intraop_scan.mesh_filenames
        res = {
                "preop": preop_scan.volume,
                "intraop": intraop_scan.surface,
                "preop_landmarks": preop_landmarks,
                "intraop_landmarks": intraop_landmarks,
                "preop_mesh_filenames": preop_scan.mesh_filenames,
                "intraop_mesh_filenames": intraop_scan.mesh_filenames,
                "partial_surface_filenames": intraop_scan.partial_surface_filenames,
                # "partial_surface_list_np": intraop_scan.partial_surface_list_np,
                "partial_surface_area_list": intraop_scan.partial_surface_area_list,
                "partial_surface_area_list_percentage": intraop_scan.partial_surface_area_list_percentage,
                "center_offset": center_offset,
                "preop_scan_name": preop_scan.scan_name,
                "intraop_scan_name": intraop_scan.scan_name,
                "random_transform": H if self.perturb_preop_randomly else None,
        }

        if self.load_internals:
            res["preop_internal"] = preop_scan.internal_full
            res["intraop_internal"] = intraop_scan.internal_slices
        return res

        #preop_full_surface_file = os.path.join( self.folder_preprocessed, preop_scan_name, "
        #preop_volume = vtk_utils.create_random_internal_points(preop_volume, points_to_create=preop_volume.GetNumberOfPoints() * 3 )


    def get_meshes(self,):
        mesh_list = []
        for idx_scan in range(len(self.loaded_scans)):
            preop_surface = self.loaded_scans[idx_scan].original_surface
            mesh_list.append({
                "time": self.all_scans[idx_scan],
                "mesh": preop_surface,
            })
        return mesh_list


def create_random_transform( max_ang_rad=0.25, max_translation=0.01 ):

    ang = random.uniform( -max_ang_rad, max_ang_rad )
    axis = np.random.rand( 3 )
    axis = axis / np.linalg.norm( axis )

    R = scipy.spatial.transform.Rotation.from_rotvec( ang*axis ).as_matrix()
    print(R)

    t = np.random.rand(3) * max_translation*2 - max_translation

    H = np.eye(4)
    H[0:3,0:3] = R
    H[0:3,3] = t

    print("Random transformation H:")
    print(H)
    return torch.Tensor( H )

    

def apply_transform( points, H ):

    _, N = points.shape
    print("N", N)

    points = torch.concatenate( (points, torch.ones(1,N)), dim = 0 )

    #transformed = (points.T @ H).T
    transformed = H @ points

    return transformed[0:3, :]


class HHLBMDataset(Dataset):
    def __init__(self,
            data_folder,
            npoints = 2500,
            scale = 1e-3,
            # preop_folder_name = None,
            filename_liversurface = None, 
            filename_partial_surface = None,   
            preop_folder_name_list = None,
            preop_folder_idx = None,
            folder_name_partial_surface = "",
            load_internals = False,
            pre_align_surfaces = False,
            perturb_preop_randomly = False,
            ) -> None:
        """
        Args:
            preop_folder_name: If given, use only folders matching this as preoperative scan. Otherwise,
                pairwise match all folders, i.e. use each folder as pre- and as intra-operative.
            pre_align_surfaces: Use ICP to align volumes before passing data on
        """
        super().__init__()
        reset_seed(1)
        self.data_folder = data_folder
        self.data_folder_preprocessed = os.path.join( self.data_folder, "Preprocessed" )
        self.data_folder_landmarks = os.path.join( self.data_folder, "Landmarks" )

        self.pre_align_surfaces = pre_align_surfaces

        self.patients = []
        self.index2patientfolder_map = {}

        patient_id = 0
        pair_id = 0

        for f in sorted(os.listdir( self.data_folder_preprocessed) ):
            folder_suffix = int(f.split("_")[-1]) if f.split("_")[-1].isdigit() else -1
            full_path_p = os.path.join( self.data_folder_preprocessed, f )
            full_path_l = os.path.join( self.data_folder_landmarks, f )
            if os.path.isdir( full_path_p ) and os.path.isdir( full_path_l ):
                if ("HHLBM_" in f and not "_00" in f and folder_suffix <= 12 and folder_suffix >= 0 and f in preop_folder_name_list.keys() ) or "Phantom" in f or "Pat" in f:
                    try:
                        p = Patient( full_path_p, full_path_l, 
                            npoints = npoints, 
                            scale=scale,
                            filename_liversurface=filename_liversurface, 
                            filename_partial_surface=filename_partial_surface,  
                            preop_folder_name = preop_folder_name_list[f] if preop_folder_name_list is not None else None, 
                            preop_folder_idx=preop_folder_idx,
                            load_internals = load_internals, 
                            folder_name_partial_surface=folder_name_partial_surface,
                            pre_align_surfaces = pre_align_surfaces,
                            perturb_preop_randomly = perturb_preop_randomly,
                        )
                        self.patients.append( p )
                        for i in range( len(p.scan_pairs ) ):
                            self.index2patientfolder_map[pair_id + i] = ( patient_id, i )
                        pair_id += len(p.scan_pairs)
                        patient_id += 1
                    except IOError as err:
                        print(f"Could not load patient {f}. Error: {err}")


        self.num_scan_pairs = sum( [len(p.scan_pairs) for p in self.patients] )

        #for k, v in self.index2patientfolder_map.items():
        #    print(k,v)
        print(f"Found {len(self.patients)} patients and {self.num_scan_pairs} preop<->intraop pairs")

    def __len__(self):
        return self.num_scan_pairs

    #def valid_index( self, index ):
    #    while index in self.images_with_bad_shape:
    #        index += 1
    #        if index == self.num_pairs:
    #            index = 0
    #    return index
    
    def __getitem__(self, index):

        #index = self.valid_index( index )
        patient_id, pair_id = self.index2patientfolder_map[index]

        data = self.patients[patient_id][pair_id]

        return data

    
    def get_num_patients(self):
        return len(self.patients)

    def get_meshes(self, index):
        # get meshes of all scans of a patient
        patient = self.patients[index]
        return patient.get_meshes()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/home/liupeng/ceph_home/data/HHLBM/test_various_visible_amounts/")
    args = parser.parse_args()

    dataset = HHLBMDataset( 
        args.data_folder,
         folder_name_partial_surface="visible_surfaces",
    )
    print(len(dataset))
    # for i in range(len(dataset)):
    for i in range(1):
        print("----------loadng sample", i)
        print(len(dataset[i]),  dataset[i].keys()) #d[i][2].shape, d[i][3], d[i][4].shape, d[i][5].shape, d[i][6].shape, d[i][7], d[i][8].shape)
        # print("type(dataset[i]):", type(dataset[i]))
        # print("len(dataset[i][8]):", len(dataset[i][8]))
    #     # print("
        # print(len(dataset[i][0])) # preop scan
        # print(type(dataset[i][0]))
        # print(dataset[i][0][0].shape) # preop volume
        # print(dataset[i][0][1].shape) # preop surface

        # print(len(dataset[i][1])) # intraop scan
        # print(dataset[i][1][0].shape) # intraop volume
        # print()
        # print(len(dataset[i]["partial_surface_list_np"]))
