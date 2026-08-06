import os
import vtk
import random
import math
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtk import *
import numpy as np
from vtk.util import numpy_support
import torch

class ErrorObserver:

   def __init__(self):
       self.__ErrorOccurred = False
       self.__ErrorMessage = None
       self.CallDataType = 'string0'

   def __call__(self, obj, event, message):
       self.__ErrorOccurred = True
       self.__ErrorMessage = message
       raise IOError( message )

   def ErrorOccurred(self):
       occ = self.__ErrorOccurred
       self.__ErrorOccurred = False
       return occ

   def ErrorMessage(self):
       return self.__ErrorMessage


def load_mesh(filename):
    """
    Loads a mesh using VTK. Supported file types: stl, ply, obj, vtk, vtu, vtp, pcd.

    Arguments:
    ---------
    filename (str)

    Returns:
    --------
    vtkDataSet
            which is a vtkUnstructuredGrid or vtkPolyData, depending on the file type of the mesh.
    """

    # Load the input mesh:
    fileType = filename[-4:].lower()
    if fileType == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".vtk": #.vtk can have different types of data
        # reader = vtk.vtkUnstructuredGridReader()
        # reader = vtk.vtkXMLUnstructuredGridReader()
        # # reader = vtk.vtkPolyDataReader()
        # reader.SetFileName(filename)
        # reader.Update()
        # mesh = reader.GetOutput()
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh = reader.GetOutput()
    elif fileType == ".pcd":
        try:
            import pcl
            pc = pcl.load(filename)
        except:
            import open3d as o3d
            pc = o3d.io.read_point_cloud(filename)
            pc = np.asarray(pc.points)
        pts = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        for i in range(pc.shape[0]):
            pts.InsertNextPoint(pc[i][0], pc[i][1], pc[i][2])
            verts.InsertNextCell(1, (i,))
        mesh = vtk.vtkPolyData()
        mesh.SetPoints(pts)
        mesh.SetVerts(verts)

    else:
        raise IOError(
            "Mesh should be .vtk, .vtu, .vtp, .obj, .stl, .ply or .pcd file!")

    if mesh.GetNumberOfPoints() == 0:
        raise IOError("Could not load a valid mesh from {}".format(filename))
    return mesh


def load_structured_grid(filename):
    """
    Loads a structured grid using VTK. Supported file types: vtk, vts.

    Arguments:
    ---------
    filename (str)

    Returns:
    --------
    vtkDataSet
            of type vtkStructuredGrid.
    """
    fileType = filename[-4:].lower()
    if fileType == ".vtk":
        reader = vtkStructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
    elif fileType == ".vts":
        reader = vtkXMLStructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        grid = reader.GetOutput()
    else:
        raise IOError(filename + " should be .vtk or .vts")
    return grid


def write_mesh(mesh, filename, verbose=False):
    """
    Saves a VTK mesh to file. 
    Supported file types: stl, ply, obj, vtk, vtu, vtp, pcd, vts.

    Arguments:
    ---------
    mesh (vtkDataSet):
            mesh to save
    filename (str): 
            name of the file where to save the input mesh. MUST contain the desired extension.

    """
    if verbose:
        print("writing mesh ({}) to {}".format(mesh.GetNumberOfPoints(), filename))
    if mesh.GetNumberOfPoints() == 0:
        raise IOError("Input mesh has no points!")

    # Get file format
    fileType = filename[-4:].lower()
    if fileType == ".stl":
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".obj":
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".ply":
        writer = vtk.vtkPLYWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vtk":
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vtu":
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vts":
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    elif fileType == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(mesh)
        writer.Update()
    else:
        raise IOError(
            "Supported extensions are .vtk, .vtu, .vts, .vtp, .obj, .stl, .ply!")



def remove_duplicates( mesh, clean_polys = True ):
# Ensure no duplicated points:
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData( mesh )
    cleaner.SetTolerance( 0.0001 )       # Fraction of the boundary box
    cleaner.ToleranceIsAbsoluteOff()
    cleaner.PointMergingOn()
    cleaner.Update()

    if clean_polys:
        # Ensure no duplicated polys:
        cleaner2 = vtk.vtkRemoveDuplicatePolys()
        cleaner2.SetInputData( cleaner.GetOutput() )
        cleaner2.Update()

        clean_mesh = cleaner2.GetOutput()
    else:
        clean_mesh = cleaner.GetOutput()

    return clean_mesh



def extract_surface(
    mesh: vtk.vtkDataSet,
) -> vtk.vtkDataSet:
    """  
    Given an input mesh, extracts its outer surface.

    Args:
        mesh: The vtkDataSet where we want to retain the external surface only.
    
    Returns:
        A vtkDataSet with surface elements only.
    """
    # WARNING: Both vtkDataSetSurfaceFilter and vtkGeometryFilter may produce non-manifold geometry.
    # I have not found a consistent behaviour, and I think it may actually have changed between vtk versions (my
    # code worked with python3.8 but not with python3.10...!)
    # As a workaround, I now use a vtkCleanPolyData Filter to remove duplicated vertices and a 
    # vtkRemoveDuplicatePolys filter to remove duplicated polygons.
    #surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter = vtk.vtkGeometryFilter()
    surface_filter.SetInputData(mesh)
    surface_filter.SetMerging( True )
    surface_filter.Update()
    surface = surface_filter.GetOutput()

    # Ensure no duplicated points:
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData( surface )
    cleaner.SetTolerance( 0.0001 )       # Fraction of the boundary box
    cleaner.ToleranceIsAbsoluteOff()
    cleaner.PointMergingOn()
    cleaner.Update()

    # Ensure no duplicated polys:
    cleaner2 = vtk.vtkRemoveDuplicatePolys()
    cleaner2.SetInputData( cleaner.GetOutput() )
    cleaner2.Update()

    clean_surface = cleaner2.GetOutput()

    #normals = compute_point_normals( clean_surface ) 
    #clean_surface.GetPointData().SetNormals( normals )

    return clean_surface

def split_surface_and_internal_points( 
    mesh: vtk.vtkDataSet
    ) -> (vtk.vtkPolyData, vtk.vtkPolyData):

    #print("INPUT MESH")
    #print(mesh)

 #   writer = vtkXMLUnstructuredGridWriter()
 #   writer.SetFileName( "/tmp/mesh1.vtu" )
 #   writer.SetInputData( mesh )
 #   writer.Update()
 #
    surface_mesh = extract_surface( mesh )

    #print("INPUT EXTRACT MESH")
    #print(mesh)
    #print("INPUT EXTRACT SURFACE_MESH")
    #print(surface_mesh)

    #assert surface_mesh.IS
    assert vtk.vtkSelectEnclosedPoints.IsSurfaceClosed(surface_mesh)

    extract = vtk.vtkExtractEnclosedPoints()
    extract.SetSurfaceData( surface_mesh );
    extract.SetInputData( mesh );
    extract.SetTolerance(.0001);
    extract.CheckSurfaceOn();

    extract.Update()

    # DEBUG:
    #writer = vtkXMLUnstructuredGridWriter()
    #writer.SetFileName( "/home/pfeiffemi/mesh.vtu" )
    #writer.SetInputData( mesh )
    #writer.Update()
    #writer = vtkOBJWriter()
    #writer.SetFileName( "/home/pfeiffemi/surface.obj" )
    #writer.SetInputData( surface_mesh )
    #writer.Update()
    #writer = vtkXMLPolyDataWriter()
    #writer.SetFileName( "/home/pfeiffemi/internal.vtp" )
    #writer.SetInputData( extract.GetOutput() )
    #writer.Update()

    internal_points = extract.GetOutput()

    if internal_points.GetNumberOfPoints() == 0:
        msg = "ERROR: The result of splitting the mesh into surface and internal points might be wrong." +\
                " This often happens when the surface contains non-manifolds."
        raise IOError( msg )

    return surface_mesh, internal_points

def compute_point_normals( surface: vtk.vtkPolyData, remove_pre_normals = False):
    if remove_pre_normals and surface.GetPointData().HasArray("Normals"):
        surface.GetPointData().RemoveArray("Normals")
    f = vtk.vtkPolyDataNormals()
    f.SetComputePointNormals( True )
    f.SetComputeCellNormals( False )
    f.SetAutoOrientNormals( True ) ## IMPORTANT! otherwise the normal vectors might point inwards
    # Do not split sharp angles:
    f.SetSplitting( False )

    f.SetInputData( surface )
    f.Update()
    # print("f.GetOutputPointsPrecision()", f.GetOutputPointsPrecision())

    normals = f.GetOutput().GetPointData().GetNormals()
    # print(f.GetOutput().GetPointData().HasArray("Normals"))
    # print(normals.GetNumberOfTuples())

    assert normals.GetNumberOfTuples() == surface.GetNumberOfPoints()

    return f.GetOutput().GetPointData().GetNormals()



def df( mesh, surface=None, return_idx=False ):
    """ Given a (volume) mesh, for each point calc the distance to the surface.

    Creates a new array 'df' (distance field) with N entries, where each entry
    is the distance to the surface from the corresponding point in mesh.

    Args:
        mesh: vtkDataSet for which to calculate the distances.
        surface: the vtkPolyData representing the surface. If None, will extract
            the surface from mesh instead.
    """

    # print("df1")
    if not surface:
        surface = extract_surface( mesh )
    # print("df2")

    dists = vtk.vtkFloatArray()
    dists.SetNumberOfTuples( mesh.GetNumberOfPoints() )
    dists.SetNumberOfComponents(1)
    dists.SetName( "df" )
    # print("df3")

    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet( surface )
    # print("df4")
    assert mesh.GetNumberOfPoints() > 0
    # print("df5")
    mesh_points = mesh.GetPoints()
    # print("df6")
    closest_id_list = []
    for i in range(mesh_points.GetNumberOfPoints()):
        mesh_point = mesh_points.GetPoint( i )
        closest_id = locator.FindClosestPoint( mesh_point )
        closest_id_list.append( closest_id )
        closest_surface_point = surface.GetPoints().GetPoint( closest_id )
        dist = math.sqrt( (mesh_point[0]-closest_surface_point[0])**2 +
            (mesh_point[1]-closest_surface_point[1])**2 +
            (mesh_point[2]-closest_surface_point[2])**2 )

        dists.SetTuple1( i, dist )
    # print("df7")

    if return_idx:
        dists, closest_id_list
    else:
        return dists



def create_random_internal_points( surface_mesh, points_to_create=100, append_surface=True ):
    """ Given a (closed) surface mesh, create random points which lie inside the mesh """

    ## Get the bounds of the orginal object:
    bounds = [0]*6;
    surface_mesh.GetBounds(bounds);
    x0 = bounds[0]
    x1 = bounds[1]
    y0 = bounds[2]
    y1 = bounds[3]
    z0 = bounds[4]
    z1 = bounds[5]

    poly_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    poly_data.SetPoints(points)
    points.SetNumberOfPoints(points_to_create)
    for i in range(points_to_create):
        x = random.random()*(x1-x0)+x0
        y = random.random()*(y1-y0)+y0
        z = random.random()*(z1-z0)+z0
        points.SetPoint(i, x,y,z)

    extract = vtk.vtkExtractEnclosedPoints()
    extract.SetSurfaceData(surface_mesh);
    extract.SetInputData(poly_data);
    extract.SetTolerance(.0001);
    extract.CheckSurfaceOn();

    extract.Update()
    #print( poly_data.GetNumberOfPoints(), "->", extract.GetOutput().GetNumberOfPoints() )

    if append_surface:
        combined = combine_datasets( surface_mesh, extract.GetOutput() )
        
        # Return the combined internal points and surface points together:
        return combined
    else:
        # Return only the internal points:
        return extract.GetOutput()

def combine_datasets( a: vtk.vtkPolyData, b: vtk.vtkPolyData ):

    append = vtk.vtkAppendPolyData()
    append.AddInputData( a )
    append.AddInputData( b )
    append.Update()

    return append.GetOutput()


def combine_meshes_list(mesh_list):
    append = vtk.vtkAppendPolyData()
    for mesh in mesh_list:
        append.AddInputData(mesh)
    append.Update()
    return append.GetOutput()


def to_pointcloud(coords, features=None, features_name="features"):
    # if coords [N, 3] is tensor, convert to vtk
    if isinstance(coords, torch.Tensor):
        coords = numpy_support.numpy_to_vtk(coords.cpu().numpy())
    # if coords [N, 3], is numpy, convert to vtk
    if isinstance(coords, np.ndarray):
        coords = numpy_support.numpy_to_vtk(coords)

    # Create a new vtk point data from the given data:
    points = vtk.vtkPoints()
    points.SetData(coords)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)

    #for i in range(pd.GetNumberOfPoints()):
    #pd.InsertNextCell( vtk.VTK_VERTEX, i )

    verts = vtk.vtkCellArray()
    for i in range(pd.GetNumberOfPoints()):
        verts.InsertNextCell( vtk.VTK_VERTEX, (i,) )
    pd.SetVerts( verts )

    #p = os.path.join(self.path, "preop_surface_w_displacement.vtu")
    #preop_mesh = DataSample.load_vtu(p)

    if features is not None:
        # features = (features/self.scale).permute(1,0).cpu().numpy()
        if isinstance(features, list):
            for i, f in enumerate(features):
                print(features_name[i], f.shape)
                arr = numpy_to_vtk(f)
                arr.SetNumberOfComponents(f.shape[-1])
                arr.SetName(features_name[i])
                pd.GetPointData().AddArray(arr)
        else:
            arr = numpy_to_vtk(features)
            arr.SetNumberOfComponents(features.shape[-1])
            arr.SetName(features_name)

            pd.GetPointData().AddArray(arr)
    return pd


def closest_vetices(source, target, discard_duplicate=True, subset=None):
    """Find index list of the closest vertices on the target mesh

    Args:
        source (vtk.mesh): the source vtk mesh
        target (vtk.mesh): the target vtk mesh

    Returns:
        edict(): return a list of indices of the closest vertices on the target mesh
    """
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(target )
    locator.SetNumberOfPointsPerBucket(1)
    locator.BuildLocator()

    closest_point_idx_list = []
    if subset is None:
        subset = range(source.GetNumberOfPoints())

    for idx in subset:
        vert = source.GetPoint(idx)
        closest_point_idx = locator.FindClosestPoint(vert)
        # If we want to discard duplicate indices and we have already found it, skip append
        if discard_duplicate and (closest_point_idx in closest_point_idx_list):
            pass
        else:
            closest_point_idx_list.append(closest_point_idx)
    return closest_point_idx_list


def poly_to_unstructured_grid(poly):
    appendFilter = vtk.vtkAppendFilter()
    appendFilter.SetInputData(poly)
    appendFilter.Update()
    return appendFilter.GetOutput()


def unstructured_grid_to_poly(ugrid):
    surfaceFilter = vtk.vtkDataSetSurfaceFilter()
    surfaceFilter.SetInputData(ugrid)
    surfaceFilter.Update()
    return surfaceFilter.GetOutput()


def calc_geodesic_distance(
    mesh: vtkDataSet, 
    node_id: int,
) ->vtkDoubleArray:
    """
    Computes the geodesic distance of each point in the input mesh with respect to the 
    point with index node_id.

    Args:
        mesh: The vtkDataSet object. 
        node_id: The ID of the node for which neighbors are to be found.

    Returns:
        A vtkDoubleArray named "geodesic_distance" containing the geodesic distance 
        of each point in the mesh to the point node_id.
    """

    # pre-compute cell neighbors:
    neighbors = {}
    for i in range(mesh.GetNumberOfPoints()):
        neighbors[i] = get_connected_vertices(mesh, i)

    distance = vtkDoubleArray()
    distance.SetNumberOfTuples(mesh.GetNumberOfPoints())
    distance.SetNumberOfComponents(1)
    distance.Fill(1e10)   # initialize with large numbers
    distance.SetName("geodesic_distance")

    front = [node_id]
    distance.SetTuple1(node_id, 0)

    while len(front) > 0:

        cur_id = front.pop(0)
        cur_pt = mesh.GetPoint(cur_id)
        cur_dist = distance.GetTuple1(cur_id)
        cur_neighbors = neighbors[cur_id]

        # Go through all neighboring points. Check if the distance in those points
        # is still up to date or whether there is a shorter path to them:
        for n_id in cur_neighbors:

            # Find distance between this neighbour and the current point:
            n_pt = mesh.GetPoint(n_id)
            dist = math.sqrt(vtkMath.Distance2BetweenPoints(n_pt, cur_pt))

            new_dist = dist + cur_dist
            if new_dist < distance.GetTuple1(n_id):
                distance.SetTuple1(n_id, new_dist)
                if not n_id in front:
                    # This neighbor node needs to be checked again!
                    front.append(n_id)

    return distance



def point_cloud_to_poly(point_coords):
    a = np.asarray(point_coords)
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()

    for p in a:
        id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)

    # Create a polydata object
    poly = vtk.vtkPolyData()

    # Set the points and vertices we created as the geometry and topology of the polydata
    poly.SetPoints(points)
    poly.SetVerts(vertices)

    return poly



def create_poly_using_points_and_faces(coords, faces):
    """
    coords: [N_points, 3]
    faces: [N_cells, N_points_per_cell], usually N_points_per_cell is 3, i.e. a cell is a triangle
    """
    poly = vtk.vtkPolyData()
    
    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(coords))        
    poly.SetPoints(points)

    # print("mesh", mesh.GetNumberOfPoints())
    cellsArray = vtk.vtkCellArray()
    for c in enumerate(np.transpose(faces, axes=(1, 0)).astype(np.int64).T):
        cellsArray.InsertNextCell( 3, c[1] )
    poly.SetPolys(cellsArray)
    
    return poly


def calc_surface_area(
    mesh: vtkDataSet
) -> float:
    """
    Computes the surface area of the input mesh and adds it to the mesh as a single entry
    field data float array "surfaceArea".

    The surface area is computed by summing up the areas of all the triangular elements
    of the mesh, done by vtk internally.

    Args:
        mesh: A vtkDataSet object.
    
    Returns:
        The computed surface area.
    """
    computation_filter = vtkIntegrateAttributes()
    computation_filter.SetInputData(mesh)
    computation_filter.Update()
    result_mesh = computation_filter.GetOutputDataObject(0)
    result = result_mesh.GetCellData().GetArray("Area")

    # 2D input data prompt filter to compute area
    if result is not None:
        val = result.GetValue(0)
    # 3D input data prompt filter to compute volume: try again with extracted surface
    elif result_mesh.GetCellData().GetArray("Volume") is not None:
        mesh_surface = extract_surface(mesh)
        computation_filter.SetInputData(mesh_surface)
        computation_filter.Update()
        result_mesh = computation_filter.GetOutputDataObject(0)
        result = result_mesh.GetCellData().GetArray("Area")
        if result is not None:
            val = result.GetValue(0)
        else:
            print(f"Warning: 3D type passed to area computation, but calculation for surface extracted from it with" +
                  f"vtkutils.extract_surface() failed. Returning 0.")
            val = 0.0
    # fail
    else:
        print(f"Warning: non-2D type {type(mesh)} passed to area computation. Returning 0.")
        val = 0.0

    # add to the mesh
    area_arr = make_single_float_array(val, "surfaceArea")
    mesh.GetFieldData().AddArray(area_arr)
    return val

def make_single_float_array(
    value: float,
    name: str = "array", 
) ->vtkFloatArray:
    """  
    Converts a float into a vtkFloatArray.

    Args:
        value: Value associated to the array.
        name: Name that will be associated to the created vtk array.

    Returns:
        A vtkFloatArray filled with the specified value and with the specified name.
    """
    arr = vtkFloatArray()
    arr.SetNumberOfTuples(1)
    arr.SetNumberOfComponents(1)
    arr.SetTuple1(0, value)
    arr.SetName(name)
    return arr



def get_connected_vertices(
    mesh: vtkDataSet, 
    node_id: int,
) ->list:
    """
    Find the neighbor vertices of the node node_id in the provided mesh.

    Args:
        mesh: The vtkDataSet object.
        node_id: The ID of the node for which neighbors are to be found.

    Returns:
        The IDs of the vertices that are neighbors of node_id.
    """
    connected_vertices = []

    # get all cells that vertex 'id' is a part of
    cell_id_list = vtkIdList()
    mesh.GetPointCells(node_id, cell_id_list)

    for i in range(cell_id_list.GetNumberOfIds()):
        c = mesh.GetCell(cell_id_list.GetId(i))
        point_id_list = vtkIdList()
        mesh.GetCellPoints(cell_id_list.GetId(i), point_id_list)
        for j in range(point_id_list.GetNumberOfIds()):
            neighbor_id = point_id_list.GetId(j)
            if neighbor_id != node_id:
                if not neighbor_id in connected_vertices:
                    connected_vertices.append(neighbor_id)
    return connected_vertices





def generate_point_normals(
    mesh: vtkPolyData,
) ->vtkPolyData:
    """  
    Generates point normals for the input mesh.

    Args:
        mesh: A vtkPolyData object. Please be aware that the input must be a polydata in order 
        to calculate point normals.

    Returns:
        The same vtkPolyData provided as input, with the additional "Normals" array.
    """

    # Check if the point normals already exist
    if mesh.GetPointData().HasArray("Normals"):
        return mesh

    # If no normals were found, generate them:
    normal_gen = vtkPolyDataNormals()
    normal_gen.SetInputData(mesh)
    normal_gen.ComputePointNormalsOn()
    normal_gen.ComputeCellNormalsOff()
    # normal_gen.ComputeCellNormalsOn()
    normal_gen.AutoOrientNormalsOn()
    normal_gen.SplittingOff()        # Don't allow generator to add points at sharp edges
    normal_gen.Update()
    return normal_gen.GetOutput()


def interpolate_deformation( mesh, displacement_mesh, displacement_array_name, radius = 0.1, sharpness = 10 ):

    interpolator = vtkPointInterpolator()
    gaussian_kernel = vtkGaussianKernel()
    gaussian_kernel.SetRadius( radius )
    gaussian_kernel.SetSharpness( sharpness )
    interpolator.SetSourceData( displacement_mesh )
    interpolator.SetKernel( gaussian_kernel )
    # If a point falls outside the mesh, we can't "interpolate" the displacement field at this point.
    # The NullPointsStrategy tells vtk how to handle this case:
    interpolator.SetNullPointsStrategy(vtkPointInterpolator.CLOSEST_POINT)

    interpolator.SetInputData( mesh )
    interpolator.Update()
    interpolated = interpolator.GetOutput()

    return interpolated


# spread arrays from voxel grid to mesh:
# eg. spreading estimated displacement field back to mesh in order to examine the deformation in paraview
# eg. spreading displacement error back to mesh in order to extract displacement hints based on displacement field error
def interpolate_deformation_from_voxelgrid(
        mesh, 
        field, 
        output_dir=None, 
        # tf_flag=False, 
        scale = 1, # default
        output_filename=None,
        preop_surface_name="preoperativeSurface",
):
    
    # if tf_flag:
    #     try:
    #         tf = loadTransformationMatrix( field )
    #         tf.Inverse()
    #         # print("Applying transform")
    #         tfFilter = vtkTransformFilter()
    #         tfFilter.SetTransform( tf )
    #         tfFilter.SetInputData( field )
    #         tfFilter.Update()
    #         field = tfFilter.GetOutput()
            
    #         # Apply transformation also to all vector fields:
    #         applyTransformation( field, tf )
            
    #         scale = tf.GetMatrix().GetElement(0,0)

    #     except Exception as e:
    #         print(e)
    #         print("Could not find or apply transformation. Skipping.")
    try:
        # writer = vtkXMLStructuredGridWriter()
        # writer.SetInputData( field )
        # writer.SetFileName( os.path.join( output_dir, "field.vts" ) )
        # writer.Update()
        # print("Written1")

        # Threshold to ignore all points outside of field:
        threshold = vtkThreshold()
        threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, preop_surface_name)
        threshold.ThresholdByLower(0)
        threshold.SetInputData( field )
        threshold.Update()
        fieldInternal = threshold.GetOutput()
        
        # if fieldInternal.GetNumberOfPoints() == 0:
        #     print("\033[93m field internal point number is zero, recording and go next....\033[0m")
        #     with open("/home/liupeng/1_Code/springsimulation_newpipeline/HintsExtraction/list_voxel_without_preop.txt", "a+") as f:
        #         without_preop_list = f.readlines()
        #         if not "{}\n".format(os.path.basename(output_dir)) in without_preop_list:
        #             f.write("{}\n".format(os.path.basename(output_dir)))
        #         f.close()
        #     return None

        # print("Scale", scale)

        if mesh.GetPointData().HasArray("estimatedDisplacement"):
            # print("found former estimatedDisplacement field, removing...")
            mesh.GetPointData().RemoveArray("estimatedDisplacement")

        kernel = vtkGaussianKernel()
        kernel.SetRadius(0.01*scale) 
        kernel.SetKernelFootprintToRadius()
        #kernel.SetKernelFootprintToNClosest()
        #kernel.SetNumberOfPoints( 4 )

        interpolator = vtkPointInterpolator()
        interpolator.SetKernel( kernel )
        interpolator.SetNullPointsStrategyToMaskPoints()
        interpolator.SetValidPointsMaskArrayName( "validInternalPoints" )
        #interpolator.SetNullPointsStrategyToClosestPoint()
        interpolator.SetSourceData( fieldInternal )
        interpolator.SetInputData( mesh )
        interpolator.Update()
        output = interpolator.GetOutput()

        # writer = vtkXMLUnstructuredGridWriter()
        # writer.SetInputData( fieldInternal )
        # writer.SetFileName( os.path.join( output_dir, "fieldInternal.vtu" ) )
        # writer.Update()
        # print(333)

        append = vtkAppendFilter()
        append.AddInputData( output )
        append.Update()
        output = append.GetOutput()

        # if save == True:
        # if not output_filename:
        #     output_filename = "initSurface_hints_withDispl.vtu"
        # else:
        #     output_filename = output_filename
        if output_filename and output_dir:
            print("write mesh:", os.path.join( output_dir, output_filename ))
            write_mesh(output, os.path.join( output_dir, output_filename ))
        # writer = vtkXMLUnstructuredGridWriter()
        # writer.SetInputData( output )

        # writer.SetFileName( os.path.join( output_dir, output_filename ) )
        # writer.Update()

        return output
    except Exception as e:
        raise e




def scale_model(model, scale_matrix=(1,1,1), store_transform=False):

    transform = vtk.vtkTransform()
    transform.Scale(scale_matrix)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(model)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    model_scaled = transformFilter.GetOutput()
    print("model scaled by matrix:", scale_matrix)
    if store_transform:
        storeTransformationMatrix(grid=model, tf=transform)
    return model_scaled


def center_mesh(mesh):
    tf = vtk.vtkTransform()
    bounds = [0]*6
    mesh.GetBounds(bounds)
    dx = -(bounds[1]+bounds[0])*0.5
    dy = -(bounds[3]+bounds[2])*0.5
    dz = -(bounds[5]+bounds[4])*0.5
    offset = ( dx,dy,dz )
    # print("Moving point cloud by:", offset )
    tf.Translate( offset )

    tfFilter = vtk.vtkTransformFilter()
    tfFilter.SetTransform( tf )
    tfFilter.SetInputData( mesh )
    tfFilter.Update()
    mesh = tfFilter.GetOutput()
    return mesh, offset


def transform_mesh(mesh, trans):
    # trans: tuple (dx, dy, dz)
    tf = vtk.vtkTransform()
    tf.Translate(trans)

    tfFilter = vtk.vtkTransformFilter()
    tfFilter.SetTransform( tf )
    tfFilter.SetInputData( mesh )
    tfFilter.Update()
    mesh = tfFilter.GetOutput()
    return mesh



def compute_curvature(mesh):

    curvature_filter = vtk.vtkCurvatures()
    curvature_filter.SetInputData(mesh)
    curvature_filter.SetCurvatureTypeToGaussian()  
    curvature_filter.Update()


    curvature_data = curvature_filter.GetOutput().GetPointData().GetScalars()
    # somehow I need to inverse it... double check this later
    gaussian_curvatures = np.array([ - curvature_data.GetValue(i) for i in range(curvature_data.GetNumberOfTuples())])
    
    gaussian_curvatures = (gaussian_curvatures - gaussian_curvatures.min()) / (gaussian_curvatures.max() - gaussian_curvatures.min())

    gaussian_curvature_array = vtk.vtkDoubleArray()
    gaussian_curvature_array.SetName("Curvature")
    for i in range(len(gaussian_curvatures)):
        gaussian_curvature_array.InsertNextValue(gaussian_curvatures[i])
    mesh.GetPointData().AddArray(gaussian_curvature_array)

    return gaussian_curvatures, mesh



def apply_deformation( mesh, displacement_mesh, displacement_array_name, radius = 0.1, sharpness = 10 ):
    if mesh.GetPointData().HasArray( displacement_array_name ):
        mesh.GetPointData().RemoveArray( displacement_array_name )
    interpolated = interpolate_deformation( mesh, displacement_mesh, displacement_array_name, radius, sharpness )

    # Apply the displacement:
    warp = vtkWarpVector()
    warp.SetInputData( interpolated )
    warp.SetScaleFactor( 1 )

    interpolated.GetPointData().SetActiveVectors( displacement_array_name )
    warp.Update()
    displaced = warp.GetOutput()

    return displaced

def compute_landmark_error( source_points, target_points ):
    # Given two VTK point clouds, calculate the TRE between them. Assumes that the landmarks correspond.

    assert source_points.GetNumberOfPoints() == target_points.GetNumberOfPoints()
    assert source_points.GetNumberOfPoints() > 0

    dists = []
    for i in range(source_points.GetNumberOfPoints()):
        s = source_points.GetPoint(i)
        t = target_points.GetPoint(i)

        dist = math.sqrt( (s[0]-t[0])**2 +
            (s[1]-t[1])**2 +
            (s[2]-t[2])**2 )

        dists.append( dist )

    avg_dist = sum(dists)/len(dists)
    return avg_dist, dists
        

def copy_normals( full_source_mesh, partial_target_mesh ):
    """ Copy the normals that exist on source_mesh to target_mesh.

    Note: This expects partial_target_mesh to be a subset of full_source_mesh!
    """

    locator = vtk.vtkPointLocator()
    locator.SetDataSet( full_source_mesh )
    locator.BuildLocator()
    target_normals = vtk.vtkFloatArray()
    target_normals.SetNumberOfComponents( 3 )
    target_normals.SetNumberOfTuples( partial_target_mesh.GetNumberOfPoints() )
    target_normals.SetName( "Normals" ) 
    source_normals = full_source_mesh.GetPointData().GetNormals()
    for i in range( partial_target_mesh.GetNumberOfPoints() ):
        closest_id = locator.FindClosestPoint( partial_target_mesh.GetPoints().GetPoint(i) )
        n = source_normals.GetTuple( closest_id )

        target_normals.SetTuple( i, n )

    partial_target_mesh.GetPointData().SetNormals( target_normals )


def copy_curvature( full_source_mesh, partial_target_mesh ):
    """ Copy the curvature that exist on source_mesh to target_mesh.

    Note: This expects partial_target_mesh to be a subset of full_source_mesh!
    """

    locator = vtk.vtkPointLocator()
    locator.SetDataSet( full_source_mesh )
    locator.BuildLocator()
    target_curvature = vtk.vtkFloatArray()
    target_curvature.SetNumberOfComponents( 1 )
    target_curvature.SetNumberOfTuples( partial_target_mesh.GetNumberOfPoints() )
    target_curvature.SetName( "Curvature" ) 
    source_curvature = full_source_mesh.GetPointData().GetArray( "Curvature" )
    for i in range( partial_target_mesh.GetNumberOfPoints() ):
        closest_id = locator.FindClosestPoint( partial_target_mesh.GetPoints().GetPoint(i) )
        c = source_curvature.GetTuple1( closest_id )

        target_curvature.SetTuple1( i, c )

    partial_target_mesh.GetPointData().AddArray( target_curvature )
    return partial_target_mesh


def estimate_volume_poly( surface_mesh ):

    f = vtk.vtkMassProperties()
    f.SetInputData( surface_mesh )
    vol = f.GetVolume()
    return vol


def estimate_volume_unstructured_grid(
        mesh: vtkUnstructuredGrid
) -> float:
    """
    Computes the volume of the input mesh using the vtkIntegrateAttributes filter.

    The volume is computed by summing up the volumes of all 3D elements of the mesh. The used filter does
    not compute enclosed volumes. Therefore, if there are no volumetric elements the result will be invalid
    (the filter will compute the surface area or line length unsolicited).

    Args:
        mesh: A vtkDataSet object with volumetric elements.

    Returns:
        The computed volume. 0.0 if no volumetric elements can be found in the input.
    """
    computation_filter = vtkIntegrateAttributes()
    computation_filter.SetInputData(mesh)
    computation_filter.Update()
    # from https://gitlab.kitware.com/vtk/vtk/-/blob/v9.2.0/Filters/Parallel/Testing/Python/TestIntegrateAttributes.py
    result = computation_filter.GetOutputDataObject(0)
    val = result.GetCellData().GetArray("Volume")
    if val is not None:
        return val.GetValue(0)
    else:
        print(f"Warning: non-volumetric type {type(mesh)} passed to volume computation. Returning 0.")
        return 0.0



def calc_registration_error_interpolated(
        preop_volume,
        intraop_volume,
        preop_array,
        displ_array,
        # radius,
        # sharpness,
        displacement_gt_array_name = "displacement",
        displacement_predicted_array_name = "displacement_predicted",
        displacemment_error_array_name = "displacement_error"
):
    """Interpolate displacement field to the preop_volume, then deform it, calculate the error with intraop_volume

    """

    # print("displ_array", displ_array.shape)
    # print("preop_array", preop_array.shape)

    assert preop_volume.GetNumberOfPoints() == intraop_volume.GetNumberOfPoints()
    assert preop_array.shape == displ_array.shape
    if preop_volume.GetPointData().HasArray( displacement_predicted_array_name ):
        preop_volume.GetPointData().RemoveArray( displacement_predicted_array_name )


    preop_mesh_with_displacement = to_pointcloud(
        coords=preop_array,
        features=displ_array,
        features_name=displacement_predicted_array_name,
    )

    preop_volume_estimated = apply_deformation(
        mesh = preop_volume, 
        displacement_mesh=preop_mesh_with_displacement, 
        displacement_array_name = displacement_predicted_array_name,
    )

    # Target displacement:
    displ_gt = vtk.vtkFloatArray()
    displ_gt.SetNumberOfComponents(3)
    displ_gt.SetNumberOfTuples(preop_volume.GetNumberOfPoints())
    displ_gt.SetName(displacement_gt_array_name)
    for i in range(preop_volume.GetNumberOfPoints()):
        d = [intraop_volume.GetPoint(i)[c] - preop_volume.GetPoint(i)[c] for c in range(3)]
        displ_gt.SetTuple(i, d)
    if preop_volume.GetPointData().HasArray(displacement_gt_array_name):
        preop_volume.GetPointData().RemoveArray(displacement_gt_array_name)
    preop_volume.GetPointData().AddArray(displ_gt)
    displ_gt_np = numpy_support.vtk_to_numpy(displ_gt)

                                                   
    # Displacement error:
    displ_err_vtk_array = vtk.vtkFloatArray()
    displ_err_vtk_array.SetNumberOfComponents(3)
    displ_err_vtk_array.SetNumberOfTuples(preop_volume_estimated.GetNumberOfPoints())
    displ_err_vtk_array.SetName(displacemment_error_array_name)
    for i in range(preop_volume_estimated.GetNumberOfPoints()):
        d = [intraop_volume.GetPoint(i)[c] - preop_volume_estimated.GetPoint(i)[c] for c in range(3)]
        displ_err_vtk_array.SetTuple(i, d)
    if preop_volume_estimated.GetPointData().HasArray(displacemment_error_array_name):
        preop_volume_estimated.GetPointData().RemoveArray(displacement_predicted_array_name)
    preop_volume_estimated.GetPointData().AddArray(displ_err_vtk_array)
    displ_array_np = numpy_support.vtk_to_numpy(displ_err_vtk_array)

    # print("dsipl_gt_np", displ_gt_np.shape, )
    # print("displ_array_np", displ_array_np.shape)

    # Calculate the groundtruth:
    displ_gt_magnitude = np.linalg.norm(displ_gt_np, axis=1)
    displ_gt_mean_magnitude = np.mean(displ_gt_magnitude)

    # Calculate the error:
    displ_error_magnitude = np.linalg.norm(displ_array_np, axis=1)
    displ_error_mean_magnitude = np.mean(displ_error_magnitude)

    return displ_gt_mean_magnitude, displ_error_mean_magnitude, preop_volume_estimated


def apply_transform( points, H ):

    _, N = points.shape
    print("N", N)

    points = torch.concatenate( (points, torch.ones(1,N)), dim = 0 )

    #transformed = (points.T @ H).T
    transformed = H @ points

    return transformed[0:3, :]


def apply_transform_to_vtk_mesh(mesh, H):
    """
    Apply a transformation matrix H to a vtk mesh.

    Parameters:
        vtk_mesh (vtk.vtkPolyData): Input VTK mesh.
        H (torch.Tensor): 4x4 transformation matrix.
    
    Returns:
        vtk.vtkPolyData: Transformed vtk mesh.
    """
    # Extract mesh points as a numpy array of shape (N, 3)
    vtk_points = mesh.GetPoints()
    points_np = vtk_to_numpy(vtk_points.GetData())
    
    # Convert numpy array to torch tensor and transpose to shape (3, N)
    points_torch = torch.from_numpy(points_np.astype(np.float32).T)
    
    # Apply the transformation using your function
    transformed_points = apply_transform(points_torch, H)
    
    # Convert transformed points back to numpy array of shape (N, 3)
    transformed_np = transformed_points.T.cpu().numpy()
    
    # Create new vtkPoints and set the transformed data
    new_vtk_points = vtk.vtkPoints()
    new_vtk_points.SetData(numpy_to_vtk(transformed_np, deep=True))
    
    # Update the input vtk mesh with the new points
    mesh.SetPoints(new_vtk_points)
    
    return mesh



def save_output_as_vtk(coords_pre, coords_intra, displ=None, displ_gt = None,
        features_pre=None, features_intra=None,
        coords_pre_internal=None, coords_intra_internal=None,
        features_pre_internal=None, features_intra_internal=None,
        landmarks_preop=None, landmarks_intraop=None,
        return_deformed_landmarks_preop=False,
        preop_meshes=[], intraop_meshes=[], perturbation=None, 
        scale=1e-3, center_offset=[0,0,0],
        folder = "tmp_out", verbose=True ):

    if len(coords_pre.shape) > 2:
        coords_pre = coords_pre.squeeze(0)
    if len(coords_intra.shape) > 2:
        coords_intra = coords_intra.squeeze(0)
    if features_pre is not None and len(features_pre.shape) > 2:
        features_pre = features_pre.squeeze(0)
    if features_intra is not None and len(features_intra.shape) > 2:
        features_intra = features_intra.squeeze(0)
    if coords_pre_internal is not None and len(coords_pre_internal.shape) > 2:
        coords_pre_internal = coords_pre_internal.squeeze(0)
    if coords_intra_internal is not None and len(coords_intra_internal.shape) > 2:
        coords_intra_internal = coords_intra_internal.squeeze(0)
    if features_pre_internal is not None and len(features_pre_internal.shape) > 2:
        features_pre_internal = features_pre_internal.squeeze(0)
    if features_intra_internal is not None and len(features_intra_internal.shape) > 2:
        features_intra_internal = features_intra_internal.squeeze(0)

    # Remove dummy points!
    # TODO: Also remove for other point clouds?
    mask_pre= np.absolute(coords_pre[:, 0]) < 1e3
    coords_pre = coords_pre[mask_pre]
    if displ is not None:
        displ = displ[mask_pre]
    if features_pre is not None:
        features_pre = features_pre[mask_pre]

    mask_intra = np.absolute(coords_intra[:, 0]) < 1e3
    coords_intra = coords_intra[mask_intra]
    if features_intra is not None:
        features_intra = features_intra[mask_intra]

    coords_pre = numpy_support.numpy_to_vtk(coords_pre)
    coords_intra = numpy_support.numpy_to_vtk(coords_intra)

    preop_vtk = to_pointcloud(
        coords_pre, features_pre
    )
    intraop_vtk = to_pointcloud(
        coords_intra, features_intra
    )

    preop_internal_vtk = None
    if coords_pre_internal is not None:
        coords_pre_internal = numpy_support.numpy_to_vtk(coords_pre_internal)
        preop_internal_vtk = to_pointcloud(
            coords_pre_internal, features_pre_internal
        )
    intraop_internal_vtk = None
    if coords_intra_internal is not None:
        coords_intra_internal = numpy_support.numpy_to_vtk(coords_intra_internal)
        intraop_internal_vtk = to_pointcloud(
            coords_intra_internal, features_intra_internal
        )


    #if features_pre is not None:
    #    for i in range( features_pre.shape[1] ):
    #        features_pre_vtk_array = numpy_support.numpy_to_vtk(features_pre[:,i])
    #        features_pre_vtk_array.SetName( f"f_{i}" )
    #        preop_vtk.GetPointData().AddArray(features_pre_vtk_array)
    #if features_intra is not None:
    #    for i in range( features_intra.shape[1] ):
    #        features_intra_vtk_array = numpy_support.numpy_to_vtk(features_intra[:,i])
    #        features_intra_vtk_array.SetName( f"f_{i}" )
    #        intraop_vtk.GetPointData().AddArray(features_intra_vtk_array)

    if displ is not None:
        displ = numpy_support.numpy_to_vtk(displ)
        displ_pred_vtk_array = numpy_support.numpy_to_vtk(displ)
        displ_pred_vtk_array.SetName("displacement_predicted")
        preop_vtk.GetPointData().AddArray(displ_pred_vtk_array)

    # Ground-Truth usually only available for synthetic data:
    if displ_gt is not None:
        displ_gt = displ_gt[mask_pre]
        # print("preop_vtk.GetNumberOfPoints()", preop_vtk.GetNumberOfPoints(), "displ_gt.shape", displ_gt.shape)
        displ_gt_vtk_array = numpy_support.numpy_to_vtk(displ_gt)
        displ_gt_vtk_array.SetName("displacement")
        preop_vtk.GetPointData().AddArray(displ_gt_vtk_array)

        if displ is not None:
            displ_error = displ_gt - displ
            displ_error_vtk_array = numpy_support.numpy_to_vtk(displ_error)
            displ_error_vtk_array.SetName("displacement_error")
            preop_vtk.GetPointData().AddArray(displ_error_vtk_array)


    # Landmarks may be provided. If so, use them to calculate an error:
    mean_orig_displ_err = 0
    mean_est_displ_err = 0
    orig_displ_err_list = []
    est_displ_err_list = []
    if landmarks_preop is not None and landmarks_intraop is not None:

        landmarks_preop = numpy_support.numpy_to_vtk(landmarks_preop)
        landmarks_intraop = numpy_support.numpy_to_vtk(landmarks_intraop)

        landmarks_preop_vtk = to_pointcloud(
            landmarks_preop,
        )
        landmarks_intraop_vtk = to_pointcloud(
            landmarks_intraop,
        )
        # print("landmarks_preop_vtk.GetNumberOfPoints()", landmarks_preop_vtk.GetNumberOfPoints(), "landmarks_intraop_vtk.GetNumberOfPoints()", landmarks_intraop_vtk.GetNumberOfPoints())
        # Apply the estimated displacement field to the preoperative landmarks:
        landmarks_intraop_estimated = apply_deformation(
                landmarks_preop_vtk, preop_vtk, displacement_array_name = "displacement_predicted" )

        # Calculate original, undisplaced error from preop to intraop:
        mean_orig_displ_err, orig_displ_err_list = compute_landmark_error(
            landmarks_preop_vtk.GetPoints(),
            landmarks_intraop_vtk.GetPoints(),
        )

        # Calculate new error after applying the estimated displacement:
        mean_est_displ_err, est_displ_err_list = compute_landmark_error(
            landmarks_intraop_estimated.GetPoints(),
            landmarks_intraop_vtk.GetPoints(),
        )

        if verbose:
            print( "TRE:" )
            print("\tBefore estimation:", mean_orig_displ_err)
            print("\tAfter estimation:", mean_est_displ_err)

    #print("preop_vtk.GetNumberOfPoints()", preop_vtk.GetNumberOfPoints(), "intraop_vtk.GetNumberOfPoints()", intraop_vtk.GetNumberOfPoints())
    #print(preop_vtk.GetPointData().HasArray("displacement_predicted"), preop_vtk.GetPointData().HasArray("displacement"), preop_vtk.GetPointData().HasArray("displacement_error"))


    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        output_filename_preop = os.path.join(folder, "preop_with_displ.vtp")
        print("Saving:", output_filename_preop)
        write_mesh(preop_vtk, output_filename_preop, verbose=True)
        output_filename_intraop = os.path.join(folder, "intraop.vtp")
        print("Saving:", output_filename_intraop)
        write_mesh(intraop_vtk, output_filename_intraop, verbose=True)

        if preop_internal_vtk:
            output_filename = os.path.join(folder, "preop_internal.vtp")
            print("Saving:", output_filename)
            write_mesh( preop_internal_vtk, output_filename, verbose=True)
        if intraop_internal_vtk:
            output_filename = os.path.join(folder, "intraop_internal.vtp")
            print("Saving:", output_filename)
            write_mesh( intraop_internal_vtk, output_filename, verbose=True)


        if landmarks_preop is not None and landmarks_intraop is not None:

            output_filename_preop = os.path.join(folder, "preop_landmarks.vtp")
            write_mesh( landmarks_preop_vtk, output_filename_preop, verbose=True )

            output_filename_intraop = os.path.join(folder, "intraop_landmarks.vtp")
            write_mesh( landmarks_intraop_vtk, output_filename_intraop, verbose=True )

            output_filename_intraop = os.path.join(folder, "intraop_landmarks_estimated.vtp")
            write_mesh( landmarks_intraop_estimated, output_filename_intraop, verbose=True )

        # scale = 1e-3
        # sf = vtkTransformFilter()
        # First, scale point cloud from mm to m:
        t_scale = vtkTransform()
        t_scale.Scale( scale, scale, scale )
        tf_scale = vtk.vtkTransformFilter()
        tf_scale.SetTransform( t_scale )

        t_center = vtkTransform()
        t_center.Translate( center_offset )
        tf_center = vtk.vtkTransformFilter()
        tf_center.SetTransform(t_center)
            
        for mesh_filename in preop_meshes:

            p, f = os.path.split( mesh_filename )

            mesh = load_mesh( mesh_filename )
        
            tf_scale.SetInputData( mesh ) 
            tf_scale.Update()
            mesh = tf_scale.GetOutput()

            tf_center.SetInputData( mesh )
            tf_center.Update()
            mesh = tf_center.GetOutput()

            if perturbation is not None:
                # print("!!!!!!Applying perturbation to", f)
                # print("perturbation", perturbation)
                mesh = apply_transform_to_vtk_mesh( mesh, perturbation)

            f_base, f_ext = os.path.splitext( f )
            output_filename = os.path.join( folder, f"{f_base}_preop{f_ext}" )
            write_mesh( mesh, output_filename, verbose=True )

            mesh_estimated = apply_deformation(
                    mesh, preop_vtk, displacement_array_name = "displacement_predicted" )

            output_filename = os.path.join( folder, f"{f_base}_intraop_estimated{f_ext}" )
            write_mesh( mesh_estimated, output_filename, verbose=True )

        for mesh_filename in intraop_meshes:

            p, f = os.path.split( mesh_filename )

            mesh = load_mesh( mesh_filename )
            tf_scale.SetInputData( mesh ) 
            tf_scale.Update()
            mesh = tf_scale.GetOutput()

            tf_center.SetInputData( mesh )
            tf_center.Update()
            mesh = tf_center.GetOutput()

            f_base, f_ext = os.path.splitext( f )
            output_filename = os.path.join( folder, f"{f_base}_intraop{f_ext}" )
            write_mesh( mesh, output_filename, verbose=True )

    if return_deformed_landmarks_preop:
        return mean_orig_displ_err, mean_est_displ_err, orig_displ_err_list, est_displ_err_list, vtk_to_numpy(landmarks_intraop_estimated.GetPoints().GetData())
    else:
        return mean_orig_displ_err, mean_est_displ_err, orig_displ_err_list, est_displ_err_list






def save_output_as_vtk_dry(
        coords_pre, coords_intra,
        displ=None, 
        displ_gt = None,
        displ_name="displacement_predicted", displ_gt_name="displacement", displ_error_name="displacement_error",
        features_pre=None, features_intra=None,
        features_pre_names=None, features_intra_names=None,
        coords_pre_internal=None, coords_intra_internal=None,
        features_pre_internal=None, features_intra_internal=None,
        landmarks_preop=None, landmarks_intraop=None,
        preop_meshes=[], intraop_meshes=[],
        scale = 1, #1e-3
        transform_center = None,
        folder = "tmp_out",
        output_preop_vtk_filename = "preop_with_displ.vtp",
        output_intraop_vtk_filename = "intraop.vtp",
        output_preop_internal_vtk_filename = "preop_internal.vtp",
        output_intraop_internal_vtk_filename = "intraop_internal.vtp",  
        output_preop_mesh_vtk_filename_prefix = "mesh_preop",  
        output_intraop_mesh_vtk_filename_prefix = "mesh_intraop_estimated",  
        output_preop_landmarks_vtk_filename = "preop_landmarks.vtp",
        output_intraop_landmarks_vtk_filename = "intraop_landmarks.vtp",
        output_intraop_landmarks_estimated_vtk_filename = "intraop_landmarks_estimated.vtp",
):

    """No displacement errors are calculated, only used to deform meshes and save them to disk.
    Then return the deformed meshes
    """

    if len(coords_pre.shape) > 2:
        coords_pre = coords_pre.squeeze(0)
    if len(coords_intra.shape) > 2:
        coords_intra = coords_intra.squeeze(0)
    # if features_pre is not None and len(features_pre.shape) > 2:
    #     features_pre = features_pre.squeeze(0)
    # if features_intra is not None and len(features_intra.shape) > 2:
    #     features_intra = features_intra.squeeze(0)
    if coords_pre_internal is not None and len(coords_pre_internal.shape) > 2:
        coords_pre_internal = coords_pre_internal.squeeze(0)
    if coords_intra_internal is not None and len(coords_intra_internal.shape) > 2:
        coords_intra_internal = coords_intra_internal.squeeze(0)
    if features_pre_internal is not None and len(features_pre_internal.shape) > 2:
        features_pre_internal = features_pre_internal.squeeze(0)
    if features_intra_internal is not None and len(features_intra_internal.shape) > 2:
        features_intra_internal = features_intra_internal.squeeze(0)

    # Remove dummy points!
    # TODO: Also remove for other point clouds?
    mask_pre= np.absolute(coords_pre[:, 0]) < 1e3
    coords_pre = coords_pre[mask_pre]
    if displ is not None:
        displ = displ[mask_pre]
    if features_pre is not None:
        features_pre = features_pre[mask_pre]

    mask = np.absolute(coords_intra[:, 0]) < 1e3
    coords_intra = coords_intra[mask]
    if features_intra is not None:
        if isinstance(features_intra, list):
            for i, f in enumerate(features_intra):
                features_intra[i] = features_intra[i][mask]
        else:
            features_intra = features_intra[mask]

    # print("coords_pre.shape", coords_pre.shape)
    # print("displ.shape", displ.shape)
    # print("displ_gt.shape", displ_gt.shape)
    # print("coords_intra.shape", coords_intra.shape)
    # for f in features_pre:
    #     print("f.shape", f.shape)

    coords_pre = numpy_support.numpy_to_vtk(coords_pre)
    coords_intra = numpy_support.numpy_to_vtk(coords_intra)


    preop_vtk = to_pointcloud(
        coords=coords_pre, 
        features=features_pre,
        features_name=features_pre_names,
    )
    intraop_vtk = to_pointcloud(
        coords=coords_intra, 
        features=features_intra,
        features_name=features_intra_names,
    )

    preop_internal_vtk = None
    if coords_pre_internal is not None:
        coords_pre_internal = numpy_support.numpy_to_vtk(coords_pre_internal)
        preop_internal_vtk = to_pointcloud(
            coords_pre_internal, features_pre_internal
        )
    intraop_internal_vtk = None
    if coords_intra_internal is not None:
        coords_intra_internal = numpy_support.numpy_to_vtk(coords_intra_internal)
        intraop_internal_vtk = to_pointcloud(
            coords_intra_internal, features_intra_internal
        )

    if displ is not None:
        displ = numpy_support.numpy_to_vtk(displ)
        displ_pred_vtk_array = numpy_support.numpy_to_vtk(displ)
        displ_pred_vtk_array.SetName(displ_name)
        preop_vtk.GetPointData().AddArray(displ_pred_vtk_array)

    # preop_vtk_deformed = apply_deformation( preop_vtk, displ_pred_vtk_array, "displacement_predicted" )

    # Ground-Truth usually only available for synthetic data:
    if displ_gt is not None:
        displ_gt = displ_gt[mask_pre]
        print("displ_gt.shape", displ_gt.shape)
        displ_gt_vtk_array = numpy_support.numpy_to_vtk(displ_gt)
        displ_gt_vtk_array.SetName(displ_gt_name)
        preop_vtk.GetPointData().AddArray(displ_gt_vtk_array)

        if displ is not None:
            displ_error = displ_gt - displ
            displ_error_vtk_array = numpy_support.numpy_to_vtk(displ_error)
            displ_error_vtk_array.SetName(displ_error_name)
            preop_vtk.GetPointData().AddArray(displ_error_vtk_array)


    # Convert landmarks to VTK point clouds:
    # mean_orig_displ_err = 0
    # mean_est_displ_err = 0
    # orig_displ_err_list = []
    # est_displ_err_list = []
    landmarks_preop_vtk, landmarks_intraop_vtk, landmarks_intraop_estimated = None, None, None
    if landmarks_preop is not None and landmarks_intraop is not None:

        landmarks_preop = numpy_support.numpy_to_vtk(landmarks_preop)
        landmarks_intraop = numpy_support.numpy_to_vtk(landmarks_intraop)

        landmarks_preop_vtk = to_pointcloud(
            landmarks_preop,
        )
        landmarks_intraop_vtk = to_pointcloud(
            landmarks_intraop,
        )

        # Apply the estimated displacement field to the preoperative landmarks:
        landmarks_intraop_estimated = apply_deformation(
                landmarks_preop_vtk, preop_vtk, displacement_array_name = displ_name )

        # # Calculate original, undisplaced error from preop to intraop:
        # mean_orig_displ_err, orig_displ_err_list = compute_landmark_error(
        #         landmarks_preop_vtk.GetPoints(),
        #         landmarks_intraop_vtk.GetPoints() )

        # # Calculate new error after applying the estimated displacement:
        # mean_est_displ_err, est_displ_err_list = compute_landmark_error(
        #         landmarks_intraop_estimated.GetPoints(),
        #         landmarks_intraop_vtk.GetPoints() )


        # print( "TRE:" )
        # print("\tBefore estimation:", mean_orig_displ_err)
        # print("\tAfter estimation:", mean_est_displ_err)

    #print("preop_vtk.GetNumberOfPoints()", preop_vtk.GetNumberOfPoints(), "intraop_vtk.GetNumberOfPoints()", intraop_vtk.GetNumberOfPoints())
    #print(preop_vtk.GetPointData().HasArray("displacement_predicted"), preop_vtk.GetPointData().HasArray("displacement"), preop_vtk.GetPointData().HasArray("displacement_error"))


    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        output_filename_preop = os.path.join(folder, output_preop_vtk_filename)
        print("Saving:", output_filename_preop)
        write_mesh(preop_vtk, output_filename_preop, verbose=True)
        output_filename_intraop = os.path.join(folder, output_intraop_vtk_filename)
        print("Saving:", output_filename_intraop)
        write_mesh(intraop_vtk, output_filename_intraop, verbose=True)

        if preop_internal_vtk:
            output_filename = os.path.join(folder, output_preop_internal_vtk_filename)
            print("Saving:", output_filename)
            write_mesh( preop_internal_vtk, output_filename, verbose=True)
        if intraop_internal_vtk:
            output_filename = os.path.join(folder, output_intraop_internal_vtk_filename)
            print("Saving:", output_filename)
            write_mesh( intraop_internal_vtk, output_filename, verbose=True)


        if landmarks_preop is not None and landmarks_intraop is not None:

            output_filename_preop = os.path.join(folder, output_preop_landmarks_vtk_filename)
            write_mesh( landmarks_preop_vtk, output_filename_preop, verbose=True )

            output_filename_intraop = os.path.join(folder, output_intraop_landmarks_vtk_filename)
            write_mesh( landmarks_intraop_vtk, output_filename_intraop, verbose=True )

            output_filename_intraop = os.path.join(folder, output_intraop_landmarks_estimated_vtk_filename)
            write_mesh( landmarks_intraop_estimated, output_filename_intraop, verbose=True )

        
        if scale != 1:
            sf = vtkTransformFilter()
            # First, scale point cloud from mm to m:
            t = vtkTransform()
            t.Scale( scale, scale, scale )
            tf = vtk.vtkTransformFilter()
            tf.SetTransform( t )
            print("Applying scale:", scale)
 
        if transform_center is not None: 
            t = vtk.vtkTransform()
            t.Translate( transform_center )
            tf_center = vtk.vtkTransformFilter()
            tf_center.SetTransform( t )
            print("Applying translation:", transform_center)
                
        for m in preop_meshes:
            if isinstance(m, str):
                mesh_filename = m
                p, f = os.path.split( mesh_filename )

                mesh = load_mesh( mesh_filename )
                if scale != 1:
                    tf.SetInputData( mesh ) 
                    tf.Update()
                    mesh = tf.GetOutput()
                if transform_center is not None:
                    tf_center.SetInputData( mesh )
                    tf_center.Update()
                    mesh = tf_center.GetOutput()
                f_base, f_ext = os.path.splitext( f )
                output_filename_preop = os.path.join( folder, f"{f_base}_preop{f_ext}" )
                output_filename_intraop_estimated = os.path.join( folder, f"{f_base}_intraop_estimated{f_ext}" )

            else:
                mesh = m

                #if scale != 1:
                #    tf.SetInputData( mesh ) 
                #    tf.Update()
                #    mesh = tf.GetOutput()   
                #if transform_center is not None:
                #    tf_center.SetInputData( mesh )
                #    tf_center.Update()
                #    mesh = tf_center.GetOutput()

                if mesh.GetPointData().HasArray(displ_name):
                    mesh.GetPointData().RemoveArray(displ_name)
                # output_filename_preop = os.path.join( folder, "{}.vtu".format(output_preop_mesh_vtk_filename_prefix) )
                # output_filename_intraop_estimated = os.path.join( folder, "{}.vtu".format(output_intraop_mesh_vtk_filename_prefix) )
                output_filename_preop = os.path.join( folder, "{}.vtp".format(output_preop_mesh_vtk_filename_prefix) )
                output_filename_intraop_estimated = os.path.join( folder, "{}.vtp".format(output_intraop_mesh_vtk_filename_prefix) )

            mesh = unstructured_grid_to_poly( mesh )
            write_mesh( mesh, output_filename_preop, verbose=True )

            mesh_estimated = apply_deformation(
                    mesh, preop_vtk, displacement_array_name = displ_name )
            mesh_estimated = unstructured_grid_to_poly( mesh_estimated )
            write_mesh( mesh_estimated, output_filename_intraop_estimated, verbose=True )

        for mesh_filename in intraop_meshes:

            p, f = os.path.split( mesh_filename )

            mesh = load_mesh( mesh_filename )
            if scale != 1:
                tf.SetInputData( mesh ) 
                tf.Update()
                mesh = tf.GetOutput()
            if transform_center is not None:
                tf_center.SetInputData( mesh )
                tf_center.Update()
                mesh = tf_center.GetOutput()

            f_base, f_ext = os.path.splitext( f )
            output_filename = os.path.join( folder, f"{f_base}_intraop{f_ext}" )
            write_mesh( mesh, output_filename, verbose=True )

    return preop_vtk, intraop_vtk, preop_internal_vtk, intraop_internal_vtk, landmarks_preop_vtk, landmarks_intraop_vtk, landmarks_intraop_estimated

 


def resample_polydata( poly_data, target_distance = 0.005, max_num_points = None, clean=True ):

    sampler = vtk.vtkPolyDataPointSampler()
    sampler.SetInputData( poly_data )
    sampler.SetDistance( target_distance )
    sampler.SetInterpolatePointData( True )
    sampler.SetGenerateEdgePoints( False )
    sampler.SetGenerateInteriorPoints( True )
    sampler.SetGenerateVertexPoints( True )
    #sampler.SetPointGenerationModeToRandom()
    sampler.Update()

    resampled_points = sampler.GetOutput()

    if max_num_points != None:
        mask = vtk.vtkMaskPoints()
        mask.SetInputData( resampled_points )
        mask.SetRandomMode( True )
        mask.SetMaximumNumberOfPoints( max_num_points )
        mask.Update()

        resampled_points = mask.GetOutput()

    if clean:
        resampled_points = remove_duplicates( resampled_points )

    return resampled_points


def decimate_polydata(poly_data, reduction_ratio=None, num_points=None):
    assert reduction_ratio is not None or num_points is not None, "Either ratio or num_points must be provided."

    if reduction_ratio is not None and reduction_ratio > 0 and reduction_ratio < 1:
        target_reduction = reduction_ratio
    elif num_points is not None:
        num_points_initial = poly_data.GetNumberOfPoints()
        # Compute the ratio of points to keep
        # Note: vtkDecimatePro uses a target reduction fraction based on the number of polygons.
        # Here, we approximate the fraction of points to keep from the original dataset.
        points_to_keep_ratio = num_points / num_points_initial

        # The target reduction is the fraction of polygons to remove.
        # This is an approximation that might need some experimentation.
        target_reduction = 1.0 - points_to_keep_ratio
    print("Target reduction (fraction of polygons to remove):", target_reduction)

    # Set up vtkDecimatePro
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(poly_data)

    # Set the target reduction based on our approximation.
    decimator.SetTargetReduction(target_reduction)

    # Preserve topology to help ensure that the mesh’s faces and connectivity remain intact.
    decimator.PreserveTopologyOn()

    # Prevent deletion of boundary vertices so that the outer shape is preserved.
    decimator.SetBoundaryVertexDeletion(False)

    # Execute the decimation filter
    decimator.Update()

    # Retrieve the decimated polydata
    decimated_polydata = decimator.GetOutput()
    print("Number of points after decimation:", decimated_polydata.GetNumberOfPoints())
    print("Number of polygons after decimation:", decimated_polydata.GetNumberOfCells())
    return decimated_polydata



def triangulate_point_cloud_delaunary_3d_Pyvista():
    import pyvista as pv
    import numpy as np

    pts = np.random.rand(512*3).reshape(-1,3)

    # Make vtkPolyData of the points array
    point_cloud = pv.PolyData(pts)
    # point_cloud.plot(render_points_as_spheres=True, point_size=10)


    # runs the delaunay 3D algorithm
    mesh = point_cloud.delaunay_3d(alpha=0.25)

    # Make a wireframe with some data
    # wires = mesh.compute_cell_sizes(length=False, area=False, volume=True).wireframe()
    # Plot it
    # wires.plot(line_width=2)
    print(mesh.verts.shape, mesh.faces.shape)
    # mesh.plot(show_edges=True)



def triangulate_point_cloud_delaunary_2d_vtk(mesh):
    # Create a Delaunay2D object
    delaunay = vtk.vtkDelaunay2D()
    delaunay.SetInputData(mesh)
    delaunay.Update()

    # Get the resulting polydata
    mesh = delaunay.GetOutput()

    return mesh



if __name__ == "__main__":
    # pass
    # path = "/mnt/ceph/tco/TCO-All/SharedDatasets/SparseDataChallenge/DatasetVoxelized/Set001/preop_mesh_centered_scaled_1.vtp"
    # mesh = load_mesh(path)
    # print(mesh.GetNumberOfPoints())

    volume = "/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/100k_nh/000006/liver_volume_f0.vtk"
    mesh = load_mesh(volume)
    # print(mesh)
    print(mesh.GetNumberOfPoints())
    surface, internal = split_surface_and_internal_points(mesh)


