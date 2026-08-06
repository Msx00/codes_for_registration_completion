import os
import numpy as np
import scipy
import vtk_utils
from vtk.util import numpy_support
import yaml
from data_ear import EarDIOMEDataset
import vtk


def transform_ear(
    mesh_coords,
    rotation,
    translation,
    transformed_output_path=None,
):
    R = scipy.spatial.transform.Rotation.from_quat(rotation)
    T = np.array(translation)
    T = np.expand_dims(T, axis=0)

    mesh_coords_tranformed = R.apply(mesh_coords) + T
    # mesh_coords_tranformed = R.apply(mesh_coords + T)
    print("mesh_coords_tranformed.shape:", mesh_coords_tranformed.shape)

    if transformed_output_path:
        mesh_transformed_poly = vtk_utils.to_pointcloud(
            coords=numpy_support.numpy_to_vtk(mesh_coords_tranformed),
        )
        vtk_utils.write_mesh(
            mesh=mesh_transformed_poly,
            filename=transformed_output_path,
        )
        print("Saved transformed mesh to:", transformed_output_path)

    return mesh_coords_tranformed


def test_transform_all_ear(
    data_folder,
    rotation,
    translation,
):
    
    dataset = EarDIOMEDataset(
        folder=data_folder,
        num=43,
        load_landmarks=True,
        load_downsampled=True,
    )

    print(dataset[0].keys())

    for idx, data in enumerate(dataset):
        print("===============Loading", idx)
        side = data["meta"]["patient_info"]["side"]
        # if side == "right":
            

def test_mirror(
    mesh_path,
):
    mesh = vtk_utils.load_mesh(mesh_path)
    mesh_coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    print("mesh_coords.shape:", mesh_coords.shape)
    center_origin = np.mean(mesh_coords, axis=0)

    # mirror
    mesh_coords[:, 0] = -mesh_coords[:, 0]
    center_flipped = np.mean(mesh_coords, axis=0)
    output_path = os.path.join(os.path.dirname(mesh_path), "test_mirror.vtp")
    mesh_flipped_poly = vtk_utils.to_pointcloud(
        coords=numpy_support.numpy_to_vtk(mesh_coords),
    )
    vtk_utils.write_mesh(
        mesh=mesh_flipped_poly,
        filename=output_path,
    )
    print("Saved mirrored mesh to:", output_path)

    # dist = center_origin - center_flipped
    dist = [15, 0, 0]
    mesh_coords = mesh_coords + dist
    output_path = os.path.join(os.path.dirname(mesh_path), "test_mirror_transformed.vtp")
    mesh_flipped_transformed_poly = vtk_utils.to_pointcloud(
        coords=numpy_support.numpy_to_vtk(mesh_coords),
    )
    vtk_utils.write_mesh(
        mesh=mesh_flipped_transformed_poly,
        filename=output_path,
    )
    print("Saved mirrored transformed mesh to:", output_path)






def generate_normals_and_downsample(
        folder="/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models",
        n_points=10000,
):
    # label_name = label_name.append("combined_no_promontory")
    for idx in range(0, 43):
    # for idx in range(39, 43):
        print("========== sample_{} ==========".format(idx))
        cur_dir = os.path.join(folder, "sample_{}".format(idx))
        seg_list = []
        # for seg_name in label_name:
        #     print("loading:", seg_name)
        #     cur_seg_path = os.path.join(cur_dir, "{}.stl".format(seg_name))
        cur_seg_path = os.path.join(cur_dir, "combined_no_promontory.stl")
        if os.path.exists(cur_seg_path):
            # load mesh
            seg = vtk_utils.load_mesh(cur_seg_path)
            # downsample
            
            seg = vtk_utils.generate_point_normals(seg)
            print(seg.GetPointData().HasArray("Normals"))
            seg_normals = seg.GetPointData().GetArray("Normals")
            seg_normals_np = numpy_support.vtk_to_numpy(seg_normals)

            seg_coords = numpy_support.vtk_to_numpy(seg.GetPoints().GetData())
            seg_coords_idx_downsampled = np.random.choice(range(seg_coords.shape[0]), n_points, replace=False)
            seg_coords_downsampled = seg_coords[seg_coords_idx_downsampled]
            seg_normals_downsampled = seg_normals_np[seg_coords_idx_downsampled, :]
            print("downsampled from {} to {}".format(seg_coords.shape[0], seg_coords_downsampled.shape[0]))

            seg_downsampled = vtk_utils.to_pointcloud(
                coords=numpy_support.numpy_to_vtk(seg_coords_downsampled),
            )
            # seg_downsampled.GetPointData().SetNormals(numpy_support.numpy_to_vtk(seg_normals_downsampled))
            # print(seg_downsampled.GetPointData().HasArray("Normals"))
            normals_array = vtk.vtkFloatArray()
            normals_array.SetNumberOfComponents(3)
            normals_array.SetName("Normals")
            for i in range(seg_normals_downsampled.shape[0]):
                normals_array.InsertNextTuple(seg_normals_downsampled[i])
            seg_downsampled.GetPointData().SetNormals(normals_array)
            print(seg_downsampled.GetPointData().HasArray("Normals"))
            output_path = os.path.join(cur_dir, "combined_no_promontory_downsampled_with_normals.vtp")

            vtk_utils.write_mesh(
                mesh=seg_downsampled,
                filename=output_path,
            )
            print("Saved downsampled mesh to:", output_path)
            # points = vtk.vtkPoints()
            # points.SetData(numpy_support.numpy_to_vtk(seg_coords_downsampled))
            # pd = vtk.vtkPolyData()
            # pd.SetPoints(points)

            # verts = vtk.vtkCellArray()
            # for i in range(pd.GetNumberOfPoints()):
            #     verts.InsertNextCell( vtk.VTK_VERTEX, (i,) )
            # pd.SetVerts( verts )
            # appendFilter = vtk.vtkAppendFilter()
            # appendFilter.SetInputData(pd)
            # appendFilter.Update()
            # seg_downsampled = appendFilter.GetOutput()
            # print("seg_downsampled:", seg_downsampled.GetNumberOfPoints())

            # # output_path = os.path.join(cur_dir, "downsampled", "{}_downsampled.stl".format(seg_name))
            # output_path = os.path.join(cur_dir, "combined_no_promontory_downsampled.vtu")
            # # if not os.path.exists(os.path.dirname(output_path)):
            # #     os.makedirs(os.path.dirname(output_path))
            # write_vtu(seg_downsampled, output_path)
            # print("saved to:", output_path)






if __name__ == "__main__":
    generate_normals_and_downsample()
    # test_mirror(
    #     mesh_path="/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/sample_0/combined_no_promontory_downsampled.vtu"
    # )

    1 / 0

    sample_folder = "/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/sample_1/"
    transformation_path = os.path.join(sample_folder, "manual_alignment/output_transformation.yaml")
    # source_path = os.path.join(sample_folder, "combined_no_promontory_downsampled.vtu")
    # target_path = os.path.join("/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/D2EAR_data/SSM/TMOSS3/TMOSS3_template.stl")

    # source_mesh = vtk_utils.load_mesh(source_path)
    # print("source_mesh.GetNumberOfPoints():", source_mesh.GetNumberOfPoints())
    # source_mesh_coords = numpy_support.vtk_to_numpy(source_mesh.GetPoints().GetData())
    # print("source_mesh_coords.shape:", source_mesh_coords.shape)

    with open(transformation_path, "r") as f:
        transformation = yaml.safe_load(f)
        f.close()
    print("transformation:", transformation)

    # quaternion = 
    rotation_dict = transformation["transformation"]["rotation"]
    translation_dict = transformation["transformation"]["translation"]

    # print("rotation_quaternion:", rotation_quaternion)
    # print("translation:", translation)

    rotation = [
        # rotation_dict["w"],
        # rotation_dict["x"],
        # rotation_dict["y"],
        # rotation_dict["z"],
        rotation_dict["x"],
        rotation_dict["y"],
        rotation_dict["z"],
        rotation_dict["w"],
    ]
    translation = [
        translation_dict["x"],
        translation_dict["y"],
        translation_dict["z"],
    ]

    print("rotation:", rotation)
    print("translation:", translation)
    

    # transform_ear(
    #     mesh_coords=source_mesh_coords,
    #     rotation=rotation,
    #     translation=translation,
    #     transformed_output_path=os.path.join(sample_folder, "manual_alignment/test_transformed_source.vtp"),
    # )

    test_transform_all_ear(
        data_folder="/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/",
        rotation=rotation,
        translation=translation,
    )

