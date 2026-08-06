import os
import sys
import numpy as np

# sys.path.append("../")
# from data import vtk_utils
import vtk_utils




def convert_landmarks():
    source_folder = "/mnt/ceph/tco/TCO-All/SharedDatasets/LiverDIRDataset/NIfTI/"
    target_folder = "/mnt/ceph/tco/TCO-All/SharedDatasets/LiverDIRDataset/LandmarksVTK"
    for idx in range(30):
        print(f"Processing {idx}")
        landmarks_path_1 = os.path.join(source_folder, "case{}_landmarks1.txt".format(idx + 1))
        landmarks_path_2 = os.path.join(source_folder, "case{}_landmarks2.txt".format(idx + 1))

        with open(landmarks_path_1, "r") as f:
            lines = f.readlines()
            landmarks_1 = np.array([list(map(float, line.strip().split(","))) for line in lines])
        with open(landmarks_path_2, "r") as f:
            landmarks_2 = np.array([list(map(float, line.strip().split(","))) for line in lines])

        print(landmarks_1.shape)
        print(landmarks_2.shape)

        landmarks_1_vtk = vtk_utils.to_pointcloud(
            coords=landmarks_1,
        )
        output_path = os.path.join(target_folder, f"case{idx + 1}_landmarks1.vtp")
        vtk_utils.write_mesh(
            mesh=landmarks_1_vtk,
            filename=output_path,
            verbose=True,
        )

        landmarks_2_vtk = vtk_utils.to_pointcloud(
            coords=landmarks_2,
        )
        output_path = os.path.join(target_folder, f"case{idx + 1}_landmarks2.vtp")
        vtk_utils.write_mesh(
            mesh=landmarks_2_vtk,
            filename=output_path,
            verbose=True,
        )

convert_landmarks()




