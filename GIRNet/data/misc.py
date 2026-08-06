import os
import yaml
import sys
sys.path.append("../")
from data import vtk_utils

def create_statistics_yaml_manually():
    """This function iterates each sample folder, try to find preoperative and intraoperative
    surfaces, i.e. valid samples. It then creates a yaml file with the NO statistics.
    
    This function is created due to problems of automatic generation of statistics with the 
    pipeline
    """

    data_folder = "/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/10k_one_frame"

    stats = {}
    num_valid = 0
    num_invalid = 0
    for idx in range(0, 10000):
        sample_folder = os.path.join(data_folder, "{:06d}".format(idx))
        print("============Sample folder: ", sample_folder)
        if not os.path.exists(sample_folder):
            continue

        filename_volume_f0 = "liver_volume_f0.vtk"
        filename_volume_f1 = "liver_volume_f1.vtk"

        filename_partial_surface_f0 = "liver_surface_partial_noisy_f1.vtp"

        if os.path.exists(os.path.join(sample_folder, filename_volume_f0)) and \
            os.path.exists(os.path.join(sample_folder, filename_volume_f1)) and \
            os.path.exists(os.path.join(sample_folder, filename_partial_surface_f0)):
            # valid sample
            stats[idx] = {
                "place_holder": "place_holder",
            }

            print("valid")
            num_valid += 1
        else:
            print("not valid")
            num_invalid += 1

        # Create statistics yaml

    print(f"Number of valid samples: {num_valid}")
    print(f"Number of invalid samples: {num_invalid}")
    output_folder = "/mnt/cluster/datasets/V2SDataNewPipeline/liupeng/10k_one_frame"
    statistics_file = os.path.join(output_folder, "stats_manual.yaml")
    with open(statistics_file, "w") as f:
        yaml.dump(stats, f)

        # print(f"Created statistics for sample {idx}")


if __name__ == "__main__":
    # create_statistics_yaml_manually()
    

    # test loding vtk
    path = "/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/10k_one_frame/000041/liver_volume_f0.vtk"
    mesh = vtk_utils.load_mesh(path)
    print(mesh.GetNumberOfPoints())
