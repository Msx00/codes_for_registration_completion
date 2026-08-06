import torch
import os
import pickle as pkl
from prettytable import PrettyTable
import json, yaml
import math
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt

from log.logger import Logger
from data.data import DisplDataset, resample_by_given_stats
from data.data_syn import LiverSampleCombined
from data.data_amos import DisplDatasetAMOS, LiverSampleAMOS

from models.P_V2S_Net_V5_downsampled_intraop import PV2SNetV5DownsampledIntraop
from models.P_V2S_Net_V5_downsampled_intraop_v2 import PV2SNetV5DownsampledIntraopV2
from models.P_V2S_Net_V5_downsampled_intraop_v2_I2P import PV2SNetV5DownsampledIntraopV2I2P
from models.P_V2S_Net_V5_downsampled_intraop_v2_I2P_v2 import PV2SNetV5DownsampledIntraopV2I2PV2
from models.P_V2S_Net_V5_downsampled_intraop_multi_res import PV2SNetV5DownsampledIntraopMultiRes
from models.P_V2S_Net_V5_downsampled_intraop_v2_I2P_dgcnn import PV2SNetV5DownsampledIntraopV2I2PDGCNN


from models.v2s_net import subsampled_prediction_loss_2
from models.loss import calc_loss_pinns

from data import vtk_utils
from metrics import metrics


train_param_map = {
    "without_rigid": {
        "filenames":{
            "preop": "liver_volume_f0.vtk",
            "intraop": {
                "noisy":{
                    "random": "liver_surface_partial_noisy_f1.vtp",
                    "camera": "liver_surface_partial_cam_noisy_f1.vtp"
                },
                "clean":{
                    "random": "liver_surface_partial_f1.stl",
                    "camera": "liver_surface_partial_cam_f1.stl"
                }

            },
            "intraop_full": "liver_volume_f1.vtk",
        },
        "resample_stats_key": {
            "max_displ": "DisplacementStatisticsBlock_max_displacement_f0_to_f1", #"DisplacementStatisticsBlock_max_displacement_f1",
            "mean_displ": "DisplacementStatisticsBlock_mean_displacement_f0_to_f1", #"DisplacementStatisticsBlock_mean_displacement_f1",
            "vis_area": {
                "random" : "AddSurfaceNoiseBlock_estimated_noisy_surface_fraction_f1",                
                "camera" : "AddSurfaceNoiseFromCameraBlock_estimated_noisy_surface_fraction_f1",
            },
            "vis_point": { 
                "random" : "AddSurfaceNoiseBlock_number_of_remaining_points_1",
                "camera" : "AddSurfaceNoiseFromCameraBlock_number_of_remaining_points_1",
            },
        },
    },
    "with_rigid": {
        "filenames":{
            "preop": "liver_volume_preop_f0.vtk",
            "intraop": {
                "noisy":{
                    "random": "liver_surface_partial_noisy_f1.vtp",
                    "camera": "liver_surface_partial_cam_noisy_f1.vtp"
                },
                "clean":{
                    "random": "liver_surface_partial_f1.stl",
                    "camera": "liver_surface_partial_cam_f1.stl"
                }
            },
            "intraop_full": "liver_volume_f1.vtk",
        },
        "resample_stats_key": {
            "max_displ": "DisplacementStatisticsWithRigidDisplacementBlock_max_displacement_f0_to_f1",
            "mean_displ": "DisplacementStatisticsWithRigidDisplacementBlock_mean_displacement_f0_to_f1",
            "vis_area": {
                "random" : "AddSurfaceNoiseBlock_estimated_noisy_surface_fraction_f1",                
                "camera" : "AddSurfaceNoiseFromCameraBlock_estimated_noisy_surface_fraction_f1",
            },
            "vis_point": { 
                "random" : "AddSurfaceNoiseBlock_number_of_remaining_points_1",
                "camera" : "AddSurfaceNoiseFromCameraBlock_number_of_remaining_points_1",
            },
        },
    },
    "old_dataset_0": { # us_vessels_combined_20000
        "preop_filename" : "preop_volume_with_displacement.vtu",
        # "intraop_filename" : "partial_surface_intraop.obj",
        "intraop_filename": "deformed.stl",
        "intraop_full_filename" : "deformed.vtu",
    },

}



def data_filter( stats, intraop_type="random", cleanliness="noisy",  ):
    """function used to filter out "wrong" samples based on statistics

    Args:
        stats (dict): statistics of the sample 
        intraop_type (str, optional): the type of the intraoperative partial surafce, can be random or camera. Defaults to "random".

    Returns:
        bool: whether the sample is valid or not
    """
    # NOTE: This filter is a bit dangerous, because it does not complain when keys are missing, i.e. if
    # we change the data generation, this may fail to load samples. Hopefully, in that case, all samples
    # will fail, and that's something we'll notice.
    # try:
    valid = False
        # max_displ_condition = stats["CalcDisplacementBlock_0"]["max_displacement"] < 0.11
        # max_displ_condition = stats["SofaSimulationBlock_max_deformation"] < 0.11
        # min_intraop_mesh_points = stats["SofaSimulationBlock_intraop_mesh_num_points"] > 100
    try:
        if intraop_type == "random":
            if cleanliness == "noisy":
                min_intraop_mesh_points = (stats["AddSurfaceNoiseBlock_number_of_remaining_points_1"] > 100) \
                and (stats["SofaSimulationBlock_intraop_mesh_num_points"] > 150)\
                and (stats["PartialSurfaceExtractionBlock_partial_surface_amount_first_frame"] > 0.09)
                # min_intraop_mesh_points = (stats["AddSurfaceNoiseBlock_number_of_remaining_points_1"] > 2500) \
                # and (stats["SofaSimulationBlock_intraop_mesh_num_points"] > 2500)\
                # and (stats["PartialSurfaceExtractionBlock_partial_surface_amount_first_frame"] > 0.09)
                # min_intraop_mesh_points = (stats["AddSurfaceNoiseBlock_number_of_remaining_points_1"] > 2000) \
                # and (stats["SofaSimulationBlock_intraop_mesh_num_points"] > 150)\
                # and (stats["PartialSurfaceExtractionBlock_partial_surface_amount_first_frame"] > 0.09)
                valid = min_intraop_mesh_points
                print("random + noisy", "AddSurfaceNoiseBlock_number_of_remaining_points_1", stats["AddSurfaceNoiseBlock_number_of_remaining_points_1"], valid)
            elif cleanliness == "clean":
                valid = (stats["PartialSurfaceExtractionBlock_extracted_surface_points_first_frame"] > 100) \
                and (stats["SofaSimulationBlock_intraop_mesh_num_points"] > 150)\
                and (stats["PartialSurfaceExtractionBlock_partial_surface_amount_first_frame"] > 0.09)
                print("random + clean", valid)
        elif intraop_type == "camera":
            if cleanliness == "noisy":
                min_intraop_mesh_points = stats["AddSurfaceNoiseFromCameraBlock_number_of_remaining_points_1"] > 100 \
                and (stats["SofaSimulationBlock_intraop_mesh_num_points"] > 150)\
                and (stats["PartialSurfaceExtractionFromCameraBlock_extracted_surface_points_first_frame"] > 0.09)
                valid = min_intraop_mesh_points
                print("camera + noisy", valid)
            elif cleanliness == "clean":
                valid = (stats["PartialSurfaceExtractionFromCameraBlock_extracted_surface_points_first_frame"] > 100) \
                and (stats["SofaSimulationBlock_intraop_mesh_num_points"] > 150)\
                and (stats["PartialSurfaceExtractionFromCameraBlock_partial_surface_amount_first_frame"] > 0.09)
                print("camera + clean", valid)
        else:
            raise KeyError("intraop_type must be either random or camera")
        # TODO: the intraop_surface_num_points name may change (misnomer)
        # min_surface_points_condition = stats["SurfaceExtractionBlock_1"]["intraop_surface_num_points_A"] > 100
        # return max_displ_condition and min_surface_points_condition
        # return max_displ_condition and min_intraop_mesh_points
            
        return valid
    except:
        try:
            print("might be old datset")
            valid = (stats["SurfaceExtractionBlock_1"]["intraop_surface_num_points_A"] > 100) \
                and (stats["SimulationBlock_0"]["intraop_mesh_num_points"] > 2000)
            return valid
        except:
            raise KeyError("Could not find the correct keys in the statistics")



def data_filter_receptive_field_test( stats, intraop_type="random", cleanliness="noisy",  ):
    """function used to filter out "wrong" samples based on statistics

    Args:
        stats (dict): statistics of the sample 
        intraop_type (str, optional): the type of the intraoperative partial surafce, can be random or camera. Defaults to "random".

    Returns:
        bool: whether the sample is valid or not
    """
    # NOTE: This filter is a bit dangerous, because it does not complain when keys are missing, i.e. if
    # we change the data generation, this may fail to load samples. Hopefully, in that case, all samples
    # will fail, and that's something we'll notice.
    # try:
    valid = False
        # max_displ_condition = stats["CalcDisplacementBlock_0"]["max_displacement"] < 0.11
        # max_displ_condition = stats["SofaSimulationBlock_max_deformation"] < 0.11
        # min_intraop_mesh_points = stats["SofaSimulationBlock_intraop_mesh_num_points"] > 100
    try:
        valid = (stats["AddSurfaceNoiseBlock_number_of_remaining_points_1"] > 100) \
        and (stats["SofaSimulationBlock_intraop_mesh_num_points"] > 150)\
        and (stats["PartialSurfaceExtractionBlock_partial_surface_amount_first_frame"] > 0.09) \
        and (stats["DisplacementStatisticsBlock_mean_displacement_f0_to_f1"] > 0.03)
        print("random + noisy", "AddSurfaceNoiseBlock_number_of_remaining_points_1", stats["AddSurfaceNoiseBlock_number_of_remaining_points_1"], valid)

        return valid
    except:
        try:
            print("might be old datset")
            valid = (stats["SurfaceExtractionBlock_1"]["intraop_surface_num_points_A"] > 100) \
                and (stats["SimulationBlock_0"]["intraop_mesh_num_points"] > 150)
            return valid
        except:
            raise KeyError("Could not find the correct keys in the statistics")





class PV2SNetTrainer():
    def __init__(
        self,
        # config_path=None,
        config
    ):
        self.config = config
        self.train_param_map = train_param_map
        
        self.logger = Logger(
            base_path=self.config.save_path,
            n_samples=self.config.n_samples,
            comment=self.config.comment,
        )
        self.logger.save_run_details(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_samples_stats_list = []

        # self.dataset_train, self.dataloader_train, self.dataset_test, self.dataloader_test = self.__init_dataloader_4dmatch()
        # # self.dataset_test = self.dataset_train
        # # self.dataloader_test = self.dataloader_train
        # self.dataset_test_camera = self.dataset_train
        # self.dataloader_test_camera = self.dataloader_train
        # self.dataset_train_list = []
        # self.dataset_test_list = []

        self.dataset_train, self.dataset_test, self.dataset_test_camera,\
        self.dataset_train_list, self.dataset_test_list, self.dataset_test_camera_list, \
        self.dataloader_train, self.dataloader_test, self.dataloader_test_camera = self.__init_dataloader()

        # self.dataset_train, self.dataset_test, self.dataset_test_camera,\
        # self.dataloader_train, self.dataloader_test, self.dataloader_test_camera = self.__init_dataloader_test_amos()

        self.min_num_valid_points = -1        
        self.model = None
        self.start_epoch = 0


    def __init_argparser(self,):
        # TODO: maybe make this outside of the Trainer
        ...

    def __init_dataloader_test_amos(self):
        dataset_train = DisplDatasetAMOS(
            path = "/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/AMOS-validation_const_noise_amplitude/",
            start_sample=0,
            nsamples=950, #1080,
            sample_class=LiverSampleAMOS,
            preop_filename = "liver_volume_f0.vtk", 
            intraop_surface_filename_prefix= "liver_surface_partial_noisy_pl",
            intraop_full_filename= "liver_volume_f1.vtk",
            augmentation=True,
            return_all_intraop=False
        )

        dataset_test = DisplDatasetAMOS(
            path = "/mnt/cluster/workspaces/pfeiffemi/V2SData/NewPipeline/AMOS-validation_const_noise_amplitude/",
            start_sample=950,
            nsamples=130, #1080,
            sample_class=LiverSampleAMOS,
            preop_filename = "liver_volume_f0.vtk", 
            intraop_surface_filename_prefix= "liver_surface_partial_noisy_pl",
            intraop_full_filename= "liver_volume_f1.vtk",
            augmentation=False,
            return_all_intraop=False
        )

        dataloader_train = torch.utils.data.DataLoader(
                dataset = dataset_train,
                batch_size = self.config.batch_size,
                num_workers = 8,
                pin_memory = True,
                shuffle = False, #True, 
        )

        dataloader_test = torch.utils.data.DataLoader(
                dataset = dataset_test,
                batch_size = self.config.batch_size,
                num_workers = 8,
                pin_memory = True,
                shuffle = False,
        )

        return dataset_train, dataset_test, dataset_test, dataloader_train, dataloader_test, dataloader_test

    def __init_dataloader_4dmatch(self):
        from data.data_4dmatch import _4DMatch
        from easydict import EasyDict as edict
        # config_path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/organ-deformation-net/configs/config_4dmatch.yaml"
        # with open(config_path, 'r') as f:
        #     config = yaml.load(f, Loader=yaml.FullLoader)
        # config = edict(config)

        dataset_train = _4DMatch(
            config=self.config,
            split="train",
            data_augmentation=False,
            num=self.config.n_samples[0],
            cache_folder=self.config.data_cached_train_root,
        )

        dataloader_train = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            # num_workers = 8,
            pin_memory = True,
        )

        dataset_test = _4DMatch(
            config=self.config,
            split="val",
            data_augmentation=False,
            num=self.config.n_samples[1],
            cache_folder=self.config.data_cached_val_root,
        )

        dataloader_test = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=self.config.batch_size,
            shuffle=False,
            # num_workers = 8,
            pin_memory = True,
        )

        table = PrettyTable()
        table.field_names = ["Datasets", "Size"]
        table.add_rows([
            ["Train", len(dataset_train)],
            ["Validation", len(dataset_test)],
            # ["Validation (camera)", len(dataset_test_camera)],
        ])
        print(table)

        return dataset_train, dataloader_train, dataset_test, dataloader_test


    def __init_dataloader(self):
        if "old_dataset_0" in self.config.keys():
            deformation = "old_dataset_0"
        else:
            deformation = "with_rigid" if self.config.with_rigid else "without_rigid"
        cleanliness = "noisy" if self.config.noisy_intraop else "clean"
        intraop_type = "random"
        if "intraop_type" in self.config.keys() and self.config.intraop_type is not None:
            if self.config.intraop_type in ["random", "camera", "mixed", "mixed_deformation"]:
                intraop_type = self.config.intraop_type

        if "data_filter_type" in self.config.keys():
            if self.config.data_filter_type == "receptive_field_test":
                data_filter_selected = data_filter_receptive_field_test
                print("Using receptive field test filter")
            elif self.config.data_filter_type == "default":
                data_filter_selected = data_filter
                print("Using default filter")
        else:
            data_filter_selected = data_filter

        print("Initializing training dataset:")
        stats_list = []
        stats_path_list = [ os.path.join(data_folder, "statistics.pkl") for data_folder in self.config.data_path ]
        for idx_stats, stats_path in enumerate(stats_path_list):
            # if stats_path and os.path.exists(stats_path):
            print("Loading statistics from", stats_path)
            with open(stats_path, 'rb') as f:
                stats_list.append(pkl.load(f))
                f.close()

        # print("args.data_path", args.data_path)

        n_sample_list = [int(n) for n in self.config.n_samples]
        n_sample_train_list = [int(n*0.9) for n in n_sample_list]
        n_sample_test_list = [n - n_train for n, n_train in zip(n_sample_list, n_sample_train_list)]
        # n_samples_train = int(args.n_samples*0.9)
        # n_samples_test = args.n_samples - n_samples_train 

        n_single_dataset = 100000

        dataset_train_list = []
        # n_samples_train_sub = n_samples_train // len(args.data_path)
        
        for idx_d, data_path in enumerate(self.config.data_path):
            print("Creating dataset from", data_path)
            if idx_d >= len(n_sample_train_list) or n_sample_train_list[idx_d] == 0:
                continue
            
            if deformation == "old_dataset_0":
                preop_filename = self.train_param_map[deformation]["preop_filename"]
                intraop_filename = self.train_param_map[deformation]["intraop_filename"]
                intraop_full_filename = self.train_param_map[deformation]["intraop_full_filename"]
            else:
                preop_filename = self.train_param_map[deformation]["filenames"]["preop"]
                intraop_full_filename= self.train_param_map[deformation]["filenames"]["intraop_full"]
                if intraop_type == "mixed":
                    intraop_filename = [
                        self.train_param_map[deformation]["filenames"]["intraop"][cleanliness]["random"],
                        self.train_param_map[deformation]["filenames"]["intraop"][cleanliness]["camera"],
                    ]
                elif intraop_type == "mixed_deformation":
                    intraop_filename = [
                        self.train_param_map["without_rigid"]["filenames"]["intraop"][cleanliness]["random"],
                        self.train_param_map["with_rigid"]["filenames"]["intraop"][cleanliness]["random"],
                    ]
                else:
                    intraop_filename = self.train_param_map[deformation]["filenames"]["intraop"][cleanliness][intraop_type]

            dataset_train_sub = DisplDataset( 
                path = data_path, 
                npoints = self.config.n_points,
                sample_class = LiverSampleCombined,
                start_sample = self.config.start_sample + idx_d * n_single_dataset,
                nsamples = n_sample_train_list[idx_d],
                preop_filename = preop_filename, 
                intraop_filename= intraop_filename,
                intraop_full_filename= intraop_full_filename,
                criteria=data_filter_selected,
                # augmentation=True if "augmentation" in self.config.keys() and self.config.augmentation else False,
                augmentation=False if "augmentation" in self.config.keys() and not self.config.augmentation else True,
                #frame = "LAST",
                #frame = "NONE",
                frame = self.config.frame,
                ratio = self.config.ratio if "ratio" in self.config.keys() else 0.3,
                #quick_preload=(n_samples_train < 200)
                #quick_preload = True,
                # resample_stats_key=resample_stats_key,
                append_curvature=True if "curvature" in self.config.keys() and self.config.curvature else False,
                stats=stats_list[idx_d],
            )
            # print("Created dataset {} from {}".format(len(dataset_train_sub), data_path))
            # n_samples_train_loaded += n_single_dataset
            # idx_data_partition += 1

            dataset_train_list.append(dataset_train_sub)
            self.train_samples_stats_list.extend(dataset_train_sub.get_statistics())
        
        dataset_train = torch.utils.data.ConcatDataset(dataset_train_list) if len(dataset_train_list) > 1 else dataset_train_list[0]
        print("Concatenated dataset {} from {}".format(len(dataset_train), self.config.data_path))

        # 1/ 0
        print("Initializing test dataset:")
        dataset_test_list = []
        dataset_test_camera_list = []
        for idx_d, data_path in enumerate(self.config.data_path):
            if idx_d >= len(n_sample_test_list) or n_sample_test_list[idx_d] == 0:
                continue

            if deformation == "old_dataset_0":
                preop_filename = self.train_param_map[deformation]["preop_filename"]
                intraop_filename_random = self.train_param_map[deformation]["intraop_filename"]
                intraop_filename_camera = intraop_filename_random
                intraop_full_filename = self.train_param_map[deformation]["intraop_full_filename"]
            else:
                preop_filename = self.train_param_map[deformation]["filenames"]["preop"]
                intraop_full_filename= self.train_param_map[deformation]["filenames"]["intraop_full"]
                intraop_filename_random = self.train_param_map[deformation]["filenames"]["intraop"][cleanliness]["random"]
                intraop_filename_camera = self.train_param_map[deformation]["filenames"]["intraop"][cleanliness]["camera"]

            dataset_test = DisplDataset( 
                path = data_path, 
                npoints = self.config.n_points,
                sample_class = LiverSampleCombined,
                start_sample = n_sample_train_list[idx_d] + self.config.start_sample + idx_d * n_single_dataset,
                nsamples = n_sample_test_list[idx_d],
                preop_filename = preop_filename, #"liver_volume_preop_f0.vtk",
                intraop_filename= intraop_filename_random, #"liver_surface_partial_noisy_f1.vtp",
                intraop_full_filename= intraop_full_filename, #"liver_volume_f1.vtk",
                criteria=data_filter_selected,
                augmentation=False,
                frame = self.config.frame,
                ratio = self.config.ratio if "ratio" in self.config.keys() else 0.3,
                #quick_preload=(n_samples_test < 200)
                #quick_preload = True,
                append_curvature=True if "curvature" in self.config.keys() and self.config.curvature else False,
                stats=stats_list[idx_d],
            )
            dataset_test_camera = DisplDataset(
                path = data_path, 
                npoints = self.config.n_points,
                sample_class = LiverSampleCombined,
                start_sample = n_sample_train_list[idx_d] + self.config.start_sample + idx_d * n_single_dataset,
                nsamples = n_sample_test_list[idx_d],
                preop_filename = preop_filename, #"liver_volume_preop_f0.vtk",
                intraop_filename= intraop_filename_camera, #"liver_surface_partial_noisy_f1.vtp",
                intraop_full_filename= intraop_full_filename, #"liver_volume_f1.vtk",
                criteria=data_filter_selected, # need a different filter for camera data
                augmentation=False,
                frame = self.config.frame,
                ratio = self.config.ratio if "ratio" in self.config.keys() else 0.3,
                append_curvature=True if "curvature" in self.config.keys() and self.config.curvature else False,
                stats=stats_list[idx_d],
            )
            dataset_test_list.append(dataset_test)
            dataset_test_camera_list.append(dataset_test_camera)

        dataset_test = torch.utils.data.ConcatDataset(dataset_test_list) if len(dataset_test_list) > 1 else dataset_test_list[0]
        dataset_test_camera = torch.utils.data.ConcatDataset(dataset_test_camera_list) if len(dataset_test_camera_list) > 1 else dataset_test_camera_list[0]
        print("Concatenated test dataset {} from {}".format(len(dataset_test), self.config.data_path))
        print("Concatenated test camera dataset {} from {}".format(len(dataset_test_camera), self.config.data_path))

        # for idx in range(len(dataset_test)):
        #     print(idx)
        #     data = dataset_test[idx]
        #     meshes = dataset_test.get_meshes(idx)
        #     print(meshes.keys())
        #     print(meshes["preop_volume"].GetNumberOfPoints())
        #     print(meshes["intraop_volume"].GetNumberOfPoints())

        del stats_list

        print("***********************************************************************************")
        print("*****************************Dataset Summary***************************************")
        print("***********************************************************************************")
        # print("\tTraining dataset size: ", len(dataset_train))
        # print("\tTest dataset size: ", len(dataset_test))
        # print("\tTest camera dataset size: ", len(dataset_test_camera))
        # print("***********************************************************************************")

        table = PrettyTable()
        table.field_names = ["Datasets", "Size"]
        table.add_rows([
            ["Train", len(dataset_train)],
            ["Validation", len(dataset_test)],
            ["Validation (camera)", len(dataset_test_camera)],
        ])
        print(table)

        dataloader_train = torch.utils.data.DataLoader(
                dataset = dataset_train,
                batch_size = self.config.batch_size,
                num_workers = 8,
                pin_memory = True,
                shuffle = False, 
        )
        dataloader_test = torch.utils.data.DataLoader(
                dataset = dataset_test,
                batch_size = self.config.batch_size,
                num_workers = 8,
                pin_memory = True,
                shuffle = False,
        )
        dataloader_test_camera = torch.utils.data.DataLoader(
                dataset = dataset_test_camera,
                batch_size = self.config.batch_size,
                num_workers = 8,
                pin_memory = True,
                shuffle = False,
        )

        return dataset_train, dataset_test, dataset_test_camera, dataset_train_list, dataset_test_list, dataset_test_camera_list, dataloader_train, dataloader_test, dataloader_test_camera


    def initialize_model(self, hparams):
        """Initialize model based on the configuration, can be used for later training or inference

        """
        
        self.model_selected = None
        self.model_version = None
        self.model_pinn = False
        print(self.config.model)
        if self.config.model == "PV2SNetV5DownsampledIntraop":
            self.model_selected = PV2SNetV5DownsampledIntraop
        elif self.config.model == "PV2SNetV5DownsampledIntraopV2":
            self.model_selected = PV2SNetV5DownsampledIntraopV2
        elif self.config.model == "PV2SNetV5DownsampledIntraopV2I2P":
            self.model_selected = PV2SNetV5DownsampledIntraopV2I2P
        elif self.config.model == "PV2SNetV5DownsampledIntraopV2I2PV2":
            self.model_selected = PV2SNetV5DownsampledIntraopV2I2PV2
        elif self.config.model == "PV2SNetV5DownsampledIntraopMultiRes":
            self.model_selected = PV2SNetV5DownsampledIntraopMultiRes
        elif self.config.model == "PV2SNetV5DownsampledIntraopV2I2PDGCNN":
            self.model_selected = PV2SNetV5DownsampledIntraopV2I2PDGCNN
        # PINN
        # elif self.config.model == "PV2SNetV2DownsampledIntraopV2PINN":
        #     self.model_selected = PV2SNetV2DownsampledIntraopV2PINN
        #     self.model_pinn = True
        # elif self.config.model == "PV2SNetV1DownsampledIntraopV2PINN":
        #     self.model_selected = PV2SNetV1DownsampledIntraopV2PINN
        #     self.model_pinn = True
        elif self.config.model is None:
            raise ValueError("Model is None!")
        else:
            raise ValueError("Unknown model: {}".format(self.config.model))
        
        if "PV2SNetV1" in self.config.model:
            n_intermediate_features = [200, 100, 50]
            # n_intermediate_points = [hparams["n_bottleneck_points"], 100]
            n_intermediate_points = [
                hparams["n_layer_0_pre_points"], 
                hparams["n_layer_1_pre_points"], 
            ]
        elif "PV2SNetV2" in self.config.model:
            n_intermediate_features = [200, 150, 80, 50]
            n_intermediate_points = [
                hparams["n_layer_0_pre_points"], 
                hparams["n_layer_1_pre_points"], 
                hparams["n_layer_2_pre_points"],
            ]
        elif "PV2SNetV5" in self.config.model:
            n_intermediate_features = [200, 150, 110, 80, 60, 50]
            n_intermediate_points = [
                hparams["n_layer_0_pre_points"], 
                hparams["n_layer_1_pre_points"], 
                hparams["n_layer_2_pre_points"],
                hparams["n_layer_3_pre_points"],
                hparams["n_layer_4_pre_points"],
                hparams["n_layer_5_pre_points"],
            ]
        if "MLPSampling" in self.config.model:
            n_mlp_sampling_features =[
                hparams["n_mlp_sampling_features_0"],
                hparams["n_mlp_sampling_features_1"],
            ]

        print("Initializing model", self.model_selected.__name__)
        print("Hparams:", json.dumps(hparams, sort_keys=True, indent=4))

        if "MLPSampling" in self.config.model:
            self.model = self.model_selected(
                n_input_features = 5, #6,
                n_preprocess_features = 50,
                n_intermediate_features = n_intermediate_features, # [200, 150, 80, 50], # [200, 100, 50],
                n_intermediate_points =  n_intermediate_points, #[hparams["n_layer_0_pre_points"], hparams["n_layer_1_pre_points"], hparams["n_layer_2_pre_points"]], # if [n_bottleneck_points, 100],
                n_mlp_sampling_features=n_mlp_sampling_features,
                n_output_features = 3,
                embedding_size = hparams["embedding_size"],
                points_per_region = hparams["points_per_region"],
                enc_freq = [2e-2, 2e-1, 2, 4, 8, 16, 32, 64], # ***preprocessor arguments start***
                enc_freq_scale=1, 
                append_df_self=True,
                append_df_cross=True,
                append_positional_encoding=True,
                compact_return=True,  # ***preprocessor arguments end***
            )
        else:
            self.model = self.model_selected(
                n_input_features = 6 if "curvature" in self.config.keys() and self.config.curvature else 5,
                n_preprocess_features = 50,
                n_intermediate_features = n_intermediate_features, # [200, 150, 80, 50], # [200, 100, 50],
                n_intermediate_points =  n_intermediate_points, #[hparams["n_layer_0_pre_points"], hparams["n_layer_1_pre_points"], hparams["n_layer_2_pre_points"]], # if [n_bottleneck_points, 100],
                n_output_features = 3,
                embedding_size = hparams["embedding_size"],
                points_per_region = hparams["points_per_region"],
                enc_freq = [2e-2, 2e-1, 2, 4, 8, 16, 32, 64], # ***preprocessor arguments start***
                enc_freq_scale=1, 
                append_df_self=True,
                append_df_cross=True,
                append_positional_encoding=True,
                compact_return=True,  # ***preprocessor arguments end***
            )
        self.model.to(self.device)

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=hparams["lr"])

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, 
            max_lr=hparams["max_lr"],
            steps_per_epoch=math.ceil(len(self.dataset_train) / self.config.batch_size),
            # steps_per_epoch=len(dataloader_train),
            epochs = self.config.epochs,
        )
        
        if "ckpt" in self.config.keys() and self.config.ckpt is not None:
            checkpoint_path = os.path.join(self.config.ckpt, "current_model.pth")
            checkpoint = torch.load( checkpoint_path )
            self.model.load_state_dict( checkpoint["model_state_dict"] )
            self.optim.load_state_dict( checkpoint["optimizer_state_dict"] )
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict( checkpoint["scheduler_state_dict"] )
            else:
                self.scheduler.last_epoch = self.optim.state[self.optim.param_groups[0]["params"][-1]]["step"]
            self.start_epoch = checkpoint['epoch'] + 1 # plus one because we start from the next epoch
            # self.num_epoch = self.config.epochs - self.start_epoch
            print("Continue training from epoch", self.start_epoch)

        if "pretrained" in self.config.keys() and self.config.pretrained is not None:
            print("Loading pretrained model from", self.config.pretrained)
            checkpoint_path = os.path.join(self.config.pretrained, "current_model.pth")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = 0

        self.logger.set_trial_params(hparams , model_name=self.model_selected.__name__ )


    def get_model(self,):
        return self.model


    def train(self, trial):
        if self.model is None:
            raise ValueError("Model is not initialized!")

        # torch.autograd.set_detect_anomaly(True)
        avg_train_errs = []
        avg_test_errs = []
        avg_train_displacement_errs = []
        avg_test_displacement_errs = []

        avg_test_errs_camera = []
        avg_test_displacement_errs_camera = []

        best_test_mean_displ_err = math.inf
        wrong_batch_list = []
        for e in range(self.start_epoch, self.config.epochs):
            torch.set_grad_enabled(True)
            self.model.train()
            # print(f"Epoch: {e}/{args.epochs}")
            train_errs = []
            # train_errs_internal = []
            displacement_errs = []
            target_displacements = []

            if "WRS" in self.config.keys() and self.config.WRS:
                resample_stats_key_list = []
                deformation = "with_rigid" if self.config.with_rigid else "without_rigid"
                # cleanliness = "noisy" if self.config.noisy_intraop else "clean"
                resample_stats_key_list = []

                if self.config.resample_key_list is not None and len(self.config.resample_key_list) > 0:
                    for rc in self.config.resample_key_list:
                        if "vis" in rc:
                            # if the key is vis_area or vis_point, we choose 'random' for now from the two 
                            # intraoperative surface types including random and camera. Will add more options later
                            resample_stats_key_list.append(train_param_map[deformation]["resample_stats_key"][rc]["random"])
                        else:
                            # if the key is max_displ or mean_displ, we choose the key directly
                            resample_stats_key_list.append(train_param_map[deformation]["resample_stats_key"][rc])

                difficulty = resample_by_given_stats(
                    stats=self.train_samples_stats_list,
                    resample_key_list=resample_stats_key_list, #["DisplacementStatisticsBlock_max_displacement_f0_to_f1",],
                    epoch=e,
                    epoch_total=self.config.epochs,
                )
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=difficulty,
                    num_samples=len(self.dataset_train),
                    replacement=False,
                )
                self.dataloader_train = torch.utils.data.DataLoader(
                    dataset = self.dataset_train,
                    batch_size = self.config.batch_size,
                    num_workers = 8,
                    pin_memory = True,
                    sampler = sampler,
                )

            for idx, data in tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train), desc="Train epoch {}".format(e)): # | displ errs {:04f}".format(e, displacement_err if 'displacement_err' in locals() else -1)):
                if idx in wrong_batch_list:
                    print("wrong batch", idx, "go next...")
                    continue
                preop = data["preop"].cuda()
                intraop = data["intraop"].cuda()
                displ = data["displ"].cuda()
                original_idx_list = data["idx"]
                youngs_modulus = data["youngs_modulus"].cuda()
                poissons_ratio = data["poissons_ratio"].cuda()
                # print("original_idx_list.shape", original_idx_list.shape)
                # print("youngs_modulus", youngs_modulus, "poissons_ratio", poissons_ratio)
                # print("preop.shape", preop.shape, "intraop.shape", intraop.shape, "displ.shape", displ.shape)

                DEBUG_PREPROCESS = False
                if DEBUG_PREPROCESS and idx == 3:
                    print("Saving debug meshes...")
                    coords_preop, features_preop, coords_intraop, features_intraop = self.model.preprocessor.preprocess(preop, intraop,)
                    vtk_utils.save_output_as_vtk(
                        coords_pre = preop.cpu().numpy()[0, :3, ...].T,
                        coords_intra = intraop.cpu().numpy()[0, :3, ...].T,
                        displ_gt = displ.cpu().numpy()[0,...].T,
                        features_pre = features_preop.cpu().numpy()[0,...].T,
                        features_intra = features_intraop.cpu().numpy()[0,...].T,
                        folder = os.path.join(self.logger.path, "{:06d}".format(idx)),
                    )

                DEBUG_DUPLICATE = False
                if DEBUG_DUPLICATE:
                    for idx_b in range(preop.shape[0]):
                        preop_coords = preop[idx_b, :3, ...]
                        preop_valid_idx = torch.all(torch.abs(preop_coords) < 100, dim=0)
                        preop_filtered = preop_coords[:, preop_valid_idx]
                        preop_unique = torch.unique(preop_filtered, dim=1)
                        has_duplicate = preop_filtered.shape[1] != preop_unique.shape[1]
                        if has_duplicate:
                            assert False, "Duplicate points in preop, preop_unique.shape:{}, preop_filtered.shape:{}".format(preop_unique.shape, preop_filtered.shape)
                        # else:
                        #     print("No duplicate points in preop")

                        intraop_coords = intraop[idx_b, :3, ...]
                        intraop_valid_idx = torch.all(torch.abs(intraop_coords) < 100, dim=0)
                        intraop_filtered = intraop_coords[:, intraop_valid_idx]
                        intraop_unique = torch.unique(intraop_filtered, dim=1)
                        has_duplicate = intraop_filtered.shape[1] != intraop_unique.shape[1]
                        if has_duplicate:
                            assert False, "Duplicate points in intraop, intraop_unique.shape:{}, intraop_filtered.shape:{}".format(intraop_unique.shape, intraop_filtered.shape)
                        # else:
                        #     print("No duplicate points in intraop")

                self.model.zero_grad()
                try:
                    predictions = self.model(
                            preop,
                            intraop,
                    )
                except:
                    wrong_sample_index_list = []
                    for idx_batch in range(preop.shape[0]):
                        wrong_sample_index_list.append(original_idx_list[idx_batch, 0])
                    raise ValueError("Wrong sample idx: {}".format(wrong_sample_index_list))

                prediction = predictions[-1]["result"]  # highest resolution output
                # print("prediction.shape", prediction.shape, "displ.shape", displ.shape)
                #loss = ((prediction - displ)**2).mean()        # MSE

                DEBUG_SAMPLING = False
                if "debug_mlp_sampling" in self.config.keys() and self.config.debug_mlp_sampling:
                    DEBUG_SAMPLING = True
                    
                if DEBUG_SAMPLING and idx == 3:
                    # point_idx_list = reversed([predictions[idx_l]["point_idx"][0] for idx_l in range(len(predictions))])
                    for idx_layer in range(len(predictions) - 1, -1, -1):
                        print("Saving downsampling points for layer", idx_layer)
                        if idx_layer == len(predictions) - 1:
                        # if idx_layer == 0:
                            points_layer = preop[0, :3, ...].cpu().numpy().T
                            # remove dummy points
                            # valid_points_mask = points_layer.max(axis=1) < 5000
                            # points_layer = points_layer[valid_points_mask]
                        else:
                            point_idx = predictions[idx_layer]["point_idx"][0].cpu().numpy().T
                            print("point_idx.shape", point_idx.shape)
                            # points_layer = preop[0, :3, ...].cpu().numpy().T
                            # print("points_layer.shape", points_layer.shape)           
                            # points_layer = preop[0, :3, point_idx].cpu().numpy().T
                            points_layer = points_layer[point_idx]
                        print("points_layer.shape", points_layer.shape)
                        # output_path = os.path.join(self.logger.path, "downsampled_points_layer_{:02d}.vtp".format(idx_layer))
                        # points_layer_vtk = vtk_utils.to_pointcloud(
                        #     coords=points_layer,
                        # )
                        # vtk_utils.write_mesh(
                        #     mesh=points_layer_vtk,
                        #     filename=output_path,
                        #     verbose=True,
                        # )
                        output_folder = os.path.join(self.logger.path, "{}".format(self.logger.trial), "debug_mlp_sampling")
                        vtk_utils.save_output_as_vtk_dry(
                            coords_pre = points_layer,
                            coords_intra=intraop[0, :3, ...].cpu().numpy().T,
                            folder = output_folder,
                            output_preop_vtk_filename= "downsampled_points_layer_{:02d}_epoch_{:03d}.vtp".format(idx_layer, e),
                            output_intraop_vtk_filename= "intraop.vtp",
                        )



                # Loss of prediction for all levels:
                loss_displ = subsampled_prediction_loss_2( displ, predictions )
                # print("loss", loss.item())
                if loss_displ.item() > 1000:
                    print("\nepoch:", idx, "loss_displ", loss_displ.item())
                    print("prediction:", prediction.min().item(), prediction.max().item(), prediction.mean().item())
                    print("displ:", displ.min().item(), displ.max().item(), displ.mean().item())

                    print("sample index:", original_idx_list)
                    
                    # now this only saves the first sample in the batch, it can be other samples
                    # maybe save all samples in the batch
                    for idx_sb in range(preop.shape[0]):
                        vtk_utils.save_output_as_vtk(
                            coords_pre = preop.cpu().numpy()[idx_sb, :3, ...].T,
                            coords_intra = intraop.cpu().numpy()[idx_sb, :3, ...].T,
                            displ_gt = displ.cpu().numpy()[idx_sb,...].T,
                            # features_pre = features_preop.cpu().numpy()[0,...].T,
                            # features_intra = features_intraop.cpu().numpy()[0,...].T,
                            # folder = os.path.join(logger.path, "{:06d}".format(idx)),
                            # folder = "/home/liupeng/ceph_home/others/debug_p-v2s-net_data_syn/wrong_sample/{:06d}".format(int(original_idx_list[idx_sb, 0])),
                        )
                    raise ValueError("Loss too high")
                    # print("Loss too high, skip this batch")
                    # raise optuna.exceptions.TrialPruned()

                if self.model_pinn:
                    loss_pinns = calc_loss_pinns(results=predictions[-1], youngs_modulus=youngs_modulus, poisson_ratio=poissons_ratio)
                    loss = loss_displ + 1e-6 * loss_pinns
                else:
                    loss = loss_displ


                # loss.backward()
                try:
                    loss.backward()
                except:
                    wrong_sample_index_list = []
                    wrong_batch_list.append(idx)
                    for idx_batch in range(preop.shape[0]):
                        wrong_sample_index_list.append(original_idx_list[idx_batch, 0])
                    print("Wrong sample idx: {}".format(wrong_sample_index_list))
                    print("Wrong batch idx: {}".format(idx), "len(wrong_batch_list)", len(wrong_batch_list))
                    raise ValueError("Wrong sample idx: {}".format(wrong_sample_index_list))

                self.optim.step()
                self.scheduler.step()

                train_errs.append( loss.item() )
                # if not args.no_internals:
                #     train_errs_internal.append( loss_internal.item() )
                displacement_err = metrics.MeanDisplacementError( prediction, displ ).item()
                displacement_errs.append( displacement_err )
                target_displacement = metrics.MeanMagnitude( displ ).item()
                target_displacements.append( target_displacement )

                # print("current learning rate:", self.optim.param_groups[0]["lr"])
                self.logger.summary_writer.add_scalar("learning_rate", self.optim.param_groups[0]["lr"], e * len(self.dataloader_train) + idx)
                # print("learning rate", optim.param_groups[0]["lr"])
                # print("current step:", optim.state[optim.param_groups[0]["params"][-1]]["step"])
            avg_train_err =  sum(train_errs)/len(train_errs)
            # if not args.no_internals:
            #     avg_train_err_internal =  sum(train_errs_internal)/len(train_errs_internal)
            avg_displacement_err = sum(displacement_errs)/len(displacement_errs)
            avg_target_displacement = sum(target_displacements)/len(target_displacements)
            print("\tTrain loss", "mean:", avg_train_err,
                    "max:", max(train_errs),
                    "Avg target displacement:", avg_target_displacement,
                    "Avg. displacement err:", avg_displacement_err )
            avg_train_errs.append( avg_train_err )
            avg_train_displacement_errs.append( avg_displacement_err )
            self.logger.summary_writer.add_scalar("AvgErr/train", avg_train_err, e)
            # if not args.no_internals:
            #     logger.summary_writer.add_scalar("AvgErrInternal/train", avg_train_err_internal, e)
            self.logger.summary_writer.add_scalar("AvgDisplacementErr/train", avg_displacement_err, e)
            if self.model_pinn:
                self.logger.summary_writer.add_scalar("LossPinns/train", loss_pinns.item(), e)

            # if args.stages:
            #     # if the dataset is curriculum, update the stage according to the epoch
            #     dataset_train.update_stage( e + 1 )

            # with torch.no_grad():
            # pinns_models = True
            # if self.model_pinn:
                # if we train PINNs models, we need to calculate the derivatives
            torch.set_grad_enabled(True)
            # else:
            #     torch.set_grad_enabled(False)

            self.model.eval()
            test_errs = []
            # test_errs_internal = []
            displacement_errs = []
            target_displacements = []

            i = 0
            for idx, data in tqdm(enumerate(self.dataloader_test), total=len(self.dataloader_test), desc="Test epoch {}".format(e)): # {} | displ errs {:04f}".format(e, displacement_err if 'displacement_err' in locals() else -1)):
                preop = data["preop"].cuda()
                intraop = data["intraop"].cuda()
                displ = data["displ"].cuda()

                predictions = self.model(
                        preop,
                        intraop,
                )

                prediction = predictions[-1]["result"]  # highest resolution output

                # Loss of prediction for other levels:
                loss_displ = subsampled_prediction_loss_2( displ, predictions )
                if self.model_pinn:
                    loss_pinns = calc_loss_pinns(results=predictions[-1], youngs_modulus=youngs_modulus, poisson_ratio=poissons_ratio)
                    loss = loss_displ + 1e-6 * loss_pinns
                else:
                    loss = loss_displ

                test_errs.append( loss.item() ) 

                displacement_err = metrics.MeanDisplacementError( prediction, displ ).item()
                displacement_errs.append( displacement_err )
                target_displacement = metrics.MeanMagnitude( displ ).item()
                target_displacements.append( target_displacement )

                if i == 0 and False:
                    # For debugging purposes, safe the point clouds for the first sample in the test:
                    preop_combined = torch.cat( (
                            preop[0,:,:],
                            displ[0,:,:],
                            prediction[0,:,:],
                            features_preop[0,:,:],
                            ), dim=0 )
                    save_point_cloud( preop_combined, "preop_combined.csv" )
                    intraop_combined = torch.cat( (intraop[0,:,:], features_intraop[0,:,:]), dim=0 )
                    save_point_cloud( intraop_combined,
                            "intraop_combined.csv" )
                    save_point_cloud( coords_preop[0,:,:], "coords_preop.csv" )
                    save_point_cloud( coords_intraop[0,:,:], "coords_intraop.csv" )

            # Errors of the test set:
            avg_test_err = sum(test_errs)/len(test_errs)
            avg_displacement_err = sum(displacement_errs)/len(displacement_errs)
            avg_target_displacement = sum(target_displacements)/len(target_displacements)
            print("\tTest loss", "mean:", avg_test_err,
                    "max:", max(test_errs),
                    "Avg target displacement:", avg_target_displacement, 
                    "Avg. displacement err:", avg_displacement_err )
            avg_test_errs.append( avg_test_err )
            avg_test_displacement_errs.append( avg_displacement_err )
            self.logger.summary_writer.add_scalar("AvgErr/test", avg_test_err, e)
            self.logger.summary_writer.add_scalar("AvgDisplacementErr/test", avg_displacement_err, e)
            if self.model_pinn:
                self.logger.summary_writer.add_scalar("LossPinns/test", loss_pinns.item(), e)

            trial.report(avg_test_err, e)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # test on camera intraoperative surfaces:    
            if "test_camera" in self.config.keys() and self.config.test_camera:    
                test_errs_camera = []
                # test_errs_internal = []
                displacement_errs_camera = []
                target_displacements_camera = []
                for idx, data in tqdm(enumerate(self.dataloader_test_camera), total=len(self.dataloader_test_camera), desc="Test camera epoch {}".format(e)): # {} | displ errs {:04f}".format(e, displacement_err if 'displacement_err' in locals() else -1)):
                    preop = data["preop"].cuda()
                    intraop = data["intraop"].cuda()
                    displ = data["displ"].cuda()

                    predictions = self.model(
                            preop,
                            intraop,
                    )

                    prediction = predictions[-1]["result"]  # highest resolution output

                    # Loss of prediction for other levels:
                    loss = subsampled_prediction_loss_2( displ, predictions )

                    test_errs_camera.append( loss.item() ) 

                    displacement_err_camera = metrics.MeanDisplacementError( prediction, displ ).item()
                    displacement_errs_camera.append( displacement_err_camera )
                    target_displacement_camera = metrics.MeanMagnitude( displ ).item()
                    target_displacements_camera.append( target_displacement_camera )

                    i = i + 1
            
                # Errors of the camera test set:
                avg_test_err_camera = sum(test_errs_camera)/len(test_errs_camera)
                avg_displacement_err_camera = sum(displacement_errs_camera)/len(displacement_errs_camera)
                avg_target_displacement_camera = sum(target_displacements_camera)/len(target_displacements_camera)
                print("\tTest camera loss", "mean:", avg_test_err_camera,
                        "max:", max(test_errs_camera),
                        "Avg target displacement:", avg_target_displacement_camera, 
                        "Avg. displacement err:", avg_displacement_err_camera,
                )
                avg_test_errs_camera.append( avg_test_err_camera )
                avg_test_displacement_errs_camera.append( avg_displacement_err_camera )
                self.logger.summary_writer.add_scalar("AvgErr/test_camera", avg_test_err_camera, e)
                self.logger.summary_writer.add_scalar("AvgDisplacementErr/test_camera", avg_displacement_err_camera, e)


            # Interpolate estimated dispalcements to the original mesh then calculate the displacement error
            # as dataset is used instead of dataloader, we need to iterate over the dataset
            if "test_mesh" in self.config.keys() and self.config.test_mesh:
                displacement_errs_meshes = []
                target_displacements_meshes = [] 
                for idx in tqdm(range(len(self.dataset_test))):
                    data = self.dataset_test[idx]
                    meshes = self.dataset_test.get_meshes(idx)
                    preop_volume = meshes["preop_volume"]
                    intraop_volume = meshes["intraop_volume"]

                    preop = torch.FloatTensor(data["preop"]).unsqueeze(0).to(self.device)
                    intraop = torch.FloatTensor(data["intraop"]).unsqueeze(0).to(self.device)

                    predictions = self.model(
                        preop,
                        intraop,
                    )

                    prediction = predictions[-1]["result"]  # highest resolution output

                    target_displacement, displacement_err, _ = vtk_utils.calc_registration_error_interpolated(
                        preop_volume=preop_volume,
                        intraop_volume=intraop_volume,
                        preop_array=preop[:, :3, ...].squeeze(0).cpu().numpy().T,
                        displ_array=prediction.squeeze(0).cpu().numpy().T,
                    )

                    displacement_errs_meshes.append( displacement_err )
                    target_displacements_meshes.append( target_displacement )

                avg_displacement_err_meshes = sum(displacement_errs_meshes)/len(displacement_errs_meshes)
                avg_target_displacement_meshes = sum(target_displacements_meshes)/len(target_displacements_meshes)

                print("\tTest mesh loss",
                    "Avg target displacement:", avg_target_displacement_meshes, 
                    "Avg. displacement err:", avg_displacement_err_meshes,
                )
                # self.logger.summary_writer.add_scalar("AvgErr/test_mesh", avg_test_err_camera, e)
                self.logger.summary_writer.add_scalar("AvgDisplacementErr/test_mesh", avg_displacement_err_meshes, e)


            # Save result every epoch, overwrite the one from the previous epoch
            self.logger.save_model(
                name = "current_model",
                epoch = e,
                model = self.model,
                optimizer = self.optim,
                scheduler = self.scheduler,
                # logger = logger,
                train_mean_displ_err = avg_train_errs[-1],
                test_mean_displ_err = avg_test_errs[-1]
            )
            if avg_test_errs[-1] < best_test_mean_displ_err:
                best_test_mean_displ_err = avg_test_errs[-1]
                self.logger.save_model(
                    name = "best_model",
                    epoch = e,
                    model = self.model,
                    optimizer = self.optim,
                    scheduler = self.scheduler,
                    # logger = logger,
                    train_mean_displ_err = avg_train_errs[-1],
                    test_mean_displ_err = avg_test_errs[-1]
                )


        # Save final model result:
        self.logger.save_model(
            name = "final_model",
            epoch = self.config.epochs,
            model = self.model,
            optimizer = self.optim,
            scheduler = self.scheduler,
            # logger = logger,
            train_mean_displ_err = avg_train_errs[-1],
            test_mean_displ_err = avg_test_errs[-1]
        )

        self.stats = {
            "avg_train_errs": avg_train_errs,
            "avg_test_errs": avg_test_errs,
            "avg_train_displacement_errs": avg_train_displacement_errs,
            "avg_test_displacement_errs": avg_test_displacement_errs,
            "avg_test_errs_camera": avg_test_errs_camera,
            "avg_test_displacement_errs_camera": avg_test_displacement_errs_camera,
        }

        self.logger.save_stats(self.stats)
        self.debug_validation()
        return avg_displacement_err


    def visualize_stats(self,):
        ...

        
    def train_optuna(self,):
        """Train the model using Optuna for hyperparameter optimization
        """
        def objective(trial):
            if "ckpt" in self.config.keys() and self.config.ckpt is not None:
                self.logger.continue_old_trial(old_trial_folder=self.config.ckpt)
            else:
                self.logger.start_new_trial()
            print( f"Training on {len(self.dataset_train)} samples, validating on {len(self.dataset_test)} samples" )

            embedding_size = trial.suggest_int( "embedding_size", 20, 50 )
            points_per_region = trial.suggest_int( "points_per_region", 20, 40 )
            lr = trial.suggest_float( "lr", 1e-6, 1e-4, log=True )
            max_lr = trial.suggest_float( "max_lr", 1e-5, 1e-3, log=True )

            if "PV2SNetV1" in self.config.model:
                n_layer_0_pre_points = trial.suggest_int( "n_layer_0_pre_points", 3, 50) # 25, 60,) #10, 35 )
                n_layer_1_pre_points = trial.suggest_int( "n_layer_1_pre_points",  80, 150) #70, 150,) # 50, 70 )

                hparams = {
                    # "n_bottleneck_points": n_bottleneck_points,
                    "n_layer_0_pre_points": n_layer_0_pre_points,
                    "n_layer_1_pre_points": n_layer_1_pre_points,
                    "embedding_size": 37, #embedding_size,
                    "points_per_region": 30, #points_per_region,
                    "lr": lr,
                    "max_lr": max_lr,
                }
                self.min_num_valid_points = n_layer_1_pre_points
                print("hparams (train):", hparams)
            elif "PV2SNetV2" in self.config.model:
                # new ranges considering the added last layer extra layer
                n_layer_0_pre_points = trial.suggest_int( "n_layer_0_pre_points", 7, 50) #3, 20) # 25, 60,) #10, 35 )
                n_layer_1_pre_points = trial.suggest_int( "n_layer_1_pre_points",  80, 120) #70, 150,) # 50, 70 )
                n_layer_2_pre_points = trial.suggest_int( "n_layer_2_pre_points", 170, 220) # 200, 300,)# 100, 150)

                # Ensamble hparams:
                hparams = {
                    # "n_intermediate_points": [n_layer_0_pre_points, n_layer_1_pre_points, n_layer_2_pre_points],
                    "n_layer_0_pre_points": n_layer_0_pre_points,
                    "n_layer_1_pre_points": n_layer_1_pre_points,
                    "n_layer_2_pre_points": n_layer_2_pre_points,
                    "embedding_size": embedding_size,
                    "points_per_region": points_per_region,
                    "lr": lr,
                    "max_lr": max_lr,
                }
                self.min_num_valid_points = n_layer_2_pre_points
            elif "PV2SNetV5" in self.config.model:
                n_layer_0_pre_points = trial.suggest_int( "n_layer_0_pre_points", 3, 30) #3, 20)
                n_layer_1_pre_points = trial.suggest_int( "n_layer_1_pre_points",  40, 70,) #30, 50)
                n_layer_2_pre_points = trial.suggest_int( "n_layer_2_pre_points", 80, 120,) #70, 100)
                n_layer_3_pre_points = trial.suggest_int( "n_layer_3_pre_points", 150, 200,) #130, 170)
                n_layer_4_pre_points = trial.suggest_int( "n_layer_4_pre_points", 230, 300) #210, 250)
                n_layer_5_pre_points = trial.suggest_int( "n_layer_5_pre_points", 330, 400) #300, 350)

                hparams = {
                    "n_layer_0_pre_points": n_layer_0_pre_points,
                    "n_layer_1_pre_points": n_layer_1_pre_points,
                    "n_layer_2_pre_points": n_layer_2_pre_points,
                    "n_layer_3_pre_points": n_layer_3_pre_points,
                    "n_layer_4_pre_points": n_layer_4_pre_points,
                    "n_layer_5_pre_points": n_layer_5_pre_points,
                    "embedding_size": embedding_size,
                    "points_per_region": points_per_region,
                    "lr": lr,
                    "max_lr": max_lr,
                }
                # made a mistake before, layer 5 is not actually used...orz...
                self.min_num_valid_points = n_layer_4_pre_points
            if "MLPSampling" in self.config.model:
                n_mlp_sampling_features_0 = trial.suggest_int( "n_mlp_sampling_features_0", 16, 64 )
                n_mlp_sampling_features_1 = trial.suggest_int( "n_mlp_sampling_features_1", 3, 16 )

                hparams.update({
                    "n_mlp_sampling_features_0": n_mlp_sampling_features_0,
                    "n_mlp_sampling_features_1": n_mlp_sampling_features_1,
                })

            # Update min_num_valid_points for train and test datasets:
            print("Setting min number of valid points in train and test datasets to:", self.min_num_valid_points)
            # self.dataset_train.set_min_num_valid_points(self.min_num_valid_points)
            # self.dataset_test.set_min_num_valid_points(self.min_num_valid_points)
            for dataset_sub in self.dataset_train_list:
                dataset_sub.set_min_num_valid_points(self.min_num_valid_points)
            for dataset_sub in self.dataset_test_list:
                dataset_sub.set_min_num_valid_points(self.min_num_valid_points)
            
            self.initialize_model(hparams)
            err = self.train(trial=trial)
            return err
        
        if self.config.n_trials > 1:
            pruner = optuna.pruners.PercentilePruner(
                percentile=25,          
                n_startup_trials=3,     
                n_warmup_steps=50,      
                interval_steps=5,
            )
            study = optuna.create_study( direction="minimize", pruner=pruner )
        else:
            assert self.config.n_trials == 1, "n_trials must be 1 or more"
            study = optuna.create_study( direction="minimize" )

        if "ckpt" in self.config.keys() and self.config.ckpt is not None:
            # if continue is set, load the hparams from previous run:
            hparam_path = os.path.join(self.config.ckpt, "params.yaml")
            with open(hparam_path, 'r') as f:
                hparams = yaml.safe_load(f)
                f.close()
            study.enqueue_trial(hparams)
            
        elif self.config.q_best:# or True:
            # if not continue and q_best is set, enqueue the best hparams from previous runs

            # best from 100k runs
        # study.enqueue_trial(
        #     {
        #         "n_layer_0_pre_points": 10,
        #         "n_layer_1_pre_points": 54,
        #         "n_layer_2_pre_points": 198,
        #         "embedding_size": 43,
        #         "points_per_region": 39,
        #         "lr": 0.000027380,
        #         "max_lr": 0.0004, # 0.00039894
        #     },
        # )
            hparam_path = "configs/p_v2s_net_best_hparams.yaml"
            with open(hparam_path, 'r') as f:
                hparam_list = yaml.safe_load(f)
                f.close()
            # hparams = hparam_list[self.config.model[:9]]
            hparams = hparam_list[self.config.model]
            print(self.config.model)
            # print("hparams:", hparams)
            study.enqueue_trial(hparams)
            print("enqueued best hparams from", hparams)


        study.optimize( objective, n_trials = self.config.n_trials )

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("\tNumber of finished trials: ", len(study.trials))
        print("\tNumber of pruned trials: ", len(pruned_trials))
        print("\tNumber of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("\tValue: ", trial.value)

        print("\tParams: ")
        for key, value in trial.params.items():
            print("\t\t{}: {}".format(key, value))

        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig( os.path.join( self.logger.path, "optuna_param_importances.ps" ) )
        except Exception as e:
            print("plot_param_importances failed. Reason:", e)

        try:
            optuna.visualization.matplotlib.plot_edf(study)
            plt.savefig( os.path.join( self.logger.path, "optuna_edf.ps" ) )
        except Exception as e:
            print("plot_edf failed. Reason:", e)


        try:
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.savefig( os.path.join( self.logger.path, "optuna_parallel_coordinate.ps" ) )
        except Exception as e:
            print("plot_param_importances failed. Reason:", e)

        #try:
        #    optuna.visualization.matplotlib.plot_contour(study)
        #    plt.savefig( os.path.join( logger.path, "optuna_contour.ps" ) )
        #except Exception as e:
        #    print("plot_contour failed. Reason:", e)

        #try:
            #optuna.visualization.matplotlib.plot_pareto_front(study)
            #plt.savefig( os.path.join( logger.path, "optuna_pareto_front.ps" ) )
        #except:
        #    pass
        try:
            optuna.visualization.matplotlib.plot_slice(study)
            plt.savefig( os.path.join( self.logger.path, "optuna_slice.ps" ) )
        except Exception as e:
            print("plot_slice failed. Reason:", e)
        #try:
        #    optuna.visualization.matplotlib.plot_intermediate_values(study)
        #    plt.savefig( os.path.join( logger.path, "optuna_intermediate_values.ps" ) )
        #except:
        #    pass
        try:
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig( os.path.join( self.logger.path, "optuna_optimization_history.ps" ) )
        except Exception as e:
            print("plot_optimization_history failed. Reason:", e)



    def debug_validation(self,):
        print("\nSaving samples for visual inspection....")
        try:
            with torch.no_grad():
                self.model.eval()
                idx_sample = 0
                for idx, data in enumerate(self.dataloader_test):
                    print("Batch:", idx)
                    if idx_sample > 3:
                        break
                    preop = data["preop"].cuda()
                    intraop = data["intraop"].cuda()
                    displ = data["displ"].cuda()

                    predictions = self.model(
                        preop,
                        intraop,
                    )

                    prediction = predictions[-1]["result"]
                    print("preop.shape", preop.shape, "intraop.shape", intraop.shape, "displ.shape", displ.shape)
                    print("prediction.shape", prediction.shape, "displ.shape", displ.shape)
                    
                    for idx_b in range(preop.shape[0]):
                        print("Sample:", idx_sample)
                        vtk_utils.save_output_as_vtk_dry(
                            coords_pre = preop.cpu().numpy()[idx_b, :3, ...].T,
                            coords_intra = intraop.cpu().numpy()[idx_b, :3, ...].T,
                            displ=prediction.cpu().numpy()[idx_b, ...].T,
                            displ_gt = displ.cpu().numpy()[idx_b, ...].T,
                            folder = os.path.join(self.logger.current_path, "validation", "{:06d}".format(idx_sample)),
                        )
                        idx_sample += 1

        except Exception as e:
            print("Failed to save samples:", e)



