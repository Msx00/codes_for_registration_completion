import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from glob import glob
import pickle as pkl
import yaml, json
import nrrd
try:
    from . import vtk_utils
    from . import pc_utils
except:
    try:
        import vtk_utils
        import pc_utils
    except:
        from data import vtk_utils
        from data import pc_utils

from vtk.util import numpy_support 
from simpleicp import SimpleICP, PointCloud
import scipy


class SSMEarDataset(Dataset):
    def __init__(self,
            folder,
            num,
            offset=0,
        ) -> None:
        super().__init__()
        
        self.folder = folder
        self.num = num
        self.offset = offset

    def __len__(self):
        return self.num
    
    def get_mean_model(self):
        import h5py
        ssm_path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/D2EAR_data/SSM/TMOSS3/TMOSS3_ssm.h5"
        h5 = h5py.File(ssm_path,'r')
        mean_model = h5["representer"]["points"]
        return mean_model

    def __getitem__(self, idx):
        cur_dir = os.path.join(self.folder, "{:06d}".format(idx + self.offset))
        # print("loading", cur_dir)
        with open(os.path.join(cur_dir, "intra.pkl"), "rb") as f:
            data = pickle.load(f)
            f.close()
        # print(data)
        # print(data.keys())

        intra_noisy = data["intra_noisy_combined"]
        intra_full = data["ssm_deformed"]
        pc = data["pc_random"]
        mean_model = self.get_mean_model()

        # print("intra_noisy.shape", intra_noisy.shape)
        # print("intra_full.shape", intra_full.shape)
        # print("pc.shape", pc.shape)

        intra_noisy_size = intra_noisy.shape[0]
        intra_noisy_padded = np.zeros(intra_full.shape)
        intra_noisy_padded[:intra_noisy_size, :] = intra_noisy
        # print(type(intra_noisy))

        intra_noisy_padded = np.transpose(intra_noisy_padded, (1, 0))
        intra_full = np.transpose(intra_full, (1, 0))

        return intra_noisy_padded, intra_full, pc, intra_noisy_size, mean_model




class EarSynDataset(Dataset):
    def __init__(self,
            folder,
            num,
            offset=0,
            data_filename = "data_cached_with_internals.pkl",
            surface_only = False,
            n_pre_points=2500,
            n_intra_points=2500,
            ratio_pre_internals=0.3,
            pre_load=False,
        ) -> None:
        super().__init__()
        
        self.folder = folder
        self.num = num
        self.offset = offset
        self.data_filename = data_filename
        self.sample_folder_list = [i.replace('\\', '/') for i in sorted(glob('{}/??????'.format(self.folder)))[offset : offset+num]]
        self.surface_only = surface_only
        self.n_pre_points = n_pre_points    
        self.n_intra_points = n_intra_points
        self.ratio_pre_internals = ratio_pre_internals
        self.pre_load = pre_load
        self.data_list = []
        if self.pre_load:
            self.data_list = self.__pre_load()

    def __len__(self):
        return self.num

    # TODO
    def __pre_load(self):

        return None

    def augmentation(self, intraop):
    
        return intraop


    def __getitem__(self, index):
        # print("loading", index)
        sample_folder = self.sample_folder_list[index] 
        with open(os.path.join(sample_folder, self.data_filename), "rb") as f:
            data = pkl.load(f)
            f.close()
        
        # print(data.keys())
        preop_surface = data["preop_surface"]
        preop_internals = data["preop_internals"]
        # intraop_surface = data["intraop_surface"]
        intraop_internals = data["intraop_internals"]
        intraop_noisy = data["intraop_noisy_with_internals"]
        # print("preop_surface.shape:", preop_surface.shape)
        # print("preop_internals.shape:", preop_internals.shape)
        # print("intraop_surface.shape:", intraop_surface.shape)
        # print("intraop_internals.shape:", intraop_internals.shape)
        # print("intraop_noisy.shape:", intraop_noisy.shape)

        preop_surface_normals = data["preop_surface_normals"]
        preop_internal_normals = np.zeros(preop_internals.shape, dtype=float)
        # intraop_surface_normals = data["intraop_surface_normals"]
        # intraop_internal_normals = np.zeros(intraop_internals.shape, dtype=float)
        intraop_normals = data["intraop_noisy_normals"]
        # print("preop_surface_normals.shape:", preop_surface_normals.shape)
        # print("preop_internal_normals.shape:", preop_internal_normals.shape)
        # print("intraop_surface_normals.shape:", intraop_surface_normals.shape)

        preop_surface_displ = data["preop_surface_displ"]
        preop_internals_displ = data["preop_internals_displ"]
        # print("preop_surface_displ.shape:", preop_surface_displ.shape)
        # print("preop_internals_displ.shape:", preop_internals_displ.shape)

        dist_internals_to_surface = vtk_utils.df(
            mesh=vtk_utils.point_cloud_to_poly(point_coords=preop_internals),
            surface=vtk_utils.point_cloud_to_poly(point_coords=preop_surface),
        )
        dist_internals_to_surface_np = np.expand_dims(numpy_support.vtk_to_numpy(dist_internals_to_surface), axis=1)
        dist_surface_to_surface_np = np.zeros([preop_surface.shape[0], 1], dtype=float)
        # dist_field_np = np.concatenate((dist_surface_to_surface_np, dist_internals_to_surface_np), axis=0)

        # if self.surface_only:
        npoints_internal = int( self.n_pre_points * self.ratio_pre_internals )
        npoints_surface = self.n_pre_points - npoints_internal
    
        assert npoints_internal + npoints_surface == self.n_pre_points

        # Preoperative (surface + internals)
        preop_surface_resampled, preop_surface_displ_resampled, preop_surface_normals_resampled, dist_surface_to_surface_np_resampled = pc_utils.set_npoints( 
            [preop_surface, preop_surface_displ, preop_surface_normals, dist_surface_to_surface_np], 
            npoints_surface,
        )
        preop_internals_resampled, preop_internals_displ_resampled, preop_internal_normals_resampled, dist_internals_to_surface_np_resampled = pc_utils.set_npoints( 
            [preop_internals, preop_internals_displ, preop_internal_normals, dist_internals_to_surface_np], 
            npoints_internal,
        )

        # Intraoperative (surface only)
        # intraop_surface_resampled, intraop_surface_normals_resampled, = pc_utils.set_npoints(
        #     [intraop_surface, intraop_surface_normals], 
        #     self.n_intra_points 
        # )

        # Intraoperative with internals
        intraop_resampled, intraop_normals_resampled = pc_utils.set_npoints(
            # [intraop_surface, intraop_surface_normals], 
            [intraop_noisy, intraop_normals],
            self.n_intra_points 
        )
        
        

        # print("preop_surface_resampled.shape:", preop_surface_resampled.shape)
        # print("preop_internals_resampled.shape:", preop_internals_resampled.shape)
        # # print("intraop_surface_resampled.shape:", intraop_surface_resampled.shape)
        # print("intraop_resampled.shape:", intraop_resampled.shape)
        # print("preop_surface_displ_resampled.shape:", preop_surface_displ_resampled.shape)
        # print("preop_internals_displ_resampled.shape:", preop_internals_displ_resampled.shape)
        # print("preop_surface_normals_resampled.shape:", preop_surface_normals_resampled.shape)
        # print("preop_internal_normals_resampled.shape:", preop_internal_normals_resampled.shape)
        # # print("intraop_surface_normals_resampled.shape:", intraop_surface_normals_resampled.shape)
        # # print("intraop_internal_normals_resampled.shape:", intraop_internal_normals_resampled.shape)
        # print("intraop_normals_resampled.shape:", intraop_normals_resampled.shape)
        
        # else:
        #     pass

        preop_coords = np.concatenate( (preop_surface_resampled, preop_internals_resampled), axis = 0 ).transpose(1, 0)
        # intraop_coords = intraop_surface_resampled.transpose(1, 0)
        intraop_coords = intraop_resampled.transpose(1, 0)
        preop_dist_field = np.concatenate( (dist_surface_to_surface_np_resampled, dist_internals_to_surface_np_resampled), axis = 0 ).transpose(1, 0)
        preop_normals = np.concatenate( (preop_surface_normals_resampled, preop_internal_normals_resampled), axis = 0 ).transpose(1, 0)
        # intraop_normals = intraop_surface_normals_resampled.transpose(1, 0)
        intraop_normals = intraop_normals_resampled.transpose(1, 0)
        preop_displ = np.concatenate( (preop_surface_displ_resampled, preop_internals_displ_resampled), axis = 0 ).transpose(1, 0)

        preop = np.concatenate([preop_coords, preop_normals, preop_dist_field], axis=0)
        intraop = np.concatenate([intraop_coords, intraop_normals], axis=0)

        # print("preop.shape", preop.shape)
        # print("intraop.shape", intraop.shape)
        # print("preop_displ.shape", preop_displ.shape)

        res = {
            "preop": preop.astype(np.float32),
            "intraop": intraop.astype(np.float32),
            "displ": preop_displ.astype(np.float32),
            # "dist_field": preop_dist_field,
            # "preop_normals": preop_normals,
            # "intraop_normals": intraop_normals,
        }

        return res




class EarSynOldDataset(Dataset):
    def __init__(self,
            folder,
            num,
            offset=0,
            data_filename = "data_cached.pkl",
            surface_only = False,
            n_pre_points=2500,
            n_intra_points=2500,
            ratio_pre_internals=0.3,
            pre_load=False,
        ) -> None:
        super().__init__()
        
        self.folder = folder
        self.num = num
        self.offset = offset
        self.data_filename = data_filename
        self.sample_folder_list = [i.replace('\\', '/') for i in sorted(glob('{}/??????'.format(self.folder)))[offset : offset+num]]
        self.surface_only = surface_only
        self.n_pre_points = n_pre_points    
        self.n_intra_points = n_intra_points
        self.ratio_pre_internals = ratio_pre_internals
        self.pre_load = pre_load
        self.data_list = []
        self.preop_normals = None
        if self.pre_load:
            self.data_list = self.__pre_load()


    def __len__(self):
        return self.num


    def get_normals(self, points, faces):
        print("generating normals...")
        poly = vtk_utils.create_poly_using_points_and_faces(
            coords=points,
            faces=faces,
        )
        # print(poly.GetNumberOfPoints())
        poly = vtk_utils.generate_point_normals(poly)
        self.preop_normals = numpy_support.vtk_to_numpy(poly.GetPointData().GetNormals())
        # print("self.preop_normals.shape:", self.preop_normals.shape)


    def __getitem__(self, index):
        sample_folder = self.sample_folder_list[index] 
        with open(os.path.join(sample_folder, self.data_filename), "rb") as f:
            data = pkl.load(f)
            f.close()
        
        if self.preop_normals is None:
            self.get_normals(
                points=data["points_pre"],
                faces=data["faces"],
            )

        # print(data.keys())
        # for k in data.keys():
        #     print(k, data[k].shape)

        preop_coords = data["points_pre"]
        intraop_coords = data["points_intra_noisy"]
        preop_normals = self.preop_normals
        intra_index = data["intra_inds"]
        # print("intra_index.shape:", intra_index.shape)
        intraop_normals = self.preop_normals[intra_index]
        displ = data["displacement"]

        # print("preop_coords.shape:", preop_coords.shape)
        # print("intraop_coords.shape:", intraop_coords.shape)
        # print("preop_normals.shape:", preop_normals.shape)
        # print("intraop_normals.shape:", intraop_normals.shape)

        dist_surface_to_surface_np = np.zeros([preop_coords.shape[0], 1], dtype=float)


        preop_coords, preop_normals, dist_surface_to_surface_np, displ = pc_utils.set_npoints(
            [preop_coords, preop_normals, dist_surface_to_surface_np, displ],
            self.n_pre_points,
        )
        intraop_coords, intraop_normals = pc_utils.set_npoints(
            [intraop_coords, intraop_normals],
            self.n_intra_points,
        )

        # print("preop_coords.shape:", preop_coords.shape)
        # print("preop_normals.shape:", preop_normals.shape)
        # print("dist_surface_to_surface_np.shape:", dist_surface_to_surface_np.shape)
        # print("intraop_coords.shape:", intraop_coords.shape)
        # print("intraop_normals.shape:", intraop_normals.shape)
        # print("displ.shape:", displ.shape)

        preop = np.concatenate([preop_coords, preop_normals, dist_surface_to_surface_np], axis=1).transpose(1, 0)
        intraop = np.concatenate([intraop_coords, intraop_normals], axis=1).transpose(1, 0)
        displ = displ.transpose(1, 0)

        # print("preop.shape", preop.shape)
        # print("intraop.shape", intraop.shape)
        # print("displ.shape", displ.shape)

        res = {
            "preop": preop.astype(np.float32),
            "intraop": intraop.astype(np.float32),
            "displ": displ.astype(np.float32),
        }

        return res
        





class EarDIOMEDataset(Dataset):
    def __init__(self,
            folder,
            start=0,
            num=43,
            # landmarks_folder="",
            diome_folder="/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/DIOME_FanShapeCorr/",
            # downsample=1,
            n_pre_points=2500,
            n_intra_points=2500,
            ratio_pre_internals=0.3,
            initial_align_icp=False,
            load_segments=False,
            load_landmarks=False,
            load_downsampled=True,
            flip_left=True,
            combined_filename="combined_no_promontory.stl",
            combined_downsampled_filename="combined_no_promontory_downsampled_with_normals.vtp", #"combined_no_promontory_downsampled.vtu",
            # template_model_path = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/D2EAR_data/SSM/TMOSS3/TMOSS3_template.stl",
            template_model_path="/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/D2EAR_data/SimulationData/data_100k_non-rigid_rigid/000000/data_cached_with_internals_uniform.pkl",
            initial_align_transformation = {
                "rotation": [0.5432588, 0.5019521, -0.45625, -0.4948601],
                "translation": [-10.27293, -20.97817, 17.2715],
            },
            debug=False,
        ):
        super().__init__()
        self.folder = folder
        self.num = num
        self.start=start
        self.diome_folder=diome_folder
        self.initial_align_icp = initial_align_icp
        self.load_segments = load_segments
        self.load_landmarks = load_landmarks
        self.load_downsampled=load_downsampled
        self.n_pre_points = n_pre_points
        self.n_intra_points = n_intra_points
        self.ratio_pre_internals = ratio_pre_internals
        self.flip_left = flip_left
        self.template_model_path = template_model_path
        # self.template_model_path_cached = template_model_path_cached
        self.combined_downsampled_filename = combined_downsampled_filename
        self.initial_align_transformation = initial_align_transformation

        self.segments_filename_list = [
            "1_Tympanic_Membrane.stl"  ,
            "2_Malleus.stl"  ,
            "3_Incus.stl"  ,
            "4_Stapes.stl"  ,
            "5_Promontory.stl",
        ]

        self.landmarks_filename_list = [
            "annulus.json",
            "umbo.json",
            "short_process_of_malleus.json",
            "malleus_handle.json",
            "long_process_of_incus.json",
            "incus.json",
        ]
        self.combined_filename = combined_filename
        self.preop, self.preop_landmark_list = self.__load_source_model()
        self.debug = debug
    

    def __len__(self): 
        return self.num

    def __load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        return metadata

    def __load_source_model(self,):
        # TODO this only load the surface points, internal points needed....
        # template_model = vtk_utils.load_mesh(self.template_model_path)
        if self.template_model_path.endswith(".pkl"):
            with open(self.template_model_path, "rb") as f:
                data = pickle.load(f)
                f.close()
            preop_surface = data["preop_surface"]
            # preop_model = vtk_utils.load_mesh(self.template_model_path)
            # preop_surface = numpy_support.vtk_to_numpy(preop_model.GetPoints().GetData())
            # preop_model = vtk_utils.generate_point_normals(preop_model)
            # preop_normals = numpy_support.vtk_to_numpy(preop_model.GetPointData().GetNormals())

            # preop = np.concatenate([preop_surface, preop_normals], axis=1).transpose(1, 0)
            preop_internals = data["preop_internals"]
            preop_surface_normals = data["preop_surface_normals"]
            preop_internal_normals = np.zeros(preop_internals.shape, dtype=float)

            dist_internals_to_surface = vtk_utils.df(
                mesh=vtk_utils.point_cloud_to_poly(point_coords=preop_internals),
                surface=vtk_utils.point_cloud_to_poly(point_coords=preop_surface),
            )
            dist_internals_to_surface_np = np.expand_dims(numpy_support.vtk_to_numpy(dist_internals_to_surface), axis=1)
            dist_surface_to_surface_np = np.zeros([preop_surface.shape[0], 1], dtype=float)
            # dist_field_np = np.concatenate((dist_surface_to_surface_np, dist_internals_to_surface_np), axis=0)

            # if self.surface_only:
            npoints_internal = int( self.n_pre_points * self.ratio_pre_internals )
            npoints_surface = self.n_pre_points - npoints_internal
        
            assert npoints_internal + npoints_surface == self.n_pre_points

            # Preoperative (surface + internals)
            preop_surface_resampled, preop_surface_normals_resampled, dist_surface_to_surface_np_resampled = pc_utils.set_npoints( 
                [preop_surface, preop_surface_normals, dist_surface_to_surface_np], 
                npoints_surface,
            )
            preop_internals_resampled, preop_internal_normals_resampled, dist_internals_to_surface_np_resampled = pc_utils.set_npoints( 
                [preop_internals, preop_internal_normals, dist_internals_to_surface_np], 
                npoints_internal,
            )

            # test generating normals using downsampled preop_surface
            # preop_surface_resampled_poly = vtk_utils.to_pointcloud(
            #     coords=numpy_support.numpy_to_vtk(preop_surface_resampled),
            
            # )
            # print("preop_surface_resampled_poly.GetNumberOfPoints()", preop_surface_resampled_poly.GetNumberOfPoints())
            # preop_surface_resampled_poly = vtk_utils.generate_point_normals(preop_surface_resampled_poly)
            # print("preop_surface_resampled_poly.GetNumberOfPoints()", preop_surface_resampled_poly.GetNumberOfPoints())
            # print(preop_surface_resampled_poly.GetPointData().HasArray("Normals"))
            # preop_surface_normals_resampled = numpy_support.vtk_to_numpy(preop_surface_resampled_poly.GetPointData().GetNormals())
            
            # save surface and normals for debugging
            # preop_downsampled_with_normals_mesh = vtk_utils.to_pointcloud(
            #     coords=numpy_support.numpy_to_vtk(preop_surface_resampled),
            #     features=preop_surface_normals_resampled,
            #     features_name="Normals",
            # )
            # output_path = os.path.join("/home/liupeng/ceph_home/others/debug/test_preop_downsampled_with_normals.vtp")
            # vtk_utils.write_mesh(
            #     preop_downsampled_with_normals_mesh,
            #     output_path,
            # )
            # print("Saved", output_path)


            preop_coords = np.concatenate( (preop_surface_resampled, preop_internals_resampled), axis = 0 ).transpose(1, 0)
            preop_dist_field = np.concatenate( (dist_surface_to_surface_np_resampled, dist_internals_to_surface_np_resampled), axis = 0 ).transpose(1, 0)
            preop_normals = np.concatenate( (preop_surface_normals_resampled, preop_internal_normals_resampled), axis = 0 ).transpose(1, 0)
            preop = np.concatenate([preop_coords, preop_normals, preop_dist_field], axis=0)
        elif self.template_model_path.endswith(".stl"):
            preop_model = vtk_utils.load_mesh(self.template_model_path)
            preop_coords = numpy_support.vtk_to_numpy(preop_model.GetPoints().GetData())
            preop_model = vtk_utils.generate_point_normals(preop_model)
            preop_normals = numpy_support.vtk_to_numpy(preop_model.GetPointData().GetNormals())
            preop = np.concatenate([preop_coords, preop_normals], axis=1).transpose(1, 0)

        landmarks_folder = "/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/D2EAR_data/SSM/TMOSS3"
        filename_list = [
            "landmark_anulus.ply",
            "landmark_umbo.ply",
            "landmark_lateral_process.vtu",
            "landmark_malleus.ply",
            "landmark_incus.ply",
            # "landmark_stapes.ply",
            "landmark_stapes_0.vtu",
        ]
        landmarks_list = []
        for f in filename_list:
            cur_path = os.path.join(landmarks_folder, f)
            landmark_coords = numpy_support.vtk_to_numpy(vtk_utils.load_mesh(cur_path).GetPoints().GetData())
            landmarks_list.append(landmark_coords.astype(np.float32))
        # return template_model_coords, landmarks_list, template_model_normals

        # Find the closest points on preop to the landmarks


        return preop, landmarks_list


    def __load_landmarks(self, landmarks_folder):
        assert os.path.exists(landmarks_folder) == True, "Landmarks folder does not exist:{}".format(landmarks_folder)
        landmarks_list = []
        for filename in self.landmarks_filename_list:
            # print(os.path.join(landmarks_folder, filename))
            if os.path.exists(os.path.join(landmarks_folder, filename)):
                # print("File exists: ", os.path.join(landmarks_folder, filename))
                with open(os.path.join(landmarks_folder, filename), 'r') as f:
                    landmarks_json = json.load(f)
                    # landmarks_json_list.append(json.load(f))
                    landmarks_points = [ c["position"]  for c in landmarks_json["markups"][0]["controlPoints"]]
                    # print(filename, len(landmarks_points))
                # landmarks_list.append(landmarks_points)
                landmarks_list.append(np.array(landmarks_points, dtype=np.float32))
            else:
                landmarks_list.append([])
        return landmarks_list   


    def __initial_rigid_alignment(self, source, target, iter=100, threshold=1e-6):
        # Not working so well...
        # if using complete template and partial target, ICP cannot find correct correspondence
        # if using landmarks, the number of lanmark points are not balanced, still fall into
        # local minima.
        # consider either do it manually or use weighted ICP.
        source_pc = PointCloud( {
            "x":source[:, 0], 
            "y":source[:, 1], 
            "z":source[:, 2],
        })
        target_pc = PointCloud( {
            "x":target[:, 0], 
            "y":target[:, 1], 
            "z":target[:, 2],
        })

        icp = SimpleICP()
        icp.add_point_clouds( target_pc, source_pc )
        H, X_mov_transformed, rigid_body_transformation_params, distance_residuals = icp.run()

        print(H)
        source = np.concatenate([source, np.ones((source.shape[0], 1))], axis=1).transpose(1, 0)
        source = np.matmul(H, source)
        source_transformed = source[0:3, :].transpose(1, 0)

        return H, source_transformed


    def __getitem__(self, index):
        cur_dir = os.path.join(self.folder, "sample_{}".format(index))
        # print("loading", cur_dir)
        if self.load_downsampled:
            # print("Loading downsampled model...")
            model_combined_path = os.path.join(cur_dir, self.combined_downsampled_filename)
        else:
            model_combined_path = os.path.join(cur_dir, self.combined_filename )
        intraop_model = vtk_utils.load_mesh(model_combined_path)
    
        intraop_coords = numpy_support.vtk_to_numpy(intraop_model.GetPoints().GetData())
        intraop_normals = numpy_support.vtk_to_numpy(intraop_model.GetPointData().GetNormals())
        
        
                
        segment_list = {}
        if self.load_segments:
            for segment_filename in self.segments_filename_list:
                segment_list[segment_filename] = None,
                segment_path = os.path.join(cur_dir, segment_filename)
                # print(segment_path)
                if os.path.exists(segment_path):
                    print("loading", segment_filename)
                    segment = vtk_utils.load_mesh(segment_path)
                    segment_coords = numpy_support.vtk_to_numpy(segment.GetPoints().GetData())
                    segment_list[segment_filename] = segment_coords

        landmark_list = []
        if self.load_landmarks:
            # print("Loading landmarks..")
            landmarks_folder=os.path.join(self.diome_folder, "sample_{}".format(index), "annotations/merged/landmarks")
            if index > 37:
                landmarks_folder=os.path.join(self.diome_folder, "sample_{}".format(index), "annotations/landmarks")
            landmark_list = self.__load_landmarks(
                landmarks_folder=landmarks_folder,
            )
            # print(len(landmark_list))

        meta_path = os.path.join(self.diome_folder, "sample_{}".format(index), 'meta_{}.yaml'.format(index))
        metadata = self.__load_metadata(meta_path)
        if self.flip_left and metadata["patient_info"]["side"] == "left":
            print("Flipping left side...")
            intraop_coords[:, 0] = -intraop_coords[:, 0]
            intraop_normals[:, 0] = -intraop_normals[:, 0]
            intraop_coords = intraop_coords + [15, 0, 0] # plus the size of the volume
            if landmark_list is not None:
                for idx_l in range(len(landmark_list)):
                    if len(landmark_list[idx_l]) > 0:
                        landmark_list[idx_l][:, 0] = -landmark_list[idx_l][:, 0]
                        landmark_list[idx_l] = landmark_list[idx_l] + [15, 0, 0] # plus the size of the volume

        H, intraop_landmarks_transformed = [], []
        # if self.initial_align_icp:
        #     # source_landmarks_np = np.concatenate(self.template_model_landmarks, axis=0)
        #     # target_landmarks_np = np.concatenate(landmark_list, axis=0)
        #     target_landmarks_np = np.concatenate(self.preop_landmark_list, axis=0)
        #     source_landmarks_np = np.concatenate(landmark_list, axis=0)
        #     H, intraop_landmarks_transformed = self.__initial_rigid_alignment(
        #         source = source_landmarks_np,
        #         target = target_landmarks_np,
        #     )
        #     print("intraop_landmarks_transformed.shape", intraop_landmarks_transformed.shape)
        #     if self.debug:
        #         intraop_landmarks_transformed_poly = vtk_utils.to_pointcloud(
        #             coords=numpy_support.numpy_to_vtk(intraop_landmarks_transformed),
        #         )
        #         output_path_debug = "/home/liupeng/ceph_home/others/debug/test_transformed_landmarks.ply"
        #         vtk_utils.write_mesh(intraop_landmarks_transformed_poly, output_path_debug)
        #         print("Saved to", output_path_debug)


        if self.initial_align_transformation is not None:
            rotation = self.initial_align_transformation["rotation"]
            rotation = scipy.spatial.transform.Rotation.from_quat(rotation)
            translation = self.initial_align_transformation["translation"]
            translation = np.array(translation)
            translation = np.expand_dims(translation, axis=0)
            intraop_coords = rotation.apply(intraop_coords) + translation
            intraop_normals = rotation.apply(intraop_normals) 
            if landmark_list is not None:
                for idx_l in range(len(landmark_list)):
                    if len(landmark_list[idx_l]) > 0:
                        landmark_list[idx_l] = rotation.apply(landmark_list[idx_l]) + translation

        intraop_resampled, intraop_normals_resampled = pc_utils.set_npoints(
                [intraop_coords, intraop_normals],
                self.n_intra_points 
        )
        intraop = np.concatenate([intraop_resampled, intraop_normals_resampled], axis=1).transpose(1, 0)


        res = {
            "preop": self.preop.astype(np.float32),
            "preop_landmarks": self.preop_landmark_list,
            "intraop": intraop.astype(np.float32),
            "intraop_landmarks": landmark_list,
            "initial_align_H": H,
            "intraop_landmarks_transformed": intraop_landmarks_transformed,
            "segments": segment_list,
            "meta": metadata,
            "is_flipped": self.flip_left and metadata["patient_info"]["side"] == "left",
            "idx": index,
        }

        return res




if __name__ == "__main__":
    import torch
    # ear_dataset = EarSynDataset(
    #     folder="/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/D2EAR_data/SimulationData/data_100k_non-rigid_rigid/",
    #     offset=0, 
    #     num=10,
    # )

    # ear_dataset = EarSynOldDataset(
    #     folder="/mnt/ceph/tco/TCO-Staff/Homes/liupeng/data/D2EAR_data/SimulationData/data_100k_non-rigid_rigid/",
    #     offset=0, 
    #     num=10,
    # )

    # model_path = ""

    import random
    random.seed(1234)

    ear_dataset = EarDIOMEDataset(
        folder="/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/",
        # num=3,
        # start=0,
        # load_segments=True,
        load_landmarks=True,
        # initial_align=True,
        flip_left=True,
        debug=True,
    )

    # print(len(ear_dataset))

    # for idx, data in enumerate(ear_dataset):
    #     print("============", idx)
    #     print(data["intraop"].shape)
    #     print("is_flipped", data["is_flipped"])

    # output_folder_flipped = "/tmp/test_flipped"
    # if not os.path.exists(output_folder_flipped):
    #     os.makedirs(output_folder_flipped)

    for idx in range(len(ear_dataset)):
    # for idx in range(0, len(ear_dataset)):
        print("============", idx)
        data = ear_dataset[idx]
        for k in data.keys():
            if isinstance(data[k], list):
                print(k, len(data[k]))
            elif data[k] is None:
                print(k, "None")
            elif isinstance(data[k], np.ndarray):
                print(k, data[k].shape)
        # print(data["intraop"].shape)
        # print("is_flipped", data["is_flipped"])

        # if data["is_flipped"]:
        #     intraop = data["intraop"]
        #     intraop_poly = vtk_utils.to_pointcloud(
        #         coords=numpy_support.numpy_to_vtk(intraop),
        #     )
        #     output_path = os.path.join("/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/", "test_mirror_transformed_{}.vtp".format(idx))
        #     vtk_utils.write_mesh(
        #         mesh=intraop_poly,
        #         filename=output_path,
        #     )
        #     print("Saved to", output_path)

        preop = data["preop"][:3, ...].transpose(1, 0)
        preop_normals = data["preop"][3:6, ...].transpose(1, 0)
        output_path = os.path.join("/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/sample_{}".format(idx), "test_preop_{}.vtp".format(idx))
        preop_poly = vtk_utils.to_pointcloud(
            coords=numpy_support.numpy_to_vtk(preop),
            features=preop_normals,
            features_name="Normals",
        )
        vtk_utils.write_mesh(
            mesh=preop_poly,
            filename=output_path,
        )
        print("Saved to", output_path)


        intraop = data["intraop"][:3, ...].transpose(1, 0)
        intraop_normals = data["intraop"][3:, ...].transpose(1, 0)
        output_path = os.path.join("/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/sample_{}".format(idx), "test_init_align_{}.vtp".format(idx))
        intraop_poly = vtk_utils.to_pointcloud(
            coords=numpy_support.numpy_to_vtk(intraop),
            features=intraop_normals,
            features_name="Normals",
        )
        vtk_utils.write_mesh(
            mesh=intraop_poly,
            filename=output_path,
        )
        print("Saved to", output_path)


        # Save landmarks:
        preop_landmarks = data["preop_landmarks"]
        intraop_landmarks = data["intraop_landmarks"]
        for idx_l, landmark in enumerate(preop_landmarks):
            if landmark is not None and landmark is not []:
                output_path = os.path.join("/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/sample_{}".format(idx), "preop_landmark_{}_{}.vtp".format(idx, idx_l))
                landmark_poly = vtk_utils.to_pointcloud(
                    coords=numpy_support.numpy_to_vtk(landmark),
                )
                vtk_utils.write_mesh(
                    mesh=landmark_poly,
                    filename=output_path,
                )
                print("Saved to", output_path)
        
        for idx_l, landmark in enumerate(intraop_landmarks):
            if landmark is not None and landmark is not []:
                output_path = os.path.join("/mnt/ceph/tco/TCO-All/SharedDatasets/DIOME/3D_Models/sample_{}".format(idx), "intraop_landmark_{}_{}.vtp".format(idx, idx_l))
                landmark_poly = vtk_utils.to_pointcloud(
                    coords=numpy_support.numpy_to_vtk(landmark),
                )
                vtk_utils.write_mesh(
                    mesh=landmark_poly,
                    filename=output_path,
                )
                print("Saved to", output_path)


    # res = ear_dataset[1]
    # print(res.keys())
    # print(res["meta"]["patient_info"]["side"])
    # res = ear_dataset[1]
    # for k in res.keys():
    #     print(k, res[k].shape)

    # preop = torch.FloatTensor(res["preop"]).unsqueeze(0).to("cuda")
    # intraop = torch.FloatTensor(res["intra"]).unsqueeze(0).to("cuda")
    # displ = torch.FloatTensor(res["displ"]).unsqueeze(0).to("cuda")

    # print(preop.shape)
    # print(intraop.shape)
    # print(displ.shape)

    # coords_preop, features_preop, coords_intraop, features_intraop = pc_utils.preprocess_data(preop, intraop, )

    # print("coords_preop.shape", coords_preop.shape)
    # print("features_preop.shape", features_preop.shape)
    # print("coords_intraop.shape", coords_intraop.shape)
    # print("features_intraop.shape", features_intraop.shape)


    