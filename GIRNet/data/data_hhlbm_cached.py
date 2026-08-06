import pickle as pkl
import numpy as np
import torch
import os
import json



class HHLBMDatasetCached():
    def __init__(self,
            cache_path,
        ) -> None:
        self.cache_path = cache_path
        if os.path.isdir(cache_path):
            # try to find all pkl files in the directory
            pkl_file_list = [os.path.join(cache_path, f) for f in sorted(os.listdir(cache_path)) if f.endswith('.pkl')]
            assert len(pkl_file_list) > 0, f"no pkl files found in {cache_path}"
            self.data = []
            for pkl_file in pkl_file_list:
                print(f"loading {pkl_file}")
                self.data.extend(pkl.load(open(pkl_file, "rb")))
        else:
            self.data = pkl.load(open(self.cache_path, "rb"))
    
    def __len__(self):
        return len(self.data)
    
    # def center(self, points):
    #     centroid = np.mean(points[:, :3], axis=0)
    #     points_centered = points[:, :3] - centroid
    #     return points_centered
    
    # def center_w_offset(self, points, offset):
    #     points_centered = points[:, :3] - offset
    #     return points_centered


    def __getitem__(self, idx):
        return self.data[idx]

