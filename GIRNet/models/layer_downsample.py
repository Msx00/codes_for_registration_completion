import torch
import torch.nn as nn
from models.attention import MultiRegionAttention
from models.farthest_point_sampling import farthest_point_sampling
from models.k_nearest_neighbors import k_nearest_neighbors
from models.radius_nearest_neighbors import radius_nearest_neighbors
from models.select import select_points, select_point_regions
from data.plot import plot_pointcloud_2d

class LayerDownsample(nn.Module):
    """A downsample layer which takes the point coordinates and features as input, first
    group the points, then apply attention for the grouped/downsampled points. 
    """

    def __init__(self,
            n_kneighbors,
            n_high_res_features,
            n_low_res_features,
            n_output_features,
            embedding_size=64,
            radius = 0.1,
            concat_coords=True,
            use_relative_coords = True,
            num_attention_modules = 1,
            randomize_subsampling = True,
            n_kernels=0,
            ) -> None:
        super().__init__()

        self.n_kernels = n_kernels
        self.n_kneighbors = n_kneighbors
        self.radius = radius
        self.n_output_features = n_output_features
        self.n_high_res_featurs = n_high_res_features
        self.n_low_res_features = n_low_res_features
        self.randomize_subsampling = randomize_subsampling

        self.att = nn.ModuleList()
        for i in range( num_attention_modules ):
            att = MultiRegionAttention(
                n_value_features=self.n_high_res_featurs,
                n_query_features=self.n_low_res_features,
                n_output_features=self.n_output_features,
                embedding_size=embedding_size,
                concat_absolute_coords_raw = concat_coords,
                concat_absolute_coords_encoded = concat_coords,
                concat_relative_coords_raw = use_relative_coords and concat_coords,
                concat_relative_coords_encoded = use_relative_coords and concat_coords,
                concat_queries_to_values = True,
            )
            self.att.append( att )

        self.conv_in = nn.Conv1d( self.n_high_res_featurs, self.n_output_features, kernel_size=1)
        #self.max_pool_1d = nn.AdaptiveMaxPool1d(output_size=num_output_features)

        # f_out = (self.n_output_features) * num_attention_modules + self.n_output_features
        f_out = self.n_output_features * num_attention_modules
        self.conv_out = nn.Conv1d( f_out, self.n_output_features, kernel_size=1)
        self.non_lin = nn.LeakyReLU()


    def forward(self, coords, features, coords_low_res=None, features_low_res=None):
        return_coords = False
        if coords_low_res is None or features_low_res is None:
            # sample the input point clouds, 
            assert self.n_kernels > 0, "If no low res point cloud is given, the number of kernels must be > 0"
            idx_coords_low_res, coords_low_res = farthest_point_sampling(
                points=coords, 
                num_samples=self.n_kernels,
                random = self.randomize_subsampling 
            )
            features_low_res = select_points(
                points=features, 
                idx=idx_coords_low_res,
            )
            return_coords = True

        # group the high resolution points
        coords_high_res, idx_grouped = k_nearest_neighbors(
                pos_source=coords,
                pos_queries=coords_low_res,
                k=self.n_kneighbors,
        )
        #coords_grouped, idx_grouped = radius_nearest_neighbors(pos_source=coords, pos_queries=coords_kernels, k=self.num_kneighbors, radius = self.radius)
        features_high_res = select_point_regions(features, idx_grouped)
        #for i in range(self.num_kernels):
        #    plot_pointcloud_2d( coords_grouped[0,:,:,i], features_grouped[0,:,:,i] )

        features_out_att = []
        for att in self.att:
            f = att( 
                value_coords=coords_high_res, #coords_grouped, 
                value_features=features_high_res, #features_grouped, 
                queries_coords=coords_low_res, 
                queries_features=features_low_res, 
            )
            features_out_att.append( f )

        ##### Additionally add a simple network inspired by point net which computes a convolution and then
        # max pooling for each region:
        # B, F, Nr, Nq = features_high_res.shape
        # features_high_res = features_high_res.view( B, F, Nr*Nq )
        # features_mlp = self.non_lin( self.conv_in( features_high_res ) )
        # features_mlp = features_mlp.view( B, self.n_output_features, Nr, Nq )
        # features_out_pool = torch.max( features_mlp, dim=2 )[1]
        # features_out = torch.cat( features_out_att + [features_out_pool], dim=1 )
        # features_out = self.non_lin( self.conv_out( features_out ) )

        features_out = torch.cat( features_out_att, dim=1 )
        features_out = self.non_lin( self.conv_out( features_out ) )

        if return_coords:
            return coords_low_res, features_out, idx_coords_low_res
        else:
            return features_out


