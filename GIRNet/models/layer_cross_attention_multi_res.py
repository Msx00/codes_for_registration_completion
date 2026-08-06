import torch
import torch.nn as nn

from models.attention import MultiRegionAttention, MultiRegionAgentAttention
from models.attention_variants import MultiRegionAttention as MultiRegionAttentionForOldModels
from models.k_nearest_neighbors import k_nearest_neighbors
from models.radius_nearest_neighbors import radius_nearest_neighbors
from models.select import select_point_regions

class LayerCrossAttentionMultiResolution(nn.Module):
    """An cross attention layer which perform attention from source preoperative points to 
    target intraperative points in different resolutions
    
    """

    def __init__( self,
            n_kneighbors,
            n_query_features,
            n_value_features_list,
            n_output_features,
            radius = None,
            embedding_size=64,
            use_relative_coords = False,
            use_positional_encoding = True,
            n_attention_modules = 1,
            attention_type = 0,
            attention_mode = "additive"
            ) -> None:
        nn.Module.__init__( self )

        if attention_type == 0:
            attention = MultiRegionAttention
        elif attention_type == 1:
            attention = MultiRegionAgentAttention
        elif attention_type == 2:
            attention = MultiRegionAttentionForOldModels
        else:
            raise ValueError("Unknown attention mode: {}".format(attention_mode))
        
        self.att_list = nn.ModuleList()
        for idx_att, n_value_features in enumerate( n_value_features_list ):
            # att = MultiRegionAttention(
            att = attention(
                n_value_features = n_value_features,
                n_query_features = n_query_features,
                n_output_features = n_output_features,
                embedding_size = embedding_size,
                concat_absolute_coords_raw = True,
                concat_absolute_coords_encoded = use_positional_encoding,
                concat_relative_coords_raw = use_relative_coords,
                concat_relative_coords_encoded = use_relative_coords and use_positional_encoding,
                concat_queries_to_values = True,
                attention_mode = attention_mode,
                )
            self.att_list.append( att )

        self.n_query_features = n_query_features
        self.n_value_features_list = n_value_features_list
        self.n_kneighbors = n_kneighbors
        self.radius = radius

        self.conv_out = nn.Conv1d( n_output_features * len(n_value_features_list), n_output_features, kernel_size=1)

        #self.conv = nn.Conv1d( n_output_features, n_output_features, kernel_size=1 )
        self.non_lin = nn.LeakyReLU()

    def forward(self, value_coords_list, value_features_list, query_coords, query_features, ): #low_res_features_skipconn=None):
        # if low_res_features_skipconn is not None:
        #     # print("Concatenating skip connection...")
        #     assert low_res_features_skipconn.shape[2] == low_res_coords.shape[2], "number of points not matching, {} != {}".format(low_res_features_skipconn.shape[2], low_res_coords.shape[2])
        #     low_res_features = torch.cat( [low_res_features, low_res_features_skipconn], dim=1 )
        # Determine how many points to actually look up in the k nearest neighbor search.
        # If the number of available points N is lower than the chosen k, only choose
        # N points:

        new_query_features = []
        for idx_val in range(len(self.n_value_features_list)):
            N = value_coords_list[idx_val].shape[2]
            k = min(N, self.n_kneighbors)

            if self.radius:
                # Select all points within a radius:
                value_coords_grouped, idx_grouped = radius_nearest_neighbors(
                    pos_source = value_coords_list[idx_val],
                    pos_queries = query_coords,
                    radius = self.radius,
                    k = k)
            else:
                # Select the k nearest neighbors in low_res_coords for every point in high_res_coords.
                # This selects the "region" of low-res coords around every high-res (output) coord
                value_coords_grouped, idx_grouped = k_nearest_neighbors(
                    pos_source = value_coords_list[idx_val],
                    pos_queries = query_coords,
                    k = k)

            value_features_grouped = select_point_regions( value_features_list[idx_val], idx_grouped )

            # Use attention to compute an aggregation for every high-res point, using values from the
            # low-res points as input

            # for att in self.att_list:
            f = self.att_list[idx_val]( 
                value_coords = value_coords_grouped, 
                value_features = value_features_grouped, 
                queries_coords = query_coords,
                queries_features = query_features, 
            )
            new_query_features.append( f )

        #new_high_res_features = self.conv( new_high_res_features )
        new_query_features = torch.cat( new_query_features, dim=1 )
        new_query_features = self.non_lin( self.conv_out( new_query_features ) )

        return new_query_features
 
