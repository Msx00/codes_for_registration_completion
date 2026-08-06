import torch
import torch.nn as nn

from models.attention import MultiRegionAttention, MultiRegionAgentAttention
from models.attention_variants import MultiRegionAttention as MultiRegionAttentionForOldModels
from models.attention_variants import MultiRegionAttentionMHA
from models.k_nearest_neighbors import k_nearest_neighbors
from models.radius_nearest_neighbors import radius_nearest_neighbors
from models.select import select_point_regions

class LayerUpsampleMHA(nn.Module):
    """An upsample layer which upscales info from a low resolution to a (given) higher resolution
    with ordinary Multi-Head Attention
    
    """

    def __init__( self,
            n_kneighbors,
            n_low_res_features,
            n_high_res_features,
            n_output_features,
            radius = None,
            embedding_size=64,
            use_relative_coords = False,
            use_positional_encoding = True,
            n_attention_modules = 2,
            n_attention_heads = 4,
            attention_type = 0,
            attention_mode = "additive"
            ) -> None:
        nn.Module.__init__( self )

        # if attention_type == 0:
        #     attention = MultiRegionAttention
        # elif attention_type == 1:
        #     attention = MultiRegionAgentAttention
        # elif attention_type == 2:
        #     attention = MultiRegionAttentionForOldModels
        # else:
        #     raise ValueError("Unknown attention mode: {}".format(attention_mode))
        attention = MultiRegionAttentionMHA

        self.att = nn.ModuleList()
        for i in range( n_attention_modules ):
            att = attention(
                n_value_features = n_low_res_features,
                n_query_features = n_high_res_features,
                n_output_features = n_output_features,
                embedding_size = embedding_size,
                concat_absolute_coords_raw = True,
                concat_absolute_coords_encoded = use_positional_encoding,
                concat_relative_coords_raw = use_relative_coords,
                concat_relative_coords_encoded = use_relative_coords and use_positional_encoding,
                concat_queries_to_values = True,
                attention_mode = attention_mode,
                n_head=n_attention_heads,
            )
            self.att.append( att )

        self.n_kneighbors = n_kneighbors
        self.radius = radius

        self.conv_out = nn.Conv1d( n_output_features * n_attention_modules, n_output_features, kernel_size=1)

        #self.conv = nn.Conv1d( n_output_features, n_output_features, kernel_size=1 )
        self.non_lin = nn.LeakyReLU()

    def forward(self, low_res_coords, low_res_features, high_res_coords, high_res_features, low_res_features_skipconn=None):
        if low_res_features_skipconn is not None:
            # print("Concatenating skip connection...")
            assert low_res_features_skipconn.shape[2] == low_res_coords.shape[2], "number of points not matching, {} != {}".format(low_res_features_skipconn.shape[2], low_res_coords.shape[2])
            low_res_features = torch.cat( [low_res_features, low_res_features_skipconn], dim=1 )
        # Determine how many points to actually look up in the k nearest neighbor search.
        # If the number of available points N is lower than the chosen k, only choose
        # N points:
        N = low_res_coords.shape[2]
        k = min(N, self.n_kneighbors)

        if self.radius:
            # Select all points within a radius:
            low_res_coords_grouped, idx_grouped = radius_nearest_neighbors(
                pos_source = low_res_coords,
                pos_queries = high_res_coords,
                radius = self.radius,
                k = k)
        else:
            # Select the k nearest neighbors in low_res_coords for every point in high_res_coords.
            # This selects the "region" of low-res coords around every high-res (output) coord
            low_res_coords_grouped, idx_grouped = k_nearest_neighbors(
                pos_source = low_res_coords,
                pos_queries = high_res_coords,
                k = k)

        low_res_features_grouped = select_point_regions( low_res_features, idx_grouped )

        # Use attention to compute an aggregation for every high-res point, using values from the
        # low-res points as input

        new_high_res_features = []
        for att in self.att:
            f = att( 
                value_coords = low_res_coords_grouped, 
                value_features = low_res_features_grouped, 
                queries_coords = high_res_coords,
                queries_features = high_res_features, 
            )
            new_high_res_features.append( f )
        # new_high_res_features = self.att( 
        #     value_coords = low_res_coords_grouped, 
        #     value_features = low_res_features_grouped, 
        #     queries_coords = high_res_coords,
        #     queries_features = high_res_features, 
        # )

        #new_high_res_features = self.conv( new_high_res_features )
        new_high_res_features = torch.cat( new_high_res_features, dim=1 )
        new_high_res_features = self.non_lin( self.conv_out( new_high_res_features ) )

        return new_high_res_features
 
