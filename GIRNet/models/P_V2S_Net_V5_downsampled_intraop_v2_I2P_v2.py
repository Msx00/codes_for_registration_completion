import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layer_upsample import LayerUpsample
from models.layer_cross_attention import LayerCrossAttention
from models.k_nearest_neighbors import k_nearest_neighbors
from models.select import select_point_regions, select_points
from models.attention import MultiRegionAttention, GroupedAttention, FusionGate
from models.preprocessor import TensorPreprocessor

from models.farthest_point_sampling import farthest_point_sampling


class PV2SNetV5DownsampledIntraopV2I2PV2(nn.Module):
    """P-V2S-Net V5:
    intraoperative points first cross attend to preoperative points, then preoperative points cross attend to intraoperative points.
    I2P V2 refers to intraoperative upsampling happens after the cross attention from intraop to preop (similar to preoperative branch 
    in the previous P-V2S-Net V5 model).

    """
    def __init__( self, 
                n_input_features, 
                # n_input_features_internals,
                n_preprocess_features, 
                n_intermediate_features,
                n_intermediate_points, 
                n_output_features, 
                embedding_size, 
                points_per_region, 
                enc_freq = [2e-2, 2e-1, 2, 4, 8, 16, 32, 64], # ***preprocessor arguments start***
                enc_freq_scale=1, 
                append_df_self=True,
                append_df_cross=True,
                append_positional_encoding=True,
                adapt_to_old_models=False,
                compact_return=True,   # ***preprocessor arguments end***
                # use_internals = True, 
        ):
        nn.Module.__init__( self )

        print("Building {} with parameters:".format(self.__class__.__name__))
        print("\tn_input_features", n_input_features)
        # if use_internals:
        #     print("\tn_input_features_internals", n_input_features_internals)
        # else:
        #     print("\tn_input_features_internals", n_input_features_internals, "(disabled)")
        print("\tn_preprocess_features", n_preprocess_features)
        print("\tn_intermediate_features", n_intermediate_features)
        print("\tn_intermediate_points", n_intermediate_points)
        print("\tn_output_features", n_output_features)
        print("\tembedding_size", embedding_size)
        print("\tpoints_per_region", points_per_region)
        # print("\tuse_internals", use_internals)

        self.n_intermediate_points = n_intermediate_points
        self.non_lin = nn.ReLU()
        self.points_per_region = points_per_region
        # self.use_internals = use_internals

        ########################
        ## Create Preprocessor
        self.preprocessor = TensorPreprocessor(
            enc_freq =enc_freq, 
            enc_freq_scale=enc_freq_scale, 
            append_df_self=append_df_self,
            append_df_cross=append_df_cross,
            append_positional_encoding=append_positional_encoding,
            adapt_to_old_models=adapt_to_old_models,
            compact_return=compact_return, # wether return concatenated features or a list of separate features [df_self, df_cross, pos_enc, input_features]
        )

        # self.n_tensorpreprocess_features = self.preprocessor.num_output_features
        # print("self.n_tensorpreprocess_features:", self.n_tensorpreprocess_features )

        # n_input_features += self.n_tensorpreprocess_features

        ########################
        ## Self-attention on each input (preop and intraop):
        self._init_self_attention(embedding_size, n_input_features, n_preprocess_features)

        ########################
        ## Build a branch of the network which processes the internal structures separately.
        ## When executed, this branch will output an estimated displacement field for the preop features and 
        # if use_internals:
        #     self.internals_processor = InternalsProcessor( n_input_features_internals, n_preprocess_features,
        #             n_intermediate_features, n_output_features, embedding_size, points_per_region )


        ########################
        ## Simple processing of the intraop. surface features between stages:
        # self.level_0_processing = nn.Sequential(self.non_lin,
        #                           nn.Conv1d(n_preprocess_features, n_intermediate_features[0], kernel_size=1))
        # self.level_1_processing = nn.Sequential(self.non_lin,
        #                           nn.Conv1d(n_intermediate_features[0], n_intermediate_features[1], kernel_size=1))
        # self.level_2_processing = nn.Sequential(self.non_lin,
        #                           nn.Conv1d(n_intermediate_features[1], n_intermediate_features[2], kernel_size=1))
        # self.level_3_processing = nn.Sequential(self.non_lin,
        #                           nn.Conv1d(n_intermediate_features[2], n_intermediate_features[3], kernel_size=1))
        # self.level_4_processing = nn.Sequential(self.non_lin,
        #                           nn.Conv1d(n_intermediate_features[3], n_intermediate_features[4], kernel_size=1))
        # self.level_5_processing = nn.Sequential(self.non_lin,
        #                           nn.Conv1d(n_intermediate_features[4], n_intermediate_features[5], kernel_size=1))
        # _, self.conv_intraop_0 = self._init_intraop_propagation_layer(
        #     n_value_features=n_preprocess_features,
        #     n_query_features=n_preprocess_features,
        #     embedding_size=embedding_size,
        #     n_output_features=n_intermediate_features[5],
        #     highest_res=True,
        # )
        # self.att_intraop_1, self.conv_intraop_1 = self._init_intraop_propagation_layer(
        #     n_value_features=n_intermediate_features[5],
        #     n_query_features=n_preprocess_features,
        #     embedding_size=embedding_size,
        #     n_output_features=n_intermediate_features[4],
        # )
        # self.att_intraop_2, self.conv_intraop_2 = self._init_intraop_propagation_layer(
        #     n_value_features=n_intermediate_features[4],
        #     n_query_features=n_preprocess_features,
        #     embedding_size=embedding_size,
        #     n_output_features=n_intermediate_features[3],
        # )
        # self.att_intraop_3, self.conv_intraop_3 = self._init_intraop_propagation_layer(
        #     n_value_features=n_intermediate_features[3],
        #     n_query_features=n_preprocess_features,
        #     embedding_size=embedding_size,
        #     n_output_features=n_intermediate_features[2],
        # )
        # self.att_intraop_4, self.conv_intraop_4 = self._init_intraop_propagation_layer(
        #     n_value_features=n_intermediate_features[2],
        #     n_query_features=n_preprocess_features,
        #     embedding_size=embedding_size,
        #     n_output_features=n_intermediate_features[1],
        # )
        # self.att_intraop_5, self.conv_intraop_5 = self._init_intraop_propagation_layer(
        #     n_value_features=n_intermediate_features[1],
        #     n_query_features=n_preprocess_features,
        #     embedding_size=embedding_size,
        #     n_output_features=n_intermediate_features[0],
        # )



        ########################
        self.cross_att_I2P_0, self.cross_att_P2I_0, self.level_0_out, self.upsample_0_to_1_intraop, self.upsample_0_to_1_preop = self._init_layer(
            embedding_size=embedding_size, 
            n_cross_att_intraop_features = n_preprocess_features, # n_intermediate_features[0], 
            n_cross_att_preop_features = n_preprocess_features, 
            n_intermediate_features = n_intermediate_features[0], 
            n_level_output_features = n_output_features,
            n_upsample_high_res_features = n_preprocess_features,
            n_kneighbors = min(self.points_per_region, self.n_intermediate_points[0]),
        )
        self.cross_att_I2P_1, self.cross_att_P2I_1, self.level_1_out, self.upsample_1_to_2_intraop, self.upsample_1_to_2_preop = self._init_layer(
            embedding_size = embedding_size, 
            n_cross_att_intraop_features = n_intermediate_features[0],
            n_cross_att_preop_features = n_intermediate_features[0], 
            n_intermediate_features = n_intermediate_features[1], 
            n_level_output_features = n_output_features, 
            n_upsample_high_res_features = n_preprocess_features,
            n_kneighbors = self.points_per_region,
        )
        self.cross_att_I2P_2, self.cross_att_P2I_2, self.level_2_out, self.upsample_2_to_3_intraop, self.upsample_2_to_3_preop = self._init_layer(
            embedding_size = embedding_size, 
            n_cross_att_intraop_features = n_intermediate_features[1],
            n_cross_att_preop_features = n_intermediate_features[1], 
            n_intermediate_features = n_intermediate_features[2], 
            n_level_output_features = n_output_features, 
            n_upsample_high_res_features = n_preprocess_features,
            n_kneighbors = self.points_per_region,
        )
        self.cross_att_I2P_3, self.cross_att_P2I_3, self.level_3_out, self.upsample_3_to_4_intraop, self.upsample_3_to_4_preop = self._init_layer(
            embedding_size = embedding_size, 
            n_cross_att_intraop_features = n_intermediate_features[2],
            n_cross_att_preop_features = n_intermediate_features[2], 
            n_intermediate_features = n_intermediate_features[3], 
            n_level_output_features = n_output_features, 
            n_upsample_high_res_features = n_preprocess_features,
            n_kneighbors = self.points_per_region,
        )   
        self.cross_att_I2P_4, self.cross_att_P2I_4, self.level_4_out, self.upsample_4_to_5_intraop, self.upsample_4_to_5_preop = self._init_layer(
            embedding_size = embedding_size, 
            n_cross_att_intraop_features = n_intermediate_features[3],
            n_cross_att_preop_features = n_intermediate_features[3], 
            n_intermediate_features = n_intermediate_features[4], 
            n_level_output_features = n_output_features, 
            n_upsample_high_res_features = n_preprocess_features,
            n_kneighbors = self.points_per_region,
        )
        self.cross_att_I2P_5, self.cross_att_P2I_5, self.level_5_out, _, _ = self._init_layer(
            embedding_size = embedding_size, 
            n_cross_att_intraop_features = n_intermediate_features[4],
            n_cross_att_preop_features = n_intermediate_features[4], 
            n_intermediate_features = n_intermediate_features[5], 
            n_level_output_features = n_output_features, 
            n_upsample_high_res_features = n_preprocess_features,
            n_kneighbors = self.points_per_region,
            last_layer=True,
        )


    def _init_self_attention(self, embedding_size, n_input_features, n_preprocess_features):
        """Initialize the self-attention layers for pre- and intraoperative data.
        """
        self.self_attention_preop = GroupedAttention(
            n_value_features=n_input_features,
            n_query_features=n_input_features,
            embedding_size=embedding_size,
            n_output_features=n_preprocess_features,
            n_kneighbors=self.points_per_region,
        )
        self.self_attention_intraop = GroupedAttention(
            n_value_features=n_input_features,
            n_query_features=n_input_features,
            embedding_size=embedding_size,
            n_output_features=n_preprocess_features,
            n_kneighbors=self.points_per_region,
        )

    def _attention_preprocessing(self, coords_intraop, coords_preop,
                                 features_intraop, features_preop):
        """Do self attention
        """
        features_preop = self.self_attention_preop(coords_preop, features_preop, coords_preop, features_preop)

        features_intraop = self.self_attention_intraop(coords_intraop, features_intraop,
                                                       coords_intraop, features_intraop)
        return features_intraop, features_preop


    def _init_intraop_propagation_layer(self, n_value_features, n_query_features, embedding_size, n_output_features, highest_res=False):
        """Initialize the intraop. propagation layers for intraoperative point clouds in different resolutions.
        """
        if not highest_res:
            att = GroupedAttention(
                n_value_features=n_value_features,
                n_query_features=n_query_features,
                embedding_size=embedding_size,
                n_output_features=n_output_features,
                n_kneighbors=self.points_per_region,
            )
            conv = nn.Conv1d(n_output_features, n_output_features, kernel_size=1)
        else:
            att = None
            conv = nn.Conv1d(n_query_features, n_output_features, kernel_size=1)
        return att, conv



    def _init_layer(
        self,
        embedding_size, 
        n_cross_att_intraop_features, # value features
        n_cross_att_preop_features, 
        n_intermediate_features, 
        n_level_output_features, # output of this level, usually displacement field at current resolution
        n_upsample_high_res_features,
        n_kneighbors,
        last_layer=False,
    ):
        """Initialize each layer of the network, including 
            1: a cross attention
            2: cross attention feature convolution
            3: non-linearity
            4: upsampling
        """
        # fusion = FusionGate(n_input_features_internals, n_input_features,
        #                     n_query_features, embedding_size,
        #                     n_intermediate_features, self.points_per_region,
        #                     use_internals = self.use_internals )
        # TODO: Check if this non-lin was really only used on the output features?
        # cross_att = GroupedAttention(
        #     n_value_features=n_input_features,
        #     n_query_features=n_query_features,
        #     embedding_size=embedding_size,
        #     n_output_features=n_intermediate_features,
        #     n_kneighbors=n_kneighbors,
        # )

        cross_att_I2P = LayerCrossAttention(
            n_query_features = n_cross_att_intraop_features,
            n_value_features = n_cross_att_preop_features,
            n_output_features = n_intermediate_features,
            embedding_size=embedding_size,
            n_kneighbors = n_kneighbors,
            use_relative_coords = True,
            use_positional_encoding = True,
            n_attention_modules = 4,
            attention_type = 0,
            attention_mode = "additive",
        )

        cross_att_P2I = LayerCrossAttention(
            n_query_features = n_cross_att_preop_features,
            n_value_features = n_intermediate_features,
            n_output_features = n_intermediate_features,
            embedding_size=embedding_size,
            n_kneighbors = n_kneighbors,
            use_relative_coords = True,
            use_positional_encoding = True,
            n_attention_modules = 4,
            attention_type = 0,
            attention_mode = "additive",
        )

        level_out = nn.Sequential(
            self.non_lin, # output of each level for multi-resolution displacement error calculation
            nn.Conv1d(n_intermediate_features, n_level_output_features, kernel_size=1)
        )

        upsampling_intraop = None
        upsampling_preop = None
        if not last_layer:
            upsampling_intraop = LayerUpsample(
                n_kneighbors=self.points_per_region,
                n_low_res_features=n_intermediate_features,
                n_high_res_features=n_upsample_high_res_features,
                n_output_features=n_intermediate_features,
                use_relative_coords=True, 
                use_positional_encoding=True,
                embedding_size=embedding_size, 
                n_attention_modules=1,
            )
            upsampling_preop = LayerUpsample(
                n_kneighbors=self.points_per_region,
                n_low_res_features=n_intermediate_features,
                n_high_res_features=n_upsample_high_res_features,
                n_output_features=n_intermediate_features,
                use_relative_coords=True, 
                use_positional_encoding=True,
                embedding_size=embedding_size, 
                n_attention_modules=4,
            )
        # return cross_att, cross_att_conv, level_out, upsampling
        return cross_att_I2P, cross_att_P2I, level_out, upsampling_intraop, upsampling_preop


    def forward(self, 
                preop, 
                intraop,
    ):

        #########################
        # preprocess the input original tensors, including the self distance field, across distance field and positional encodings
        coords_preop, features_preop, \
        coords_intraop, features_intraop = self.preprocessor.preprocess( preop, intraop)
        # coords_preop_internal, features_preop_internal, \
        # coords_intraop_internal, features_intraop_internal = self.preprocessor.preprocess_internals( preop_internal, intraop_internal )
        # print("coords_preop.shape:", coords_preop.shape)
        # print("features_preop.shape:", features_preop.shape)
        # print("coords_intraop.shape:", coords_intraop.shape)
        # print("features_intraop.shape:", features_intraop.shape)
        # coords_preop_enc = self.preprocessor.coords_preop_enc
        # coords_intraop_enc = self.preprocessor.coords_intraop_enc

        #########################
        ## Preprocess the input data by using a (regional) self-attention on preop and intraop data (separately):
        features_intraop, features_preop\
            = self._attention_preprocessing(coords_intraop,
                                            coords_preop,
                                            features_intraop,
                                            features_preop)

        #print("Preprocessed:")
        #print("\tfeatures_intraop", features_intraop.shape)
        #print("\tfeatures_intraop_internal", features_intraop_internal.shape)
        #print("\tfeatures_preop", features_preop.shape)
        #print("\tfeatures_preop_internal", features_preop_internal.shape)

        # print("downsample...")
        #########################
        # At the highest level, the features and coordinates correspond to the (preop) input data:
        coords_preop_5 = coords_preop
        features_preop_5 = features_preop

        ids_preop_4, coords_preop_4 = farthest_point_sampling( coords_preop_5, self.n_intermediate_points[4], random=False )
        features_preop_4 = select_points( features_preop_5, ids_preop_4 )

        ## Downsample:
        ids_preop_3, coords_preop_3 = farthest_point_sampling( coords_preop_4, self.n_intermediate_points[3], random=False )
        features_preop_3 = select_points( features_preop_4, ids_preop_3 )

        ids_preop_2, coords_preop_2 = farthest_point_sampling( coords_preop_3, self.n_intermediate_points[2], random=False )
        features_preop_2 = select_points( features_preop_3, ids_preop_2 )
        # ids_2, coords_preop_2 = farthest_point_sampling( coords_preop, self.n_intermediate_points[1], random=False )
        # features_preop_2 = select_points( features_preop, ids_2 )

        ids_preop_1, coords_preop_1 = farthest_point_sampling( coords_preop_2, self.n_intermediate_points[1], random=False )
        features_preop_1 = select_points( features_preop_2, ids_preop_1 )
        # features_coords_preop_enc_1 = select_points( coords_preop_enc, ids_1 )

        ids_preop_0, coords_preop_0 = farthest_point_sampling( coords_preop_1, self.n_intermediate_points[0], random=False )
        features_preop_0 = select_points( features_preop_1, ids_preop_0 )
        # features_coords_preop_enc_0 = select_points( features_coords_preop_enc_1, ids_0 )
        # features_coords_preop_enc_2 = coords_preop_enc


        # Downsample intraoperative points:
        coords_intraop_5 = coords_intraop
        features_intraop_5 = features_intraop

        ids_intraop_4, coords_intraop_4 = farthest_point_sampling( coords_intraop_5, self.n_intermediate_points[4], random=False )
        features_intraop_4 = select_points( features_intraop_5, ids_intraop_4 )

        ids_intraop_3, coords_intraop_3 = farthest_point_sampling( coords_intraop_4, self.n_intermediate_points[3], random=False )
        features_intraop_3 = select_points( features_intraop_4, ids_intraop_3 )

        ids_intraop_2, coords_intraop_2 = farthest_point_sampling( coords_intraop_3, self.n_intermediate_points[2], random=False )
        features_intraop_2 = select_points( features_intraop_3, ids_intraop_2 )

        ids_intraop_1, coords_intraop_1 = farthest_point_sampling( coords_intraop_2, self.n_intermediate_points[1], random=False )
        features_intraop_1 = select_points( features_intraop_2, ids_intraop_1 )

        ids_intraop_0, coords_intraop_0 = farthest_point_sampling( coords_intraop_1, self.n_intermediate_points[0], random=False )
        features_intraop_0 = select_points( features_intraop_1, ids_intraop_0 )

        # intraoperative cross attention
        # features_intraop_5 = self.non_lin(self.conv_intraop_0(features_intraop_5))

        # features_intraop_4 = self.att_intraop_1(
        #     value_coords=coords_intraop_5,
        #     value_features=features_intraop_5,
        #     queries_coords=coords_intraop_4,
        #     queries_features=features_intraop_4,
        # )
        # features_intraop_4 = self.non_lin(self.conv_intraop_1(features_intraop_4))

        # features_intraop_3 = self.att_intraop_2(
        #     value_coords=coords_intraop_4,
        #     value_features=features_intraop_4,
        #     queries_coords=coords_intraop_3,
        #     queries_features=features_intraop_3,
        # )
        # features_intraop_3 = self.non_lin(self.conv_intraop_2(features_intraop_3))

        # features_intraop_2 = self.att_intraop_3(
        #     value_coords=coords_intraop_3,
        #     value_features=features_intraop_3,
        #     queries_coords=coords_intraop_2,
        #     queries_features=features_intraop_2,
        # )
        # features_intraop_2 = self.non_lin(self.conv_intraop_3(features_intraop_2))

        # features_intraop_1 = self.att_intraop_4(
        #     value_coords=coords_intraop_2,
        #     value_features=features_intraop_2,
        #     queries_coords=coords_intraop_1,
        #     queries_features=features_intraop_1,
        # )
        # features_intraop_1 = self.non_lin(self.conv_intraop_4(features_intraop_1))

        # features_intraop_0 = self.att_intraop_5(
        #     value_coords=coords_intraop_1,
        #     value_features=features_intraop_1,
        #     queries_coords=coords_intraop_0,
        #     queries_features=features_intraop_0,
        # )
        # features_intraop_0 = self.non_lin(self.conv_intraop_5(features_intraop_0))
        


        ##########################
        ## Initialize a list of results, a result will be collected at each level:
        results = []


        # print("Level 0...")
        ##########################
        ## Level 0:
        # features_intraop_0 = self.level_0_processing( features_intraop )
        features_intraop_0 = self.cross_att_I2P_0(
            value_coords=coords_preop_0,
            value_features=features_preop_0,
            query_coords=coords_intraop_0,
            query_features=features_intraop_0,
        )   
        features_preop_0 = self.cross_att_P2I_0(
            value_coords=coords_intraop_0, 
            value_features=features_intraop_0, 
            query_coords=coords_preop_0, 
            query_features=features_preop_0,
        )
        # features_preop_0 = self.non_lin(self.cross_att_conv_0(features_preop_0))
        features_preop_0_out = self.level_0_out(features_preop_0)

        results.append({"result": features_preop_0_out, "point_idx": ids_preop_0})


        # print("Level 1...")
        ##########################
        ## Level 1:
        # Upsample to the next level:
        features_intraop_1 = self.upsample_0_to_1_intraop(
            low_res_coords=coords_intraop_0, 
            low_res_features=features_intraop_0, 
            high_res_coords=coords_intraop_1, 
            high_res_features=features_intraop_1,
        )

        features_preop_1 = self.upsample_0_to_1_preop(coords_preop_0, features_preop_0,
                                                coords_preop_1, features_preop_1)

        features_intraop_1 = self.cross_att_I2P_1(
            value_coords=coords_preop_1,
            value_features=features_preop_1,
            query_coords=coords_intraop_1,
            query_features=features_intraop_1,
        )
        features_preop_1 = self.cross_att_P2I_1(
                value_coords=coords_intraop_1, 
                value_features=features_intraop_1, 
                query_coords=coords_preop_1, 
                query_features=features_preop_1,
        )
        # features_preop_1 = self.non_lin(self.cross_att_conv_1(features_preop_1))
        features_preop_1_out = self.level_1_out(features_preop_1)

        results.append({"result": features_preop_1_out, "point_idx": ids_preop_1,})


        # print("Level 2...")
        ##########################
        ## Level 2:
        features_intraop_2 = self.upsample_1_to_2_intraop(
            low_res_coords=coords_intraop_1, 
            low_res_features=features_intraop_1, 
            high_res_coords=coords_intraop_2, 
            high_res_features=features_intraop_2,
        )
        features_preop_2 = self.upsample_1_to_2_preop(coords_preop_1, features_preop_1,
                                                coords_preop_2, features_preop_2)

        features_intraop_2 = self.cross_att_I2P_2(
            value_coords=coords_preop_2,
            value_features=features_preop_2,
            query_coords=coords_intraop_2,
            query_features=features_intraop_2,
        )
        features_preop_2 = self.cross_att_P2I_2(
                value_coords=coords_intraop_2, 
                value_features=features_intraop_2, 
                query_coords=coords_preop_2, 
                query_features=features_preop_2,
        )
        # features_preop_2 = self.non_lin(self.cross_att_conv_2(features_preop_2))
        features_preop_2_out = self.level_2_out(features_preop_2)
        results.append({"result": features_preop_2_out, "point_idx": ids_preop_2,})

        ##########################
        ## Level 3:
        features_intraop_3 = self.upsample_2_to_3_intraop(
            low_res_coords=coords_intraop_2, 
            low_res_features=features_intraop_2, 
            high_res_coords=coords_intraop_3, 
            high_res_features=features_intraop_3,
        )
        features_preop_3 = self.upsample_2_to_3_preop(coords_preop_2, features_preop_2,
                                                coords_preop_3, features_preop_3)
        
        features_intraop_3 = self.cross_att_I2P_3(
            value_coords=coords_preop_3,
            value_features=features_preop_3,
            query_coords=coords_intraop_3,
            query_features=features_intraop_3,
        )
        features_preop_3 = self.cross_att_P2I_3(
                value_coords=coords_intraop_3, 
                value_features=features_intraop_3, 
                query_coords=coords_preop_3, 
                query_features=features_preop_3,
        )
        # features_preop_3 = self.non_lin(self.cross_att_conv_3(features_preop_3))
        features_preop_3_out = self.level_3_out(features_preop_3)
        results.append({"result": features_preop_3_out, "point_idx": ids_preop_3,})


        ##########################
        ## Level 4:
        features_intraop_4 = self.upsample_3_to_4_intraop(
            low_res_coords=coords_intraop_3, 
            low_res_features=features_intraop_3, 
            high_res_coords=coords_intraop_4, 
            high_res_features=features_intraop_4,
        )
        features_preop_4 = self.upsample_3_to_4_preop(coords_preop_3, features_preop_3,
                                                coords_preop_4, features_preop_4)
        
        features_intraop_4 = self.cross_att_I2P_4(
            value_coords=coords_preop_4,
            value_features=features_preop_4,
            query_coords=coords_intraop_4,
            query_features=features_intraop_4,
        )
        features_preop_4 = self.cross_att_P2I_4(
                value_coords=coords_intraop_4, 
                value_features=features_intraop_4, 
                query_coords=coords_preop_4, 
                query_features=features_preop_4,
        )
        # features_preop_4 = self.non_lin(self.cross_att_conv_4(features_preop_4))
        features_preop_4_out = self.level_4_out(features_preop_4)
        results.append({"result": features_preop_4_out, "point_idx": ids_preop_4,})

        ##########################
        ## Level 5:
        features_intraop_5 = self.upsample_4_to_5_intraop(
            low_res_coords=coords_intraop_4, 
            low_res_features=features_intraop_4, 
            high_res_coords=coords_intraop_5, 
            high_res_features=features_intraop_5,
        )
        features_preop_5 = self.upsample_4_to_5_preop(coords_preop_4, features_preop_4,
                                                coords_preop_5, features_preop_5)
        
        features_intraop_5 = self.cross_att_I2P_5(
            value_coords=coords_preop_5,
            value_features=features_preop_5,
            query_coords=coords_intraop_5,
            query_features=features_intraop_5,
        )    
        features_preop_5 = self.cross_att_P2I_5(
                value_coords=coords_intraop_5, 
                value_features=features_intraop_5, 
                query_coords=coords_preop_5, 
                query_features=features_preop_5,
        )
        # features_preop_5 = self.non_lin(self.cross_att_conv_5(features_preop_5))
        features_preop_5_out = self.level_5_out(features_preop_5)


        # Mask out all dummy points. For those, we don't want to return any displacement field (i.e.
        # we don't care what the value is there):
        dummy_point_mask = coords_preop.abs().max( dim = 1, keepdim = True )[0] > 5000

        features_preop_5_out[dummy_point_mask.repeat( (1,3,1) )] = 0

        results.append({"result": features_preop_5_out})

        return results

