import torch
import torch.nn as nn

try:
    from models.positional_encoding import PositionalEncoder
    # from models.k_nearest_neighbors import k_nearest_neighbors
    # from models.select import select_point_regions, select_points
    # from models.farthest_point_sampling import farthest_point_sampling
except:
    from positional_encoding import PositionalEncoder
    # from k_nearest_neighbors import k_nearest_neighbors
    # from select import select_point_regions, select_points
    from data.vtk_utils import save_output_as_vtk


class TensorPreprocessor():
    def __init__(self, 
            enc_freq = [1e-2, 1e-1, 2, 4, 8, 16, 32, 64], 
            enc_freq_scale=1, 
            # pad_features_to=0,
            append_df_self=True,
            append_df_cross=True,
            append_positional_encoding=True,
            no_internals=True,
            adapt_to_old_models=False,
            compact_return=True,
        ):
        self.enc_freq = enc_freq
        self.enc_freq_scale = enc_freq_scale
        # self.pad_features_to = pad_features_to
        self.append_df_self = append_df_self
        self.append_df_cross = append_df_cross
        self.append_positional_encoding = append_positional_encoding
        self.no_internals = no_internals
        self.adapt_to_old_models = adapt_to_old_models
        self.compact_return = compact_return
        self.num_output_features = 0
        self.pe = None
        
        # self.num_output_features is not in use...
        if adapt_to_old_models:
            self.num_output_features += 1
        if self.append_df_self:
            self.num_output_features += 1
        if self.append_df_cross:
            self.num_output_features += 1
        if self.append_positional_encoding:
            self.pe = PositionalEncoder( self.enc_freq, self.enc_freq_scale, )
            self.num_output_features += self.pe.num_features  
        
        self.df_preop = None
        self.df_intraop = None

        self.dists_preop_to_intraop = None
        self.dists_intraop_to_preop = None

        self.coords_preop_enc = None
        self.coords_intraop_enc = None


    def preprocess(self, preop, intraop,):
        """Calculate preprocessed features for the preop and intraop point clouds, including three features
        according to the configuration of the preprocessor:: 
        - The distance field (for preop points only)
        - The mutual closest distance to the other point cloud (for both preop and intraop points)
        - The encoded point coordinates (postional encoding)

        Args:
            preop (_type_): preoperative point cloud
            intraop (_type_): intraoperative partial point cloud

        Returns:
            if compact_return:
                coords_preop: preop coordinates
                features_preop: concatenated preop features (including, distance field, mutual closest distance, positional encoding)
                coords_intraop: intraop coordinates
                features_intraop: concatenated intraop features (including distance field, mutual closest distance, positional encoding)
            else:
                return the same as compact_return, but as separate lists of features
        """

        preop_feature_list = []
        intraop_feature_list = []

        # Compute how many features we already have (everything is considered a feature except the first three
        # values, which are the point coordinates)
        n_preop_features_input = preop.shape[1] - 3
        n_intraop_features_input = intraop.shape[1] - 3

        # First 3 values are the coordinates:
        coords_preop = preop[:,0:3,:]
        coords_intraop = intraop[:,0:3,:]

        # get mask for dummy points
        coords_preop_mask = coords_preop[:, 0, :] < 1000
        coords_intraop_mask = coords_intraop[:, 0, :] < 1000
        coords_preop_mask = coords_preop_mask.unsqueeze(1)
        coords_intraop_mask = coords_intraop_mask.unsqueeze(1)


        if self.adapt_to_old_models:
            # some old trained models used -1 and 1 as indicators for preop and intraop points
            point_type_preop = (-1)*torch.ones( (preop.shape[0], 1, preop.shape[2]) ).cuda()
            point_type_intraop = torch.ones( (intraop.shape[0], 1, intraop.shape[2]) ).cuda()
            preop_feature_list.append(point_type_preop)
            intraop_feature_list.append(point_type_intraop)

        # why not align these two features at dataloader?
        if self.append_df_self:
            self.df_preop = preop[:,3,:]     # distance field
            self.df_preop = self.df_preop.unsqueeze(1)
            self.df_intraop = torch.zeros_like(self.df_preop)

            self.df_preop = self.df_preop * coords_preop_mask
            self.df_intraop = self.df_intraop * coords_intraop_mask
            preop_feature_list.append(self.df_preop)
            intraop_feature_list.append(self.df_intraop)
            

        if self.append_df_cross:
            dists = torch.cdist( coords_preop.permute(0,2,1), coords_intraop.permute(0,2,1) )
            self.dists_preop_to_intraop, _ = torch.min( dists, dim=2 )
            self.dists_intraop_to_preop, _ = torch.min( dists, dim=1 )

            self.dists_preop_to_intraop = self.dists_preop_to_intraop.unsqueeze(1)
            self.dists_intraop_to_preop = self.dists_intraop_to_preop.unsqueeze(1)
            
            self.dists_preop_to_intraop = self.dists_preop_to_intraop * coords_preop_mask
            self.dists_intraop_to_preop = self.dists_intraop_to_preop * coords_intraop_mask

            preop_feature_list.append(self.dists_preop_to_intraop)
            intraop_feature_list.append(self.dists_intraop_to_preop)
            
        # Positional encoding
        if self.append_positional_encoding:            
            self.coords_preop_enc = self.pe( coords_preop )
            self.coords_intraop_enc = self.pe( coords_intraop )
            self.coords_preop_enc = self.coords_preop_enc * coords_preop_mask
            self.coords_intraop_enc = self.coords_intraop_enc * coords_intraop_mask
            coords_preop = torch.cat( (coords_preop, self.coords_preop_enc), dim=1 )
            coords_intraop = torch.cat( (coords_intraop, self.coords_intraop_enc), dim=1 )

        # concatenate features along with the input preop and intraop tensors
        if n_preop_features_input > 1:
            features_preop_input = preop[:,4:,:]
            # features_preop = torch.cat( (features_preop, features_preop_input), dim=1 )
            # print("preop.shape", preop.shape)
            # print("features_preop_input", features_preop_input.shape)
            preop_feature_list.append(features_preop_input)
        if n_intraop_features_input > 0:
            features_intraop_input = intraop[:,3:,:]
            # features_intraop = torch.cat( (features_intraop, features_intraop_input), dim=1 )
            # print("features_intraop_input", features_intraop_input.shape)
            intraop_feature_list.append(features_intraop_input)

        features_preop = torch.cat( preop_feature_list, dim=1 )
        features_intraop = torch.cat( intraop_feature_list, dim=1 )

        # return coords_preop, features_preop, coords_intraop, features_intraop
        if self.compact_return:
            # features_preop = torch.cat( (self.coords_preop_enc, features_preop), dim=1 )
            # features_intraop = torch.cat( (self.coords_intraop_enc, features_intraop), dim=1 )
            return coords_preop, features_preop, coords_intraop, features_intraop
        else:
            # return coords_preop, coords_preop_enc, features_preop, coords_intraop, coords_intraop_enc, features_intraop
            return coords_preop, preop_feature_list, coords_intraop, intraop_feature_list


    def preprocess_internals(self, preop_internals=None, intraop_internals=None):
        if self.no_internals or preop_internals is None or intraop_internals is None:
            return None, None, None, None
        else:
            return self.preprocess(
                preop=preop_internals,
                intraop=intraop_internals,
            )
        
    def preprocess_4dmatch(self, preop, intraop ):
        preop_feature_list = []
        intraop_feature_list = []

        # Compute how many features we already have (everything is considered a feature except the first three
        # values, which are the point coordinates)
        n_preop_features_input = preop.shape[1] - 3
        n_intraop_features_input = intraop.shape[1] - 3

        # First 3 values are the coordinates:
        coords_preop = preop[:,0:3,:]
        coords_intraop = intraop[:,0:3,:]

        if self.adapt_to_old_models:
            # some old trained models used -1 and 1 as indicators for preop and intraop points
            point_type_preop = (-1)*torch.ones( (preop.shape[0], 1, preop.shape[2]) ).cuda()
            point_type_intraop = torch.ones( (intraop.shape[0], 1, intraop.shape[2]) ).cuda()
            preop_feature_list.append(point_type_preop)
            intraop_feature_list.append(point_type_intraop)

        # why not align these two features at dataloader?
        if self.append_df_self:
            self.df_preop = preop[:,3,:]     # distance field
            self.df_preop = self.df_preop.unsqueeze(1)
            self.df_intraop = torch.zeros_like(self.df_preop)

            preop_feature_list.append(self.df_preop)
            intraop_feature_list.append(self.df_intraop)
            

        if self.append_df_cross:
            dists = torch.cdist( coords_preop.permute(0,2,1), coords_intraop.permute(0,2,1) )
            self.dists_preop_to_intraop, _ = torch.min( dists, dim=2 )
            self.dists_intraop_to_preop, _ = torch.min( dists, dim=1 )

            self.dists_preop_to_intraop = self.dists_preop_to_intraop.unsqueeze(1)
            self.dists_intraop_to_preop = self.dists_intraop_to_preop.unsqueeze(1)
            
            preop_feature_list.append(self.dists_preop_to_intraop)
            intraop_feature_list.append(self.dists_intraop_to_preop)
            
        # Positional encoding
        if self.append_positional_encoding:            
            self.coords_preop_enc = self.pe( coords_preop )
            self.coords_intraop_enc = self.pe( coords_intraop )
            coords_preop = torch.cat( (coords_preop, self.coords_preop_enc), dim=1 )
            coords_intraop = torch.cat( (coords_intraop, self.coords_intraop_enc), dim=1 )

        # concatenate features along with the input preop and intraop tensors
        if n_preop_features_input > 1:
            features_preop_input = preop[:,3:,:]
            # features_preop = torch.cat( (features_preop, features_preop_input), dim=1 )
            # print("preop.shape", preop.shape)
            # print("features_preop_input", features_preop_input.shape)
            preop_feature_list.append(features_preop_input)
        if n_intraop_features_input > 0:
            features_intraop_input = intraop[:,3:,:]
            # features_intraop = torch.cat( (features_intraop, features_intraop_input), dim=1 )
            # print("features_intraop_input", features_intraop_input.shape)
            intraop_feature_list.append(features_intraop_input)

        features_preop = torch.cat( preop_feature_list, dim=1 )
        features_intraop = torch.cat( intraop_feature_list, dim=1 )

        # return coords_preop, features_preop, coords_intraop, features_intraop
        if self.compact_return:
            # features_preop = torch.cat( (self.coords_preop_enc, features_preop), dim=1 )
            # features_intraop = torch.cat( (self.coords_intraop_enc, features_intraop), dim=1 )
            return coords_preop, features_preop, coords_intraop, features_intraop
        else:
            # return coords_preop, coords_preop_enc, features_preop, coords_intraop, coords_intraop_enc, features_intraop
            return coords_preop, preop_feature_list, coords_intraop, intraop_feature_list

def preprocess_data(preop, intraop, enc_freq, enc_freq_scale=1, compact_return=False):
    pe = PositionalEncoder( enc_freq, enc_freq_scale, )

    #print("preop", preop.shape)
    #print("intraop", intraop.shape)
    n_points = preop.shape[-1]
    batch_size = preop.shape[0]

    # Compute how many features we already have (everything is considered a feature except the first three
    # values, which are the point coordinates)
    n_preop_features_input = preop.shape[1] - 3
    n_intraop_features_input = intraop.shape[1] - 3

    # Calculate point coordinates related features
    # First 3 values are the coordinates:
    coords_preop = preop[:,0:3,:]
    coords_intraop = intraop[:,0:3,:]

    # 1. preoperative distance field
    df_preop = preop[:,3,:]     # distance field
    df_preop = df_preop.unsqueeze(1)

    # 2. mutual closest distance to the other point cloud
    dists = torch.cdist( coords_preop.permute(0,2,1), coords_intraop.permute(0,2,1) )
    dists_preop_to_intraop, _ = torch.min( dists, dim=2 )
    dists_intraop_to_preop, _ = torch.min( dists, dim=1 )

    dists_preop_to_intraop = dists_preop_to_intraop.unsqueeze(1)
    dists_intraop_to_preop = dists_intraop_to_preop.unsqueeze(1)

    # 3. Positional encoding
    coords_preop_enc = pe( coords_preop )
    coords_intraop_enc = pe( coords_intraop )

    features_preop = torch.cat( ( df_preop, dists_preop_to_intraop,), dim=1 )
    features_intraop = torch.cat( ( torch.zeros_like(df_preop), dists_intraop_to_preop,), dim=1 )

    #print("features_preop", features_preop.shape)
    #print("features_intraop", features_intraop.shape)

    # concatenate the input features
    if n_preop_features_input > 1:
        features_preop_input = preop[:,4:,:]
        features_preop = torch.cat( (features_preop, features_preop_input), dim=1 )
    if n_intraop_features_input > 0:
        features_intraop_input = intraop[:,3:,:]
        features_intraop = torch.cat( (features_intraop, features_intraop_input), dim=1 )

    #print("features_preop final", features_preop.shape)
    #print("features_intraop final", features_intraop.shape)
 
    if compact_return:
        features_preop = torch.cat( (features_preop, coords_preop_enc ), dim=1 )
        features_intraop = torch.cat( ( features_intraop, coords_intraop_enc), dim=1 )
        return coords_preop, features_preop, coords_intraop, features_intraop
    else:
        return coords_preop, coords_preop_enc, features_preop, coords_intraop, coords_intraop_enc, features_intraop







if __name__ == "__main__":
    # test the preprocessor
    # preop = torch.rand( 2, 4, 100 ).cuda()
    # intraop = torch.rand( 2, 3, 100 ).cuda()
    # pp = Preprocessor(

    # )
    # print(pp.num_output_features)
    # res = pp(preop, intraop)
    # # print(res.shape)
    # for r in res:
    #     print(r.shape)
    
    # pe = PointAttentionPreprocessor(
    #     coords_preop = torch.rand( 2, 3, 100 ).cuda(),
    #     coords_intraop = torch.rand( 2, 3, 100 ).cuda(),
    # )

    res = preprocess_data(
        preop = torch.rand( 2, 4, 100 ).cuda(),
        intraop = torch.rand( 2, 3, 100 ).cuda(),
        enc_freq = [1e-2, 1e-1, 2, 4, 8, 16, 32, 64],

    )

    for r in res:
        print(r.shape)


