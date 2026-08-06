import torch
import torch.nn as nn


from models.layer_downsample import LayerDownsample
from models.layer_upsample import LayerUpsample
from models.select import select_points
from models.attention import MultiRegionAttention, SelfAttention

class U( nn.Module ):

    def __init__( self,
            n_points = [128, 32, 8, 3],
            n_features = [2, 8, 32, 128],
            radii_down = [0.05, 0.07, 0.1, 1],
            radii_up = [0.5, 0.05, 0.01, 0.003],
            kn_down = 50,
            kn_up = 50,
            n_attention_modules = 8,
            embedding_size = 16,
            use_relative_coords_down = True,
            use_relative_coords_up = True
            ):

        nn.Module.__init__( self )

        assert len(n_points) == len(n_features)
        assert len(radii_down) == len(n_features)
        assert len(radii_up) == len(n_features)
        #assert len(f_kneighbors_down) == len(n_points)-1 and \
        #        len(f_kneighbors_up) == len(n_points)-1

        self.n_points = n_points[0]
        self.n_features = n_features[0]     # Features on current level

        self.n_points_down = n_points[1]
        self.n_features_down = n_features[1]        # Features on next level

        self.radius_down = radii_down[0]
        self.radius_up = radii_up[-1]

        # Calculate the (integer) number of nearest neighbors for down- and up-sampling:
        #kn_down = max( int(n_points[0]*f_kneighbors_down[0]), 3 )
        #kn_up = max( int(n_points[1]*f_kneighbors_up[0]), 3 )
        kn_down = 50
        kn_up = 50
        
        self.down = LayerDownsample(
                num_kernels = self.n_points_down,
                num_kneighbors = kn_down,
                num_input_features = self.n_features,
                num_output_features = self.n_features_down,        # Features of next level
                embedding_size = embedding_size,
                radius = self.radius_down,
                num_attention_modules = n_attention_modules,
                use_relative_coords = use_relative_coords_down
                )
        self.conv_down = nn.Conv1d( self.n_features_down, self.n_features_down, kernel_size = 1 )

        self.self_attention = SelfAttention(
                n_value_features = self.n_features_down,
                embedding_size = embedding_size,
                n_output_features = self.n_features_down
                )

        n_sub_levels = len(n_points) - 1
        if n_sub_levels > 1:
            self.sublevel = U(
                    n_points = n_points[1:],
                    n_features = n_features[1:],
                    radii_down = radii_down[1:],
                    radii_up = radii_up[:-1],
                    kn_down = kn_down,
                    kn_up = kn_up,
                    n_attention_modules = n_attention_modules,
                    use_relative_coords_down = use_relative_coords_down,
                    use_relative_coords_up = use_relative_coords_up
                    )
        else:
            self.sublevel = None
        
        self.conv_skip = nn.Conv1d( self.n_features, self.n_features, kernel_size = 1 )

        self.up = LayerUpsample(
                n_kneighbors = kn_up,
                n_low_res_features = self.n_features_down,
                n_high_res_features = self.n_features,
                n_output_features = self.n_features,
                radius = self.radius_up,
                n_attention_modules = n_attention_modules,
                embedding_size = embedding_size,
                use_relative_coords = use_relative_coords_up
                )
        self.conv_up = nn.Conv1d( self.n_features, self.n_features, kernel_size = 1 )

        self.non_lin = nn.LeakyReLU()

    def forward( self, coords_in, features_in ):

        ##################
        ## First, downsample the coordinates and calculate new features for the points:
        coords_down, features_down, idx_down = self.down( coords_in, features_in )
        #features_down = self.non_lin( self.conv_down(features_down) )

        features_down = self.self_attention( coords_down, features_down )

        ##################
        ## If there are lower levels, let them work on the downsampled point cloud:
        if self.sublevel:
            features_down, sublevel_results = self.sublevel( coords_down, features_down )
        else:
            # We are the lowest level?
            sublevel_results = []
            
            # Repeat the input value coords/features for every query point:
            #B, D, Nq = features_down.shape
            #coords_values = coords_down.view( B, 3, 1, Nq ).repeat( 1, 1, Nq, 1 )
            #features_values = features_down.view( B, D, 1, Nq ).repeat( 1, 1, Nq, 1 )

            ## NOTE: TODO: This could maybe be replaced by a simple attention layer rather than
            ## Multi-Region attention, however the normal Attention module currently doesn't have
            ## Positional encoding or coordinate-concatenation implemented
            #features_down = self.bottleneck_self_attention(
            #        coords_values, features_values,     # values
            #        coords_down, features_down      # queries
            #        )

        # Add result for this level to the list of returned data:
        sublevel_results.append(
                {
                    "features": features_down,
                    "point_idx": idx_down,
                } )

        # Add a convolution on the skip connection as well:
        features_skip = self.non_lin( self.conv_skip( features_in ) )

        ##################
        ## Upsample point cloud again, taking features from lower levels into account:
        features_up = self.up(
                low_res_coords = coords_down,
                low_res_features = features_down,
                high_res_coords = coords_in,
                high_res_features = features_skip
                )
        features_up = self.non_lin( self.conv_up(features_up) )
        # features_up has the same number of points as coords_in and features_in, but may have a different
        # number of features per point.

        return features_up, sublevel_results





class V2SNet( nn.Module ):

    def __init__( self,
            n_input_features=2,
            n_output_features=3,
            n_points = [2500*2, 400, 100, 50, 25, 15, 7, 5],
            n_features = [3, 30, 40, 60, 80, 90, 100, 160],
            radii_down = [0.01, 0.02, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1],
            radii_up = [1, 0.03, 0.01, 0.01, 0.01, 0.005, 0.005, 0.002],
            kn_down = 50,
            kn_up = 50,
            embedding_size = 16,
            use_relative_coords_down = True,
            use_relative_coords_up = True
            ):


        print("Building V2SNet:")
        print("\tn_input_features:", n_input_features)
        print("\tn_points:", n_points)
        print("\tn_features:", n_features)
        print("\tn_output_features:", n_output_features)
        print("\tradii_down:", radii_down)
        print("\tradii_up:", radii_up)
        print("\tkn_down:", kn_down)
        print("\tkn_up:", kn_up)
        print("\tembedding_size:", embedding_size)
        print("\tuse_relative_coords_down:", use_relative_coords_down)
        print("\tuse_relative_coords_up:", use_relative_coords_up)

        nn.Module.__init__( self )

        self.n_points = n_points

        assert min(n_features) > n_input_features, "At some level, the number of features is lower than the number of input features n_input_features. This introduces a probably unwanted bottleneck!"
        assert min(n_features) > n_output_features, "At some level, the number of features is lower than the number of output features n_output_features. This introduces a probably unwanted bottleneck!"

        # During forward propagation, we concatenate preop and intraop points. To retrieve the preop features
        # later, we create an array which indexes the first half of the results, i.e. which can retrieve the
        # results for the preop points:
        #n_preop_points = int(n_points[0]*0.5)
        #preop_points_idx = torch.linspace(
        #        start = 0,
        #        end = n_preop_points-1,
        #        steps = n_preop_points-1,
        #        dtype = torch.long)
        #self.preop_points_idx = preop_points_idx.unsqueeze(0)
       
        self.non_lin = torch.nn.Softsign()
        self.conv_in = nn.Conv1d( n_input_features, n_features[0], kernel_size = 1 )

        self.net = U(
            n_points = n_points,
            n_features = n_features,
            radii_down = radii_down,
            radii_up = radii_up,
            kn_down = kn_down,
            kn_up = kn_up,
            embedding_size = embedding_size,
            use_relative_coords_down = use_relative_coords_down,
            use_relative_coords_up = use_relative_coords_up
            )

        self.conv_out = nn.ModuleList()
        for i, f in enumerate(reversed(n_features)):
            print( f"\tOutput convolution (level {i}): {f} -> {n_output_features}" )
            conv = nn.Conv1d( f, n_output_features, kernel_size = 1 )
            self.conv_out.append( conv )

        self.initialize()

    def forward( self, coords_preop, features_preop, coords_intraop, features_intraop ):

        coords_in = torch.cat( (coords_preop, coords_intraop), dim=2 )
        features_in = torch.cat( (features_preop, features_intraop), dim=2 )

        features_in = self.non_lin( self.conv_in( features_in ) )

        features, results = self.net( coords_in, features_in )

        # We're only interested in what was calculated for the preoperative points, i.e.
        # the displacement field:
        n_preop_points = int(self.n_points[0]*0.5)
        features_preop = features[:,:,:n_preop_points]
        #batch_size = coords_in.shape[0]
        #preop_points_idx = self.preop_points_idx.repeat( batch_size, 1 )
        results.append( {
            #"point_idx":preop_points_idx,
            "features":features_preop
            } )

        output = []
        for conv, result in zip( self.conv_out, results ):
            out = conv( result["features"] )
            result["result"] = out

        
        return results

    def initialize( self ):
        # Set the last convolution close to all zeros. This should be a good initial guesss:
        for conv in self.conv_out:
            #conv.bias.data.fill_(0)
            conv.bias.data *= 0.01
            conv.weight.data *= 0.1
            #conv.weight.data *= 0.01


def subsampled_prediction_loss( displ, predictions ):

    displ_intraop_dummy = torch.zeros_like( displ )
    displ_full = torch.cat( (displ, displ_intraop_dummy), dim=2 )
    
    preop_point_mask = torch.cat( (torch.ones_like(displ), torch.zeros_like(displ)), dim=2 )

    displ_current_level = displ_full

    full_loss = 0
   
    # Traverse in order from high res (original) point cloud to lower res (subsampled) levels:
    predictions = [p for p in reversed(predictions)]
    # Weigh depending on number of levels. This makes losses between runs slightly more comparable.
    # Might not be optimal, though.
    num_predictions = len(predictions)
    weights = [1/num_predictions for p in predictions]
    # Weigh the main, full-scale displacement field with a higher factor:
    weights[0] = 10

    for level, level_result in enumerate( predictions ):
        prediction = level_result["result"]
        if level == 0:
            prediction = torch.cat( (prediction, displ_intraop_dummy), dim=2 )

        #loss = (((prediction - displ_current_level)*preop_point_mask)**2).mean()        # MSE
        loss = (((prediction - displ_current_level))**2).mean()        # MSE
        print("\tlevel, loss", level, loss.item())
        full_loss += weights[level]*loss

        if level < len(predictions)-1:
            next_level_idx = predictions[level+1]["point_idx"]
            displ_current_level = select_points( points = displ_current_level, idx = next_level_idx )
            preop_point_mask = select_points( points = preop_point_mask, idx = next_level_idx )

    return full_loss

def subsampled_prediction_loss_2( displ, predictions ):
    """ Similar to subsampled_prediction_loss, but only for preop points """

    displ_current_level = displ

    full_loss = 0
   
    # Traverse in order from high res (original) point cloud to lower res (subsampled) levels:
    predictions = [p for p in reversed(predictions)]
    # Weigh depending on number of levels. This makes losses between runs slightly more comparable.
    # Might not be optimal, though.
    num_predictions = len(predictions)
    #weights = [1/num_predictions for p in predictions]
    weights = [1 for p in predictions]
    # Weigh the main, full-scale displacement field with a higher factor:
    weights[0] = 5

    for level, level_result in enumerate( predictions ):
        prediction = level_result["result"]
        #if level == 0:
        #    prediction = torch.cat( (prediction, displ_intraop_dummy), dim=2 )
        #loss = (((prediction - displ_current_level)*preop_point_mask)**2).mean()        # MSE
        loss = (((prediction - displ_current_level))**2).mean()        # MSE
        full_loss += weights[level]*loss

        if level < len(predictions)-1:
            next_level_idx = predictions[level+1]["point_idx"]
            displ_current_level = select_points( points = displ_current_level, idx = next_level_idx )
            #preop_point_mask = select_points( points = preop_point_mask, idx = next_level_idx )
    return full_loss



def subsampled_prediction_loss_3( displ, predictions ):
    """ Loss for both downsampled preop and intraop points """

    displ_current_level = displ

    full_loss = 0
   
    # Traverse in order from high res (original) point cloud to lower res (subsampled) levels:
    predictions = [p for p in reversed(predictions)]
    # Weigh depending on number of levels. This makes losses between runs slightly more comparable.
    # Might not be optimal, though.
    num_predictions = len(predictions)
    weights = [1/num_predictions for p in predictions]
    # Weigh the main, full-scale displacement field with a higher factor:
    weights[0] = 10

    for level, level_result in enumerate( predictions ):
        prediction = level_result["result"]
        #if level == 0:
        #    prediction = torch.cat( (prediction, displ_intraop_dummy), dim=2 )
        #loss = (((prediction - displ_current_level)*preop_point_mask)**2).mean()        # MSE
        loss = (((prediction - displ_current_level))**2).mean()        # MSE
        full_loss += weights[level]*loss

        if level < len(predictions)-1:
            next_level_idx = predictions[level+1]["point_preop_idx"]
            displ_current_level = select_points( points = displ_current_level, idx = next_level_idx )
            #preop_point_mask = select_points( points = preop_point_mask, idx = next_level_idx )

    return full_loss



def subsampled_prediction_loss_3_matching_cues(displ, predictions, cue_mask, cue_valid_bits, weight_cue=1):
    """ Loss for both downsampled preop and intraop points including MSE for all levels of displacement
        and loss item for matching cues 

    """

    displ_current_level = displ

    full_loss = 0
   
    # Traverse in order from high res (original) point cloud to lower res (subsampled) levels:
    predictions = [p for p in reversed(predictions)]
    # Weigh depending on number of levels. This makes losses between runs slightly more comparable.
    # Might not be optimal, though.
    num_predictions = len(predictions)
    #weights = [1/num_predictions for p in predictions]
    weights = [1 for p in predictions]
    # Weigh the main, full-scale displacement field with a higher factor:
    weights[0] = 5

    for level, level_result in enumerate( predictions ):
        prediction = level_result["result"]
        #if level == 0:
        #    prediction = torch.cat( (prediction, displ_intraop_dummy), dim=2 )
        #loss = (((prediction - displ_current_level)*preop_point_mask)**2).mean()        # MSE
        loss = (((prediction - displ_current_level))**2).mean()        # MSE
        full_loss += weights[level]*loss

        if level < len(predictions)-1:
            next_level_idx = predictions[level+1]["point_idx"]
            displ_current_level = select_points( points = displ_current_level, idx = next_level_idx )
            #preop_point_mask = select_points( points = preop_point_mask, idx = next_level_idx )

    # loss for matching cues, for now only a single pair of cues is supported
    # cue loss is calculated only using the points in the highest resolution
    # TODO 1. add support for multiple pairs of cues

    pred_level_0 = predictions[0]["result"]
    displ_level_0 = displ
    
    
    B = cue_mask.shape[0]
    N_cue = cue_mask.shape[1]
    # for idx_cue in range():
    #     mask = (mask > 0.5).repeat(1, 3, 1)
    #     cue_loss = (pred_level_0[..., mask>0.5] - displ_level_0[..., mask>0.5]).mean()
    #     print("idx_cue:", idx_cue, "cue_loss:", cue_loss)
    #     cue_loss_total += cue_loss.item()
    # for idx_cue in range(N_cue):
    #     mask = (cue_mask[:, idx_cue, :] > 0.5).unsqueeze(1).repeat(1, 3, 1)
    #     print("mask:", mask.shape, "pred_level_0[mask]:", pred_level_0[mask].shape, "displ_level_0[mask]:", displ_level_0[mask].shape)
    #     cue_loss = (pred_level_0[mask].reshape(B, 3, -1) - displ_level_0[mask].reshape(B, 3, -1)).mean()
    #     cue_loss_total += cue_loss

    cue_loss_total = 0
    for idx_b in range(B):
        cue_loss_b = 0
        N_cue_valid = 0
        for idx_cue in range(N_cue):
            if cue_valid_bits[idx_b, idx_cue] == 0:
                continue
            mask = (cue_mask[idx_b, idx_cue, :] > 0.5)
            # print("mask.shape:", mask.shape)
            # print("pred_level_0[idx_b, :, mask].shape", pred_level_0[idx_b, :, mask].shape)
            cue_loss = ((pred_level_0[idx_b, :, mask] - displ_level_0[idx_b, :, mask])**2).mean()
            # print(pred_level_0[idx_b, :, mask], displ_level_0[idx_b, :, mask])
            # print("idx_cue:", idx_cue, "cue_loss:", cue_loss)
            cue_loss_b += cue_loss
            N_cue_valid += 1

        if N_cue_valid > 0:
            cue_loss_total += cue_loss_b / N_cue_valid
        else:
            cue_loss_total += 0
    cue_loss_total = cue_loss_total / B
    # print("cue_loss_total:", cue_loss_total)
    full_loss += weight_cue * cue_loss_total
    

    # TODO 2. add support for multiple resolutions

    # pred_cue = predictions[0]["result"][:, ]
    

    return full_loss


