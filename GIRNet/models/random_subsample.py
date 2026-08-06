import numpy as np
import torch.nn as nn
import torch

class RandomSubsample( nn.Module ):
    """ Randomly subsample a point cloud.

    Optionally, choose new indices on every forward pass
    """

    def __init__( self, n_input_points, n_output_points, keep_indices = False ):
        """ 
        Args:
            n_input_points: int, number of points the forward function should expect
            n_output_points: int, number of points that should be chosen
            keep_indices: bool, whether or not to re-use the same indices on every forward call
        """
        nn.Module.__init__( self )

        assert n_input_points >= n_output_points

        self.n_input_points = n_input_points
        self.n_output_points = n_output_points

        self.sampled_inds = self.choose_indices()
        self.keep_indices = keep_indices

    def choose_indices( self ):
        """ Choose new indices, i.e. which points should stay in the output """

        sampled_inds_np = np.random.choice( range(self.n_input_points),
                size = (self.n_output_points,),
                replace = False )

        sampled_inds = torch.LongTensor( sampled_inds_np )
        sampled_inds.requires_grad = False

        return sampled_inds

    def forward( self, coords, features ):
        """ Subsample the point cloud.

        Note: The same indices will be used for 'coords' as well as 'features', so
        if a point is chosen by this module, both its coords and features will be in
        the output.

        Note: The same indices are used for every point cloud in the batch!

        Args:
            coords: Tensor [batch_size, 3, n_input_points]
            features: Tensor [batch_size, n_features, n_input_points]

        Returns:
            coords: Tensor [batch_size, 3, n_output_points]
            features: Tensor [batch_size, n_features, n_output_points]
            indices: LongTensor [n_output_points]
        """

        if not self.keep_indices:
            self.sampled_inds = self.choose_indices()

        subsampled_coords = coords[:,:,self.sampled_inds]
        subsamples_features = features[:,:,self.sampled_inds]
    
        return subsampled_coords, subsamples_features, self.sampled_inds

if __name__ == "__main__":

    batch_size = 3
    features = 5
    n_points = 10

    p = torch.rand( (batch_size, 3, n_points) )
    f = torch.rand( (batch_size, features, n_points) )

    n_output_points = 4

    print( "Keep indices:")
    subsample = RandomSubsample( n_points, n_output_points, keep_indices = True )

    for i in range(3):
        p_sub, f_sub, inds = subsample( p, f )

        print( "\tchosen indices:", inds)
        print( "\tsubsampled coords:", p_sub.shape)
        print( "\tsubsampled features:", f_sub.shape)

    print( "Shuffle indices:")
    subsample = RandomSubsample( n_points, n_output_points, keep_indices = False )

    for i in range(3):
        p_sub, f_sub, inds = subsample( p, f )

        print( "\tchosen indices:", inds)
        print( "\tsubsampled coords:", p_sub.shape)
        print( "\tsubsampled features:", f_sub.shape)


