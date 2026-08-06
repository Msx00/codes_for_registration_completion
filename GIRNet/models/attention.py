import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from models.positional_encoding import PositionalEncoder
from models.k_nearest_neighbors import k_nearest_neighbors
from models.select import select_point_regions, select_points



# def grad_hook(grad):
#     nan_count = torch.isnan(grad).sum().item()
#     if nan_count > 0:
#         print(f"NaN in gradient! Count: {nan_count}")
#     return grad



def create_grad_hook(name, mask=None):
    """生成一个带有名称标识的梯度钩子"""
    def grad_hook(grad):
        if torch.isnan(grad).any():
            nan_count = torch.isnan(grad).sum().item()
            print(f"!!!!!!!!!!!!!!!NaN detected in gradient of tensor: {name}", nan_count)
            # 可选：触发调试断点
            # import pdb; pdb.set_trace()
            if mask is not None:
                # get the number of valid / invalid points in the mask
                print("mask.shape", mask.shape, "valid:", mask.sum(), "invalid:", (~mask).sum())
            raise ValueError("NaN in gradient!", name, "number of nan", nan_count, "total number of grad", torch.numel(grad))
        # else:
        #     print(f"No NaN in gradient of tensor: {name}")
        return grad
    return grad_hook




class EmbedNetwork(nn.Module):
    def __init__(self, input_features, embedding_size=64):
        nn.Module.__init__( self )
        self.conv = nn.Conv1d(input_features, embedding_size, kernel_size=1, bias=True)

    def forward(self, inputs):
        """ Calculate an embeding for each vector in inputs, using the same weights.

        Args:
            inputs: Tensor of shape [B, input_features, N]
        Returns:
            Tensor of shape [B, embedding_size, N]
        """

        embedding = self.conv( inputs )

        return embedding

class MultiRegionEmbedNetwork(nn.Module):
    def __init__(self, input_features, embedding_size=64, use_non_lin=False):
        nn.Module.__init__( self )
        self.conv = nn.Conv2d(input_features, embedding_size, kernel_size=1, bias=True)

        if use_non_lin:
            self.non_lin = torch.nn.ReLU()
        else:
            self.non_lin = None

    def forward(self, inputs):
        """ Calculate an embeding for each vector in inputs, using the same weights.

        Args:
            inputs: Tensor of shape [B, input_features, Nv, Nq]
        Returns:
            Tensor of shape [B, embedding_size, Nv, Nq]
        """

        if self.non_lin:
            embedding = self.non_lin( self.conv( inputs ) )
        else:
            embedding = self.conv( inputs )

        return embedding



class Attention( nn.Module ):

    def __init__( self, n_value_features, n_query_features, embedding_size, n_output_features ):
        nn.Module.__init__( self )

        self.query_embedding = EmbedNetwork( n_query_features, embedding_size )
        self.key_embedding = EmbedNetwork( n_value_features, embedding_size )
        self.value_embedding = EmbedNetwork( n_value_features, n_output_features )

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features

        #self.multihead_attn = nn.MultiheadAttention(
        #        embedding_size,
        #        num_heads=1,
        #        bias=True,
        #        batch_first=True,
        #        kdim = n_value_features,
        #        vdim = n_value_features )


    def forward( self, inp_values, inp_queries ):
        """ Compute attention on features given by "inp_values", attended to/by/at "inp_queries"

        Args:
            inp_values: Tensor of shape [B, Dv, Nv]  (B: batch, Dv: n_value_features, Nv: number of input points)
            inp_queries: Tensor of shape [B, Dq, Nq]  (B: batch, Dq: n_query_features, Nq: number of query/output points)

        Returns:
            Tensor of shape [B, D', Nq] (B: batch, D': n_output_features, Nq: number of points)
        """
        n_value_points = inp_values.shape[2]      # Nv
        n_query_points = inp_queries.shape[2]     # Nq

        Q = self.query_embedding( inp_queries )     # [B, embedding_size, Nq]
        K = self.key_embedding( inp_values )       # [B, embedding_size, Nv]
        V = self.value_embedding( inp_values )     # [B, n_output_features, Nv]

        B = inp_queries.shape[0]    # batch_size

        ###############
        #inp_queries = torch.cat( (inp_queries, torch.zeros( (B,3, n_query_points), device=inp_queries.device )), dim=1 )
        #print("inp_queries", inp_queries.permute(0,2,1).shape)
        #print("inp_values", inp_values.permute(0,2,1).shape)
        #attn_output, attn_output_weights = self.multihead_attn( inp_queries.permute(0,2,1), inp_values.permute(0,2,1), inp_values.permute(0,2,1))
        #print(attn_output.shape)
        #print(attn_output_weights.shape)

        K = K.unsqueeze(3).repeat( 1, 1, 1, n_query_points )  # [B, embedding_size, Nq, Nv]
        Q = Q.unsqueeze(2).repeat( 1, 1, n_value_points, 1 )  # [B, embedding_size, Nq, Nv]

        weights = (K*Q).sum( dim=1 ) # Dot product
        # Scale: See "Attention is all you need" paper (they call the value d_k):
        weights = weights/math.sqrt( self.embedding_size )         
        weights = F.softmax( weights, dim=1 )       # Softwmax over the _value_ entries

        # Apply the attention to the values:
        #weights_rep = weights.unsqueeze(1).repeat(1, self.n_output_features, 1, 1)
        #V = V.unsqueeze(3)
        #result = V * weights_rep
        #result = result.sum( dim=2 )
        result = torch.bmm( weights.permute(0,2,1), V.permute(0,2,1) )
        result = result.permute( 0, 2, 1 )

        return result
        #return attn_output.permute( 0,2,1 )

class SelfAttention( nn.Module ):
    """ An attention module where value and query points are the same
    """
    def __init__( self, n_value_features, embedding_size, n_output_features,
            concat_coords = True, use_positional_encoding = True ):

        nn.Module.__init__( self )

        self.concat_coords = concat_coords
        self.use_positional_encoding = use_positional_encoding

        n_query_features = n_value_features

        if self.use_positional_encoding:
            self.encoder = PositionalEncoder( frequencies = [1e-2, 1e-1, 2, 4, 8, 16, 32, 64, 128, 256], scale=1 )
            n_query_features += self.encoder.num_features
            n_value_features += self.encoder.num_features

        if self.concat_coords:
            n_query_features += 3
            n_value_features += 3

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features

        self.attn = Attention( n_value_features, n_query_features, embedding_size, n_output_features )

    def forward( self, value_coords, value_features ):

        """ Compute attention on features given by "inp_values", attended to/by/at "inp_queries"

        Note: if concat_coords is False, this module will not use the value_coords and query_coords.

        Args:
            value_coords: Tensor of shape [B, 3, Nv]
            value_features: Tensor of shape [B, Dv, Nv]
                (B: batch,
                Dv: n_value_features,
                Nv: number of input points,

        Returns:
            Tensor of shape [B, D', Nv] (B: batch, D': n_output_features, Nv: number of query points)
        """

        n_value_points = value_features.shape[2]      # Nv
        n_query_points = n_value_points

        if self.concat_coords:
            values = torch.cat( (value_coords, value_features), dim=1 )
        else:   # Otherwise only use the feature vectors, not the coordinates:
            values = value_features

        if self.use_positional_encoding:
            encoded_value_coords = self.encoder( value_coords )
            # concatenate the encoded positions as features:
            values = torch.cat( (values, encoded_value_coords), dim=1 )

        result = self.attn( values, values )

        return result


class MultiRegionAttention( nn.Module ):
    """ An attention module where each query point attends to it's own set of value points.


    This is a module intended to be used for processing point clouds, where a "new" set of points
    (i.e. a subsequent layer) attends to an "old" set of points (i.e. the previous layer). Here a 
    "new" point should not attend to all points, but rather only to the points within its spatial
    neighbourhood (usually the k nearest neighbors). Note that these neighborhoods may overlap, so
    the "old" point cloud may contain a point multiple times, in different neighborhoods.

    In contrast to this, the Attention module above will let each query point attend to each value
    point.

    Note that this should effectively compute the same thing as the simpler "Attention" module, but
    for many regions at once (while the "Attention" module can only compute it for a single region).
    """

    def __init__( self,
            n_value_features,
            n_query_features,
            embedding_size,
            n_output_features,
            concat_absolute_coords_raw = True,
            concat_absolute_coords_encoded = True,
            concat_relative_coords_raw = True,
            concat_relative_coords_encoded = True,
            concat_queries_to_values = True,
            attention_mode = "additive", 
            return_weights=False ):
        """
        Args:
            concat_coords: If set to True, the coordinates of the value points will be concatinated
                to the value features and the query point coordinates will be concatinated to the 
                query features before calculating attention. In this way, point coordinates can
                play a role in the attention mechanism. If concat_coords is false, only the features
                will be used to calculate attention, not the point coordinates.
            use_relative_coords: TODO
            attention_mode: str, "additive" or "multiplicative"
        """

        nn.Module.__init__( self )

        self.concat_absolute_coords_raw = concat_absolute_coords_raw
        self.concat_absolute_coords_encoded = concat_absolute_coords_encoded
        self.concat_relative_coords_raw = concat_relative_coords_raw
        self.concat_relative_coords_encoded = concat_relative_coords_encoded
        self.concat_queries_to_values = concat_queries_to_values
        self.n_query_features_raw = n_query_features
        self.n_value_features_raw = n_value_features


        # if self.concat_absolute_coords_encoded or self.concat_relative_coords_encoded:
        if self.concat_relative_coords_encoded:
            self.encoder = PositionalEncoder( frequencies = [2e-2, 2e-1, 2, 4, 8, 16, 32, 64], scale=1 )
        if self.concat_absolute_coords_raw:
            n_query_features += 3
            n_value_features += 3
        if self.concat_relative_coords_raw:
            n_value_features += 3

        if self.concat_absolute_coords_encoded:
            n_query_features += self.encoder.num_features
            n_value_features += self.encoder.num_features
        if self.concat_relative_coords_encoded:
            n_value_features += self.encoder.num_features

        if self.concat_queries_to_values:
            n_value_features += n_query_features

        self.query_embedding = EmbedNetwork( n_query_features, embedding_size )
        self.key_embedding = MultiRegionEmbedNetwork( n_value_features, embedding_size )
        self.value_embedding = MultiRegionEmbedNetwork( n_value_features, n_output_features )

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features

        self.attention_mode = attention_mode

        # collapse embedding size to 1
        if self.attention_mode == "additive":
            self.v = nn.Conv2d(self.embedding_size, 1, kernel_size=1, bias=True)

        self.return_weights = return_weights

    def forward( self, value_coords, value_features, queries_coords, queries_features ):
        # TODO update comments here
        """ Compute attention on features given by "inp_values", attended to/by/at "inp_queries"

        Note: if concat_coords is False, this module will not use the value_coords and query_coords.

        Args:
            value_coords: Tensor of shape [B, 3, Nv, Nq]
            value_features: Tensor of shape [B, Dv, Nv, Nq]
                (B: batch,
                Dv: n_value_features,
                Nv: number of input points,
                Nq: numbr of query/output points)
            queries_coords: Tensor of shape [B, 3, Nq]
            queries_features: Tensor of shape [B, Dq, Nq]
                (B: batch,
                Dq: n_query_features,
                Nq: number of query/output points)

        Returns:
            Tensor of shape [B, D', Nq] (B: batch, D': n_output_features, Nq: number of query points)
        """
        
        # print("value_coords.shape", value_coords.shape)
        # print("value_features.shape", value_features.shape)
        # print("queries_coords.shape", queries_coords.shape)
        # print("queries_features.shape", queries_features.shape)

        n_value_points = value_features.shape[2]      # Nv
        n_query_points = queries_features.shape[2]     # Nq

        # print("self.n_query_features_raw", self.n_query_features_raw,)
        # print("queries_features.shape", queries_features.shape)
        # print("value_features.shape", value_features.shape)
        assert value_features.shape[3] == n_query_points, "There must be a set of value points for _every_ query point!"
        assert queries_features.shape[1] == self.n_query_features_raw, "The number of query features ({}) must be the same as the number of query features given in the constructor ({})!".format(queries_features.shape[1], self. n_query_features_raw)
        assert value_features.shape[1] == self.n_value_features_raw, "The number of value features ({}) must be the same as the number of value features given in the constructor ({})!".format(value_features.shape[1], self.n_value_features_raw)
    
        # By default, use only the input features as features:
        queries = queries_features
        values = value_features

        # Optionally, append the coordinates of queries and values as features:
        if self.concat_absolute_coords_raw:
            queries = torch.cat( (queries, queries_coords[:, 0:3, ...]), dim=1 )
            values = torch.cat( (values, value_coords[:, 0:3, ...]), dim=1 )
          
        # Optionally, append the encoded coordinates of queries and values as features:
        if self.concat_absolute_coords_encoded:
            # queries_coords_encoded = self.encoder( queries_coords ) # these are for the old models which have PE calculated everytime right before attention
            # value_coords_encoded = self.encoder( value_coords )
            queries_coords_encoded = queries_coords[:, 3:, ...] # these are for models with new preprocesser which PE is calcuated only once during preprocessing
            value_coords_encoded = value_coords[:, 3:, ...]

            queries = torch.cat( (queries, queries_coords_encoded), dim=1 )
            values = torch.cat( (values, value_coords_encoded), dim=1 )

        if self.concat_relative_coords_raw or self.concat_relative_coords_encoded:
            # For the value coordinates in each region, subtract the corresponding query's coordinate
            # value_coords_rel = value_coords - queries_coords.unsqueeze(2)
            value_coords_rel = value_coords[:, 0:3, ...] - queries_coords[:, 0:3, ...].unsqueeze(2)
            if self.concat_relative_coords_raw:
                values = torch.cat( (values, value_coords_rel), dim=1 )
            if self.concat_relative_coords_encoded:
                value_coords_rel_encoded = self.encoder( value_coords_rel )
                values = torch.cat( (values, value_coords_rel_encoded), dim=1 )

        if self.concat_queries_to_values:
            expanded_queries = queries.unsqueeze( dim=2 ).repeat( 1,1, n_value_points, 1 )
            values = torch.cat( (values, expanded_queries), dim=1 )


        Q = self.query_embedding( queries )     # [B, embedding_size, Nq]
        K = self.key_embedding( values )       # [B, embedding_size, Nv, Nq]

        # TODO: Is this really required here, or would it be enough to encode the vectors _after_ the
        # weighting and summation? Would be much less expensive, but if there's a non-linearity involved,
        # the operations are not cummutative... but are they, if there's no non-linearity?
        V = self.value_embedding( values )     # [B, output_features, Nv, Nq]

        #K = K.unsqueeze(3).repeat( 1, 1, 1, n_query_points )  # [B, embedding_size, Nq, Nv]
        Q = Q.unsqueeze(2).repeat( 1, 1, n_value_points, 1 )  # [B, embedding_size, Nv, Nq]

        if self.attention_mode == "additive":
            weights = self.v( torch.tanh( K + Q ) )
            weights = weights.squeeze( dim=1 )
        else:
            weights = (K*Q).sum( dim=1 ) # Dot product (sum over embedding_size dimension)
            # Scale: See "Attention is all you need" paper (they call the value d_k):
            weights = weights/math.sqrt( self.embedding_size )     # [B, Nv, Nq]   
        weights = F.softmax( weights, dim=1 )       # Softwmax over the _value_ entries

        # Apply the attention to the values:
        weights_rep = weights.unsqueeze(1).repeat(1, V.shape[1], 1, 1)  # [B, D', Nv, Nq]
        #print("weigths_rep", weights_rep.shape)
        #V = V.unsqueeze(3)
        result = V * weights_rep
        result = result.sum( dim=2 )        # Sum together the value points (for each query point)

        # print("values.max:", values.max(), "values.min:", values.min())
        # print("V.max:", V.max(), "V.min:", V.min())
        # print("Q.max:", Q.max(), "Q.min:", Q.min())
        # print("K.max:", K.max(), "K.min:", K.min())
        # print("result.max:", result.max(), "result.min:", result.min())
        if self.return_weights:
            return result, weights
        else:
            return result
        



class MultiRegionAttentionDisentangled( nn.Module ):
    """ An attention module where each query point attends to it's own set of value points.


    This is a module intended to be used for processing point clouds, where a "new" set of points
    (i.e. a subsequent layer) attends to an "old" set of points (i.e. the previous layer). Here a 
    "new" point should not attend to all points, but rather only to the points within its spatial
    neighbourhood (usually the k nearest neighbors). Note that these neighborhoods may overlap, so
    the "old" point cloud may contain a point multiple times, in different neighborhoods.

    In contrast to this, the Attention module above will let each query point attend to each value
    point.

    Note that this should effectively compute the same thing as the simpler "Attention" module, but
    for many regions at once (while the "Attention" module can only compute it for a single region).
    """

    def __init__( self,
            n_value_features,
            n_query_features,
            embedding_size,
            n_output_features,
            concat_absolute_coords_raw = True,
            concat_absolute_coords_encoded = True,
            concat_relative_coords_raw = True,
            concat_relative_coords_encoded = True,
            concat_queries_to_values = True,
            attention_mode = "additive", 
            return_weights=False ):
        """
        Args:
            concat_coords: If set to True, the coordinates of the value points will be concatinated
                to the value features and the query point coordinates will be concatinated to the 
                query features before calculating attention. In this way, point coordinates can
                play a role in the attention mechanism. If concat_coords is false, only the features
                will be used to calculate attention, not the point coordinates.
            use_relative_coords: TODO
            attention_mode: str, "additive" or "multiplicative"
        """

        nn.Module.__init__( self )

        self.concat_absolute_coords_raw = concat_absolute_coords_raw
        self.concat_absolute_coords_encoded = concat_absolute_coords_encoded
        self.concat_relative_coords_raw = concat_relative_coords_raw
        self.concat_relative_coords_encoded = concat_relative_coords_encoded
        self.concat_queries_to_values = concat_queries_to_values
        self.n_query_features_raw = n_query_features
        self.n_key_features_raw = n_value_features
        self.n_value_features_raw = n_value_features

        n_key_features = n_value_features

        # if self.concat_absolute_coords_encoded or self.concat_relative_coords_encoded:
        if self.concat_relative_coords_encoded:
            self.encoder = PositionalEncoder( frequencies = [2e-2, 2e-1, 2, 4, 8, 16, 32, 64], scale=1 )
        if self.concat_absolute_coords_raw:
            n_query_features += 3
            # n_value_features += 3
            n_key_features +=3
        if self.concat_relative_coords_raw:
            n_value_features += 3

        if self.concat_absolute_coords_encoded:
            n_query_features += self.encoder.num_features
            # n_value_features += self.encoder.num_features
            n_key_features += self.encoder.num_features
        if self.concat_relative_coords_encoded:
            n_value_features += self.encoder.num_features

        # if self.concat_queries_to_values:
        #     n_value_features += n_query_features

        self.query_embedding = EmbedNetwork( n_query_features, embedding_size )
        self.key_embedding = MultiRegionEmbedNetwork( n_value_features, embedding_size )
        self.value_embedding = MultiRegionEmbedNetwork( n_value_features, n_output_features )

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features

        self.attention_mode = attention_mode

        # collapse embedding size to 1
        if self.attention_mode == "additive":
            self.v = nn.Conv2d(self.embedding_size, 1, kernel_size=1, bias=True)

        self.return_weights = return_weights

    def forward( self, value_coords, value_features, queries_coords, queries_features ):
        # TODO update comments here
        """ Compute attention on features given by "inp_values", attended to/by/at "inp_queries"

        Note: if concat_coords is False, this module will not use the value_coords and query_coords.

        Args:
            value_coords: Tensor of shape [B, 3, Nv, Nq]
            value_features: Tensor of shape [B, Dv, Nv, Nq]
                (B: batch,
                Dv: n_value_features,
                Nv: number of input points,
                Nq: numbr of query/output points)
            queries_coords: Tensor of shape [B, 3, Nq]
            queries_features: Tensor of shape [B, Dq, Nq]
                (B: batch,
                Dq: n_query_features,
                Nq: number of query/output points)

        Returns:
            Tensor of shape [B, D', Nq] (B: batch, D': n_output_features, Nq: number of query points)
        """
        
        # print("value_coords.shape", value_coords.shape)
        # print("value_features.shape", value_features.shape)
        # print("queries_coords.shape", queries_coords.shape)
        # print("queries_features.shape", queries_features.shape)

        n_value_points = value_features.shape[2]      # Nv
        n_query_points = queries_features.shape[2]     # Nq

        # print("self.n_query_features_raw", self.n_query_features_raw,)
        # print("queries_features.shape", queries_features.shape)
        # print("value_features.shape", value_features.shape)
        assert value_features.shape[3] == n_query_points, "There must be a set of value points for _every_ query point!"
        assert queries_features.shape[1] == self.n_query_features_raw, "The number of query features ({}) must be the same as the number of query features given in the constructor ({})!".format(queries_features.shape[1], self. n_query_features_raw)
        assert value_features.shape[1] == self.n_value_features_raw, "The number of value features ({}) must be the same as the number of value features given in the constructor ({})!".format(value_features.shape[1], self.n_value_features_raw)
    
        # By default, use only the input features as features:
        queries = queries_features
        values = value_features
        keys = value_features

        # Optionally, append the coordinates of queries and values as features:
        if self.concat_absolute_coords_raw:
            queries = torch.cat( (queries, queries_coords[:, 0:3, ...]), dim=1 )
            # values = torch.cat( (values, value_coords[:, 0:3, ...]), dim=1 )
            keys = torch.cat( (keys, value_coords[:, 0:3, ...]), dim=1 )
          
        # Optionally, append the encoded coordinates of queries and values as features:
        if self.concat_absolute_coords_encoded:
            # queries_coords_encoded = self.encoder( queries_coords ) # these are for the old models which have PE calculated everytime right before attention
            # value_coords_encoded = self.encoder( value_coords )
            queries_coords_encoded = queries_coords[:, 3:, ...] # these are for models with new preprocesser which PE is calcuated only once during preprocessing
            # value_coords_encoded = value_coords[:, 3:, ...]
            keys_coords_encoded = value_coords[:, 3:, ...]

            queries = torch.cat( (queries, queries_coords_encoded), dim=1 )
            # values = torch.cat( (values, value_coords_encoded), dim=1 )
            keys = torch.cat( (keys, keys_coords_encoded), dim=1 )

        if self.concat_relative_coords_raw or self.concat_relative_coords_encoded:
            # For the value coordinates in each region, subtract the corresponding query's coordinate
            # value_coords_rel = value_coords - queries_coords.unsqueeze(2)
            value_coords_rel = value_coords[:, 0:3, ...] - queries_coords[:, 0:3, ...].unsqueeze(2)
            if self.concat_relative_coords_raw:
                values = torch.cat( (values, value_coords_rel), dim=1 )
            if self.concat_relative_coords_encoded:
                value_coords_rel_encoded = self.encoder( value_coords_rel )
                values = torch.cat( (values, value_coords_rel_encoded), dim=1 )

        # if self.concat_queries_to_values:
        #     expanded_queries = queries.unsqueeze( dim=2 ).repeat( 1,1, n_value_points, 1 )
        #     values = torch.cat( (values, expanded_queries), dim=1 )


        Q = self.query_embedding( queries )     # [B, embedding_size, Nq]
        # K = self.key_embedding( values )       # [B, embedding_size, Nv, Nq]
        K = self.key_embedding( keys )       # [B, embedding_size, Nv, Nq]

        # TODO: Is this really required here, or would it be enough to encode the vectors _after_ the
        # weighting and summation? Would be much less expensive, but if there's a non-linearity involved,
        # the operations are not cummutative... but are they, if there's no non-linearity?
        V = self.value_embedding( values )     # [B, output_features, Nv, Nq]

        #K = K.unsqueeze(3).repeat( 1, 1, 1, n_query_points )  # [B, embedding_size, Nq, Nv]
        Q = Q.unsqueeze(2).repeat( 1, 1, n_value_points, 1 )  # [B, embedding_size, Nv, Nq]

        if self.attention_mode == "additive":
            weights = self.v( torch.tanh( K + Q ) )
            weights = weights.squeeze( dim=1 )
        else:
            weights = (K*Q).sum( dim=1 ) # Dot product (sum over embedding_size dimension)
            # Scale: See "Attention is all you need" paper (they call the value d_k):
            weights = weights/math.sqrt( self.embedding_size )     # [B, Nv, Nq]   
        weights = F.softmax( weights, dim=1 )       # Softwmax over the _value_ entries

        # Apply the attention to the values:
        weights_rep = weights.unsqueeze(1).repeat(1, V.shape[1], 1, 1)  # [B, D', Nv, Nq]
        #print("weigths_rep", weights_rep.shape)
        #V = V.unsqueeze(3)
        result = V * weights_rep
        result = result.sum( dim=2 )        # Sum together the value points (for each query point)

        # print("values.max:", values.max(), "values.min:", values.min())
        # print("V.max:", V.max(), "V.min:", V.min())
        # print("Q.max:", Q.max(), "Q.min:", Q.min())
        # print("K.max:", K.max(), "K.min:", K.min())
        # print("result.max:", result.max(), "result.min:", result.min())
        if self.return_weights:
            return result, weights
        else:
            return result




class MultiRegionAttentionDry( nn.Module ):
    """ An attention module where each query point attends to it's own set of value points.
    
    !!!Note: this is a "dryer" verision of MultiRegionAttention, which means no preprocessing featurres
    are calculated, and the input features are directly used for attention calculation.!!!

    This is a module intended to be used for processing point clouds, where a "new" set of points
    (i.e. a subsequent layer) attends to an "old" set of points (i.e. the previous layer). Here a 
    "new" point should not attend to all points, but rather only to the points within its spatial
    neighbourhood (usually the k nearest neighbors). Note that these neighborhoods may overlap, so
    the "old" point cloud may contain a point multiple times, in different neighborhoods.

    In contrast to this, the Attention module above will let each query point attend to each value
    point.

    Note that this should effectively compute the same thing as the simpler "Attention" module, but
    for many regions at once (while the "Attention" module can only compute it for a single region).
    """

    def __init__( self,
            n_value_features,
            n_query_features,
            embedding_size,
            n_output_features,
            attention_mode = "additive", 
            return_weights=False ):
        """
        Args:
            concat_coords: If set to True, the coordinates of the value points will be concatinated
                to the value features and the query point coordinates will be concatinated to the 
                query features before calculating attention. In this way, point coordinates can
                play a role in the attention mechanism. If concat_coords is false, only the features
                will be used to calculate attention, not the point coordinates.
            use_relative_coords: TODO
            attention_mode: str, "additive" or "multiplicative"
        """

        nn.Module.__init__( self )

        self.n_query_features_raw = n_query_features
        self.n_value_features_raw = n_value_features

        # if self.concat_absolute_coords_encoded or self.concat_relative_coords_encoded:
        # if self.concat_relative_coords_encoded:
        #     self.encoder = PositionalEncoder( frequencies = [1e-2, 1e-1, 2, 4, 8, 16, 32, 64], scale=1 )
        # if self.concat_absolute_coords_raw:
        #     n_query_features += 3
        #     n_value_features += 3
        # if self.concat_relative_coords_raw:
        #     n_value_features += 3

        # if self.concat_absolute_coords_encoded:
        #     n_query_features += self.encoder.num_features
        #     n_value_features += self.encoder.num_features
        # if self.concat_relative_coords_encoded:
        #     n_value_features += self.encoder.num_features

        # if self.concat_queries_to_values:
        #     n_value_features += n_query_features

        self.query_embedding = EmbedNetwork( n_query_features, embedding_size )
        self.key_embedding = MultiRegionEmbedNetwork( n_value_features, embedding_size )
        self.value_embedding = MultiRegionEmbedNetwork( n_value_features, n_output_features )

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features

        self.attention_mode = attention_mode

        if self.attention_mode == "additive":
            self.v = nn.Conv2d(self.embedding_size, 1, kernel_size=1, bias=True)

        self.return_weights = return_weights

    def forward( self, value_coords, value_features, queries_coords, queries_features ):

        """ Compute attention on features given by "inp_values", attended to/by/at "inp_queries"

        Note: if concat_coords is False, this module will not use the value_coords and query_coords.

        Args:
            value_coords: Tensor of shape [B, 3, Nv, Nq]
            value_features: Tensor of shape [B, Dv, Nv, Nq]
                (B: batch,
                Dv: n_value_features,
                Nv: number of input points,
                Nq: numbr of query/output points)
            query_coords: Tensor of shape [B, 3, Nq]
            queries_features: Tensor of shape [B, Dq, Nq]
                (B: batch,
                Dq: n_query_features,
                Nq: number of query/output points)

        Returns:
            Tensor of shape [B, D', Nq] (B: batch, D': n_output_features, Nq: number of query points)
        """

        n_value_points = value_features.shape[2]      # Nv
        n_query_points = queries_features.shape[2]     # Nq

        assert value_features.shape[3] == n_query_points, "There must be a set of value points for _every_ query point!"
        assert queries_features.shape[1] == self.n_query_features_raw, "The number of query features must be the same as the number of query features given in the constructor!"
        assert value_features.shape[1] == self.n_value_features_raw, "The number of value features must be the same as the number of value features given in the constructor!"
    
        # By default, use only the input features as features:
        queries = queries_features
        values = value_features

        # # Optionally, append the coordinates of queries and values as features:
        # if self.concat_absolute_coords_raw:
        #     queries = torch.cat( (queries, queries_coords), dim=1 )
        #     values = torch.cat( (values, value_coords), dim=1 )
          
        # # Optionally, append the encoded coordinates of queries and values as features:
        # if self.concat_absolute_coords_encoded:
        #     queries_coords_encoded = self.encoder( queries_coords )
        #     value_coords_encoded = self.encoder( value_coords )

        #     queries = torch.cat( (queries, queries_coords_encoded), dim=1 )
        #     values = torch.cat( (values, value_coords_encoded), dim=1 )

        # if self.concat_relative_coords_raw or self.concat_relative_coords_encoded:
        #     # For the value coordinates in each region, subtract the corresponding query's coordinate
        #     value_coords_rel = value_coords - queries_coords.unsqueeze(2)
        #     if self.concat_relative_coords_raw:
        #         values = torch.cat( (values, value_coords_rel), dim=1 )
        #     if self.concat_relative_coords_encoded:
        #         value_coords_rel_encoded = self.encoder( value_coords_rel )
        #         values = torch.cat( (values, value_coords_rel_encoded), dim=1 )

        # if self.concat_queries_to_values:
        #     expanded_queries = queries.unsqueeze( dim=2 ).repeat( 1,1, n_value_points, 1 )
        #     values = torch.cat( (values, expanded_queries), dim=1 )


        Q = self.query_embedding( queries )     # [B, embedding_size, Nq]
        K = self.key_embedding( values )       # [B, embedding_size, Nv, Nq]

        # TODO: Is this really required here, or would it be enough to encode the vectors _after_ the
        # weighting and summation? Would be much less expensive, but if there's a non-linearity involved,
        # the operations are not cummutative... but are they, if there's no non-linearity?
        V = self.value_embedding( values )     # [B, output_features, Nv, Nq]

        #K = K.unsqueeze(3).repeat( 1, 1, 1, n_query_points )  # [B, embedding_size, Nq, Nv]
        Q = Q.unsqueeze(2).repeat( 1, 1, n_value_points, 1 )  # [B, embedding_size, Nv, Nq]

        if self.attention_mode == "additive":
            weights = self.v( torch.tanh( K + Q ) )
            weights = weights.squeeze( dim=1 )
        else:
            weights = (K*Q).sum( dim=1 ) # Dot product (sum over embedding_size dimension)
            # Scale: See "Attention is all you need" paper (they call the value d_k):
            weights = weights/math.sqrt( self.embedding_size )     # [B, Nv, Nq]   
        weights = F.softmax( weights, dim=1 )       # Softwmax over the _value_ entries

        # Apply the attention to the values:
        weights_rep = weights.unsqueeze(1).repeat(1, V.shape[1], 1, 1)  # [B, D', Nv, Nq]
        #print("weigths_rep", weights_rep.shape)
        #V = V.unsqueeze(3)
        result = V * weights_rep
        result = result.sum( dim=2 )        # Sum together the value points (for each query point)

        if self.return_weights:
            return result, weights
        else:
            return result







class MultiRegionAttentionMask( nn.Module ):
    """ An attention module where each query point attends to it's own set of value points.


    This is a module intended to be used for processing point clouds, where a "new" set of points
    (i.e. a subsequent layer) attends to an "old" set of points (i.e. the previous layer). Here a 
    "new" point should not attend to all points, but rather only to the points within its spatial
    neighbourhood (usually the k nearest neighbors). Note that these neighborhoods may overlap, so
    the "old" point cloud may contain a point multiple times, in different neighborhoods.

    In contrast to this, the Attention module above will let each query point attend to each value
    point.

    Note that this should effectively compute the same thing as the simpler "Attention" module, but
    for many regions at once (while the "Attention" module can only compute it for a single region).
    """

    def __init__( self,
            n_value_features,
            n_query_features,
            embedding_size,
            n_output_features,
            concat_absolute_coords_raw = True,
            concat_absolute_coords_encoded = True,
            concat_relative_coords_raw = True,
            concat_relative_coords_encoded = True,
            concat_queries_to_values = True,
            attention_mode = "additive", 
            # attention_mode = "multi",
            return_weights=False,
            func_name="", ):
        """
        Args:
            concat_coords: If set to True, the coordinates of the value points will be concatinated
                to the value features and the query point coordinates will be concatinated to the 
                query features before calculating attention. In this way, point coordinates can
                play a role in the attention mechanism. If concat_coords is false, only the features
                will be used to calculate attention, not the point coordinates.
            use_relative_coords: TODO
            attention_mode: str, "additive" or "multiplicative"
        """

        nn.Module.__init__( self )

        self.concat_absolute_coords_raw = concat_absolute_coords_raw
        self.concat_absolute_coords_encoded = concat_absolute_coords_encoded
        self.concat_relative_coords_raw = concat_relative_coords_raw
        self.concat_relative_coords_encoded = concat_relative_coords_encoded
        self.concat_queries_to_values = concat_queries_to_values
        self.n_query_features_raw = n_query_features
        self.n_value_features_raw = n_value_features


        if self.concat_absolute_coords_encoded or self.concat_relative_coords_encoded:
        # if self.concat_relative_coords_encoded:
            self.encoder = PositionalEncoder( frequencies = [2e-2, 2e-1, 2, 4, 8, 16, 32, 64], scale=1 )
        if self.concat_absolute_coords_raw:
            n_query_features += 3
            n_value_features += 3
        if self.concat_relative_coords_raw:
            n_value_features += 3

        if self.concat_absolute_coords_encoded:
            n_query_features += self.encoder.num_features
            n_value_features += self.encoder.num_features
        if self.concat_relative_coords_encoded:
            n_value_features += self.encoder.num_features

        if self.concat_queries_to_values:
            n_value_features += n_query_features

        self.query_embedding = EmbedNetwork( n_query_features, embedding_size )
        self.key_embedding = MultiRegionEmbedNetwork( n_value_features, embedding_size )
        self.value_embedding = MultiRegionEmbedNetwork( n_value_features, n_output_features )

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features

        self.attention_mode = attention_mode

        # collapse embedding size to 1
        if self.attention_mode == "additive":
            self.v = nn.Conv2d(self.embedding_size, 1, kernel_size=1, bias=True)

        self.return_weights = return_weights
        self.func_name = func_name

    def forward( self, value_coords, value_features, queries_coords, queries_features ):
        # TODO update comments here
        """ Compute attention on features given by "inp_values", attended to/by/at "inp_queries"

        Note: if concat_coords is False, this module will not use the value_coords and query_coords.

        Args:
            value_coords: Tensor of shape [B, 3, Nv, Nq]
            value_features: Tensor of shape [B, Dv, Nv, Nq]
                (B: batch,
                Dv: n_value_features,
                Nv: number of input points,
                Nq: numbr of query/output points)
            queries_coords: Tensor of shape [B, 3, Nq]
            queries_features: Tensor of shape [B, Dq, Nq]
                (B: batch,
                Dq: n_query_features,
                Nq: number of query/output points)

        Returns:
            Tensor of shape [B, D', Nq] (B: batch, D': n_output_features, Nq: number of query points)
        """
        
        # print("value_coords.shape", value_coords.shape)
        # print("value_features.shape", value_features.shape)
        # print("queries_coords.shape", queries_coords.shape)
        # print("queries_features.shape", queries_features.shape)

        n_value_points = value_features.shape[2]      # Nv
        n_query_points = queries_features.shape[2]     # Nq

        # print("self.n_query_features_raw", self.n_query_features_raw,)
        # print("queries_features.shape", queries_features.shape)
        # print("value_features.shape", value_features.shape)
        assert value_features.shape[3] == n_query_points, "There must be a set of value points for _every_ query point!"
        assert queries_features.shape[1] == self.n_query_features_raw, "The number of query features ({}) must be the same as the number of query features given in the constructor ({})!".format(queries_features.shape[1], self. n_query_features_raw)
        assert value_features.shape[1] == self.n_value_features_raw, "The number of value features ({}) must be the same as the number of value features given in the constructor ({})!".format(value_features.shape[1], self.n_value_features_raw)
    
        # By default, use only the input features as features:
        queries = queries_features
        values = value_features

        query_mask = (queries_coords.abs()[:, 0:3, ...] <= 1000).all(dim=1) 
        value_mask = (value_coords.abs()[:, 0:3, ...] <= 1000).all(dim=1)
        mask_combined = query_mask.unsqueeze(1).repeat(1, n_value_points, 1) & value_mask
        # n_queries_coords = queries_coords.shape[1]
        # n_value_coords = value_coords.shape[1]
        # queries_coords[~query_mask.unsqueeze(1).repeat(1, n_queries_coords, 1)] = 0.0
        # value_coords[~value_mask.unsqueeze(1).repeat(1, n_value_coords, 1, 1)] = 0.0

        # n_queries_features = queries_features.shape[1]
        # n_value_features = value_features.shape[1]
        # queries_features[~query_mask.unsqueeze(1).repeat(1, n_queries_features, 1)] = 0
        # value_features[~value_mask.unsqueeze(1).repeat(1, n_value_features, 1, 1)] = 0

        # print("queries_coords[~query_mask]", queries_coords[:, 0:3, ...][queries_coords.abs()[:, 0:3, ...]  > 1000])
        # print("queries_coords.max()", queries_coords[:, 0:3, ...].max(), "queries_coords.min()", queries_coords[:, 0:3, ...].min())

        # print("query_mask", query_mask.any(), "valid:", query_mask.float().sum(), "total", "invalid:", (~query_mask).float().sum(),  query_mask.numel())
        # print("value_mask", value_mask.any(), "valid:", value_mask.float().sum(), "total", "invalid:", (~value_mask).float().sum(), value_mask.numel())
        # print("mask_combined", mask_combined.any(), "valid:", mask_combined.float().sum(), "invalid:", (~mask_combined).float().sum(), "total", mask_combined.numel())



        # Optionally, append the coordinates of queries and values as features:
        if self.concat_absolute_coords_raw:
            queries = torch.cat( (queries, queries_coords[:, 0:3, ...]), dim=1 )
            values = torch.cat( (values, value_coords[:, 0:3, ...]), dim=1 )
          
        # Optionally, append the encoded coordinates of queries and values as features:
        if self.concat_absolute_coords_encoded:
            # queries_coords_encoded = self.encoder( queries_coords ) # these are for the old models which have PE calculated everytime right before attention
            # value_coords_encoded = self.encoder( value_coords )
            queries_coords_encoded = queries_coords[:, 3:, ...] # these are for models with new preprocesser which PE is calcuated only once during preprocessing
            value_coords_encoded = value_coords[:, 3:, ...]

            queries = torch.cat( (queries, queries_coords_encoded), dim=1 )
            values = torch.cat( (values, value_coords_encoded), dim=1 )

        if self.concat_relative_coords_raw or self.concat_relative_coords_encoded:
            # For the value coordinates in each region, subtract the corresponding query's coordinate
            # value_coords_rel = value_coords - queries_coords.unsqueeze(2)
            value_coords_rel = value_coords[:, 0:3, ...] - queries_coords[:, 0:3, ...].unsqueeze(2)
            value_coords_rel[~mask_combined.unsqueeze(1).repeat(1, 3, 1, 1)] = 0
            if self.concat_relative_coords_raw:
                values = torch.cat( (values, value_coords_rel), dim=1 )
            if self.concat_relative_coords_encoded:
                value_coords_rel_encoded = self.encoder( value_coords_rel )
                value_coords_rel_encoded[~mask_combined.unsqueeze(1).repeat(1, self.encoder.num_features, 1, 1)] = 0
                values = torch.cat( (values, value_coords_rel_encoded), dim=1 )

        # n_queries_features = queries.shape[1]
        # n_value_features = values.shape[1]
        queries[~query_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)] = 0
        values[~value_mask.unsqueeze(1).repeat(1, values.shape[1], 1, 1)] = 0

        if self.concat_queries_to_values:
            expanded_queries = queries.unsqueeze( dim=2 ).repeat( 1,1, n_value_points, 1 )
            values = torch.cat( (values, expanded_queries), dim=1 )

        # print("queries.shape", queries.shape)
        # print("values.shape", values.shape)


        # queries[~query_mask.unsqueeze(1).repeat(1, n_queries_features, 1)] = 0
        # values[~value_mask.unsqueeze(1).repeat(1, n_value_features, 1, 1)] = 0

        Q = self.query_embedding( queries )     # [B, embedding_size, Nq]
        K = self.key_embedding( values )       # [B, embedding_size, Nv, Nq]

        # TODO: Is this really required here, or would it be enough to encode the vectors _after_ the
        # weighting and summation? Would be much less expensive, but if there's a non-linearity involved,
        # the operations are not cummutative... but are they, if there's no non-linearity?
        V = self.value_embedding( values )     # [B, output_features, Nv, Nq]

        #K = K.unsqueeze(3).repeat( 1, 1, 1, n_query_points )  # [B, embedding_size, Nq, Nv]
        Q = Q.unsqueeze(2).repeat( 1, 1, n_value_points, 1 )  # [B, embedding_size, Nv, Nq]

        # Q.register_hook(create_grad_hook("Q"))
        # K.register_hook(create_grad_hook("K"))
        # V.register_hook(create_grad_hook("V"))

        # print("K.max()", K.max(), "K.min()", K.min())
        # print("Q.max()", Q.max(), "Q.min()", Q.min())

        if self.attention_mode == "additive":
            weights = self.v( torch.tanh( K + Q ) )
            weights = weights.squeeze( dim=1 )
        else:
            weights = (K*Q).sum( dim=1 ) # Dot product (sum over embedding_size dimension)
            # Scale: See "Attention is all you need" paper (they call the value d_k):
            weights = weights/math.sqrt( self.embedding_size )     # [B, Nv, Nq]   
        
        weights.register_hook(create_grad_hook("weights"))

        # mask the weights, so that the attention is only calculated for the valid points
        # dummy points only exists in query points, as value points are from FPS, where exception will be
        # raised if there are dummy points in value points
        # query_mask = queries_coords.abs()[:, 0, ...] < 1000  # [B, Nq]
        

        # query_coords_filtered = queries_coords[:, 0:3, ...] * query_mask.float()
        # print("query_coords_filtered", query_coords_filtered.shape, query_coords_filtered.max(), query_coords_filtered.min())

        # query_mask = query_mask.unsqueeze(1).repeat(1, n_value_points, 1) # [B, Nv, Nq]
        # print("query_mask", query_mask.shape, query_mask.any())

        # value_mask = value_coords.abs()[:, 0, ...] < 1000 # [B, Nv, Nq]
        
        # double check mask_combined by selecting coordinates from value_coords then check if there are any dummy points
        # value_coords_filtered = value_coords[:, 0:3, ...] * value_mask.unsqueeze(1).repeat((1,3,1,1,)).float()
        # print("value_coords_filtered", value_coords_filtered.shape, value_coords_filtered.max(), value_coords_filtered.min())

        # value_mask = value_mask.unsqueeze(2).repeat(1, 1, n_query_points)
        # print("value_mask", value_mask.shape, value_mask.any())
        
        # query mask and value mask should at least have one valid points:
        # if not query_mask.any() or not value_mask.any():
        #     print("query_mask", query_mask.any(), query_mask.float().sum())
        #     print("value_mask", value_mask.any(), value_mask.float().sum())
        #     raise ValueError("All points are dummy points, no valid points to calculate attention")



        # mask_combined = value_mask
        # print("num valid: mask_combined.sum()", mask_combined.sum(), "num dummy: (~mask_combined).sum()", (~mask_combined).sum())




        # counting dummy points and valid points
        # print("num valid: query_mask.sum()", query_mask.sum(), "num dummy: (~query_mask).sum()", (~query_mask).sum())
        if query_mask.sum().item() < 1:
            print(query_mask)
            print(queries_coords[:, 0, ...])

        weights = weights.masked_fill(~mask_combined, -1e9)
        # weights = weights.masked_fill(~mask_combined, -float('inf'))
        # query_mask = query_mask.float() 
        # weights = torch.where(query_mask, weights, torch.tensor(-1e9).to(queries_coords.device))
        # weights = weights.masked_fill(~query_mask, -1e9)
        # weights = weights.masked_fill(~value_mask, -1e9)
        # print("weights.max()", weights.max(), "weights.min()", weights.min())
        
        # weights = torch.clamp(weights, min=-1e6, max=1e6)

        weights.register_hook(create_grad_hook("weights_masked_{}".format(self.func_name)))
        # print("before: weights.max()", weights.max(), "weights.min()", weights.min())
        weights = F.softmax( weights, dim=1 )      # Softwmax over the _value_ entries
        # print("after: weights.max()", weights.max(), "weights.min()", weights.min())

        weights.register_hook(create_grad_hook("weights_softmax_{}".format(self.func_name)))
        # Apply the attention to the values:
        weights_rep = weights.unsqueeze(1).repeat(1, V.shape[1], 1, 1)  # [B, D', Nv, Nq]
        #print("weigths_rep", weights_rep.shape)
        #V = V.unsqueeze(3)
        result = V * weights_rep
        result.register_hook(create_grad_hook("result"))
        result = result.sum( dim=2 )        # Sum together the value points (for each query point)


        query_mask = (queries_coords.abs()[:, 0:3, ...] <= 1000).all(dim=1) 
        # print("result.shape", result.shape, "query_mask.shape", query_mask.shape)
        result = result.masked_fill(query_mask.unsqueeze(1).repeat(1, result.shape[1], 1), 0.0)

        if self.return_weights:
            return result, weights
        else:
            return result










class MultiRegionAgentAttention(nn.Module):
    def __init__(self, 
            n_value_features,
            n_query_features,
            embedding_size,
            n_output_features,
            n_agnets=10,
            concat_absolute_coords_raw = True,
            concat_absolute_coords_encoded = True,
            concat_relative_coords_raw = True,
            concat_relative_coords_encoded = True,
            concat_queries_to_values = True,
            # attention_mode = "additive", 
            return_weights=False ,
        ) -> None:
        super().__init__()
        """
        Args:
            concat_coords: If set to True, the coordinates of the value points will be concatinated
                to the value features and the query point coordinates will be concatinated to the 
                query features before calculating attention. In this way, point coordinates can
                play a role in the attention mechanism. If concat_coords is false, only the features
                will be used to calculate attention, not the point coordinates.
            use_relative_coords: TODO
            attention_mode: str, "additive" or "multiplicative"
        """

        self.concat_absolute_coords_raw = concat_absolute_coords_raw
        self.concat_absolute_coords_encoded = concat_absolute_coords_encoded
        self.concat_relative_coords_raw = concat_relative_coords_raw
        self.concat_relative_coords_encoded = concat_relative_coords_encoded
        self.concat_queries_to_values = concat_queries_to_values
        self.n_query_features_raw = n_query_features
        self.n_value_features_raw = n_value_features

        if self.concat_absolute_coords_encoded or self.concat_relative_coords_encoded:
            self.encoder = PositionalEncoder( frequencies = [1e-2, 1e-1, 2, 4, 8, 16, 32, 64], scale=1 )
        if self.concat_absolute_coords_raw:
            n_query_features += 3
            n_value_features += 3
        if self.concat_relative_coords_raw:
            n_value_features += 3

        if self.concat_absolute_coords_encoded:
            n_query_features += self.encoder.num_features
            n_value_features += self.encoder.num_features
        if self.concat_relative_coords_encoded:
            n_value_features += self.encoder.num_features

        if self.concat_queries_to_values:
            n_value_features += n_query_features

        self.query_embedding = EmbedNetwork( n_query_features, embedding_size )
        self.key_embedding = MultiRegionEmbedNetwork( n_value_features, embedding_size )
        self.value_embedding = MultiRegionEmbedNetwork( n_value_features, n_output_features )

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features
        self.n_agents = n_agnets

        # self.attention_mode = attention_mode

        # if self.attention_mode == "additive":
        #     self.v = nn.Conv2d(self.embedding_size, 1, kernel_size=1, bias=True)

        self.return_weights = return_weights
        self.pool = nn.AdaptiveAvgPool1d(output_size=(n_agnets))

        # seperate Nv, and Nq
        # self.pos_bias = nn.Parameter(torch.zeros(1, self.n_agents, 7, 7))
        # self.pos_bias_nq = nn.Parameter(torch.zeros(1, self.n_agents, 1, 7,))
        # self.pos_bias_nv = nn.Parameter(torch.zeros(1, self.n_agents, 7, 1,))

        # self.agent_bias = nn.Parameter(torch.zeros(1, self.n_agents, 1, 1, 1))
        # self.agent_bias_nq = nn.Parameter(torch.zeros(1, self.n_agents, 1, 1, 1))
        # self.agent_bias_nv = nn.Parameter(torch.zeros(1, self.n_agents, 1, 1, 1))

        # combine Nv, and Nq
        self.pos_bias = nn.Parameter(torch.zeros(1, self.n_agents, 7, 7))
        self.pos_bias_nq = nn.Parameter(torch.zeros(1, self.n_agents, 1, 7,))
        self.pos_bias_nv = nn.Parameter(torch.zeros(1, self.n_agents, 7, 1,))

        self.agent_bias = nn.Parameter(torch.zeros(1, self.n_agents, 7))
        self.agent_bias_nq = nn.Parameter(torch.zeros(1, 1, self.n_agents))
        self.agent_bias_nv = nn.Parameter(torch.zeros(1, 1, self.n_agents))

        trunc_normal_(self.pos_bias, std=.02)
        trunc_normal_(self.pos_bias_nq, std=.02)
        trunc_normal_(self.pos_bias_nv, std=.02)
        trunc_normal_(self.agent_bias, std=.02)
        trunc_normal_(self.agent_bias_nq, std=.02)
        trunc_normal_(self.agent_bias_nv, std=.02)

    def forward(self, 
        value_coords, 
        value_features, 
        queries_coords, 
        queries_features,       
    ):
        """ Compute attention on features given by "inp_values", attended to/by/at "inp_queries"

        Note: if concat_coords is False, this module will not use the value_coords and query_coords.

        Args:
            value_coords: Tensor of shape [B, 3, Nv, Nq]
            value_features: Tensor of shape [B, Dv, Nv, Nq]
                (B: batch,
                Dv: n_value_features,
                Nv: number of input points,
                Nq: numbr of query/output points)
            query_coords: Tensor of shape [B, 3, Nq]
            queries_features: Tensor of shape [B, Dq, Nq]
                (B: batch,
                Dq: n_query_features,
                Nq: number of query/output points)

        Returns:
            Tensor of shape [B, D', Nq] (B: batch, D': n_output_features, Nq: number of query points)
        """
        n_bs = value_features.shape[0]      # B
        n_features = value_features.shape[1]      # Dv
        n_value_points = value_features.shape[2]      # Nv
        n_query_points = queries_features.shape[2]     # Nq

        assert value_features.shape[3] == n_query_points, "There must be a set of value points for _every_ query point!"
        assert queries_features.shape[1] == self.n_query_features_raw, "The number of query features must be the same as the number of query features given in the constructor!"
        assert value_features.shape[1] == self.n_value_features_raw, "The number of value features must be the same as the number of value features given in the constructor!"
    
        # By default, use only the input features as features:
        queries = queries_features
        values = value_features

        # Optionally, append the coordinates of queries and values as features:
        if self.concat_absolute_coords_raw:
            queries = torch.cat( (queries, queries_coords), dim=1 )
            values = torch.cat( (values, value_coords), dim=1 )
          
        # Optionally, append the encoded coordinates of queries and values as features:
        if self.concat_absolute_coords_encoded:
            queries_coords_encoded = self.encoder( queries_coords )
            value_coords_encoded = self.encoder( value_coords )

            queries = torch.cat( (queries, queries_coords_encoded), dim=1 )
            values = torch.cat( (values, value_coords_encoded), dim=1 )

        if self.concat_relative_coords_raw or self.concat_relative_coords_encoded:
            # For the value coordinates in each region, subtract the corresponding query's coordinate
            value_coords_rel = value_coords - queries_coords.unsqueeze(2)
            if self.concat_relative_coords_raw:
                values = torch.cat( (values, value_coords_rel), dim=1 )
            if self.concat_relative_coords_encoded:
                value_coords_rel_encoded = self.encoder( value_coords_rel )
                values = torch.cat( (values, value_coords_rel_encoded), dim=1 )

        # Keep the information about the queries in the values, so that it will not be lost after attention
        if self.concat_queries_to_values:
            expanded_queries = queries.unsqueeze( dim=2 ).repeat( 1,1, n_value_points, 1 )
            values = torch.cat( (values, expanded_queries), dim=1 )

        Q = self.query_embedding( queries )     # [B, embedding_size, Nq]
        K = self.key_embedding( values )       # [B, embedding_size, Nv, Nq]

        # TODO: Is this really required here, or would it be enough to encode the vectors _after_ the
        # weighting and summation? Would be much less expensive, but if there's a non-linearity involved,
        # the operations are not cummutative... but are they, if there's no non-linearity?
        V = self.value_embedding( values )     # [B, output_features, Nv, Nq]       
        #print("values.max:", values.max(), "values.min:", values.min())
        #print("V.max:", V.max(), "V.min:", V.min())

        # ######
        # seperate Nv, and Nq
        ########
        # initialize Agent tokens:
        # agent_tokens = self.pool(Q) # [B, embedding_size, n_agents]
        # # agent_tokens = agent_tokens.unsqueeze(2).repeat(1,1,n_value_points,1) # [B, embedding_size, Nv, n_agents]
        # # print("agent_tokens.shape:", agent_tokens.shape)
        
        # # pos_bias_0 = nn.functional.interpolate(self.pos_bias, size=[self.n_agents, n_query_points], mode='bilinear').repeat(n_bs, 1, 1, 1)
        # # pos_bias_nv = nn.functional.interpolate(self.pos_bias_nv, size=[self.n_agents, 1], mode='bilinear')
        # # pos_bias_nq = nn.functional.interpolate(self.pos_bias_nq, size=[1, n_query_points], mode='bilinear')
        # # pos_bias_1 = (pos_bias_nv + pos_bias_nq).repeat(n_bs, 1, 1, 1)
        # # pos_bias_nv = self.pos_bias_nv.repeat(n_bs, 1, 1, 1)
        # # pos_bias_nq = self.pos_bias_nq.repeat(n_bs, 1, 1, 1)
        # # print("pos_bias_0.shape:", pos_bias_0.shape)
        # # print("pos_bias_nv.shape:", pos_bias_nv.shape)
        # # pos_bias = pos_bias_0 + pos_bias_1
        # # print("pos_bias.shape:", pos_bias.shape)
        # # print("agent_tokens.shape", agent_tokens.shape, "K.shape:", K.shape, "pos_bias.shape:", pos_bias.shape)

        # # 1. agent attention: Agent * K
        # # agent_tokens = agent_tokens.unsqueeze(2).repeat( 1, 1, n_value_points, 1 )  # [B, embedding_size, Nq, Nv]
        # agent_tokens = agent_tokens.unsqueeze(3).repeat( 1, 1, 1, n_query_points )  # [B, embedding_size, n_agents, Nq]
        # # print("agent_tokens.shape", agent_tokens.shape, "K.shape:", K.shape, "pos_bias.shape:", pos_bias.shape)
        # weights = (agent_tokens * K).sum( dim=1 ) # Dot product (sum over embedding_size dimension)
        # # weights = (agent_tokens @ K.transpose(-2, -1)) 
        # # weights = (agent_tokens.transpose(-2, -1) @ K)    # [B, embedding_size, n_agents]
        # print("weights.shape:", weights.shape)
        # # Scale: See "Attention is all you need" paper (they call the value d_k):
        # # weights = weights/math.sqrt( self.embedding_size )     # [B, Nv, Nq]   
        # print("weights.shape:", weights.shape)
        # # weights = weights + pos_bias
        # weights = F.softmax( weights, dim=1 )
        # print("weights.shape:", weights.shape)

        # agent_v = weights.transpose(-2, -1) @ V


        # 2. Agent attention: Q * Agent_att


        #####
        ## combine Nv and Nq
        #####
        agent_tokens = self.pool(Q).permute(0, 2, 1) # [B, n_agents, embedding_size]
        pos_bias_0 = nn.functional.interpolate(self.pos_bias, size=[n_value_points, n_query_points], mode='bilinear').repeat(n_bs, 1, 1, 1)
        pos_bias_nv = nn.functional.interpolate(self.pos_bias_nv, size=[n_value_points, 1], mode='bilinear')
        pos_bias_nq = nn.functional.interpolate(self.pos_bias_nq, size=[1, n_query_points], mode='bilinear')
        pos_bias_1 = (pos_bias_nv + pos_bias_nq).repeat(n_bs, 1, 1, 1)
        # pos_bias_nv = self.pos_bias_nv.repeat(n_bs, 1, 1, 1)
        # pos_bias_nq = self.pos_bias_nq.repeat(n_bs, 1, 1, 1)
        # print("pos_bias_0.shape:", pos_bias_0.shape)
        # print("pos_bias_nv.shape:", pos_bias_nv.shape)
        # print("pos_bias_nq.shape:", pos_bias_nq.shape)
        # print("pos_bias_1.shape:", pos_bias_1.shape)
        pos_bias = pos_bias_0 + pos_bias_1
        # print("pos_bias.shape:", pos_bias.shape)

        pos_bias = pos_bias.reshape(n_bs, self.n_agents, n_value_points * n_query_points) # [B, n_agents, Nv*Nq]

        agent_att = ( agent_tokens @ K.reshape(n_bs, self.embedding_size, n_value_points * n_query_points) ) # [B, n_agents, Nv*Nq]
        # agent_att = agent_att / math.sqrt( self.embedding_size )  +   pos_bias # [B, n_agents, Nv*Nq]
        agent_att =  agent_att / math.sqrt( self.embedding_size )  # [B, n_agents, Nv*Nq]
        # agent_att = agent_att.reshape(n_bs, self.n_agents, n_value_points, n_query_points) # [B, n_agents, Nv, Nq]
        # print("agent_att.shape:", agent_att.shape)
        # print("target_shape:", n_bs, self.n_agents, n_value_points * n_query_points)
        # agent_att = F.softmax( agent_att, dim=2 ).reshape(n_bs, self.n_agents, n_value_points * n_query_points)       # Softwmax over the _value_ entries
        agent_att = F.softmax( agent_att, dim=2 ) 
        #print("agent_att.max", torch.max(agent_att), "agent_att.min", torch.min(agent_att))
        # print("V.shape:", V.shape)
        agent_v = agent_att @ V.reshape(n_bs, self.n_output_features, n_value_points * n_query_points).transpose(-2, -1) # [B, n_agents, self.n_output_features,]
        #print("V.max", torch.max(V), "V.min", torch.min(V))
        agent_bias1 = nn.functional.interpolate(self.agent_bias, size=[n_query_points], mode='linear').repeat(n_bs, 1, 1).permute(0, 2, 1) # [B, Nq, n_agents]
        agent_bias2 = self.agent_bias_nq.repeat(n_bs, n_query_points, 1)# [B, Nq, n_agents]
        # print("agent_bias1.shape:", agent_bias1.shape)
        # print("agent_bias2.shape:", agent_bias2.shape)
        agent_bias = agent_bias1 + agent_bias2 # [B, Nq, n_agents]

        # print("Q.shape:", Q.shape)
        # print("agent_tokens.shape:", agent_tokens.shape)
        q_agent = Q.transpose(-2, -1) @ agent_tokens.permute(0, 2, 1) # [B, Nq, self.n_agents]
        # q_att = q_agent / math.sqrt( self.embedding_size ) + agent_bias # [B, Nq, self.n_agents]
        q_att = q_agent / math.sqrt( self.embedding_size ) # [B, Nq, self.n_agents]
        q_att = F.softmax( q_att, dim=-1 ) # [B, Nq, self.n_agents]

        # print("q_att.max", torch.max(q_att), "q_att.min", torch.min(q_att))
        # print("agent_v.max", torch.max(agent_v), "agent_v.min", torch.min(agent_v))
        result = q_att @ agent_v # [B, Nq, self.n_output_features]
        result = result.permute(0, 2, 1) # [B, self.n_output_features, Nq]
        # print(result)
        #print("result.shape:", result.shape, torch.max(result), torch.min(result))
        return result


class GroupedAttention(MultiRegionAttention):
    """ A model first perform grouping and feature selection then apply attention function 
    from the parent.  (works like a decorator)

    """
    def __init__(self, n_value_features,
                 n_query_features, embedding_size, n_output_features, n_kneighbors, concat_absolute_coords_raw=True,
                 concat_absolute_coords_encoded=True, concat_relative_coords_raw=True,
                 concat_relative_coords_encoded=True, concat_queries_to_values=True,
                 attention_mode="additive", return_weights=False):
        super().__init__(n_value_features, n_query_features, embedding_size, n_output_features,
                         concat_absolute_coords_raw, concat_absolute_coords_encoded, concat_relative_coords_raw,
                         concat_relative_coords_encoded, concat_queries_to_values, attention_mode, return_weights)
        self.n_kneighbors = n_kneighbors

    def forward(self, value_coords, value_features, queries_coords, queries_features):
        value_coords_grouped, idx_grouped = k_nearest_neighbors(pos_source=value_coords,
                                                                pos_queries=queries_coords,
                                                                k=self.n_kneighbors)
        value_features_grouped = select_point_regions(value_features, idx_grouped)
        attention = super().forward(value_coords_grouped, value_features_grouped,
                                    queries_coords, queries_features)
        return attention


class GroupedAttentionDisentangled(MultiRegionAttentionDisentangled):
    """ A model first perform grouping and feature selection then apply attention function 
    from the parent.  (works like a decorator)

    """
    def __init__(self, n_value_features,
                 n_query_features, embedding_size, n_output_features, n_kneighbors, concat_absolute_coords_raw=True,
                 concat_absolute_coords_encoded=True, concat_relative_coords_raw=True,
                 concat_relative_coords_encoded=True, concat_queries_to_values=True,
                 attention_mode="additive", return_weights=False):
        super().__init__(n_value_features, n_query_features, embedding_size, n_output_features,
                         concat_absolute_coords_raw, concat_absolute_coords_encoded, concat_relative_coords_raw,
                         concat_relative_coords_encoded, concat_queries_to_values, attention_mode, return_weights)
        self.n_kneighbors = n_kneighbors

    def forward(self, value_coords, value_features, queries_coords, queries_features):
        value_coords_grouped, idx_grouped = k_nearest_neighbors(pos_source=value_coords,
                                                                pos_queries=queries_coords,
                                                                k=self.n_kneighbors)
        value_features_grouped = select_point_regions(value_features, idx_grouped)
        attention = super().forward(value_coords_grouped, value_features_grouped,
                                    queries_coords, queries_features)
        return attention


class GroupedAttentionMask(MultiRegionAttentionMask):
    """ A model first perform grouping and feature selection then apply attention function 
    from the parent.  (works like a decorator)

    """
    def __init__(self, n_value_features,
                 n_query_features, embedding_size, n_output_features, n_kneighbors, concat_absolute_coords_raw=True,
                 concat_absolute_coords_encoded=True, concat_relative_coords_raw=True,
                 concat_relative_coords_encoded=True, concat_queries_to_values=True,
                 attention_mode="additive", return_weights=False, 
                 func_name=""
                 ):
        super().__init__(n_value_features, n_query_features, embedding_size, n_output_features,
                         concat_absolute_coords_raw, concat_absolute_coords_encoded, concat_relative_coords_raw,
                         concat_relative_coords_encoded, concat_queries_to_values, attention_mode, return_weights, 
                         func_name)
        self.n_kneighbors = n_kneighbors

    def forward(self, value_coords, value_features, queries_coords, queries_features):
        value_coords_grouped, idx_grouped = k_nearest_neighbors(pos_source=value_coords,
                                                                pos_queries=queries_coords,
                                                                k=self.n_kneighbors)
        value_features_grouped = select_point_regions(value_features, idx_grouped)
        attention = super().forward(value_coords_grouped, value_features_grouped,
                                    queries_coords, queries_features)
        return attention



class FusionGate(nn.Module):
    def __init__(self, n_internal_value_features, n_surface_value_features,
                 n_query_features, embedding_size, n_output_features, n_kneighbors,
                 concat_absolute_coords_raw=True, concat_absolute_coords_encoded=True,
                 concat_relative_coords_raw=True, concat_relative_coords_encoded=True,
                 concat_queries_to_values=True, attention_mode="additive", return_weights=False,
                 use_internals = True ):

        nn.Module.__init__( self )

        self.non_lin = nn.ReLU()
        self.use_internals = use_internals

        print("Building FusionGate:")
        print("\tn_internal_value_features", n_internal_value_features)
        print("\tn_surface_value_features", n_surface_value_features)
        print("\tn_query_features", n_query_features)
        print("\tembedding_size", embedding_size)
        print("\tn_output_features", n_output_features)
        print("\tn_kneighbors", n_kneighbors)
        print("\tuse_internals", use_internals)

        if use_internals:
            self.internal_attention = GroupedAttention(n_internal_value_features,
                                        n_query_features, embedding_size, n_output_features, n_kneighbors,
                                        concat_absolute_coords_raw, concat_absolute_coords_encoded,
                                        concat_relative_coords_raw, concat_relative_coords_encoded,
                                        concat_queries_to_values, attention_mode, return_weights)
            self.post_internal_attention_conv = nn.Conv1d(n_output_features, n_output_features, kernel_size=1)
        self.surface_attention = GroupedAttention(n_surface_value_features,
                                        n_query_features, embedding_size, n_output_features, n_kneighbors,
                                        concat_absolute_coords_raw, concat_absolute_coords_encoded,
                                        concat_relative_coords_raw, concat_relative_coords_encoded,
                                        concat_queries_to_values, attention_mode, return_weights)

        self.post_surface_attention_conv = nn.Conv1d(n_output_features, n_output_features, kernel_size=1)

        if use_internals:
            # If internals are active, we'll need to process twice the amount of features:
            self.post_combined_conv = nn.Conv1d(2*n_output_features, n_output_features, kernel_size=1)
        else:
            self.post_combined_conv = nn.Conv1d(n_output_features, n_output_features, kernel_size=1)

    def forward(self,
            query_positions, query_features,
            intraop_surf_positions, intraop_surf_features, 
            internals_positions = None, internals_features = None):
        #print("internals_positions", internals_positions.shape)
        #print("internals_features", internals_features.shape)
        #print("query_positions", query_positions.shape)
        #print("query_features", query_features.shape)
        if self.use_internals:
            attended_internals = self.internal_attention(internals_positions, internals_features,
                                                         query_positions, query_features)
            internals_conv = self.non_lin(self.post_internal_attention_conv(attended_internals))

        attended_intraop = self.surface_attention(intraop_surf_positions, intraop_surf_features,
                                                  query_positions, query_features)
        intraop_conv = self.non_lin(self.post_surface_attention_conv(attended_intraop))

        if self.use_internals:
            all_info = torch.cat((internals_conv, intraop_conv), dim=1)
        else:
            all_info = intraop_conv #torch.cat((internals_conv, intraop_conv), dim=1)

        all_info = self.non_lin(self.post_combined_conv(all_info))

        return all_info


