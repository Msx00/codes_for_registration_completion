import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from models.positional_encoding import PositionalEncoder
from models.k_nearest_neighbors import k_nearest_neighbors
from models.select import select_point_regions, select_points
from models.attention import EmbedNetwork, MultiRegionEmbedNetwork




class Local_op(nn.Module):
    ### Can be used as EmbedNetwork
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("input:", x.shape) # B, D, N, Ns
        B, D, Ns, N = x.size()  # torch.Size([32, 512, 32, 6]) 
        # B, D, N, Ns = x.size()
        x = x.permute(0, 3, 1, 2).reshape(B * N, D, Ns) # switch D and N
        # x = x.permute(0, 1, 3, 2)
        # x = x.permute(0, 1, 3, 2).reshape(B, D * Ns, N)
        # x = x.reshape(-1, d, s)
        print("reshape:", x.shape)
        batch_size, _, _ = x.size() # new bach size after reshaping
        # x = self.relu(self.bn1(self.conv1(x))) # B*N, D, Ns
        # x = self.relu(self.bn2(self.conv2(x))) # B*N, D, Ns
        x = self.relu(self.conv1(x)) # B*N, D, Ns
        x = self.relu(self.conv2(x)) # B*N, D, Ns
        print("conv2:", x.shape)
        x = torch.max(x, 2)[0] # maxpooling over Ns
        print("torch.max:", x.shape)
        x = x.view(batch_size, -1) # no change in shape
        print("view:", x.shape)
        x = x.reshape(B, N, -1).permute(0, 2, 1) # switch back to original shape
        print("output:", x.shape)
        return x




class MultiRegionAttentionMaxpoolSelfAttention(nn.Module):
    ### This is self attention only
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    


class MultiRegionAttentionMaxpool( nn.Module ):
    """ 
    !!!NOTE: This attention module is for the old P-V2S-Net models, where positional encoding is 
    calculated within the attetion module and right before attention.!!!

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
        # self.key_embedding = MultiRegionEmbedNetwork( n_value_features, embedding_size )
        # self.value_embedding = MultiRegionEmbedNetwork( n_value_features, n_output_features )
        # self.key_embedding = EmbedNetwork( n_value_features, embedding_size )
        # self.value_embedding = EmbedNetwork( n_value_features, n_output_features )
        self.key_embedding = Local_op(in_channels=n_value_features, out_channels=embedding_size)
        self.value_embedding = Local_op(in_channels=n_value_features, out_channels=n_output_features)

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
        assert value_features.shape[1] == self.n_value_features_raw, "The number of value features must be the same as the number of value features given in the constructor!"
    
        # By default, use only the input features as features:
        queries = queries_features
        values = value_features

        # Optionally, append the coordinates of queries and values as features:
        if self.concat_absolute_coords_raw:
            queries = torch.cat( (queries, queries_coords[:, 0:3, ...]), dim=1 )
            values = torch.cat( (values, value_coords[:, 0:3, ...]), dim=1 )
          
        # Optionally, append the encoded coordinates of queries and values as features:
        if self.concat_absolute_coords_encoded:
            queries_coords_encoded = self.encoder( queries_coords ) # these are for the old models which have PE calculated everytime right before attention
            value_coords_encoded = self.encoder( value_coords )
            # queries_coords_encoded = queries_coords[:, 3:, ...] # these are for models with new preprocesser which PE is calcuated only once during preprocessing
            # value_coords_encoded = value_coords[:, 3:, ...]

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

        print("queries.shape", queries.shape)
        print("values.shape", values.shape)
        Q = self.query_embedding( queries )     # [B, embedding_size, Nq]
        K = self.key_embedding( values )       # [B, embedding_size, Nv, Nq]

        # TODO: Is this really required here, or would it be enough to encode the vectors _after_ the
        # weighting and summation? Would be much less expensive, but if there's a non-linearity involved,
        # the operations are not cummutative... but are they, if there's no non-linearity?
        V = self.value_embedding( values )     # [B, output_features, Nv, Nq]

        print("Q.shape", Q.shape)
        print("K.shape", K.shape)
        print("V.shape", V.shape)

        Q = Q.permute(0, 2, 1) # B, Nq, D
        energy = Q @ K # B, Nq, Nv
        attention = F.softmax(energy, dim=2)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = V @ attention # b, c, n 
        # x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        # x = x + x_r
        return x_r

        # return x

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
        



class MultiRegionAttention( nn.Module ):
    """ 
    !!!NOTE: This attention module is for the old P-V2S-Net models, where positional encoding is 
    calculated within the attetion module and right before attention.!!!

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
        assert value_features.shape[1] == self.n_value_features_raw, "The number of value features must be the same as the number of value features given in the constructor!"
    
        # By default, use only the input features as features:
        queries = queries_features
        values = value_features

        # Optionally, append the coordinates of queries and values as features:
        if self.concat_absolute_coords_raw:
            queries = torch.cat( (queries, queries_coords[:, 0:3, ...]), dim=1 )
            values = torch.cat( (values, value_coords[:, 0:3, ...]), dim=1 )

        # Optionally, append the encoded coordinates of queries and values as features:
        if self.concat_absolute_coords_encoded:
            queries_coords_encoded = self.encoder( queries_coords ) # these are for the old models which have PE calculated everytime right before attention
            value_coords_encoded = self.encoder( value_coords )
            # queries_coords_encoded = queries_coords[:, 3:, ...] # these are for models with new preprocesser which PE is calcuated only once during preprocessing
            # value_coords_encoded = value_coords[:, 3:, ...]

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




class MultiRegionAttentionMHA( nn.Module ):
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
            n_head=4,
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
        # self.value_embedding = MultiRegionEmbedNetwork( n_value_features, n_output_features )
        self.value_embedding = MultiRegionEmbedNetwork(n_value_features, embedding_size)

        self.embedding_size = embedding_size
        self.n_output_features = n_output_features

        self.attention_mode = attention_mode

        self.n_head = n_head
        self.head_dim = embedding_size // n_head
        assert self.head_dim * n_head == embedding_size, "Embedding size must be divisible by the number of heads"

        # collapse embedding size to 1
        if self.attention_mode == "additive":
            # self.v = nn.Conv2d(self.embedding_size, 1, kernel_size=1, bias=True)
            self.v = nn.Conv3d(self.head_dim, 1, kernel_size=1, bias=True)

        self.conv_out = nn.Conv1d(embedding_size, n_output_features, kernel_size=1)

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
        n_batch = value_coords.shape[0]
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
        V = self.value_embedding( values )     # [B, output_features, Nv, Nq]

        # Reshape for multi-head attention
        Q_mh = Q.view(n_batch, self.head_dim, self.n_head, n_query_points)
        K_mh = K.view(n_batch, self.head_dim, self.n_head, n_value_points, n_query_points)
        V_mh = V.view(n_batch, self.head_dim, self.n_head, n_value_points, n_query_points)
        # V_mh = V.unsqueeze(2).repeat(1, 1, self.n_head, 1, 1) # [B, head_dim, n_head, Nv, Nq]

        # Q = Q.unsqueeze(2).repeat( 1, 1, n_value_points, 1 )  # [B, embedding_size, Nv, Nq]
        # Q_mh = Q_mh.permute(0, 1, 3, 2) # [B, n_head, Nq, head_dim]
        Q_mh = Q_mh.unsqueeze(3).repeat(1, 1, 1, n_value_points, 1) # [B, head_dim, n_head, Nv, Nq]

        # if self.attention_mode == "additive":
        #     weights = self.v( torch.tanh( K + Q ) )
        #     weights = weights.squeeze( dim=1 )
        # else:
        #     weights = (K*Q).sum( dim=1 ) # Dot product (sum over embedding_size dimension)
        #     # Scale: See "Attention is all you need" paper (they call the value d_k):
        #     weights = weights/math.sqrt( self.embedding_size )     # [B, Nv, Nq]   
        weights = self.v( torch.tanh( K_mh + Q_mh ) )
        weights = weights.squeeze( dim=1 )
        weights = F.softmax( weights, dim=1 )       # Softwmax over the _value_ entries
        # print("weights.shape", weights.shape)

        # Apply the attention to the values:
        weights_rep = weights.unsqueeze(1).repeat(1, self.head_dim, 1, 1, 1)  # [B, head_dim, n_head, Nv, Nq]
        # print("weigths_rep", weights_rep.shape)
        # print("V.shape", V.shape)
        #V = V.unsqueeze(3)
        result = V_mh * weights_rep
        result = result.sum( dim=3 )        # Sum together the value points (for each query point)
        # print("result.shape", result.shape)


        result = result.contiguous().view(n_batch, self.embedding_size, n_query_points)
        # print("values.max:", values.max(), "values.min:", values.min())
        # print("V.max:", V.max(), "V.min:", V.min())
        # print("Q.max:", Q.max(), "Q.min:", Q.min())
        # print("K.max:", K.max(), "K.min:", K.min())
        # print("result.max:", result.max(), "result.min:", result.min())
        result = self.conv_out(result)
        if self.return_weights:
            return result, weights
        else:
            return result




# if __name__ == "__main__":
#     gather_local = Local_op(
#         in_channels=3,
#         out_channels=64,
#     )
#     gather_local = gather_local.cuda()

#     x = torch.random(32, 512, 32, 3).cuda()
#     x = gather_local(x)
#     print(x.shape)

