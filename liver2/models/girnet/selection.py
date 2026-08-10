"""Batched point and neighborhood selection helpers."""

import torch

# @timeit
def select_points_0(points, idx):
    """

    Input:
        points: input point cloud data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    print("view_shape", view_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    print("batch_indices.shape", batch_indices.shape)
    print("batch_indices", batch_indices)
    new_points = points[batch_indices, idx, :]
    return new_points


# @timeit
def select_points(points, idx):
    """
    Indexing point cloud by the index list.

    Input:
        points: input point cloud data, [B, C, Np]
        idx: sample index data, [B, Ns]
    Return:
        points_indexed:, indexed points data, [B, C, Ns]
    """
    C = points.shape[1]
    idx = idx.unsqueeze(dim=1).repeat(1, C, 1)
    
    points_indexed= torch.gather(input=points, dim=2, index=idx)
    return points_indexed



def select_point_regions( source, indices ):
    """ Assuming source contains F features for Ns points, select the features of those given be indices

    This expects 'indices' to be the result of something like a nearest neighbor search, where Nq query points
    were used and we received the indices of the k nearest neighbors. For every one of the Nq points, this will
    then select the k points from source that we're interested in.

    Args:
        source: Tensor [B, F, Ns]
        indices: Tensor [B, k, Nq]

    Returns:
        selected: Tensor [B, F, k, Nq] k selected points from source for every one of the Nq query points
    """
    # print("source.max", source.max())
    B = source.shape[0]        # batch size
    F = source.shape[1]         # Number of features per point
    Ns = source.shape[2]        # Number of source points
    k = indices.shape[1]        # Number of points to select per query point
    Nq = indices.shape[2]        # Number of query points

    out = torch.empty( (B, F, k, Nq) )
    indices = indices.unsqueeze( 1 )            # [B, 1, k, Nq]
    indices = indices.repeat( 1, F, 1, 1 )      # [B, F, k, Nq]

    source = source.unsqueeze( 2 )          # [B, F, 1, Ns]
    source = source.repeat( 1, 1, k, 1 )    # 

    selected = source.gather( dim = 3, index=indices )
    # print("selected.max", selected.max())
    return selected


