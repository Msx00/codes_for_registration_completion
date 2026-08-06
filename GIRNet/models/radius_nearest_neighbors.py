import torch

from models.select import select_point_regions


def radius_nearest_neighbors( pos_source, pos_queries, radius, k ):
    """ Return the nearest neighbors in source for each point in queries, within a radius.

    If less than k nearby points are found for a certain query point, points are repeated randomly.
    If more than k nearby points are found, points are sampled randomly.

    The function loosely follows the "query_ball_point" function in the PointNet++ implementation:
    https://gitlab.com/MichaPfeiffer/pointnet_pointnet2_pytorch/-/blob/master/models/pointnet2_utils.py

    Args:
        pos_source: Tensor, shape [B, 3, Ns]: source point positions
        pos_queries: Tensor, shape [B, 3, Nq]: query point positions
        k: int, number of points to return.
    Retrurns:
        pos: Tensor, shape [B, 3, k, Nq] k nearest points for every query point
        inds: Tensor, shape [B, k, Nq] Indices of the k nearest points for every query point

    """
    B = pos_source.shape[0]
    Ns = pos_source.shape[2]
    Nq = pos_queries.shape[2]

    #print("B", B)
    #print("Ns", Ns)
    #print("Nq", Nq)
    #print("radius", radius)
    #print("k", k)

    device = pos_source.device

    assert Ns >= k, f"pos_source only contains {Ns} points, cannot lookup k={k} points!"

    # Pairwise (squared) distances:
    p_s = pos_source.unsqueeze(3).repeat( 1, 1, 1, Nq )
    p_q = pos_queries.unsqueeze(2).repeat( 1, 1, Ns, 1 )
    dists_squared = (p_s - p_q).pow(2).sum( dim = 1 )       # [B, Ns, Nq]  pairwise distances
    dists_squared = dists_squared.permute( 0, 2, 1 )

    # Find the k nearest points. Don't sort them.
    nearest_dists_squared, idx = torch.topk( dists_squared, dim = 2, k = k, largest = False, sorted = True )

    closest_idx = idx[:,:,0].clone().unsqueeze(2)

    idx[nearest_dists_squared > radius**2] = Ns

    ## Create a list of all source point indices. Repeat for full batch and each query point:
    #orig_idx = torch.arange(Ns, dtype=torch.long, device=device).view(1, 1, Ns).repeat([B, Nq, 1])

    #idx = orig_idx.clone()
    #idx[dists_squared > radius**2] = Ns     # Set invalid index (Ns) for points outside radius
    #idx = idx.sort( dim = 2 )[0] # Sort high index (i.e. those invalid points from above (?)) to the end
    #idx = idx[:,:,:k]           # Select the indices of the k first source points for each query point

    # If too few points were found for the given radius, we could still have invalid indices Ns in idx
    # at this point. Replace those with indices of the closest point:
    #lowest_dist_inds = dists_squared.argmin( dim = 2, keepdim=True )
    #closest = torch.gather( orig_idx, 2, lowest_dist_inds )

    # Debug:
    #closest_dists = torch.gather( dists_squared, 2, lowest_dist_inds )

    closest_idx = closest_idx.repeat( 1, 1, k )

    #first = idx[:, :, 0].view(B, Nq, 1).repeat([1, 1, k])

    mask = idx == Ns
    idx[mask] = closest_idx[mask]


    #b = 1
    #q = 3
    #print(dists_squared.shape)
    #print(dists_squared[b,:,q])
    #print(nearest_inds.shape)
    #print(nearest_inds[b,:,q])      # GOOD
    #print(pos_source[b, :, :])
    #print(pos_source)
    idx = idx.permute( 0, 2, 1 )
    selected = select_point_regions( pos_source, idx )   # [B, 3, k, Nq]
    #print(selected[b, :, :, q])

    #print(selected.shape)
    #print(selected[1, :, 

        #nearest_pos = pos_source[:, :, nearest_inds]
    return selected, idx



    
