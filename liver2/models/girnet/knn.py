import torch

from .selection import select_point_regions


def k_nearest_neighbors( pos_source, pos_queries, k ):
    """ Return the k nearest neighbors in source for each point in queries

    Args:
        pos_source: Tensor, shape [B, 3, Ns]: source point positions
        pos_queries: Tensor, shape [B, 3, Nq]: query point positions
        k: int, number of (closests) source points to return per query point
    Retrurns:
        pos: Tensor, shape [B, 3, k, Nq] k nearest points for every query point
        inds: Tensor, shape [B, k, Nq] Indices of the k nearest points for every query point

    """
    B = pos_source.shape[0]
    # C = pos_source.shape[1]
    Ns = pos_source.shape[2]
    Nq = pos_queries.shape[2]
    
    pos_source_extra = None
    
    if pos_source.shape[1] > 3:
        pos_source_extra = pos_source[:, 3:, ...]
        pos_source = pos_source[:, :3, ...]
    if pos_queries.shape[1] > 3:
        pos_queries = pos_queries[:, :3, :]

    assert Ns >= k, f"pos_source only contains {Ns} points, cannot lookup k={k} points!"

    # Pairwise (squared) distances:
    p_s = pos_source.unsqueeze(3).repeat( 1, 1, 1, Nq )
    p_q = pos_queries.unsqueeze(2).repeat( 1, 1, Ns, 1 )
    dists_squared = (p_s - p_q).pow(2).sum( dim = 1 )       # [B, Ns, Nq]  pairwise distances

    # Find the k nearest points. Don't sort them.
    _, nearest_inds = torch.topk( dists_squared, dim = 1, k = k, largest = False, sorted = False )

    #b = 1
    #q = 3
    #print(dists_squared.shape)
    #print(dists_squared[b,:,q])
    #print(nearest_inds.shape)
    #print(nearest_inds[b,:,q])      # GOOD
    #print(pos_source[b, :, :])
    #print(pos_source)
    if pos_source_extra is not None:
        pos_source = torch.cat( [pos_source, pos_source_extra], dim = 1 )
    selected = select_point_regions( pos_source, nearest_inds )   # [B, 3, k, Nq]
    #print(selected[b, :, :, q])

    #print(selected.shape)
    #print(selected[1, :, 

        #nearest_pos = pos_source[:, :, nearest_inds]
    return selected, nearest_inds



if __name__ == "__main__":

    a = torch.rand(5, 51, 1000)
    b = torch.rand(5, 51, 100)
    k = 10
    pos, inds = k_nearest_neighbors(a, b, k)
    print(pos.shape)
    print(inds.shape)
