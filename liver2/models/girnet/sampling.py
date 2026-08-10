import torch
import time

from .selection import select_points

def timeit(func):
    """
    A decorator function that measures the time required to
    run the decorated function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for {func.__name__}: {elapsed_time:.6f} seconds")
        return result
    return wrapper



def farthest_point_sampling(points, num_samples, random=True):
    """
    Implementation of the farthest point sampling algorithm
    for selecting a subset of points from a larger set.

    Input:
        points: pointcloud data, [B, 3, N]
        num_samples: number of samples
    Return:
        selected: indices of the selected kernel points in the original point clouds, [B, num_samples]
        coords: coordinates of the selected kernel points
    """
    device = points.device
    B, C, N = points.shape
    points_extra = None
    if C > 3:
        points_extra = points[:, 3:, :]
        points = points[:, :3, :]
    selected = torch.zeros(B, num_samples, dtype=torch.long).to(device) # shape [B, num_samples]
    distance = torch.ones(B, N, dtype=torch.float).to(device) * 1e10  # shape [B, N]
    if random:
        #assert points.
        raise NotImplementedError( "Random farthest point fampling currently not implemented, set random=False!" )
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # shape [B],
    else:
        # Start with the point closest to the origin:
        origin = torch.zeros( (B, 3, 1), device = points.device )
        dists_from_origin = torch.sum((points - origin) ** 2, dim=1 )
        farthest = torch.argmin( dists_from_origin, dim=1 )
        #farthest = torch.zeros( (B,), dtype=torch.long, device=device ) + int(N/2)
    # Ensure this is a valid point that lies "close" to the origin:

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(num_samples):
        selected[:, i] = farthest
        centroid = points[batch_indices, :, farthest].view(B, 3, 1) # coordinates of the current centroid (selected points)
        dist = torch.sum((points - centroid) ** 2, dim=1) # distance between all points to centroid [B, N]
        dist[dist > 5000] = -1      # Exclude points that are very far away # TODO: This could be replaced by a precomputed mask that only has to be calculated once at the beginning?
        mask = dist < distance  # For all points, if the distance is getting closer to the selected set, update the distance
        distance[mask] = dist[mask] 
        # assert distance.max() > 0, f"Cannot select enough points. Too many points are dummy points or duplcates exist. {distance.max()}"
        if distance.max() <= 0.0:
            print("Cannot select enough points. Too many points are dummy points or duplcates exist.", "distance.max():", distance.max(), "i:", i)
            # check if there are any duplicate points:
            print("Checking for duplicate points...")
            for idx_b in range(points.shape[0]):
                points_single = points[idx_b, ...]
                points_single_valid_idx = torch.all(torch.abs(points_single) < 100, dim=0)
                # count number of valid points:
                num_valid_points = torch.sum(points_single_valid_idx)
                print("num_valid_points:", num_valid_points, "num_samples:", num_samples)
                if num_valid_points < num_samples:
                    raise ValueError("Not enough valid points to select from.")
                points_single_filtered = points_single[:, points_single_valid_idx]
                points_single_unique = torch.unique(points_single_filtered, dim=1)
                has_duplicate = points_single_filtered.shape[1] != points_single_unique.shape[1]
                if has_duplicate:
                    raise ValueError("Duplicate points found in batch {}".format(idx_b))
        farthest = torch.max(distance, dim=-1)[1] # update farthest point index

    if points_extra is not None:
        points = torch.cat([points, points_extra], dim=1)
    coords = select_points(points, selected)

    # Make sure we only selected valid points, and masked out any dummy-points (which have coordinates set
    # to -9999):
    assert coords.min() > -5000 and coords.max() < 5000, "coords.min(): {} coords.max(): {}".format(coords.min(), coords.max())

    return selected, coords



def farthest_point_sampling_features(points, num_samples, random=True):
    """
    Implementation of the farthest point sampling algorithm
    for selecting a subset of points from a larger set. 
    The distance is calculated based on the features of the points.

    Input:
        points: pointcloud data, [B, C, N]
        num_samples: number of samples
    Return:
        selected: indices of the selected kernel points in the original point clouds, [B, num_samples]
        coords: coordinates of the selected kernel points
    """
    device = points.device
    B, C, N = points.shape
    # points_extra = None
    # if C > 3:
    #     points_extra = points[:, 3:, :]
    #     points = points[:, :3, :]
    selected = torch.zeros(B, num_samples, dtype=torch.long).to(device) # shape [B, num_samples]
    distance = torch.ones(B, N, dtype=torch.float).to(device) * 1e10  # shape [B, N]
    if random:
        #assert points.
        raise NotImplementedError( "Random farthest point fampling currently not implemented, set random=False!" )
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # shape [B],
    else:
        # Start with the point closest to the origin:
        origin = torch.zeros( (B, 3, 1), device = points.device )
        dists_from_origin = torch.sum((points - origin) ** 2, dim=1 )
        farthest = torch.argmin( dists_from_origin, dim=1 )
        #farthest = torch.zeros( (B,), dtype=torch.long, device=device ) + int(N/2)
    # Ensure this is a valid point that lies "close" to the origin:

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(num_samples):
        selected[:, i] = farthest
        centroid = points[batch_indices, :, farthest].view(B, 3, 1) # coordinates of the current centroid (selected points)
        dist = torch.sum((points - centroid) ** 2, dim=1) # distance between all points to centroid [B, N]
        dist[dist > 5000] = -1      # Exclude points that are very far away # TODO: This could be replaced by a precomputed mask that only has to be calculated once at the beginning?
        mask = dist < distance  # For all points, if the distance is getting closer to the selected set, update the distance
        distance[mask] = dist[mask] 
        assert distance.max() > 0, f"Cannot select enough points. Too many points are dummy points or duplcates exist. {distance.max()}"
        farthest = torch.max(distance, dim=-1)[1] # update farthest point index

    # if points_extra is not None:
    #     points = torch.cat([points, points_extra], dim=1)
    coords = select_points(points, selected)

    # Make sure we only selected valid points, and masked out any dummy-points (which have coordinates set
    # to -9999):
    assert coords.min() > -5000 and coords.max() < 5000

    return selected, coords

