import torch

def collate_geometry(data):
    """Porcess data which contains geometry information

    Args:
        data (list): data from Dataset subclasses

    Returns:
        pointcloud_data: point cloud data including: preop, displ and intraop in the shape of [B, C, N] 
        geometry_data: corresponding information needed 
    """
    print("entered collate function")
    print(type(data), len(data))

    # for d in data:
    #     print(d.keys(), type(d["preop"]))

    preop = torch.cat([d["preop"].unsqueeze(0) for d in data], dim=0)
    displ = torch.cat([d["displ"].unsqueeze(0) for d in data], dim=0)
    intraop = torch.cat([d["intraop"].unsqueeze(0) for d in data], dim=0)
    geometry_data = [
        {
            "preop_subsampled_idx_on_original" :  d["preop_subsampled_idx_on_original"], 
            "intraop_subsampled_idx_on_original" : d["intraop_subsampled_idx_on_original"], 
            "preop_surface_idx_on_original" : d["preop_surface_idx_on_original"], 
            "preop_surface_verts" : d["preop_surface_verts"], 
            "preop_surface_faces" : d["preop_surface_faces"], 
        } for d in data
    ]
    pointcloud_data = {
        "preop" : preop,
        "displ" : displ,
        "intraop" : intraop,
    }
    # print("preop.shape", preop.shape)
    # print("displ.shape", displ.shape)
    # print("intraop.shape", intraop.shape)
    return pointcloud_data, geometry_data


