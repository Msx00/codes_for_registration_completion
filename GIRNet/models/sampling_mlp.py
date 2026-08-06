import torch
from torch import nn




class SamplingMLP(nn.Module):
    def __init__(self, 
        num_output_points,
        num_input_features,
        num_intermediate_features=[32, 8, 1],
        append_coords=False
    ):
        nn.Module.__init__( self )
        self.num_input_features = num_input_features
        self.num_output_points = num_output_points
        self.append_coords = append_coords
        if self.append_coords:
            self.num_input_features += 51
            # print("Appending coordinates to the input features!!!", self.num_input_features)
        self.mlp = nn.Sequential(
            nn.Conv1d(self.num_input_features, num_intermediate_features[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_intermediate_features[0], num_intermediate_features[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_intermediate_features[1], num_intermediate_features[2], kernel_size=1),
            # nn.ReLU(),
        )
        


    def forward(self, coords, features):
        """Subsampling using MLP, givin features of a point cloud in shape of [B, F, N], 
        the MLP outputs a score for each point feature, then select the top num_output_points

        Args:
            coords (torch.tensor): input coordinates in shape of [B, 3, N]
            features (torch.tensor): input features in shape of [B, F, N]

        Returns:
            idx: selected indices in shape of [B, num_output_points]
        """

        # print("coords.shape", coords.shape)
        # print("features.shape", features.shape)

        if self.append_coords:
            features = torch.cat([features, coords], dim=1)
        # print("features.shape", features.shape)

        features = self.mlp(features)
        # print("features.shape", features.shape)

        dummy_point_mask = coords.abs().max( dim = 1, keepdim = True )[0] > 5000
        # print("dummy_point_mask.shape", dummy_point_mask.shape)
        # print(dummy_point_mask)
        features[dummy_point_mask] = - torch.inf
        # mask = coords[:, 0, ...].abs() < 0.1
        # print("mask.shape", mask.shape)
        # print(mask)
        # features_masked = features * mask.unsqueeze(1).float()
        # print(features_masked)
        # print("features_masked.shape", features_masked.shape)

        idx = torch.topk(features, self.num_output_points, dim=2)[1]
        # print("idx.shape", idx.shape)
        idx = idx.squeeze(1)
        # print("idx.shape", idx.shape)
        # print("max idx", idx.max(), "min idx", idx.min())
        # check if the idx is not in mask, i.e. the point is not dummy
        # mask = mask.unsqueeze(1)
        # print("mask.shape", mask.shape)
        mask_selected = torch.gather(dummy_point_mask, 2, idx.unsqueeze(1))
        # print("mask_selected.shape", mask_selected.shape)
        # print(torch.any(mask_selected))
        # print(mask_selected)
        assert not torch.any(mask_selected), "Some selected points are dummy points..."

        return idx



if __name__ == "__main__":

    coords = torch.rand(4, 3, 2500).cuda()
    features = torch.rand(4, 60, 2500).cuda()
    sampling = SamplingMLP(
        num_output_points=100,
        num_input_features=60,
        num_intermediate_features=[32, 8, 1],
    ).cuda()

    res = sampling(coords, features)
    print(res.shape)


