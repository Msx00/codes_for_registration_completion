"""Dynamic graph convolution blocks used by the V3 encoder."""

import torch
import torch.nn as nn

class LayerDGCNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d((in_features+51)*2, out_features+51, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_features+51, out_features, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, central_feature, neighbor_features, central_coords, neighbor_coords):
        '''
        Args:
            central_feature - [B, F, Nq] Features of the query points
            neighbor_features - [B, F, K, Nq] Features of the k-nearest neighbors
            central_coords - [B, F, Nq] coords of the query points
            neighbor_coords - [B, 3, k, Nq] coords of the k-nearest neighbors

        Returns: 
            aggregated_features: Tensor [B, out_features, Nq]

        '''
        B, F, Nq = central_feature.shape
        k = neighbor_features.shape[2]

         # Concatenate features with coordinates
        # Concatenate features with coordinates
        central_feature = torch.cat([central_feature, central_coords], dim=1)  # [B, F + F', Nq]
        neighbor_features = torch.cat([neighbor_features, neighbor_coords], dim=1)  # [B, F + F', K, Nq]
        #print("central_feature.shape", central_feature.shape)
        central_feature_expanded = central_feature.unsqueeze(2).expand(-1, -1, k, -1)
        #print("central_feature_expanded", central_feature_expanded.shape)

        edge_features = torch.cat(
            [central_feature_expanded, neighbor_features-central_feature_expanded], dim=1
            )
        #print("edge_features.shape",edge_features.shape)

        edge_features = self.edge_conv(edge_features) # [B, out_features, k, Nq]

        #aggregated_features = torch.max(edge_features,dim=2)[0]
        aggregated_features = torch.mean(edge_features,dim=2)[0]

        return aggregated_features
    
class LayerDGCNN_v3(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_features*2, out_features, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, central_feature, neighbor_features):
        '''
        Args:
            central_feature - [B, F, Nq] Features of the query points
            neighbor_features - [B, F, K, Nq] Features of the k-nearest neighbors
            central_coords - [B, F, Nq] coords of the query points
            neighbor_coords - [B, 3, k, Nq] coords of the k-nearest neighbors

        Returns: 
            aggregated_features: Tensor [B, out_features, Nq]

        '''
        B, F, Nq = central_feature.shape
        k = neighbor_features.shape[2]

        # Concatenate features with coordinates
        # Concatenate features with coordinates
        #print("central_feature.shape", central_feature.shape)
        central_feature_expanded = central_feature.unsqueeze(2).expand(-1, -1, k, -1)
        #print("central_feature_expanded", central_feature_expanded.shape)

        edge_features = torch.cat(
            [central_feature_expanded, neighbor_features-central_feature_expanded], dim=1
            )
        #print("edge_features.shape",edge_features.shape)

        edge_features = self.edge_conv(edge_features) # [B, out_features, k, Nq]

        #aggregated_features = torch.max(edge_features,dim=2)[0]
        aggregated_features = torch.max(edge_features,dim=2)[0]

        return aggregated_features


