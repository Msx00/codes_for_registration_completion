import torch
from torch import nn
from models.layer_downsample import LayerDownsample
from models.layer_upsample import LayerUpsample


class Encoder(nn.Module):
    def __init__(self,
            n_points_list=[5000, 1000, 200, 50, 1], #[1000, 300, 60, 1],
            n_features_in_list=[1, 8, 32, 128], # [2, 8, 32, 128],
            n_features_out_list=[8, 32, 128, 128], # [8, 32, 128, 128],
            radii_list=[0.05, 0.07, 0.1, 1],  
            n_kneighbors_list=[50, 50, 50, 50],
            embedding_size=50,
            latent_size=16,
            use_relative_coords=True,
        ) -> None:
        super().__init__()

        # assert len(n_points_list) == len(n_points_list) == len(radii_list)
        self.num_layers = len(n_points_list) - 1

        self.n_points_list = n_points_list
        self.n_kneighbors_list = n_kneighbors_list
        self.n_features_in_list = n_features_in_list
        self.n_features_out_list = n_features_out_list

        # self.coords_list = []
        # self.features_list = []
        # self.features_skip_list = []
        self.layers = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for idx in range(self.num_layers ):
            layer_down = LayerDownsample(
                num_kernels = self.n_points_list[idx + 1],
                num_kneighbors = self.n_kneighbors_list[idx],
                num_input_features = self.n_features_in_list[idx],
                num_output_features = self.n_features_out_list[idx],        # Features of next level
                embedding_size = embedding_size,
                radius = radii_list[idx],
                use_relative_coords = use_relative_coords
            )
            self.layers.append(layer_down)
        
            conv = nn.Conv1d( 
                in_channels = self.n_features_out_list[idx], 
                out_channels = self.n_features_out_list[idx], 
                kernel_size = 1,
            )
            self.down_convs.append(conv)

            skip = nn.Conv1d( 
                self.n_features_out_list[idx], 
                self.n_features_out_list[idx], 
                kernel_size = 1,
            )
            self.skip_convs.append(skip)

        self.latent_conv = nn.Conv1d(
            self.n_features_out_list[-1],
            latent_size,
            kernel_size=1,
        )

        self.non_lin = nn.LeakyReLU()


    def forward(self, coords_in, features_in):
        coords_list = [coords_in, ]
        # features_list = [features_in,]
        features_skip_list = [features_in, ]
        for idx_l in range(self.num_layers):
            coords_out, features_out, _ = self.layers[idx_l](coords_in, features_in)
            features_skip = self.skip_convs[idx_l](features_out)
            features_out = self.down_convs[idx_l](features_out)
            features_out = self.non_lin(features_out)
            coords_in = coords_out
            features_in = features_out

            coords_list.append(coords_out)
            # features_list.append(features_out)
            if not idx_l == self.num_layers - 1:
                features_skip_list.append(features_skip)

        latent_code = self.latent_conv(features_out)
        # return self.coords_list, self.features_list, self.features_skip_list
        return coords_list, features_skip_list, latent_code



class Decoder(nn.Module):
    def __init__(self,
        # n_points_list=[1, 60, 300, 1000],
        n_low_res_features_list=[2, 8, 32, 128],
        n_high_res_features_list=[128, 64, 32, 8],
        n_output_features_list=[128, 64, 32, 16],
        n_kneighbors_list=[50, 50, 50, 50],
        radii_list=[1.0, 0.1, 0.07, 0.05 ],  
        # num_kneighbors=50,   
        embedding_size=16,
        use_relative_coords=True,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.num_layers = len(radii_list)

        for idx in range(self.num_layers):
            self.layers.append(
                LayerUpsample(
                    n_kneighbors = n_kneighbors_list[idx],
                    n_low_res_features = n_low_res_features_list[idx],
                    n_high_res_features = n_high_res_features_list[idx],
                    n_output_features = n_output_features_list[idx],
                    radius=radii_list[idx],
                    embedding_size = embedding_size,
                    use_relative_coords = use_relative_coords
                )
            )

            self.up_convs.append(
                nn.Conv1d( 
                    in_channels = n_output_features_list[idx], 
                    out_channels = n_output_features_list[idx], 
                    kernel_size = 1,
                )
            )

        self.non_lin = nn.LeakyReLU()

        self.mlp = nn.Sequential(
            nn.Conv1d(
                in_channels=n_output_features_list[-1], 
                out_channels=8,
                kernel_size=1,
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=8, 
                out_channels=3,
                kernel_size=1,
            ),
        )


    def forward(self, coords_in_list, features_skip_list, latent_code):
        coords_in = coords_in_list[0]
        features_in = latent_code
        for idx_l in range(self.num_layers):
            # print("coords_in.shape", coords_in.shape, "features_in.shape", features_in.shape, coords_in)
            features_out = self.layers[idx_l](
                low_res_coords=coords_in_list[idx_l], 
                low_res_features=features_in, 
                high_res_coords=coords_in_list[idx_l + 1], 
                high_res_features=features_skip_list[idx_l],
            )
            features_out = self.up_convs[idx_l](features_out)
            features_out = self.non_lin(features_out)
            # coords_in = coords_out
            features_in = features_out
        
        displ = self.mlp(features_out)

        return displ




class DeformPreop(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()


    def create_encoder(self):
        encoder = Encoder(
            n_points_list = self.config["encoder"]["n_points_list"],
            n_features_in_list = self.config["encoder"]["n_features_in_list"] ,
            n_features_out_list = self.config["encoder"]["n_features_out_list"]  ,
            radii_list = self.config["encoder"]["radii_list"]  ,
            n_kneighbors_list = self.config["encoder"]["n_kneighbors_list"]  ,
            embedding_size = self.config["encoder"]["embedding_size"]  ,
            use_relative_coords = self.config["encoder"]["use_relative_coords"] ,
            latent_size = self.config["encoder"]["latent_size"] ,
        )
        return encoder

    def create_decoder(self):
        decoder = Decoder(
            # n_points_list = self.config["decoder"]["n_points_list"],
            n_low_res_features_list = self.config["decoder"]["n_low_res_features_list"],
            n_high_res_features_list = self.config["decoder"]["n_high_res_features_list"],
            n_output_features_list = self.config["decoder"]["n_output_features_list"],
            n_kneighbors_list = self.config["decoder"]["n_kneighbors_list"],
            radii_list = self.config["decoder"]["radii_list"],
            embedding_size = self.config["decoder"]["embedding_size"],
            use_relative_coords = self.config["decoder"]["use_relative_coords"],
        )
        return decoder


    def forward(self, coords_in, features_in,):
        # coords_list, features_skip_list, latent_code = self.encoder(coords_in, features_in)
        coords_list, feature_skip_list, latent = self.encoder(
            coords_in=coords_in, 
            features_in=features_in  
        )
        coords_list.reverse()
        feature_skip_list.reverse()

        print("coords_list:", len(coords_list), "feature_skip_list:", len(feature_skip_list), "latent:", latent.shape)

        displ = self.decoder(
            coords_in_list = coords_list, 
            features_skip_list = feature_skip_list, 
            latent_code = latent, 
            # displ = displ_gt,
        )
        print("displ.shape:", displ.shape)


        return displ



class V2SSharedEncoder(DeformPreop):
    def __init__(self, config) -> None:
        super().__init__(config=config)

    
    def forward(self, coords_pre_in, coords_intra_in, features_pre_in, features_intra_in,):
        
        coords_pre_list, feature_skip_pre_list, latent_pre = self.encoder(
            coords_in=coords_pre_in, 
            features_in=features_pre_in,  
        )

        coords_intra_list, feature_skip_intra_list, latent_intra = self.encoder(
            coords_in=coords_intra_in, 
            features_in=features_intra_in,  
        )

        coords_pre_list.reverse()
        feature_skip_pre_list.reverse()
        coords_intra_list.reverse()
        feature_skip_intra_list.reverse()

        features_skip_list = [ feature_skip_pre_list[i] + feature_skip_intra_list[i] for i in range(len(feature_skip_pre_list)) ]
        latent = latent_pre + latent_intra

        displ = self.decoder(
            coords_in_list = coords_pre_list, 
            features_skip_list = features_skip_list, 
            latent_code = latent, 
            # displ = displ_gt,
        )
        print("displ.shape:", displ.shape)


        return displ




