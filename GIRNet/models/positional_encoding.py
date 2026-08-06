import torch
import torch.nn as nn

class PositionalEncoder( nn.Module ):

    def __init__( self, frequencies = [1e-2, 1e-1, 2, 4, 8, 16, 32, 64, 128, 256], scale=1 ):
        nn.Module.__init__( self )

        self.frequencies = [f*scale for f in frequencies]

        # Calculate how many values this encoding will produce (assumes 3D coords!):
        self.num_features = 3*len(self.frequencies)*2

    def forward( self, coords ):

        features = []
        
        for f in self.frequencies:
            s = torch.sin( coords*f )
            c = torch.cos( coords*f )
            features.append( s )
            features.append( c )

        encoding = torch.cat( features, dim = 1 )
        return encoding


if __name__ == "__main__":

    coords = torch.rand( 2, 3, 4 )

    res = PositionalEncoder()( coords )

    print(coords)
    print(res)
