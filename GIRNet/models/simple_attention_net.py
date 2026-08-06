import math
import random

import torch
import torch.nn as nn

from .attention import Attention

class SimpleAttentionNet(nn.Module):

    def __init__( self, preop_features, intraop_features, embedding_size ):
        nn.Module.__init__( self )

        self.att1 = Attention( intraop_features, preop_features, embedding_size, 7 )
        self.conv1 = nn.Conv1d( 7, 7, kernel_size=1, bias=True )

        self.att2 = Attention( 7, preop_features, embedding_size, 7 )
        self.conv2 = nn.Conv1d( 7, 3, kernel_size=1, bias=True )

        #self.att3 = Attention( 25, 25, embedding_size, 150 )
        #self.conv3 = nn.Conv1d( 25, 25, kernel_size=1, bias=True )

        #self.att4 = Attention( 150, 150, embedding_size, 150 )
        #self.conv4 = nn.Conv1d( 150, 3, kernel_size=1, bias=True )

        self.non_lin = nn.Softsign()

    def forward( self, preop, intraop ):

        res = self.att1( intraop, preop )
        res = self.non_lin( self.conv1( res ) )

        res = self.att2( res, preop )
        res = self.non_lin( self.conv2( res ) )

        #res = self.att3( intraop, preop )
        #res = self.non_lin( self.conv3( res ) )

        #res = self.att4( intraop, preop )
        #res = self.non_lin( self.conv4( res ) )

        return res
