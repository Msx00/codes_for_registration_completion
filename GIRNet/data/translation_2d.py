###################
## Simple random 2d shapes for registration task
## Tries to use the same conventions/styles as the data in data/data.py, but
## Shapes are 2D (z == 0) and randomly generated on the fly.
import random
import numpy as np
import torch
from easydict import EasyDict as edict

from torch.utils.data import Dataset, DataLoader

class DataSample():

    def __init__( self, preop_coords, preop_features ):

        self.preop_coords = preop_coords
        self.n_points = self.preop_coords.shape[0]
        self.preop_features = preop_features

        self.deform()

    def deform( self ):

        # Maximum displacement (is applied at center)
        dx_max = random.uniform( -0.1, 0.1 )
        dy_max = random.uniform( -0.1, 0.1 )
        dz_max = 0
        d_max = np.asarray( (dx_max, dy_max, dz_max) ).reshape( (1,3) )

        # Same displacement for every point:
        self.displ = np.repeat( d_max, repeats=self.n_points, axis=0 )

        self.intraop_coords = self.preop_coords + self.displ

    def load( self ):

        res = edict()
        preop = self.preop_coords
        displ = self.displ
        intraop = self.intraop_coords
        preop_features = self.preop_features

        res.update({
            "preop": preop.transpose( (1,0) ),
            "preop_features": preop_features,
            "intraop": intraop.transpose( (1,0) ),
            "displ": displ.transpose( (1,0) ),
        })

        return res

    @staticmethod
    def visualize( preop, intraop, preop_features, intraop_features, displ=None, prediction=None, block=False, n_rows=2, row=0 ):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        Fp = preop_features.shape[0]
        Fi = intraop_features.shape[0]

        N_plots = 2 + Fp + Fi

        #fig = plt.figure(figsize=(5,5))
        fig = plt.gcf()
        fig.set_size_inches( 10, 5 )
        if row == 0:
            plt.clf()
        ax = fig.add_subplot(n_rows, N_plots, row*N_plots + 1)#, projection='3d')

        xs = preop[0, ...]
        ys = preop[1, ...]
        zs = preop[2, ...]

        ax.scatter(xs, ys, c="grey" )

        xs = intraop[0, ...]
        ys = intraop[1, ...]
        zs = intraop[2, ...]

        ax.scatter(xs, ys, c="orange" )

        if displ is not None:
            for i in range(displ.shape[1]):
                d = displ[:, i]
                p = preop[:, i]
                if torch.linalg.norm(d) > 0:
                    plt.arrow( p[0], p[1], d[0], d[1], color="black",
                     #shape='full', color='b', lw=d['MAG']/2., length_includes_head=True, 
                        zorder=100, head_length=0.01, head_width=0.01)

        ax.set_aspect("equal")

        ax = fig.add_subplot(n_rows, N_plots, row*N_plots + 2)#, projection='3d')

        ax.set_aspect("equal")

        xs = preop[0, ...]
        ys = preop[1, ...]
        zs = preop[2, ...]

        ax.scatter(xs, ys, c="grey" )


        if prediction is not None:
            for i in range(prediction.shape[1]):
                d = prediction[:, i]
                p = preop[:, i]
                if torch.linalg.norm(d) > 0:
                    plt.arrow( p[0], p[1], d[0], d[1], color="black",
                     #shape='full', color='b', lw=d['MAG']/2., length_includes_head=True, 
                        zorder=100, head_length=0.01, head_width=0.01)

        # Show all preop features:
        xs = preop[0, ...]
        ys = preop[1, ...]
        for i in range( Fp ):
            ax = fig.add_subplot(n_rows, N_plots, row*N_plots + 2+i+1)#, projection='3d')
            col = preop_features[i,:].squeeze()
            data = ax.scatter(xs, ys, c=col)
            ax.set_title( f"preop f_{i}" )
            plt.colorbar(data, ax=ax)
            ax.set_aspect("equal")

        # Show all preop features:
        xs = intraop[0, ...]
        ys = intraop[1, ...]
        for i in range( Fp ):
            ax = fig.add_subplot(n_rows, N_plots, row*N_plots + 2+Fp+i+1)#, projection='3d')
            col = intraop_features[i,:].squeeze()
            data = ax.scatter(xs, ys, c=col)
            ax.set_title( f"intraop f_{i}" )
            plt.colorbar(data, ax=ax)
            ax.set_aspect("equal")

        if block:
            plt.show()
        else:
            plt.draw()
            plt.pause(0.001)


class Rect( DataSample ):

    def __init__( self, n_points=500 ):

        width = random.uniform( 0.3, 1.5 )
        height = random.uniform( 0.3, 1.5 )
        min_x = random.uniform( -1, 1-width )
        min_y = random.uniform( -1, 1-height )

        xs = np.random.rand( n_points, 1 )*width + min_x
        ys = np.random.rand( n_points, 1 )*height + min_y
        zs = np.zeros_like( xs )
        coords = np.concatenate( (xs, ys, zs ), axis = 1 )

        dists_1 = xs - min_x
        dists_2 = min_x + width - xs
        dists_3 = ys - min_y
        dists_4 = min_y + height - ys

        dists_1 = dists_1.reshape(1, -1)
        dists_2 = dists_2.reshape(1, -1)
        dists_3 = dists_3.reshape(1, -1)
        dists_4 = dists_4.reshape(1, -1)
        dists = np.concatenate( (dists_1, dists_2, dists_3, dists_4), axis=0 )
        df = dists.min( axis=0 ).reshape(1, n_points )

        DataSample.__init__( self, coords, df )


class TranslationDataset( Dataset ):

    def __init__( self, n_shapes = 1000, n_points = 500 ):

        self.shapes = []

        print(f"Generating {n_shapes} samples:")
        for i in range( n_shapes ):
            s  = Rect( n_points = n_points )
            self.shapes.append( s )
        print("\tDone")

    def __len__( self ):

        return len( self.shapes )

    def __getitem__( self, i ):

        sample = self.shapes[i]
        res = sample.load()
        for k in res.keys():
            res[k] =  torch.Tensor( res[k])

        return res
           


if __name__ == "__main__":

    Rect()


    dataset = TranslationDataset()

    print(dataset[0])
    print(dataset[1])
    for k, v in dataset[0].items():
        print( k, v.shape )

    DataSample.visualize( dataset[6]["preop"], dataset[6]["intraop"], dataset[6]["preop_features"], block=True )

