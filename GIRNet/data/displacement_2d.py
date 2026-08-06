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

    def __init__( self, coords_preop, coords_preop_border, features_preop, surface_amount=1 ):
        """  Given preoperative points (and their features), construct a full sample with displacement

        Note: If surface_amount is < 1, a partial intraoperative surface will randomly be selected.
        """

        assert coords_preop.shape == coords_preop_border.shape

        self.coords_preop = coords_preop
        self.coords_preop_border = coords_preop_border
        self.n_points = self.coords_preop.shape[0]
        self.features_preop = features_preop

        self.surface_amount = surface_amount

        self.deform()

        # Select a random part of the surface:
        n_points_in_surface = self.coords_intraop_border.shape[0]
        n_points_in_partial_surface = int(surface_amount*n_points_in_surface+0.5)
        if n_points_in_partial_surface == n_points_in_surface:
            self.coords_intraop_border_partial = self.coords_intraop_border
        else:

            n_surface_pieces = random.randint(1,3)
            selected_coords = []
            coords = self.coords_intraop_border.copy()
            n_selected_points = 0
            while len(selected_coords) < n_surface_pieces and n_selected_points < n_points_in_partial_surface:

                n_remaining_points = coords.shape[0]
    
                # Determine how many points to select for this piece:
                if len(selected_coords) < n_surface_pieces-1:
                    n_points_in_piece = random.randint( 1, n_points_in_partial_surface - n_selected_points )
                else:
                    n_points_in_piece = n_points_in_partial_surface - n_selected_points

                # Select a random point:
                start_id = random.randint( 0, n_remaining_points - 1 )
                start_pos = coords[start_id,:]

                dists_squared = ((start_pos[np.newaxis,:].repeat(n_remaining_points,0) - coords)**2).sum(1)
                inds = np.argsort( dists_squared, axis = 0 )
                coords_sorted = coords[inds,:]

                # Choose the first N points to append to the selected surface:
                selected_coords.append( coords_sorted[0:n_points_in_piece,:] )

                # Leave the rest for further selection:
                coords = coords_sorted[n_points_in_piece:,:]
                n_selected_points += selected_coords[-1].shape[0]

            self.coords_intraop_border_partial = np.concatenate( selected_coords, axis = 0 )
        assert n_points_in_partial_surface == self.coords_intraop_border_partial.shape[0], \
                f"Attempted to select {n_points_in_partial_surface} points for partial surface, " +\
                f"but selected: {self.coords_intraop_border_partial.shape[0]}"

        # Fill the partial shape with dummy points.
        # Note: this is done to ensure that every batch has the same number of points.
        # Also note: This must be handled correctly down-stream, for example in k-farthest-point sampling
        # where these points should _not_ be selected.
        if self.coords_intraop_border_partial.shape[0] < n_points_in_surface:
            n_dummy_points = n_points_in_surface - self.coords_intraop_border_partial.shape[0]
            dummy_points = np.full( (n_dummy_points, 3), 99999, dtype=float )
            self.coords_intraop_border_partial = np.concatenate(
                    (self.coords_intraop_border_partial, dummy_points),
                    axis = 0 )



    def deform( self ):

        # Center of deformation:
        cx = random.uniform( -1, 1 )
        cy = random.uniform( -1, 1 )
        cz = 0
        c = np.asarray( (cx, cy, cz) ).reshape( (1,3) )

        # Maximum displacement (is applied at center)
        dx_max = random.uniform( -0.2, 0.2 )
        dy_max = random.uniform( -0.2, 0.2 )
        dz_max = 0
        d_max = np.asarray( (dx_max, dy_max, dz_max) ).reshape( (1,3) )

        # Calculate distance for each point from center:
        diff = self.coords_preop - c
        dists = np.sqrt( (diff**2).sum( axis=1 ) )

        # linear falloff:
        f = np.maximum(1 - dists, 0)
        self.displ_preop = np.repeat( d_max, repeats=self.n_points, axis=0 )*np.reshape( f, (self.n_points, 1) )

        ########### Repeat for border points:
        diff = self.coords_preop_border - c
        dists = np.sqrt( (diff**2).sum( axis=1 ) )

        # linear falloff:
        f = np.maximum(1 - dists, 0)
        self.displ_preop_border = np.repeat( d_max, repeats=self.n_points, axis=0 )*np.reshape( f, (self.n_points, 1) )

        self.coords_intraop_border = self.coords_preop_border + self.displ_preop_border
        self.coords_intraop = self.coords_preop + self.displ_preop

        #print(self.coords_preop.shape, self.coords_preop_border.shape)

        self.coords_preop_all = np.concatenate( (self.coords_preop, self.coords_preop_border), axis=0 )
        self.coords_intraop_all = np.concatenate( (self.coords_intraop, self.coords_intraop_border), axis=0 )
        self.displ_preop_all = np.concatenate( (self.displ_preop, self.displ_preop_border), axis=0 )

    def load( self ):

        res = edict()
        #coords_preop = self.coords_preop
        #displ = self.displ
        #coords_intraop = self.coords_intraop
        #features_preop = self.features_preop
        #coords_preop_border = self.coords_preop_border
        #displ_preop_border = self.displ_preop_border
        #coords_intraop_border = self.coords_intraop_border

        res.update({
            "coords_preop": self.coords_preop.transpose( (1,0) ),
            "features_preop": self.features_preop,
            "coords_intraop": self.coords_intraop.transpose( (1,0) ),
            "displ_preop": self.displ_preop.transpose( (1,0) ),
            "coords_preop_border": self.coords_preop_border.transpose( (1,0) ),
            "displ_preop_border": self.displ_preop_border.transpose( (1,0) ),
            "coords_intraop_border": self.coords_intraop_border.transpose( (1,0) ),
            "coords_intraop_all": self.coords_intraop_all.transpose( (1,0) ),
            "coords_preop_all": self.coords_preop_all.transpose( (1,0) ),
            "displ_preop_all": self.displ_preop_all.transpose( (1,0) ),
            "coords_intraop_border_partial": self.coords_intraop_border_partial.transpose( (1,0) ),
        })

        return res

    @staticmethod
    def visualize(
            coords_preop,
            coords_intraop = None,
            features_preop = None,
            features_intraop = None,
            displ_target = None,
            displ_prediction = None,
            display = True,
            path = "/tmp/displacement_2d.ps",
            block = False ):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        #fig = plt.figure(figsize=(5,5))
        fig = plt.gcf()
        plt.clf()
        ax = fig.add_subplot(1, 3, 1)#, projection='3d')

        xs = coords_preop[0, ...]
        ys = coords_preop[1, ...]
        zs = coords_preop[2, ...]

        ax.scatter(xs, ys, c="grey" )

        if coords_intraop is not None:
            xs = coords_intraop[0, ...]
            ys = coords_intraop[1, ...]
            zs = coords_intraop[2, ...]

            ax.scatter(xs, ys, c="orange" )

        def plot_displ( ax, coords, displ, color="black" ):
            for i in range(displ.shape[1]):
                d = displ[:, i]
                p = coords[:, i]
                if torch.linalg.norm(d) > 0:
                    ax.arrow( p[0], p[1], d[0], d[1], color=color,
                     #shape='full', color='b', lw=d['MAG']/2., length_includes_head=True, 
                        zorder=100, head_length=0.01, head_width=0.01)

        if features_preop is not None:
            if features_preop.shape[0] == 3:
                plot_displ( ax, coords_preop, features_preop, color = "black" )

        ax.set_aspect("equal")

        ax = fig.add_subplot(1, 3, 2)#, projection='3d')

        ax.set_aspect("equal")

        xs = coords_intraop[0, ...]
        ys = coords_intraop[1, ...]
        zs = coords_intraop[2, ...]

        ax.scatter(xs, ys, c="orange" )

        if features_intraop is not None:
            if features_intraop.shape[0] == 3:
                plot_displ( ax, coords_intraop, features_intraop, color = "black" )

        ax = fig.add_subplot(1, 3, 3)#, projection='3d')

        xs = coords_preop[0, ...]
        ys = coords_preop[1, ...]
        zs = coords_preop[2, ...]

        ax.scatter(xs, ys, c="grey")

        ax.set_aspect("equal")

        if displ_target is not None:
            if displ_target.shape[0] == 3:
                plot_displ( ax, coords_preop, displ_target, color = "black" )
        if displ_prediction is not None:
            if displ_prediction.shape[0] == 3:
                plot_displ( ax, coords_preop, displ_prediction, color = "blue" )

        if display:
            if block:
                plt.show()
            else:
                plt.draw()
                plt.pause(0.001)
        else:
            plt.savefig( path, dpi=300 )



class Rect( DataSample ):

    def __init__( self, n_points=500, surface_amount = 1 ):

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

        border_coords = np.zeros_like( coords )
        for i in range( n_points ):
            side = random.randint(0,3)
            if side == 0:
                border_coords[i, 0] = min_x
                border_coords[i, 1] = random.uniform(min_y, min_y+height)
            elif side == 1:
                border_coords[i, 0] = min_x + width
                border_coords[i, 1] = random.uniform(min_y, min_y+height)
            elif side == 2:
                border_coords[i, 0] = random.uniform( min_x, min_x + width )
                border_coords[i, 1] = min_y
            elif side == 3:
                border_coords[i, 0] = random.uniform( min_x, min_x + width )
                border_coords[i, 1] = min_y + height


        DataSample.__init__( self, coords, border_coords, df, surface_amount = surface_amount )


class DisplDataset( Dataset ):

    def __init__( self, n_shapes = 1000, n_points = 500,
            min_intraop_surface_amount=1, max_intraop_surface_amount=1 ):

        self.shapes = []

        print(f"Generating {n_shapes} samples:")
        for i in range( n_shapes ):
            r = random.uniform( min_intraop_surface_amount, max_intraop_surface_amount )
            s  = Rect( n_points = n_points, surface_amount = r )
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


    dataset = DisplDataset( min_intraop_surface_amount = 0.2, max_intraop_surface_amount=0.8 )

    print(dataset[0])
    print(dataset[1])
    for k, v in dataset[0].items():
        print( k, v.shape )

    #DataSample.visualize( dataset[6]["preop"], dataset[6]["intraop"], dataset[6]["features_preop"], block=True )

