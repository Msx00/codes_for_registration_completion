import math
import matplotlib.pyplot as plt
import open3d as o3d

def plot_pointcloud_2d( coords, features, dimensions = [-1,-1,1,1], fig=None, block=True, title="" ):
    """ Visualize point cloud with features. Ignore 3rd dimension, if any.

    Arguments:
        coords: [3, N] or [2, N]
        features: [F, N]
    """

    num_plots = features.shape[0]
    num_plots_x = max( int(math.sqrt(num_plots)+1), 1 )
    num_plots_y = max( int(num_plots/num_plots_x+1), 1 )

    if not fig:
        fig = plt.figure( figsize=(20,20) )

    xs = coords[0,:].detach().cpu()
    ys = coords[1,:].detach().cpu()

    for i in range( num_plots ):

        f = features[i,:].detach().cpu()
        ax = fig.add_subplot( num_plots_y, num_plots_x, i+1 )
        data = ax.scatter( xs, ys, c = f )
        ax.set_title( f"f_{i}" )
        ax.set_xlim( dimensions[0], dimensions[2] )
        ax.set_ylim( dimensions[1], dimensions[3] )
        fig.colorbar( data, ax=ax )
    fig.suptitle( title )


    if block:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.001)


def plot_pointcloud_3d( coords_list, features_list ):
    pcds = []
    for coords, features in zip(coords_list, features_list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        if features is not None:
            if len(features.shape) == 2 and features.shape[1] == 0:
                pcd.colors = o3d.utility.Vector3dVector(features)
            elif len(features.shape) == 1 and features.shape[0] == 3:
                f = features.reshape(1,3).repeat( len(pcd.points), axis=0 )
                print(features.reshape(1,3).shape)
                print(f.shape)
                pcd.colors = o3d.utility.Vector3dVector(f)
            else:
                raise ValueError( "Features must be of shape (N,3) or (3,)")
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)