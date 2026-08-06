
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def visualize_points_3D(points_list, points_size_list=None, batch_list=None, output_path=None, vis_points_only=False, vis_in_one_plot=False ):
    """Visualize 3D points
        if the points_list is not batched, 

    Args:
        points_list (numpy array): points to be visualized
        points_size_list (list, optional): the list of point size in matplotlib. Defaults to None.
        batch_list (list, optional): list of batch to be visualized. Defaults to [0,].
    """
    #print(points_list.shape)
    cmap = plt.cm.get_cmap("viridis", len(points_list))
    

    # the points in points_list are NOT batched, i.e. they are in shape of [3, N]
    # show the points one after another in subplots
    if not points_size_list:
        points_size_list = [3,] * len(points_list)
    if len(points_list[0].shape) < 3 :
        fig = plt.figure(figsize=(5 * len(points_list),5))
        for idx_p, points in enumerate(points_list):
            ax = fig.add_subplot(1, len(points_list), idx_p + 1, projection='3d')
            xs = points[0, ...]
            ys = points[1, ...]
            zs = points[2, ...]
            # print(cmap(idx))
            ax.scatter(xs, ys, zs, s=points_size_list[idx_p], c=cmap(idx_p))
            ax.title.set_text("point cloud {}".format(idx_p))
            if vis_points_only:
                ax.axis('off')
                ax.grid(False)

    # the points in points_list are batched, i.e. they are in shape of [B, 3, N]
    # visualize the batches according to the given batch list
    else:
        if not batch_list:
            batch_list = np.arange(points_list[0].shape[0])
        fig = plt.figure(figsize=(5 * len(batch_list),5))
        for idx_b, b, in enumerate(batch_list):
            # TODO: this will merge all the points in one plot, rewrite in a better way
            ax = fig.add_subplot(1, len(batch_list), idx_b + 1, projection='3d')
            for idx, points in enumerate(points_list):
                xs = points[b, 0, ...]
                ys = points[b, 1, ...]
                zs = points[b, 2, ...]

                ax.scatter(xs, ys, zs, s=points_size_list[idx], c=cmap(idx))
                ax.title.set_text("batch {}".format(b))
                if vis_points_only:
                    ax.axis('off')
                    ax.grid(False)
    plt.show()
    if output_path:
        plt.savefig(output_path)


def sphere():
    x = []
    y = []
    z = []
    for i in range(2000):
        u = np.random.normal(0,1)
        v = np.random.normal(0,1)
        w = np.random.normal(0,1)
        norm = (u*u + v*v + w*w)**(0.5)
        xi,yi,zi = u/norm,v/norm,w/norm
        x.append(xi)
        y.append(yi)
        z.append(zi)
    x = np.expand_dims(np.asarray(x), axis=0)
    y = np.expand_dims(np.asarray(y), axis=0)
    z = np.expand_dims(np.asarray(z), axis=0)
    # print(x.shape)
    coords = np.concatenate([x,y,z], axis=0)
    return coords




def sphere_surface(num_points, r, rand_points=None, show=False):
    """Create point clouds of a ball surface

    Args:
        num_points: number of points
        r: radius
        show (bool, optional): show point cloud or not. Defaults to False.

    Returns:
        points: point cloud coordinates (3, N)
    """
    # num_points = 1000
    # r = 1.0
    if not isinstance(rand_points, (np.ndarray, np.generic)):
        points = np.random.randn(3, num_points)
    else:
        points = rand_points
    norms = np.sqrt(np.sum(points ** 2, axis=1))
    points = r * (points / norms[:, np.newaxis])

    r = np.random.rand(num_points, 1) * r
    r = r.transpose()

    points = points * r
    # points = points.transpose()

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        plt.show()
    return points


def sphere_2(show=False):
    """Creat point cloud inside a ball

    Args:
        show (bool, optional): show point cloud or not. Defaults to False.

    Returns:
        points: point cloud coordinates
    """
    num_points = 1000
    r = 0.5
    points = np.random.uniform(low=-r, high=r, size=(num_points, 3)) 
    # print(points.shape)

    norm = np.linalg.norm(points, axis=1)
    # print(norm.shape)

    points_ball = points[norm <= 0.5, :]
    # print(points_ball.shape)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(points_ball[:, 0], points_ball[:, 1], points_ball[:, 2], s=1)
        plt.show()
    return points_ball



def sphere_points_inside(num_points=1000, radius=1.0, show=False):
    """Creat point cloud inside a ball with a specific number of points

    Args:
        num_points (int, optional): the number of points will be generated in the sphere.
        radius (float, optional): radius of the generated sphere point cloud.
        show (bool, optional): show point cloud or not. Defaults to False.

    Returns:
        points: point cloud coordinates in (3, N)
    """

    phi = np.random.uniform(0, 2 * np.pi, size=(1, num_points))
    costheta = np.random.uniform(-1, 1, size=(1, num_points)) #
    u = np.random.uniform(0, 1, size=(1, num_points))

    theta = np.arccos( costheta )
    r = radius * np.cbrt( u )

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    # print(x, y, z)
    # print(x.shape, y.shape, z.shape)
    sphere = np.concatenate([x, y, z], axis=0)
    #print(sphere.shape)

    shell = 0.1
    center = np.asarray([0, 0, 0]).reshape([3, 1])
    # boundary = np.zeros(sphere.shape,)
    # dist = np.sqrt(np.square(sphere) - np.square(center))
    dist = np.sqrt(np.sum(np.square(sphere) - np.square(center), axis=0))
    boundary_mask = dist > radius - shell
    # boundary[mask] = 1.0

    if show:
        # visualize_points_3D(points_list=[np.expand_dims(sphere, axis=0).transpose(0, 2, 1)])
        visualize_points_3D(points_list=[sphere], vis_points_only=True)
    return sphere, boundary_mask



def cube_0():
    x_min, x_max = 0,10
    y_min, y_max = -10,10
    z_min, z_max = 20,30

    x = np.linspace(x_min,x_max,13)
    y = np.linspace(y_min,y_max,13)
    z = np.linspace(z_min,z_max,13)
    #print(x.shape)
    # using list comprehension 
    # to compute all possible permutations
    res = [[i, j, k] for i in x 
                    for j in y
                    for k in z]
    res = np.array(res)
    res = res.reshape(-1,3)
    return res




def cube(num_points=1000, side=1, shell = 0.05, show=False):
    res = np.random.uniform(- side / 2, side / 2, size=(3, num_points))
    mask_x = np.absolute(res[0, ...]) <= (side / 2 - shell)
    mask_y = np.absolute(res[1, ...]) <= (side / 2 - shell)
    mask_z = np.absolute(res[2, ...]) <= (side / 2 - shell)
    # print(mask_x.shape, np.unique(mask_x))
    # mask = np.logical_not(np.logical_and(mask_x, mask_y, mask_z))
    mask = np.logical_and(mask_x, mask_y,)
    mask = np.logical_and(mask, mask_z,)
    mask = np.logical_not(mask)
    # print(mask.shape, np.unique(mask))

    if show:
        visualize_points_3D(points_list=[res], vis_points_only=True)

    return res, mask


def rotation_matrix_from_rad(x_rad, y_rad, z_rad):
    R_x = np.array([[1,     0,                  0           ],
                    [0,     np.cos(x_rad),      -np.sin(x_rad)],
                    [0,     np.sin(x_rad),      np.cos(x_rad)]])

    R_y = np.array([[np.cos(y_rad),     0,      np.sin(y_rad)],
                    [0,                 1,      0            ],
                    [-np.sin(y_rad),    0,      np.cos(y_rad)]])

    R_z = np.array([[np.cos(z_rad),     -np.sin(z_rad),     0],
                    [np.sin(z_rad),     np.cos(z_rad),      0],
                    [0,                 0,                  1]])

    # Combine the rotations in the order of roll -> pitch -> yaw
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
    return rotation_matrix



if __name__ == "__main__":
    # res = cube()
    # res = sphere()
    # print(res.shape)
    # res = np.expand_dims(res, axis=0)
    # visualize_points_3D(points_list=[res], points_size_list=[5,], batch_list=[0])

    # sphere_2()
    # sphere_points_inside(num_points=1000, radius=0.5, show=True,)
    # s, mask = sphere_points_inside(num_points=1000)
    # print(np.unique(mask))
    c = cube(num_points=1000, show=True,)
    # # print(res.shape)
    # print(s.shape, c.shape)
    # print(s.shape, mask.shape)
    # print(np.unique(mask))
    # s_boundary = s[:, mask]
    # print(s_boundary.shape)
    # print(s_boundary)
    # visualize_points_3D(points_list=[s, c], points_size_list=[5, 5], batch_list=[0,0])
    # visualize_points_3D(points_list=[s_boundary], points_size_list=[5, ], batch_list=[0,])

    # c, mask = cube()
    # print(mask)
    # c_boundary = c[..., mask]
    # print(c.shape)
    # visualize_points_3D(points_list=[c_boundary], points_size_list=[5, ], batch_list=[0,])


    # s = sphere_surface(num_points=1000, r=1)
    # print(s.shape)

