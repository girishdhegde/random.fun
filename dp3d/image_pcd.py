import numpy as np
import cv2 
import open3d as o3d 

from linemesh import LineMesh


__author__ = "__Girish_Hegde__"


def to_pcd(points, colors=None, viz=False, filepath=None):
    """ Function to convert points array into o3d.PointCloud
        author: Girish D. Hegde - girish.dhc@gmail.com

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        colors (np.ndarray/List, optional): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.

    Returns:
        (o3d.PointCloud): point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            pcd.colors =  o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color(colors)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    if filepath is not None:
        o3d.io.write_point_cloud(filepath, pcd)
    return pcd


def get_bbox(
        mins=(0, 0, 0),
        maxs=(1, 1, 1),
        eye=None,
        color=(1, 0, 0), 
        radius=0.1,
    ):
    """ Fuction to get bbox 

    Args:
        mins (tuple, optional): [minx, miny, minz] of bbox.
        maxs (tuple, optional): [maxx, maxy, maxz] of bbox.
        eye (tuple, optional): [x, y, z] of camera center.
        color (tuple, optional): bbox edge color. Defaults to (1, 0, 0).
        radius (float, optional): bbox edge radius. Defaults to 0.1.
    """

    minx, miny, minz = mins
    maxx, maxy, maxz = maxs
    
    corners = np.array([
        [minx, miny, minz],
        [minx, miny, maxz],
        [minx, maxy, minz],
        [minx, maxy, maxz],
  
        [maxx, miny, minz],
        [maxx, miny, maxz],
        [maxx, maxy, minz],
        [maxx, maxy, maxz],
    ])

    edges = [
        [0, 1],
        [0, 2],
        [0, 4],
        [4, 5],
        [4, 6],
        [6, 7],
        [6, 2],
        [2, 3],
        [1, 5],
        [1, 3],
        [5, 7],
        [7, 3],
    ]
    vertices = corners.tolist()
    if eye is not None:
        vertices.append(eye)
        edges.extend([
            [8, 1],
            [8, 3],
            [8, 5],
            [8, 7],
        ])
    colors = [color for i in range(len(edges))]

    line_mesh_obj = LineMesh(np.array(vertices), edges, colors, radius=radius)
    bbox_mesh = line_mesh_obj.cylinder_set
        
    return bbox_mesh, corners


def custom_draw_geometries(scene, bg=(0, 0, 0)):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg)    
    for obj in scene:
        vis.add_geometry(obj)
    vis.run()
    vis.destroy_window()


def img23d(img, offset=1):
    h, w, c = img.shape
    x = np.arange(w)*offset
    y = np.arange(h)*offset
    x, y = np.meshgrid(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = np.full_like(x, 0.)

    xyz = np.vstack([x, y, z]).T
    clrs = img[::-1, :, ::-1].reshape(h*w, -1)/255
    
    pcd = to_pcd(xyz, clrs, viz=False)
    bbox, corners = get_bbox(
        mins=(0, 0, -h//4), 
        maxs=(w - 1, h - 1, h//4), 
        # eye=(w//2, h//2, h),
        # color=(1, 0, 0),
        color=(1, 1, 1),
        radius=1.,
    )
    custom_draw_geometries([pcd, bbox], (0, 0, 0))


if __name__ == '__main__':
    img = "../Dataset/lena_color_512.tif"
    # img = "./data/image.jpg"
    offset = 1.

    img = cv2.imread(img)
    img23d(img, offset)
