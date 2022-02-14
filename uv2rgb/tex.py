import numpy as np 
import open3d as o3d 
import cv2 


__author__ = "__Girish_Hegde__"


def to_mesh(points, faces, colors=None, viz=False, filepath=None):
    """ Function to convert points array into o3d.geometry.TriangleMesh
        author: girish d. hegde - girish.dhc@gmail.com

    Args:
        points (np.ndarray): [N, 3] - list of xyz of points.
        faces (np.ndarray): [M, 3] - list of triangle faces of points.
        colors (np.ndarray/List, optional): [N, 3] pcd colors or [r, g, b]. Defaults to None.
        viz (bool, optional): show point cloud. Defaults to False.
        filepath (str, optional): save point cloud as. Defaults to None.

    Returns:
        (o3d.geometry.TriangleMesh): mesh
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        colors = np.array(colors)
        if len(colors.shape) > 1:
            mesh.vertex_colors =  o3d.utility.Vector3dVector(colors)
        else:
            mesh.paint_uniform_color(colors)
    if viz:
        o3d.visualization.draw_geometries([mesh])
    if filepath is not None:
        o3d.io.write_triangle_mesh(filepath, mesh)
    return mesh


def uv2rgb(mesh, texture, rgb=False, save_as=None):
    """ Function to convert uv textured mesh to rgb point colored mesh
        author: girish d. hegde - girish.dhc@gmail.com

    Args:
        meshfile (o3d.geometry.TriangleMesh): mesh with texture.
        texture (np.ndarray): [h, w, 3] - texture image.
        rgb (bool, optional): True if texture is rgb, False if bgr. Defaults to False.
        save_as (str, optional): path to save.
    Returns:
        o3d.io.TriangleMesh: colored mesh.
        np.ndarray: [N, 3] - vertices/points.
        np.ndarray: [N, 3] - vertex colors.
        np.ndarray: [M, 3] - triangle faces.
    """
    texture = texture[..., ::-1] if not rgb else texture
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    uvs = np.asarray(mesh.triangle_uvs)
    face_vertices = faces.reshape(-1)

    h, w, c = texture.shape
    u, v = uvs.T
    u = (u * w).astype(np.int32)
    v = ((1 - v) * h).astype(np.int32)

    colors = np.empty_like(vertices)
    colors[face_vertices] = texture[v, u]/255

    colored_mesh = to_mesh(vertices, faces, colors, viz=False, filepath=save_as)
    return colored_mesh, vertices, colors, faces


if __name__ == '__main__':
    # mesh = './seg.obj'
    # tex = './seg_0.png'
    mesh = '../data/test.obj'
    tex = '../data/testtexture1.png'
    out = '../data/colored.obj'

    # Read data
    mesh = o3d.io.read_triangle_mesh(mesh, enable_post_processing=True)
    tex = cv2.imread(tex)

    # Process data
    colored_mesh, vertices, colors, faces = uv2rgb(mesh, tex, rgb=False, save_as=out)
    
    # Visualization
    o3d.visualization.draw_geometries([colored_mesh], mesh_show_back_face=True)

    # Usage
    print('\npcd.coords\n', vertices)
    print('\npcd.colors\n', colors)
    print('\nmesh.faces\n', faces)
