from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import RandomScale
from torch_geometric.transforms import *
import numpy as np
import open3d as o3d
import time
import math
from numpy.random import rand
from tqdm import tqdm 
from tkinter import *
from numpy.random import default_rng
from pathlib import Path
import os

curr_dir = Path(__file__).parent


initial_cameras_for_models = {
    "airplane": [0, 0.25, 0.9],
}

colors = rand(50, 3)

def to_pcd(point_cloud):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud.pos))
    pcd.normals = o3d.utility.Vector3dVector(point_cloud.x) 
    return pcd, point_cloud.y

def rotate(point, angle):
    ox, oy = 0, 0
    px, py = point[0], point[1]

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy, point[2]]

    
def get_view(pcd: o3d.geometry.PointCloud, cam: list):
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound() - np.asarray(pcd.get_min_bound())))

    camera = [cam[0], cam[2], cam[1]]
    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    pcd_filtered = pcd.select_by_index(pt_map)
    camera_point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([camera]))
    camera_point.paint_uniform_color([0, 0, 1])
    
    return  pcd_filtered, camera_point, pt_map


def select_by_index(pcd: o3d.geometry.PointCloud, indexes: list):
    # Open3D's function to select indexes from pcd shuffles the points and indexes
    # so the labels do not correspont. Therefore I created a simple function that does not
    # shuffle the indexes

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    pcd_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[indexes]))
    pcd_filtered.normals = o3d.utility.Vector3dVector(normals[indexes])
    # pcd_filtered.colors = o3d.utility.Vector3dVector(c[indexes])
    return pcd_filtered


def rotate_pcd(pcd, angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis == 0:
        """rot on x"""
        R = np.array([[1 ,0 ,0],[0 ,c ,-s],[0 ,s ,c]])
    elif axis == 1:
        """rot on y"""
        R = np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]])
    elif axis == 1:
        """rot on z"""
        R = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])

    return pcd.rotate(R)

def sample_pcd(pointcloud, initial_cam, cam_range, save_root, i, angles, num_points_threshold=1024):
    assert len(cam_range) < 4
    if pointcloud.x.shape[0] < num_points_threshold:
        return
    pcd, labels = to_pcd(pointcloud)
    
    if len(cam_range) == 2:
        cam_range = range(cam_range[0], cam_range[1])
    elif len(cam_range) == 1:
        cam_range = range(cam_range[0])
    
    folder = ['train', 'test']

    for j in cam_range:
        cam = rotate(initial_cam, j/10)
        pcd = rotate_pcd(pcd, angles[j + i * cam_range[1]], 0)
        pcd_filtered, _, indexes = get_view(pcd, cam)
        filtered_labels = np.asarray(labels[indexes])
        pcd_filtered = select_by_index(pcd, indexes)

        if j % 4 == 0:
            curr_folder = folder[1]
        else:
            curr_folder = folder[0]

        save_path = f"{save_root}/{curr_folder}/"
        
        if not os.path.isdir(save_path):
            os.mkdir(save_path)    

        file_name = save_path + f"plane_{i}_{j}.pcd"
        lable_name = save_path + f"plane_lbl_{i}_{j}.npy"
        np.save(lable_name, filtered_labels)
        o3d.io.write_point_cloud(file_name, pcd_filtered)


def visual_check(pointcloud, initial_cam, cam_range):
    assert len(cam_range) < 3
    pcd, labels = to_pcd(pointcloud)
    pcd.paint_uniform_color([1,0,0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    rng = default_rng()
    angles = rng.choice(360, cam_range[1], replace=False)
    # Delay so you have time to adjust the camera.
    for i in range(25):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
    
    # Creating a python range from list
    if len(cam_range) == 2:
        cam_range = range(cam_range[0], cam_range[1])
    elif len(cam_range) == 1:
        cam_range = range(cam_range[0])

    # Actual filtering and visualization
    for j in cam_range:
        cam = rotate(initial_cam, j/10)
        # pcd = rotate_pcd(pcd, angles[j + 0 * cam_range[1]], 1)

        pcd_filtered, camera_point, indexes = get_view(pcd, cam)
        pcd_filtered.paint_uniform_color([0,1,0])
        camera_point.paint_uniform_color([0,0,1])
        filtered_labels = np.asarray(labels[indexes])
        pcd_filtered = select_by_index(pcd, indexes)
        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=False)
        vis.add_geometry(pcd_filtered, reset_bounding_box=False)
        vis.add_geometry(camera_point, reset_bounding_box=False)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
    vis.destroy_window()


def continue_to_sample(a):
    '''
        Function that is called when pressing the 'Create Database' button
    '''
    r.withdraw()
    init_range = [0, 60]
    initial_cam = a.split(" ")
    if initial_cam == ['']:
        initial_cam = [0, 0.25, 0.9]
    else:
        initial_cam = [float(val) for val in initial_cam]
    rng = default_rng()
    angles = rng.choice(360, pcd_num * init_range[1], replace=True)
    for i in tqdm(range(pcd_num)):
        sample_pcd(dataset[i], initial_cam, [0 , 60], str(sampled_shapenet_root), i, angles)
        # sample_pcd(dataset[i], initial_cam, [50, 60], save_root, "test", i)

    exit()

def restart_with_new_cam(a):
    '''
        Function that is called when pressing the 'Visualize' button
    '''
    print(a)
    initial_cam = a.split(" ")
    if initial_cam == ['']:
        initial_cam = [0, 0.25, 0.9]
    else:
        initial_cam = [float(val) for val in initial_cam]
    visual_check(dataset[0], initial_cam, cam_range)

    return

if __name__ == "__main__":
    #####################################################################
    # Definition of the "User Interface"
    answer = ""
    r = Tk()
    a = StringVar()
    txt = "Visualize the sampleing or create dataset. \n Enter a new camera initial position as \n 'x y' and press 'Visualize'\n to see the result or \n press create database to continue."
    Label(r, text=txt).pack()
    Entry(r, textvariable=a).pack()
    Button(r, text="Create Database", command=lambda:continue_to_sample(a.get())).pack()
    Button(r, text="Visualiation", command=lambda:restart_with_new_cam(a.get())).pack()
    #####################################################################

    #####################################################################
    # Initial parameters of the program
    # TO DO: put them in the UI
    pcd_num = 1000
    initial_cam  = [25, 15]
    cam_range = [0, 60]

    transforms = Compose([Center(), RandomScale(scales=[0.5, 0.5])])
    shapenet_root = (curr_dir / "ShapeNet").resolve()
    
    dataset = ShapeNet(root=shapenet_root, categories="Airplane", include_normals=True, split="trainval", transform=transforms)
    # dataset = ShapeNet(root=shapenet_root, include_normals=True, split="trainval", transform=transforms)

    sampled_shapenet_root = (curr_dir / "Airplane/").resolve()

    if not os.path.isdir(str(sampled_shapenet_root)):
        os.mkdir(str(sampled_shapenet_root))
    
    #####################################################################
    # uncomment this to first see the results with default initial_camera values
    # visual_check(dataset[0], initial_cam, cam_range)

    #####################################################################
    # Run the application
    r.mainloop()
    #####################################################################   


# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd1_filtered)
# vis.add_geometry(camera_point)
# vis.poll_events()
# vis.update_renderer()

# for i in range(25):
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# for i in range(50):
#     vis.clear_geometries()
#     cam = rotate(initial_cam, i/10)
#     pcd1_filtered, camera_point, indexes = get_view(pcd1, cam)
#     pcd1_filtered.paint_uniform_color([0.5, 1, 0.2])
#     vis.add_geometry(pcd1, reset_bounding_box=False)
#     vis.add_geometry(pcd1_filtered, reset_bounding_box=False)
#     vis.add_geometry(camera_point, reset_bounding_box=False)
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# # vis.destroy_window()






########################################################################################################

# import torch
# import torch_geometric
# from torch_geometric.datasets import ShapeNet
# from torch_geometric.transforms import RandomScale
# from torch_geometric.transforms import *
# import numpy as np
# import open3d as o3d
# import time
# import math
# from numpy.random import rand
# from tqdm import tqdm 

# colors = rand(50, 3)

# def to_pcd(point_cloud):
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud.pos))
#     pcd.normals = o3d.utility.Vector3dVector(point_cloud.x) 
#     return pcd, point_cloud.y

# def rotate(point, angle):
#     ox, oy = 0, 0
#     px, py = point[0], point[1]

#     qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
#     qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
#     return [qx, qy]

    
# def get_view(pcd: o3d.geometry.PointCloud, cam: list):
#     diameter = np.linalg.norm(np.asarray(pcd.get_max_bound() - np.asarray(pcd.get_min_bound())))
#     # print(diameter)
#     camera = [cam[0], diameter/2, cam[1]]
#     radius = diameter * 100
#     _, pt_map = pcd.hidden_point_removal(camera, radius)

#     pcd_filtered = pcd.select_by_index(pt_map)
#     camera_point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([camera]))
#     camera_point.paint_uniform_color([0, 0, 1])
    
#     return  pcd_filtered, camera_point, pt_map


# def select_by_index(pcd: o3d.geometry.PointCloud, indexes: list):
#     points = np.asarray(pcd.points)
#     normals = np.asarray(pcd.normals)
#     pcd_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[indexes]))
#     pcd_filtered.normals = o3d.utility.Vector3dVector(normals[indexes])
#     # pcd_filtered.colors = o3d.utility.Vector3dVector(c[indexes])
#     return pcd_filtered


# transforms = Compose([Center(), RandomScale(scales=[0.5, 0.5])])
# root = "/home/domsa/workspace/data/ShapeNet/"
# dataset = ShapeNet(root=root, categories="Airplane", include_normals=True, split="trainval", transform=transforms)


# pcd_num = 100
# initial_cam  = [25, 15]
# save_root = "/home/domsa/workspace/data/Airplane/"


# for i in tqdm(range(pcd_num)):
#     pcd, labels = to_pcd(dataset[i])
#     for j in range(50):
#         cam = rotate(initial_cam, i/10)
#         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
#         filtered_labels = np.asarray(labels[indexes])
#         pcd_filtered = select_by_index(pcd, indexes)

#         name = f"plane_{i}_{j}.pcd"
#         file_name = save_root + "train/" + name
#         lable_name = save_root + "train/" + f"plane_lbl_{i}_{j}.npy"
#         np.save(lable_name, filtered_labels)
#         o3d.io.write_point_cloud(file_name, pcd_filtered)

#     for j in range(50, 60):
#         cam = rotate(initial_cam, i/10)
#         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
#         filtered_labels = np.asarray(labels[indexes])

#         pcd_filtered = select_by_index(pcd, indexes)
#         name = f"plane_{i}_{j}.pcd"
#         file_name = save_root + "test/" + name
#         lable_name = save_root + "test/" + f"plane_lbl_{i}_{j}.npy"
#         np.save(lable_name, filtered_labels)
#         o3d.io.write_point_cloud(file_name, pcd_filtered)


# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd1_filtered)
# vis.add_geometry(camera_point)
# vis.poll_events()
# vis.update_renderer()

# for i in range(25):
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# for i in range(50):
#     vis.clear_geometries()
#     cam = rotate(initial_cam, i/10)
#     pcd1_filtered, camera_point, indexes = get_view(pcd1, cam)
#     pcd1_filtered.paint_uniform_color([0.5, 1, 0.2])
#     vis.add_geometry(pcd1, reset_bounding_box=False)
#     vis.add_geometry(pcd1_filtered, reset_bounding_box=False)
#     vis.add_geometry(camera_point, reset_bounding_box=False)
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# # vis.destroy_window()






# import torch
# import torch_geometric
# from torch_geometric.datasets import ShapeNet
# from torch_geometric.transforms import RandomScale
# from torch_geometric.transforms import *
# import numpy as np
# import open3d as o3d
# import time
# import math
# from numpy.random import rand
# from tqdm import tqdm 
# from tkinter import *

# colors = rand(50, 3)

# def to_pcd(point_cloud):
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud.pos))
#     pcd.normals = o3d.utility.Vector3dVector(point_cloud.x) 
#     return pcd, point_cloud.y

# def rotate(point, angle):
#     ox, oy = 0, 0
#     px, py = point[0], point[1]

#     qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
#     qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
#     return [qx, qy]

    
# def get_view(pcd: o3d.geometry.PointCloud, cam: list):
#     diameter = np.linalg.norm(np.asarray(pcd.get_max_bound() - np.asarray(pcd.get_min_bound())))

#     camera = [cam[0], diameter/2, cam[1]]
#     radius = diameter * 100
#     _, pt_map = pcd.hidden_point_removal(camera, radius)

#     pcd_filtered = pcd.select_by_index(pt_map)
#     camera_point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([camera]))
#     camera_point.paint_uniform_color([0, 0, 1])
    
#     return  pcd_filtered, camera_point, pt_map


# def select_by_index(pcd: o3d.geometry.PointCloud, indexes: list):
#     # Open3D's function to select indexes from pcd shuffles the points and indexes
#     # so the labels do not correspont. Therefore I created a simple function that does not
#     # shuffle the indexes

#     points = np.asarray(pcd.points)
#     normals = np.asarray(pcd.normals)
#     pcd_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[indexes]))
#     pcd_filtered.normals = o3d.utility.Vector3dVector(normals[indexes])
#     # pcd_filtered.colors = o3d.utility.Vector3dVector(c[indexes])
#     return pcd_filtered

# def sample_pcd(pointcloud, initial_cam, cam_range, save_root, i):
#     assert len(cam_range) < 3
#     pcd, labels = to_pcd(pointcloud)
    
#     if len(cam_range) == 2:
#         cam_range = range(cam_range[0], cam_range[1])
#     elif len(cam_range) == 1:
#         cam_range = range(cam_range[0])
    
#     folder = ['train', 'test']

#     for j in cam_range:
#         cam = rotate(initial_cam, j/10)
#         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
#         filtered_labels = np.asarray(labels[indexes])
#         pcd_filtered = select_by_index(pcd, indexes)

#         if j % 4 == 0:
#             curr_folder = folder[1]
#         else:
#             curr_folder = folder[0]

#         name = f"plane_{i}_{j}.pcd"
#         file_name = save_root + f"{curr_folder}/" + name
#         lable_name = save_root + f"{curr_folder}/" + f"plane_lbl_{i}_{j}.npy"
#         np.save(lable_name, filtered_labels)
#         o3d.io.write_point_cloud(file_name, pcd_filtered)


# def visual_check(pointcloud, initial_cam, cam_range):
#     assert len(cam_range) < 3
#     pcd, labels = to_pcd(pointcloud)
#     pcd.paint_uniform_color([1,0,0])
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     vis.add_geometry(pcd)
#     vis.poll_events()
#     vis.update_renderer()
    
#     # Delay so you have time to adjust the camera.
#     for i in range(25):
#         vis.poll_events()
#         vis.update_renderer()
#         time.sleep(0.1)
    
#     # Creating a python range from list
#     if len(cam_range) == 2:
#         cam_range = range(cam_range[0], cam_range[1])
#     elif len(cam_range) == 1:
#         cam_range = range(cam_range[0])

#     # Actual filtering and visualization
#     for j in cam_range:
#         cam = rotate(initial_cam, j/10)
#         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
#         pcd_filtered.paint_uniform_color([0,1,0])
#         camera_point.paint_uniform_color([0,0,1])
#         filtered_labels = np.asarray(labels[indexes])
#         pcd_filtered = select_by_index(pcd, indexes)
#         vis.clear_geometries()
#         vis.add_geometry(pcd, reset_bounding_box=False)
#         vis.add_geometry(pcd_filtered, reset_bounding_box=False)
#         vis.add_geometry(camera_point, reset_bounding_box=False)
#         vis.poll_events()
#         vis.update_renderer()
#         time.sleep(0.1)
#     vis.destroy_window()


# def continue_to_sample(a):
#     '''
#         Function that is called when pressing the 'Create Database' button
#     '''
#     r.withdraw()

#     initial_cam = a.split(" ")
#     if initial_cam == ['']:
#         initial_cam = [0.5, 0.4]
#     else:
#         initial_cam = [float(val) for val in initial_cam]

#     for i in tqdm(range(pcd_num)):
#         sample_pcd(dataset[i], initial_cam, [0 , 60], save_root, i)
#         # sample_pcd(dataset[i], initial_cam, [50, 60], save_root, "test", i)

#     exit()

# def restart_with_new_cam(a):
#     '''
#         Function that is called when pressing the 'Visualize' button
#     '''
#     print(a)
#     initial_cam = a.split(" ")
#     if initial_cam == ['']:
#         initial_cam = [0.5, 0.4]
#     else:
#         initial_cam = [float(val) for val in initial_cam]
#     visual_check(dataset[0], initial_cam, cam_range)

#     return

# if __name__ == "__main__":
#     #####################################################################
#     # Definition of the "User Interface"
#     answer = ""
#     r = Tk()
#     a = StringVar()
#     txt = "Visualize the sampleing or create dataset. \n Enter a new camera initial position as \n 'x y' and press 'Visualize'\n to see the result or \n press create database to continue."
#     Label(r, text=txt).pack()
#     Entry(r, textvariable=a).pack()
#     Button(r, text="Create Database", command=lambda:continue_to_sample(a.get())).pack()
#     Button(r, text="Visualiation", command=lambda:restart_with_new_cam(a.get())).pack()
#     #####################################################################

#     #####################################################################
#     # Initial parameters of the program
#     # TO DO: put them in the UI
#     pcd_num = 100
#     initial_cam  = [25, 15]
#     cam_range = [0, 60]

#     transforms = Compose([Center(), RandomScale(scales=[0.5, 0.5])])
#     root = "/home/victor/workspace/thesis_ws/datasets/ShapeNet/"
#     dataset = ShapeNet(root=root, categories="Airplane", include_normals=True, split="trainval", transform=transforms)

#     save_root = "/home/victor/workspace/thesis_ws/datasets/FilteredShapeNet/"
#     #####################################################################

#     # uncomment this to first see the results with default initial_camera values
#     # visual_check(dataset[0], initial_cam, cam_range)

#     #####################################################################
#     # Run the application
#     r.mainloop()
#     #####################################################################   


# # for i in tqdm(range(pcd_num)):
# #     pcd, labels = to_pcd(dataset[i])
# #     for j in range(50):
# #         cam = rotate(initial_cam, j/10)
# #         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
# #         filtered_labels = np.asarray(labels[indexes])
# #         pcd_filtered = select_by_index(pcd, indexes)

# #         name = f"plane_{i}_{j}.pcd"
# #         file_name = save_root + "train/" + name
# #         lable_name = save_root + "train/" + f"plane_lbl_{i}_{j}.npy"
# #         np.save(lable_name, filtered_labels)
# #         o3d.io.write_point_cloud(file_name, pcd_filtered)

# #     for j in range(50, 60):
# #         cam = rotate(initial_cam, i/10)
# #         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
# #         filtered_labels = np.asarray(labels[indexes])

# #         pcd_filtered = select_by_index(pcd, indexes)
# #         name = f"plane_{i}_{j}.pcd"
# #         file_name = save_root + "test/" + name
# #         lable_name = save_root + "test/" + f"plane_lbl_{i}_{j}.npy"
# #         np.save(lable_name, filtered_labels)
# #         o3d.io.write_point_cloud(file_name, pcd_filtered)


# # vis = o3d.visualization.Visualizer()
# # vis.create_window()
# # vis.add_geometry(pcd1)
# # vis.add_geometry(pcd1_filtered)
# # vis.add_geometry(camera_point)
# # vis.poll_events()
# # vis.update_renderer()

# # for i in range(25):
# #     vis.poll_events()
# #     vis.update_renderer()
# #     time.sleep(0.1)

# # for i in range(50):
# #     vis.clear_geometries()
# #     cam = rotate(initial_cam, i/10)
# #     pcd1_filtered, camera_point, indexes = get_view(pcd1, cam)
# #     pcd1_filtered.paint_uniform_color([0.5, 1, 0.2])
# #     vis.add_geometry(pcd1, reset_bounding_box=False)
# #     vis.add_geometry(pcd1_filtered, reset_bounding_box=False)
# #     vis.add_geometry(camera_point, reset_bounding_box=False)
# #     vis.poll_events()
# #     vis.update_renderer()
# #     time.sleep(0.1)

# # # vis.destroy_window()
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd1_filtered)
# vis.add_geometry(camera_point)
# vis.poll_events()
# vis.update_renderer()

# for i in range(25):
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# for i in range(50):
#     vis.clear_geometries()
#     cam = rotate(initial_cam, i/10)
#     pcd1_filtered, camera_point, indexes = get_view(pcd1, cam)
#     pcd1_filtered.paint_uniform_color([0.5, 1, 0.2])
#     vis.add_geometry(pcd1, reset_bounding_box=False)
#     vis.add_geometry(pcd1_filtered, reset_bounding_box=False)
#     vis.add_geometry(camera_point, reset_bounding_box=False)
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# # vis.destroy_window()






########################################################################################################

# import torch
# import torch_geometric
# from torch_geometric.datasets import ShapeNet
# from torch_geometric.transforms import RandomScale
# from torch_geometric.transforms import *
# import numpy as np
# import open3d as o3d
# import time
# import math
# from numpy.random import rand
# from tqdm import tqdm 

# colors = rand(50, 3)

# def to_pcd(point_cloud):
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud.pos))
#     pcd.normals = o3d.utility.Vector3dVector(point_cloud.x) 
#     return pcd, point_cloud.y

# def rotate(point, angle):
#     ox, oy = 0, 0
#     px, py = point[0], point[1]

#     qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
#     qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
#     return [qx, qy]

    
# def get_view(pcd: o3d.geometry.PointCloud, cam: list):
#     diameter = np.linalg.norm(np.asarray(pcd.get_max_bound() - np.asarray(pcd.get_min_bound())))
#     # print(diameter)
#     camera = [cam[0], diameter/2, cam[1]]
#     radius = diameter * 100
#     _, pt_map = pcd.hidden_point_removal(camera, radius)

#     pcd_filtered = pcd.select_by_index(pt_map)
#     camera_point = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([camera]))
#     camera_point.paint_uniform_color([0, 0, 1])
    
#     return  pcd_filtered, camera_point, pt_map


# def select_by_index(pcd: o3d.geometry.PointCloud, indexes: list):
#     points = np.asarray(pcd.points)
#     normals = np.asarray(pcd.normals)
#     pcd_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[indexes]))
#     pcd_filtered.normals = o3d.utility.Vector3dVector(normals[indexes])
#     # pcd_filtered.colors = o3d.utility.Vector3dVector(c[indexes])
#     return pcd_filtered


# transforms = Compose([Center(), RandomScale(scales=[0.5, 0.5])])
# root = "/home/domsa/workspace/data/ShapeNet/"
# dataset = ShapeNet(root=root, categories="Airplane", include_normals=True, split="trainval", transform=transforms)


# pcd_num = 100
# initial_cam  = [25, 15]
# save_root = "/home/domsa/workspace/data/Airplane/"


# for i in tqdm(range(pcd_num)):
#     pcd, labels = to_pcd(dataset[i])
#     for j in range(50):
#         cam = rotate(initial_cam, i/10)
#         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
#         filtered_labels = np.asarray(labels[indexes])
#         pcd_filtered = select_by_index(pcd, indexes)

#         name = f"plane_{i}_{j}.pcd"
#         file_name = save_root + "train/" + name
#         lable_name = save_root + "train/" + f"plane_lbl_{i}_{j}.npy"
#         np.save(lable_name, filtered_labels)
#         o3d.io.write_point_cloud(file_name, pcd_filtered)

#     for j in range(50, 60):
#         cam = rotate(initial_cam, i/10)
#         pcd_filtered, camera_point, indexes = get_view(pcd, cam)
#         filtered_labels = np.asarray(labels[indexes])

#         pcd_filtered = select_by_index(pcd, indexes)
#         name = f"plane_{i}_{j}.pcd"
#         file_name = save_root + "test/" + name
#         lable_name = save_root + "test/" + f"plane_lbl_{i}_{j}.npy"
#         np.save(lable_name, filtered_labels)
#         o3d.io.write_point_cloud(file_name, pcd_filtered)


# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd1)
# vis.add_geometry(pcd1_filtered)
# vis.add_geometry(camera_point)
# vis.poll_events()
# vis.update_renderer()

# for i in range(25):
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# for i in range(50):
#     vis.clear_geometries()
#     cam = rotate(initial_cam, i/10)
#     pcd1_filtered, camera_point, indexes = get_view(pcd1, cam)
#     pcd1_filtered.paint_uniform_color([0.5, 1, 0.2])
#     vis.add_geometry(pcd1, reset_bounding_box=False)
#     vis.add_geometry(pcd1_filtered, reset_bounding_box=False)
#     vis.add_geometry(camera_point, reset_bounding_box=False)
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.1)

# # vis.destroy_window()


