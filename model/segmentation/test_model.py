from regex import R
import torch
import open3d as o3d
import numpy as np
from RGCNNSegmentation import seg_model
from numpy.random import default_rng
import os
import sys
from pathlib import Path
from torch_geometric.nn import fps
from numpy.random import rand

curr_dir = Path(__file__).parent

utils_path = (curr_dir / "../../utils").resolve()
sys.path.append(str(utils_path))

from utils_pcd import pcd_registration

model_path = (curr_dir / '../../ros_ws/src/rgcnn_models/src/segmentation').resolve()
model_name = "1024p_model_v2_1.pt"

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

def resize_pointcloud(pcd, num_points):
    data_length = torch.tensor(pcd.points).size(0)
    if data_length < num_points:
        alpha = 0.03
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        #o3d.visualization.draw_geometries([pcd, rec_mesh])
        
        pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points)

        points = np.asarray(pcd_sampled.points)
        normals = np.asarray(pcd_sampled.normals)
    else:
        index_fps = fps(points, ratio=float(num_points)/data_length, random_start=True)

        points = np.asarray(pcd.points)[index_fps]
        normals = np.asarray(pcd.normals)[index_fps]
        
    return torch.tensor(points), torch.tensor(normals)
device = 'cuda'

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, 4]
num_points = 1024
model = seg_model(num_points, F, K, M, input_dim=6)
model.load_state_dict(torch.load(f'{str(model_path)}/{model_name}'))
model.to(device)
model.eval()

color = []
rng = default_rng()
for i in range(50):
    color.append(rng.choice(254, size=3, replace=False).tolist())

colors = rand(50, 3)

pcd_path = (curr_dir / "../../dataset/test_pcd/airplanes").resolve()

pcd_list = []
label_list =[]
segmented_pcds = []
for file in sorted(os.listdir(pcd_path), reverse=True):
    if file.endswith('pcd'):
        with open(pcd_path/file) as f:
            pcd_list.append(o3d.io.read_point_cloud(f.name))
    if file.endswith('npy'):
        with open(pcd_path/file) as f:
            label_list.append(np.load(f.name))


registrator = pcd_registration()
first = True
for i, pcd in enumerate(pcd_list):
    center = pcd.get_center()
    pcd = pcd.translate(-center, relative=True)

    if first:
        first = False
        registrator.set_target(pcd)
    else:
        registrator.set_source(pcd)
        pcd = registrator.register_pcds()
        # registrator.draw_pcds()
    
    if not pcd.has_normals():
        pcd = rotate_pcd(pcd, 0, 0)
        pcd.estimate_normals(fast_normal_computation=False)
        pcd.orient_normals_consistent_tangent_plane(30)
    
    points, normals = resize_pointcloud(pcd, 1024)
        
    x = torch.cat([points, normals], 1).unsqueeze(0)
    pred, _, _ = model(x.to(torch.float32).to(device), None)
    
    labels = pred.argmax(dim=2)
    labels = labels.squeeze(0)
    labels = labels.to('cpu')

    aux_label = np.zeros([num_points, 3])
    for j in range(num_points):
        aux_label[j] = color[int(labels[j])]

    pcd.colors = o3d.utility.Vector3dVector(colors[label_list[i]])
    pcd.colors = o3d.utility.Vector3dVector(colors[labels])
    # pcd.paint_uniform_color([1,0,0])
    segmented_pcds.append(pcd)
    o3d.visualization.draw_geometries([pcd])
    
o3d.visualization.draw_geometries(segmented_pcds)
