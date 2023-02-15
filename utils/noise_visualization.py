
import open3d as o3d
# from ray import get

from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import NormalizeScale, Compose, FixedPoints, RandomRotate

import numpy as np

from pathlib import Path

model_path = (Path(__file__).parent / '../ros_ws/src/rgcnn_models/src/segmentation/').resolve()

# model_root = (Path(__file__).parent / '../model/segmentation/').resolve()
import sys
# sys.path.append(str(model_root))

sys.path.append("/home/domsa/workspace/git/rgcnn_pytorch/model/segmentation")
from RGCNNSegmentation import seg_model

from utils import GaussianNoiseTransform

# from RGCNNSegmentation import seg_model


import torch


def get_noisy_pcd(index, recompute_normals, row_nr=0, transform=None, y=None):
    colors = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1],[0.3, 0.4, 0.6]])
    
    if transform is None:
        transform = Compose([FixedPoints(512), NormalizeScale(), GaussianNoiseTransform(mu=0, sigma=0.01, recompute_normals=recompute_normals)])

    dataset_noise = ShapeNet(root="../dataset/Journal/ShapeNet", categories="Airplane", transform=transform)
    data = dataset_noise[0]
    data_noise = data.pos
    if y is None:
        y = data.y
        print(y.shape)
        print(y)
    pcd_noise = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_noise))
    pcd_noise.normals = o3d.utility.Vector3dVector(data.x)
    pcd_noise.colors = o3d.utility.Vector3dVector(colors[y])
    pcd_noise = pcd_noise.translate([2.5*index, row_nr*1.5, 0])
    return pcd_noise

    
def show_pcds(pcds):
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()   
    o3d.visualization.draw_geometries(pcds)
    
    
if __name__ == "__main__":
    num_points = 2048
    input_dim  = 22

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128]
        
    net = seg_model(num_points, F, K, M, input_dim, dropout=0.2, reg_prior=False)
    model_path = "/home/domsa/workspace/git/rgcnn_pytorch/JurnalCode/final_models/2048_seg_clean.pt"

    # net.load_state_dict(torch.load(model_path))
    net.eval()
    print(net)
    
    # mu      = [0, 0, 0, 0, 0]
    # sigma   = [0, 0.02, 0.05, 0.08, 0.1]
    # # sigma = [0, 0.002, 0.005, 0.008, 0.01]
    rotations = Compose([RandomRotate(15, 0), RandomRotate(15, 1), RandomRotate(15, 2)])
    
    transform_gn = Compose([FixedPoints(2048), NormalizeScale(), GaussianNoiseTransform(mu=0, sigma=0.01, recompute_normals=True)])
    transform_rot = Compose([FixedPoints(2048), NormalizeScale(), rotations])
    pcd_gn = get_noisy_pcd(0, row_nr=0, recompute_normals=True, transform=transform_gn)
    pcd_rot = get_noisy_pcd(0, row_nr=1, recompute_normals=True, transform=transform_rot)
    
    # pcds_original   = [get_noisy_pcd(i, mu[i], sigma[i], False, 1) for i in range(len(mu))]
    show_pcds([pcd_gn, pcd_rot])