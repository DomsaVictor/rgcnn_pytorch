
import open3d as o3d
# from ray import get

from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import NormalizeScale, Compose, FixedPoints

import numpy as np

from pathlib import Path

model_path = (Path(__file__).parent / '../ros_ws/src/rgcnn_models/src/segmentation/').resolve()

model_root = (Path(__file__).parent / '../model/segmentation/').resolve()
import sys
sys.path.append(str(model_root))
from utils import GaussianNoiseTransform

def get_noisy_pcd(index, mu, sigma, recompute_normals, row_nr=0):
    colors = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1],[0.3, 0.4, 0.6]])
    transform = Compose([FixedPoints(512),NormalizeScale(), GaussianNoiseTransform(mu=mu, sigma=sigma, recompute_normals=recompute_normals)])
    dataset_noise = ShapeNet(root="/home/domsa/workspace/git/rgcnn_pytorch/dataset/Journal/ShapeNet", categories="Airplane", transform=transform)
    data = dataset_noise[0]
    data_noise = data.pos
    y = data.y
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
    mu      = [0, 0, 0, 0, 0]
    sigma   = [0, 0.02, 0.05, 0.08, 0.1]
    # sigma = [0, 0.002, 0.005, 0.008, 0.01]
    
    pcds_recomputed = [get_noisy_pcd(i, mu[i], sigma[i], True,  0) for i in range(len(mu))]
    pcds_original   = [get_noisy_pcd(i, mu[i], sigma[i], False, 1) for i in range(len(mu))]
    show_pcds(pcds_original + pcds_recomputed)