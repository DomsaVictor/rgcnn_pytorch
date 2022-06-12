from regex import R
import torch
import open3d as o3d
import numpy as np
from RGCNNSegmentation import seg_model
from numpy.random import default_rng
import os
import sys
from pathlib import Path
# from torch_geometric.nn import fps
from numpy.random import rand

def compute_acc(output, label, num_points):
    ncorrects = np.sum(output == label)
    accuracy = ncorrects * 100 / num_points
    return ncorrects, accuracy
    
curr_dir = Path(__file__).parent

utils_path = (curr_dir / "../../utils").resolve()
sys.path.append(str(utils_path))

# from utils_pcd import pcd_registration

colors = rand(50, 3)
colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.8, 0.3, 0.4]])
pcd_path = (curr_dir / "../../dataset/test_pcd/airplanes").resolve()

pcd_list = []
label_list = []
segmented_pcds = []
output_list= []
for file in sorted(os.listdir(pcd_path), reverse=True):
    if file.endswith('out.pcd'):
        with open(pcd_path/file) as f:
            pcd_list.append(o3d.io.read_point_cloud(f.name))
    if file.endswith('.npy'):
        if file.endswith('out.npy'):
            with open(pcd_path/file) as f:
                output_list.append(np.load(f.name))
        elif file.endswith('sampled.npy'):
            with open(pcd_path/file) as f:
                label_list.append(np.load(f.name))
        # else:
        #     with open(pcd_path/file) as f:
        #         label_list.append(np.load(f.name))
                
ncorrects_list = []                


for i, pcd in enumerate(pcd_list):
    pcd.colors = o3d.utility.Vector3dVector(colors[output_list[i]])
    pcd2 = o3d.geometry.PointCloud(pcd)
    if i < len(label_list):
        pcd2.colors = o3d.utility.Vector3dVector(colors[label_list[i]])
        pcd2.translate(np.array([0.5, 0, 0]))
        ncorrects, acc = compute_acc(output_list[i], label_list[i], 512)
        ncorrects_list.append(ncorrects)
        print(acc)
        # o3d.visualization.draw_geometries([pcd, pcd2])
    else:
        # o3d.visualization.draw_geometries([pcd])
        pass

total_corrects = 0
for n in ncorrects_list:
    total_corrects += n
total_acc = total_corrects * 100 / 512 / len(ncorrects_list)
print(f"Total acc:  {total_acc}%")