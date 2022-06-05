from matplotlib import transforms
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
from torch_geometric.transforms import *
from torch_geometric.loader import DenseDataLoader

curr_dir = Path(__file__).parent

utils_path = (curr_dir / "../../utils").resolve()
sys.path.append(str(utils_path))

dataset_path = (curr_dir / "../../dataset/").resolve()
sys.path.append(str(dataset_path))

pcd_path = (dataset_path / "Airplane/test").resolve()
root_dir = (dataset_path / "Airplane").resolve()

from FilteredShapenetDataset import FilteredShapeNet

from utils_pcd import pcd_registration

model_path = (curr_dir / '../../ros_ws/src/rgcnn_models/src/segmentation').resolve()
model_name = "512p_model_v2_10.pt"

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
    index_fps = None
    if data_length < num_points:
        alpha = 0.03
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        #o3d.visualization.draw_geometries([pcd, rec_mesh])
        
        pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points)

        points = np.asarray(pcd_sampled.points)
        normals = np.asarray(pcd_sampled.normals)
    else:
        points = torch.tensor(pcd.points)
        index_fps = fps(points, ratio=float(num_points) / data_length, random_start=True)

        points = np.asarray(pcd.points)[index_fps]
        normals = np.asarray(pcd.normals)[index_fps]
        
    return torch.tensor(points), torch.tensor(normals), index_fps

device = 'cuda'

F = [128, 512, 1024]  # Outputs size of convolutional filter.
K = [6, 5, 3]         # Polynomial orders.
M = [512, 128, 4]
num_points = 512
model = seg_model(num_points, F, K, M, input_dim=6)
model.load_state_dict(torch.load(f'{str(model_path)}/{model_name}'))
model.eval()
model.to(device)

color = []
rng = default_rng()
for i in range(50):
    color.append(rng.choice(254, size=3, replace=False).tolist())

colors = rand(50, 3)


pcd_list = []
label_list = []
segmented_pcds = []
output_list= []

max_pcds = 15000
curr_pcds = 0

max_labels = 15000
curr_labels = 0

# save_path = (curr_dir / "../../dataset/test_pcd/airplanes").resolve()

# for file in sorted(os.listdir(pcd_path), reverse=True):
#     if file.endswith('pcd') and curr_pcds < max_pcds:
#         with open(pcd_path/file) as f:
#             pcd_list.append(o3d.io.read_point_cloud(f.name))
#             output_list.append(f.name)
#         curr_pcds += 1
#     if file.endswith('npy') and curr_labels < max_labels:
#         with open(pcd_path/file) as f:
#             label_list.append(np.load(f.name))
#         curr_labels += 1
        

# registrator = pcd_registration()
# first = True
# for i, pcd in enumerate(pcd_list):
#     center = pcd.get_center()
#     pcd = pcd.translate(-center, relative=True)

#     if first:
#         first = False
#         registrator.set_target(pcd)
#     else:
#         registrator.set_source(pcd)
#         pcd = registrator.register_pcds()
#         # registrator.draw_pcds()
    
#     if not pcd.has_normals():
#         pcd = rotate_pcd(pcd, 0, 0)
#         pcd.estimate_normals(fast_normal_computation=False)
#         pcd.orient_normals_consistent_tangent_plane(30)
    
#     points, normals, indexes = resize_pointcloud(pcd, num_points)
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.normals = o3d.utility.Vector3dVector(normals)

#     x = torch.cat([points, normals], 1).unsqueeze(0)
#     pred, _, _ = model(x.to(torch.float32).to(device), None)
    
#     labels = pred.argmax(dim=2)
#     labels = labels.squeeze(0)
#     labels = labels.to('cpu')
#     aux_label = np.zeros([num_points, 3])
#     for j in range(num_points):
#         aux_label[j] = color[int(labels[j])]

#     # save_path = str((Path(__file__) / "outputs").resolve())
#     # if not os.path.isdir(save_path):
#     #     os.mkdir(save_path)
    
#     # pcd.colors = o3d.utility.Vector3dVector(colors[label_list[i]])
#     pcd.colors = o3d.utility.Vector3dVector(colors[labels])
#     # pcd.paint_uniform_color([1,0,0])
#     o3d.io.write_point_cloud(str((save_path / (output_list[i].split("/")[-1][:-4]+"_out.pcd")).resolve())
# , pcd)
#     np.save(str((save_path / (output_list[i].split("/")[-1][:-4]+"_out.npy")).resolve())
# , labels)
#     if i < len(label_list):
#         np.save(str((save_path / (output_list[i].split("/")[-1][:-4]+"sampled.npy")).resolve())
# , label_list[i][indexes])
#     segmented_pcds.append(pcd)
#     # o3d.visualization.draw_geometries([pcd])
    
# # o3d.visualization.draw_geometries(segmented_pcds)


transforms = Compose([FixedPoints(num_points)])
dataset = FilteredShapeNet(root_dir=root_dir, folder='test', transform=transforms)
loader = DenseDataLoader(dataset, batch_size=8, num_workers=6)

total_correct = 0
for data in loader:
    x = torch.cat([data.pos.type(torch.float32), data.x.type(torch.float32)], dim=2)
    y = data.y
    logits, _, _ = model(x.to(device), None)
    logits = logits.cpu()
    pred = logits.argmax(dim=2)

    total_correct += int((pred == y).sum()) 
    
acc = total_correct * 100 / (num_points * len(dataset))
print(acc)
