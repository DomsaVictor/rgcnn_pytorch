from cv2 import compare
from matplotlib.pyplot import axes
import utils 
import open3d as o3d

import torch_geometric
import torch

from torch_geometric.datasets import ShapeNet
from torch_geometric.utils import dense_to_sparse
from torch_geometric.transforms import FixedPoints
import numpy as np
from utils import get_laplacian, pairwise_distance
import networkx as nx
import matplotlib
import matplotlib.cm as cm
from pathlib import Path
import copy

model_path = (Path(__file__).parent / '../ros_ws/src/rgcnn_models/src/segmentation/').resolve()

model_root = (Path(__file__).parent / '../model/segmentation/').resolve()
import sys
sys.path.append(str(model_root))

from RGCNNSegmentation import seg_model

def cheb_conv_filtering(x, L, K):
    x0=x
    x1 = torch.matmul(L, x)
    out = [x1]
    for i in range(1, K):
        x2 = 2 * x1 - x0
        out.append(x2)
        x0, x1 = x1, x2

    return x2, out

  
def cheb_conv_visualization(L, K):
    x0 = 1
    conv = []
    x1 = L
    conv.append(x1)
    for i in range(1, K):
        x2 = 2 * x1 - x0
        conv.append(x2)
        x0, x1 = x1, x2
    
    return conv
    
def get_feature_map(out):
    out = out.squeeze()
    out_sum = torch.sum(out, axis=1)
    norm = matplotlib.colors.Normalize(vmin=out_sum.min(), vmax=out_sum.max())
    colors = norm(out_sum.detach().numpy())
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = cmap(colors)[:, :3]
    return colors

def visualize_laplacian(pcd, pts_to_sample):
    points = torch.tensor(np.asarray(pcd.points))
    points = torch.unsqueeze(points, 0)
    graph = pairwise_distance(points)
    L = get_laplacian(graph, normalize=True)
    L = L.squeeze(0)
    L_sum = torch.sum(L, axis=1)
    norm = matplotlib.colors.Normalize(vmin=L_sum.min(), vmax=L_sum.max())
    colors = norm(L_sum)
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = cmap(colors)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    vis.add_geometry(pcd)
    
    
    _, indices_asc = torch.sort(L_sum)
    _, indices_desc = torch.sort(L_sum, descending=True)
    
    pcd_sampled_asc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.squeeze()[indices_asc[0:pts_to_sample]]))
    pcd_sampled_asc.translate([1,0,0])
    
    vis.add_geometry(pcd_sampled_asc)

    pcd_sampled_desc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.squeeze()[indices_desc[0:pts_to_sample]]))
    pcd_sampled_desc.translate([-1,0,0])
    
    vis.add_geometry(pcd_sampled_desc)
    
    vis.run()
    
def rotate_pcd(pcd, axis, angle): 
        c = np.cos(angle)
        s = np.sin(angle)
        
        if axis == 0:
            """rot on x"""
            R = np.array([[1 ,0 ,0],[0 ,c ,-s],[0 ,s ,c]])
        elif axis == 1:
            """rot on y"""
            R = np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]])
        elif axis == 2:
            """rot on z"""
            R = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])
        else:
            return pcd
        pcd_aux = o3d.geometry.PointCloud(copy.deepcopy(pcd))
        return pcd_aux.rotate(R)
    
def compare_laplacians(pcd):
    pcd_rot = rotate_pcd(pcd, 2, -49)
    L_o = get_laplacian(pairwise_distance(torch.tensor(np.asarray(pcd.points)).unsqueeze(0)))
    L_r = get_laplacian(pairwise_distance(torch.tensor(np.asarray(pcd_rot.points)).unsqueeze(0)))
    
    # print(L_o == L_r)
    # print(L_o.T == L_r)
    print((L_o - L_r).mean())
    
    colors = np.array([[1,0,0],[0,1,0]])
    pcd.paint_uniform_color(colors[0])
    pcd_rot.paint_uniform_color(colors[1])
    o3d.visualization.draw_geometries([pcd, pcd_rot])
    
if __name__ == "__main__":
    # dataset = ShapeNet(root="/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/dataset/ShapeNet", categories="Airplane")
    # data = dataset[0]
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    # # o3d.visualization.draw_geometries([pcd])
    
    # visualize_laplacian(pcd)    
    
    # dataset = ShapeNet(root="/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/dataset/ShapeNet", categories="Chair")
    
    # data = dataset[0]
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    # # o3d.visualization.draw_geometries([pcd])
    
    # visualize_laplacian(pcd)    
    
    # dataset = ShapeNet(root="/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/dataset/ShapeNet", categories="Airplane", transform=FixedPoints(400))
    
    # data = dataset[0]
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    # pcd.normals = o3d.utility.Vector3dVector(data.x)
    # o3d.visualization.draw_geometries([pcd])
    
    # pcd = o3d.io.read_point_cloud("/home/victor/Desktop/Dataset_pcd.pcd")
    pcd = o3d.io.read_point_cloud("/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/dataset/Plane/test/1651654248041929.pcd")

    compare_laplacians(pcd)
    
    # visualize_laplacian(pcd, 100)
    # model_name = '400p_model_v2_180.pt'

    # weights_path = str((model_path/model_name).resolve())
    
    # F = [128, 512, 1024]  # Outputs size of convolutional filter.
    # K = [6, 5, 3]         # Polynomial orders.
    # M = [512, 128, 3]
    
    # model = seg_model(400, F, K, M, input_dim=6, recompute_L=False)
    # model.load_state_dict(torch.load(weights_path))
    # model.eval()
    # xyz = torch.tensor(np.asarray(pcd.points)).to(torch.float32)
    # normals = torch.tensor(np.asarray(pcd.normals)).to(torch.float32)
    # pointcloud = torch.cat([xyz, normals], dim=1)
    # pointcloud = pointcloud.unsqueeze(0)
    # logits, x, L = model(pointcloud, None)
    # pred = logits.argmax(dim=2)
    # colors = []
    
    # for out in L:
    #     colors.append(get_feature_map(out))
        
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # for i, color in enumerate(colors):
    #     aux_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    #     aux_pcd.colors = o3d.utility.Vector3dVector(color)
    #     aux_pcd.translate([i,0,0])
    #     vis.add_geometry(aux_pcd)
        
    # vis.run()
    
    # pcd_colors = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # pcd.colors = o3d.utility.Vector3dVector(pcd_colors[pred].squeeze(0))
    # o3d.visualization.draw_geometries([pcd])
    
    points = torch.tensor(np.asarray(pcd.points))
    points = points.unsqueeze(0)
    adj_matrix = pairwise_distance(points)
    L = get_laplacian(adj_matrix)
    
    y, out = cheb_conv_filtering(points, L, 6)
    
    out_sum = out[5].squeeze().sum(axis=1)
    norm = matplotlib.colors.Normalize(vmin=out_sum.min(), vmax=out_sum.max())
    cmap = matplotlib.cm.get_cmap('Spectral')

    colors = norm(out_sum)
    colors = cmap(colors)[:, :3]

    pcd_y = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(out[2].squeeze()))
    pcd_y = pcd_y.translate(-pcd_y.get_center())
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.squeeze()))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    
    print(y)
    # convs = cheb_conv_visualization(L, 50)
    
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    
    
    # max_L = []
    # min_L = []
    # L_sums = []
    
    # for i, L in enumerate(convs):
    #     L = L.squeeze()
    #     L_sum = torch.sum(L, axis=1)
    #     # if i == 0:
        
    #     L_sums.append(L_sum)
    #     max_L.append(L_sum.max())
    #     min_L.append(L_sum.min())


    # cmap = matplotlib.cm.get_cmap('Spectral')

    # for i, L_sum in enumerate(L_sums):
    #     norm = matplotlib.colors.Normalize(vmin=min_L[i], vmax=max_L[i])

    #     colors = norm(L_sum)
    #     colors = cmap(colors)[:, :3]
        
    #     pcd_aux = o3d.geometry.PointCloud(pcd)
    #     pcd_aux.colors = o3d.utility.Vector3dVector(colors)
    #     # aux_pcd.colors = o3d.utility.Vector3dVector(color)
    #     pcd_aux.translate([i,0,0])
    #     vis.add_geometry(pcd_aux)
        
    # # vis.add_geometry(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(out.squeeze(0))))
        
    # vis.run()    
    # print(convs)
    