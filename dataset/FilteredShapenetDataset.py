import torch
import torch_geometric
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import RandomScale
from torch_geometric.transforms import *
import numpy as np
import open3d as o3d
import time
import math
from numpy.random import rand
from tqdm import tqdm 
from pathlib import Path
from torch_geometric.data.dataset import Dataset
import os 
from torch_geometric.loader import DenseDataLoader
from torch_geometric.loader import DataLoader



colors = rand(50, 3)

def default_transforms():
        return Compose([FixedPoints(512)])

class FilteredShapeNet(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms(), with_normals=True, save_path=None):
        self.root_dir = root_dir
        # folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.transforms = transform
        self.files = []
        self.labels= []
        self.with_normals = with_normals
        self.save_path = None
        self.folder = folder
        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        new_dir = root_dir/folder
        for file in sorted(os.listdir(new_dir)):
            if file.endswith('.pcd'):
                sample = {}
                sample['pcd_path'] = new_dir/file
                self.files.append(sample)
            
            if file.endswith('.npy'):
                sample={}
                sample['label_path'] = new_dir/file
                self.labels.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, lbl, idx):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        points = torch.tensor(points)

        normals = []

        if self.with_normals == True:
            # pcd.estimate_normals(fast_normal_computation=False)
            # pcd.normalize_normals()
            # pcd.orient_normals_consistent_tangent_plane(100)

            # o3d.visualization.draw_geometries([pcd])

            # print(len(points))
            # if self.save_path is not None:
            #     if len(points) < self.points:
            #         alpha = 0.03
            #         rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            #             pcd, alpha)
            #         # o3d.visualization.draw_geometries([pcd, rec_mesh])

            #         num_points_sample = self.points

            #         pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 
            #         points = pcd_sampled.points
                
            #         normals = np.asarray(pcd_sampled.normals)
            #     else:
            #         normals = np.asarray(pcd.normals)
            # else:
            #     normals = np.asarray(pcd.normals)

            normals = torch.Tensor(np.asarray(pcd.normals)) 
        labels = torch.tensor(np.load(lbl))
        # pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=labels, num_nodes=labels.size(0))

        pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=labels)

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        lbl_path = self.labels[idx]['label_path']
        with open(pcd_path, 'r') as f:
            with open(lbl_path, 'r') as l:    
                pointcloud = self.__preproc__(f.name.strip(), l.name.strip(), idx)
                if self.save_path is not None:
                    name = str(time.time())
                    name = name.replace('.', '')
                    name = str(name) + ".pcd"
                    splits = f.name.strip().split("/")
                    cat = splits[len(splits) - 3]
                    total_path = self.save_path/cat/self.folder/name
                    pcd_save = o3d.geometry.PointCloud()
                    pcd_save.points = o3d.utility.Vector3dVector(pointcloud.pos)
                    pcd_save.normals =  o3d.utility.Vector3dVector(pointcloud.x)
                    o3d.io.write_point_cloud(str(total_path), pcd_save, write_ascii=True)
        return pointcloud


def process_dataset(root, save_path, transform=None, num_points=512):
    #dataset_train = PcdDataset(root, folder="train", transform=transform, save_path=save_path, points=num_points)
    dataset_test =  FilteredShapeNet(root, folder="test", transform=transform, save_path=save_path, points=num_points)

    # print("Processing train data: ")
    # for i in tqdm(range(len(dataset_train))):
    #     _ = dataset_train[i]

    print("Processing test data: ")
    for i in tqdm(range(len(dataset_test))):
        _ = dataset_test[i]

if __name__ == '__main__':
    root = Path("/home/domsa/workspace/data/Airplane")
    dataset_train = FilteredShapeNet(root)
    
    loader = DenseDataLoader(dataset_train, batch_size=2)

    for data in loader:
        print(data)
        break

    # pointcloud = dataset_train[0]
    # pos = pointcloud.pos
    # label = pointcloud.y
    # c = colors[label]
    
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
    # pcd.colors = o3d.utility.Vector3dVector(c)

    # o3d.visualization.draw_geometries([pcd])