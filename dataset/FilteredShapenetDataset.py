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
from tkinter import *
from tkinter import ttk
import copy

colors = rand(50, 3)

def default_transforms():
    return Compose([FixedPoints(512)])

class PcdDataset(Dataset):
    def __init__(self, root_dir, points=512, valid=False, folder="train", 
                 transform=default_transforms(), with_normals=True, save_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        self.with_normals = with_normals
        self.save_path = None
        self.folder = folder
        self.points = points
        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                
                for category in self.classes.keys():
                    save_dir = save_path/Path(category)/folder
                    os.makedirs(save_dir)
                    
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.pcd'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, idx):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        points = torch.tensor(points)

        normals = []

        if self.with_normals == True:
            pcd.estimate_normals(fast_normal_computation=False)
            pcd.normalize_normals()
            pcd.orient_normals_consistent_tangent_plane(100)

            if self.save_path is not None:
                if len(points) < self.points:
                    alpha = 0.03
                    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        pcd, alpha)

                    num_points_sample = self.points

                    pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 
                    points = pcd_sampled.points
                
                    normals = np.asarray(pcd_sampled.normals)
                else:
                    normals = np.asarray(pcd.normals)
            else:
                normals = np.asarray(pcd.normals)

            normals = torch.Tensor(normals)

        pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=self.files[idx]['category'])

        if self.save_path is not None:

            pcd.estimate_normals(fast_normal_computation=False)
            pcd.normalize_normals()

            normals=np.asarray(pcd.normals)

            if len(points) < self.points:
                alpha = 0.03
                rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha)

                num_points_sample = self.points

                pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 

                points = pcd_sampled.points
                points = torch.tensor(points)
                points=points.float()
                normals = np.asarray(pcd_sampled.normals)
            else:
                nr_points_fps=self.points
                nr_points=points.shape[0]
                index_fps = fps(points, ratio=float(nr_points_fps/nr_points), 
                                random_start=True)

                index_fps=index_fps[0:nr_points_fps]

                fps_points=points[index_fps]
                fps_normals=normals[index_fps]

                points=fps_points
                normals = fps_normals
        else:
            normals=np.asarray(pcd.normals)

        normals = torch.Tensor(normals)
        normals=normals.float()

        pointcloud = torch_geometric.data.Data(normal=normals, 
                                               pos=points, 
                                               y=self.classes[self.files[idx]['category']])

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f.name.strip(), idx)
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


                # name = str(time.time())
                # name = name.replace('.', '')
                # name = str(name) + ".pcd"
                splits = f.name.strip().split("/")
                cat = splits[len(splits) - 3]

                name = splits[len(splits) - 1]

                total_path = self.save_path/cat/self.folder/name
                pcd_save = o3d.geometry.PointCloud()
                pcd_save.points = o3d.utility.Vector3dVector(pointcloud.pos)
                pcd_save.normals =  o3d.utility.Vector3dVector(pointcloud.normal)
                o3d.io.write_point_cloud(str(total_path), pcd_save, write_ascii=True)
        return pointcloud

class FilteredShapeNet(Dataset):
    def __init__(self, root_dir, folder="train", transform=FixedPoints(512), with_normals=True, save_path=None):
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
        points = torch.tensor(np.asarray(pcd.points))
        labels = torch.tensor(np.load(lbl))
        colors = np.array([[1,0,0],[0,1,0],[0,0,1],[0.3,0.5,0.7]])

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
        # pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=labels, num_nodes=labels.size(0))

        pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=labels)

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        lbl_path = self.labels[idx]['label_path']
        # print(pcd_path)
        # print(lbl_path)
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


class ShapeNetCustom(Dataset):
    def __init__(self, root_dir, folder="train", transform=FixedPoints(512), with_normals=True, save_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir/folder)) if os.path.isdir(root_dir/folder/dir)]
        self.transforms = transform
        self.files = []
        self.labels= []
        self.categories = []
        self.with_normals = with_normals
        self.save_path = None
        self.folder = folder
        self.all_categories = sorted(["Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
                "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
                "Pistol", "Rocket", "Skateboard", "Table"])
        self.seg_classes = {
                'Airplane': [0, 1, 2, 3],
                'Bag': [4, 5],
                'Cap': [6, 7],
                'Car': [8, 9, 10, 11],
                'Chair': [12, 13, 14, 15],
                'Earphone': [16, 17, 18],
                'Guitar': [19, 20, 21],
                'Knife': [22, 23],
                'Lamp': [24, 25, 26, 27],
                'Laptop': [28, 29],
                'Motorbike': [30, 31, 32, 33, 34, 35],
                'Mug': [36, 37],
                'Pistol': [38, 39, 40],
                'Rocket': [41, 42, 43],
                'Skateboard': [44, 45, 46],
                'Table': [47, 48, 49],}
        if save_path is not None:
            self.save_path = save_path
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        for fld in folders:
            new_dir = root_dir/folder/fld
            for file in sorted(os.listdir(new_dir)):
                if file.endswith('.pcd'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    self.files.append(sample)
                    self.categories.append(fld)
                
                if file.endswith('.npy'):
                    sample={}
                    sample['label_path'] = new_dir/file
                    self.labels.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, lbl, curr_class):
        pcd = o3d.io.read_point_cloud(file)
        points = torch.tensor(np.asarray(pcd.points))
        labels = torch.tensor(np.load(lbl))
        # colors = np.array([[1,0,0],[0,1,0],[0,0,1],[0.3,0.5,0.7]])

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
        # pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=labels, num_nodes=labels.size(0))
        cat = torch.tensor([self.all_categories.index(curr_class)])
        # labels += min(self.seg_classes[curr_class])
        pointcloud = torch_geometric.data.Data(x=normals, pos=points, y=labels, category=cat)

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        lbl_path = self.labels[idx]['label_path']
        curr_class = self.categories[idx]
        # print(pcd_path)
        # print(lbl_path)
        with open(pcd_path, 'r') as f:
            with open(lbl_path, 'r') as l:    
                pointcloud = self.__preproc__(f.name.strip(), l.name.strip(), curr_class)
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


class FilteredShapenetViewer():
    def __init__(self, root_dir, folder="train", transform=default_transforms(), with_normals=True):
        self.root_dir = root_dir
        unit = 0.5
        self.translations = [[[-unit],[-unit],[0]], [[0],[-unit],[0]], [[-unit],[unit],[0]], 
                             [[-unit],[0],[0]],  [[0],[0],[0]],  [[unit],[0],[0]], 
                             [[unit],[-unit],[0]],  [[0],[unit],[0]],  [[unit],[unit],[0]]]
        self.colors = np.array([[1,0,0],[0,1,0],[0,0,1],[0.3,0.5,0.7]])
        self.folder = folder
        self.transform = transform
        self.keep_view = True
        self.dataset = FilteredShapeNet(root_dir, folder, transform, with_normals=with_normals)
        self.i = 0
        self.master = Tk()
        self.first_draw = True
        self.pcds = []
        self.vis = o3d.visualization.Visualizer()
        frame = ttk.Frame(self.master, padding=10)
        frame.grid()
        self.label = Label(frame, text=str(self.i)).grid(column=0, row=0)
        Button(frame, text="Next", command=lambda:self.next_view()).grid(column=1, row=0)
        self.vis.create_window()
        self.update_pcds()
        self.view_dataset()

        
    def rotate_pcd(self, pcd, axis, angle): 
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
    
    
    def update_pcds(self):
        self.pcds = []
        for j in range(min(9, len(self.dataset) - self.i)):
            pointcloud = self.dataset[j + self.i]
            pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pointcloud.pos))
            # pcd = self.rotate_pcd(pcd, axis=0, angle=45)
            # pcd = self.rotate_pcd(pcd, axis=1, angle=-45)
            pcd = pcd.translate(np.asarray(self.translations[j]))
            pcd.colors = o3d.utility.Vector3dVector(self.colors[np.array(pointcloud.y)])
            self.pcds.append(pcd)
        self.i = self.i + min(9, len(self.dataset) - self.i)
        
    def view_dataset(self):
        self.keep_view = True
        self.vis.clear_geometries()
        for pcd in self.pcds:
            self.vis.add_geometry(pcd, reset_bounding_box=self.first_draw)
        self.first_draw = False
        while(self.keep_view):
            self.vis.poll_events()
            self.vis.update_renderer()
            self.master.update_idletasks()
            self.master.update()
            
        # self.vis.poll_events()
        # self.vis.update_renderer()
            
    def next_view(self):
        self.keep_view=False
        self.update_pcds()
        self.view_dataset()
        

def process_dataset(root, save_path, transform=None, num_points=512):
    #dataset_train = PcdDataset(root, folder="train", transform=transform, save_path=save_path, points=num_points)
    dataset_test =  FilteredShapeNet(root, folder="test", transform=transform, save_path=save_path, points=num_points)

    # print("Processing train data: ")
    # for i in tqdm(range(len(dataset_train))):
    #     _ = dataset_train[i]

    print("Processing test data: ")
    for i in tqdm(range(len(dataset_test))):
        _ = dataset_test[i]


colors = np.array([[1,0,0],[0,1,0],[0,0,1],[0.3,0.5,0.7]])

def choose_pcd(dataset):
    pointcloud = dataset[0]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud.pos))
    pcd.colors = o3d.utility.Vector3dVector(colors[pointcloud.y])
    return pcd

if __name__ == '__main__':
    root_f = (Path(__file__).parent / "Plane").resolve()
    root_s = (Path(__file__).parent / "ShapeNet").resolve()
    # dataset_f = FilteredShapeNet(root_f, folder="train")
    # dataset_s = ShapeNet(str(root_s), categories="Airplane", split="train")
    # pcd_f = choose_pcd(dataset_f)
    # pcd_s = choose_pcd(dataset_s)
    # pcd_s.translate(np.array([0.5, 0, 0]))
    # o3d.visualization.draw_geometries([pcd_f, pcd_s])
    
    # root_f = Path(str(Path(__file__).parent/"Airplane")).resolve()
    data_viewer = FilteredShapenetViewer(root_f, "test")
    
    # for data in loader:
    #     print(data)
    #     break

    # pointcloud = dataset_train[0]
    # pos = pointcloud.pos
    # label = pointcloud.y
    # c = colors[label]
    
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
    # pcd.colors = o3d.utility.Vector3dVector(c)

    # o3d.visualization.draw_geometries([pcd])