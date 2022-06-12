from nis import cat
from unicodedata import category
import numpy as np
import os

from tqdm import tqdm
import torch
# from torch.utils.data import DataLoader
from torchvision import transforms, utils

from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.data.dataset import Dataset
from pathlib import Path
import scipy.spatial.distance

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from torch_geometric.nn import fps
from math import floor 

import torch_geometric.transforms

import open3d as o3d
import random
import time

from torch_geometric.transforms import Compose

from utils import GaussianNoiseTransform
from utils import Sphere_Occlusion_Transform
from torch_geometric.transforms import RandomRotate

import probreg_functions as probreg_f
import Passthrough_function as pf_f

def default_transforms():
    return transforms.Compose([
        # transforms.PointSampler(512),
        # transforms.Normalize(),
        # transforms.ToTensor()
    ])

class PcdDataset(Dataset):
    def __init__(self, root_dir, points=512, valid=False, folder="train", transform=default_transforms(), save_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
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
        #print(file)
        points = np.asarray(pcd.points)
        points = torch.tensor(points)

        normals = []


            #print(len(points))
        if self.save_path is not None:

            pcd.estimate_normals(fast_normal_computation=False)
            pcd.normalize_normals()
            #pcd.orient_normals_consistent_tangent_plane(100)

            normals=np.asarray(pcd.normals)

            if len(points) < self.points:
                alpha = 0.03
                rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha)


                #o3d.visualization.draw_geometries([pcd, rec_mesh])

                num_points_sample = self.points

                pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 

                points = pcd_sampled.points
                points = torch.tensor(points)
                points=points.float()
                normals = np.asarray(pcd_sampled.normals)

                # print(len(points))
                # print(len(normals))
            else:

                nr_points_fps=self.points
                nr_points=points.shape[0]

                index_fps = fps(points, ratio=float(nr_points_fps/nr_points) , random_start=True)

                index_fps=index_fps[0:nr_points_fps]

                fps_points=points[index_fps]
                fps_normals=normals[index_fps]

                points=fps_points
                normals = fps_normals
        else:
            normals=np.asarray(pcd.normals)

        normals = torch.Tensor(normals)
        normals=normals.float()

        pointcloud = torch_geometric.data.Data(normal=normals, pos=points, y=self.classes[self.files[idx]['category']])

        if self.transforms:
            pointcloud = self.transforms(pointcloud)


        return pointcloud

    def __getitem__(self, idx):

       
        aligner = probreg_f.pcd_registration()
        pcd_path = self.files[idx]['pcd_path']

        

        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f.name.strip(), idx)
            if self.save_path is not None:
                # name = str(time.time())
                # name = name.replace('.', '')
                # name = str(name) + ".pcd"

                splits = f.name.strip().split("/")
                cat = splits[len(splits) - 3]

                new_splits=f.name.strip().split(cat)
                base_pcd_dir=new_splits[0]+cat

                base_pcd_dir=Path(base_pcd_dir)

                base_pcd_name = [file for file in os.listdir(base_pcd_dir) if file.endswith('.pcd')]
                base_pcd_name = base_pcd_name[0]

                with open(base_pcd_dir/base_pcd_name, 'r') as f:
                    target = o3d.io.read_point_cloud(f.name)
                    aligner.set_target(target)
                

                name = splits[len(splits) - 1]

                name_npy = name.replace('pcd','npy')


                total_path = self.save_path/cat/self.folder/name
                total_path_npy =self.save_path/cat/self.folder/name_npy

                pcd_save = o3d.geometry.PointCloud()
                pcd_save.points = o3d.utility.Vector3dVector(pointcloud.pos)

                source_pcd = pcd_save
                aligner.set_source(source_pcd)
                result = aligner.register_pcds()

                # cloud.paint_uniform_color([0.5, 0.5, 0.5])

                cloud=result

                labels=2*np.ones((np.asarray(cloud.points).shape[0],1))
                
                label_indices=range(np.asarray(cloud.points).shape[0])

                label_indices=np.expand_dims(label_indices, axis=1)

                labels=np.concatenate((labels,label_indices),axis=1)
                labels=np.int_(labels)

                L_1=0.40
                l_1=0.15
                h_1=0.05

                angle_x_1=0
                angle_y_1=10
                angle_z_1=0

                centroid_x_1=-0.2874577
                centroid_y_1= -0.14024239+0.1
                centroid_z_1=1.21766081-0.05

                color_box_1=[0.45, 0.1, 0.9]
                color_pass_cloud_1=[1, 0.3, 0.5]

                pf_filter1=o3d.geometry.PointCloud()
                pf_filter2=o3d.geometry.PointCloud()
                pf_filter3=o3d.geometry.PointCloud()

                box_1=o3d.geometry.PointCloud()
                box_2=o3d.geometry.PointCloud()
                box_3=o3d.geometry.PointCloud()

                label_nr_1=0

                pf_filter1,box_1,cloud,labels1,remaining_labels=pf_f.Passthrough_custom(L=L_1,
                                                    l=l_1,
                                                    h=h_1,
                                                    angle_x=angle_x_1,
                                                    angle_y=angle_y_1,
                                                    angle_z=angle_z_1,
                                                    centroid_x=centroid_x_1,
                                                    centroid_y=centroid_y_1,
                                                    centroid_z=centroid_z_1,
                                                    color_box=color_box_1,
                                                    color_pass_cloud=color_pass_cloud_1,
                                                    cloud=cloud,
                                                    labels=labels,
                                                    label_nr=label_nr_1)
                cloud.paint_uniform_color([0.5, 0.5, 0.5])
                                      
                # o3d.visualization.draw_geometries([cloud,box_2,pf_filter2])
                #o3d.visualization.draw_geometries([cloud])

                L_2=0.27
                l_2=1
                h_2=0.23

                angle_x_2=5
                angle_y_2=0
                angle_z_2=2.7

                centroid_x_2=-0.2874577+0.09
                centroid_y_2= -0.14024239+0.02
                centroid_z_2=1.21766081

                color_box_2=[0.2, 0.2, 0.5]
                color_pass_cloud_2=[0., 1., 1.]

                label_nr_2=1

                pf_filter2,box_2,cloud,labels2,remaining_labels = pf_f.Passthrough_custom(L=L_2,
                                                    l=l_2,
                                                    h=h_2,
                                                    angle_x=angle_x_2,
                                                    angle_y=angle_y_2,
                                                    angle_z=angle_z_2,
                                                    centroid_x=centroid_x_2,
                                                    centroid_y=centroid_y_2,
                                                    centroid_z=centroid_z_2,
                                                    color_box=color_box_2,
                                                    color_pass_cloud=color_pass_cloud_2,
                                                    cloud=cloud,
                                                    labels=remaining_labels,
                                                    label_nr=label_nr_2)

                cloud.paint_uniform_color([0.5, 0.5, 0.5])

                
                                                    
                #o3d.visualization.draw_geometries([cloud,box_1,pf_filter1])
                #o3d.visualization.draw_geometries([cloud])

                L_3=0.25
                l_3=0.27
                h_3=0.30

                angle_x_3=0
                angle_y_3=2.55
                angle_z_3=0

                centroid_x_3=-0.2874577-0.07
                centroid_y_3= -0.14024239+0.1
                centroid_z_3=1.21766081


                color_box_3=[0.2, 0.2, 0.5]
                color_pass_cloud_3=[0., 1., 1.]

                label_nr_3=1

                pf_filter3,box_3,cloud,labels3,remaining_labels=pf_f.Passthrough_custom(L=L_3,
                                                    l=l_3,
                                                    h=h_3,
                                                    angle_x=angle_x_3,
                                                    angle_y=angle_y_3,
                                                    angle_z=angle_z_3,
                                                    centroid_x=centroid_x_3,
                                                    centroid_y=centroid_y_3,
                                                    centroid_z=centroid_z_3,
                                                    color_box=color_box_3,
                                                    color_pass_cloud=color_pass_cloud_3,
                                                    cloud=cloud,
                                                    labels=remaining_labels,
                                                    label_nr=label_nr_3)

                cloud.paint_uniform_color([0.5, 0.5, 0.5])

                labels[labels1.T[1,:],0]=label_nr_1
                labels[labels2.T[1,:],0]=label_nr_2
                labels[labels3.T[1,:],0]=label_nr_3

                
                labels_final=labels[:,0]

                with open(total_path_npy, 'wb') as f_npy:
                    np.save(f_npy, labels_final)

                #o3d.visualization.draw_geometries([cloud,pf_filter1,pf_filter2,pf_filter3])

                pcd_save.normals =  o3d.utility.Vector3dVector(pointcloud.normal)
                o3d.io.write_point_cloud(str(total_path), pcd_save, write_ascii=True)

                #o3d.io.write_point_cloud(str(total_path), result, write_ascii=True)
        return pointcloud

def process_dataset(root, save_path, transform=None, num_points=512):
    dataset_train = PcdDataset(root, folder="train", transform=transform, save_path=save_path, points=num_points)
    dataset_test =  PcdDataset(root, folder="test", transform=transform, save_path=save_path, points=num_points)

    print("Processing train data: ")
    for i in tqdm(range(len(dataset_train))):
        _ = dataset_train[i]

    print("Processing test data: ")
    for i in tqdm(range(len(dataset_test))):
        _ = dataset_test[i]

if __name__ == '__main__':
    ########################3
    ######New tests

    mu=0
    sigma=0.0

    rot_x=1
    rot_y=1
    rot_z=1

    ceva=4

    radius=0.25
    percentage=0.25

    random_rotate = Compose([
    RandomRotate(degrees=rot_x*ceva*10, axis=0),
    RandomRotate(degrees=rot_y*ceva*10, axis=1),
    RandomRotate(degrees=rot_z*ceva*10, axis=2),
    ])

    test_transform = Compose([
                    #random_rotate,
                    #GaussianNoiseTransform(mu=mu,sigma=sigma)
                    #Sphere_Occlusion_Transform(radius=radius, percentage=percentage,num_points=1024)
                    ])

    ##################################################3
   
    print("Type value for selected choice---------1 for processing dataset-----all else for loading dataset")
    print("Choice=")
    choice=int(input())

    num_points = 400
    curr_dir = (Path(__file__).parent).resolve()
    root = Path("/home/alex/Alex_documents/RGCNN_git/data/dataset/")

    root_noise_1 = Path("/home/alex/Alex_documents/RGCNN_git/data/dataset2/")

    if(choice==1):
    ####Processing the datasets
         process_dataset(root=root, save_path=root_noise_1,  num_points=num_points,transform=test_transform)

    else:
    ##################################################################333333333

    ##Loading the processed dataset

        num_points_original=400
        num_points_noise=400

        
        train_dataset = PcdDataset(root,valid=False,points=num_points_original)
        train_dataset_noise_1 = PcdDataset(root_noise_1,valid=False,points=num_points_noise)

        pcd_test_path="/home/alex/Alex_documents/RGCNN_git/Git_folder/data/plane/1651654247541301.pcd"
        pcd_test=o3d.io.read_point_cloud(pcd_test_path)

        for i in range(0,200):
        
            pcd_sampled = o3d.geometry.PointCloud()
            pcd_noise_1 = o3d.geometry.PointCloud()
           
            print("PCD sampled")
            pcd_sampled.points=o3d.utility.Vector3dVector(train_dataset[i].pos)
            #pcd_sampled.normals=o3d.utility.Vector3dVector(train_dataset[i].normal)

            pcd_sampled.paint_uniform_color([0, 0, 1])

            # print("PCD noise")
            pcd_noise_1.points=o3d.utility.Vector3dVector(train_dataset_noise_1[i].pos)
            #pcd_noise.normals=o3d.utility.Vector3dVector(train_dataset_noise[i].normal)
            pcd_noise_1.paint_uniform_color([1, 0, 0])

            o3d.visualization.draw_geometries([pcd_test, pcd_noise_1])
            #o3d.visualization.draw_geometries([pcd_sampled])
            #o3d.visualization.draw_geometries([ pcd_noise_1])
  