import os
import copy
import open3d as o3d
import Passthrough_function as pf_f
from torch_geometric.transforms import RandomRotate
import numpy as np

from tqdm import tqdm
import torch
# from torch.utils.data import DataLoader
from torchvision import transforms
from torch_geometric.data.dataset import Dataset
from pathlib import Path
from torch_geometric.nn import fps
import torch_geometric.transforms
import sys
from torch_geometric.transforms import Compose

utils_path = str((Path(__file__).parent / "../utils").resolve())
sys.path.append(utils_path)
# from utils import Sphere_Occlusion_Transform

# import probreg_functions as probreg_f


def rotate_pcd(pcd, angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 0:
        """rot on x"""
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:
        """rot on y"""
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 2:
        """rot on z"""
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return pcd.rotate(R)


def default_transforms():
    return transforms.Compose([
        # transforms.PointSampler(512),
        # transforms.Normalize(),
        # transforms.ToTensor()
    ])


class PcdDataset(Dataset):
    def __init__(self, root_dir, points=512, valid=False, folder="train", transform=default_transforms(), save_path=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(
            root_dir)) if os.path.isdir(root_dir/dir)]
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
                if not os.path.isdir(save_dir):
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
        if self.save_path is not None:

            pcd.estimate_normals(fast_normal_computation=False)
            # pcd.normalize_normals()
            # pcd.orient_normals_consistent_tangent_plane(100)

            normals = np.asarray(pcd.normals)

            if len(points) < self.points:
                alpha = 0.03
                rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha)

                num_points_sample = self.points

                pcd_sampled = rec_mesh.sample_points_poisson_disk(
                    num_points_sample)

                points = pcd_sampled.points
                points = torch.tensor(points)
                points = points.float()
                normals = np.asarray(pcd_sampled.normals)
            else:
                nr_points_fps = self.points
                nr_points = points.shape[0]

                index_fps = fps(points, ratio=float(
                    nr_points_fps/nr_points), random_start=True)

                index_fps = index_fps[0:nr_points_fps]

                fps_points = points[index_fps]
                fps_normals = normals[index_fps]

                points = fps_points
                normals = fps_normals
        else:
            normals = np.asarray(pcd.normals)

        normals = torch.Tensor(normals)
        normals = normals.float()

        pointcloud = torch_geometric.data.Data(
            normal=normals, pos=points, y=self.classes[self.files[idx]['category']])

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        return pointcloud

    def __getitem__(self, idx):

        pcd_path = self.files[idx]['pcd_path']

        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f.name.strip(), idx)
            if self.save_path is not None:
                splits = f.name.strip().split("/")
                cat = splits[len(splits) - 3]

                new_splits = f.name.strip().split(cat)
                base_pcd_dir = new_splits[0]+cat

                base_pcd_dir = Path(base_pcd_dir)

                base_pcd_name = [file for file in os.listdir(
                    base_pcd_dir) if file.endswith('.pcd')]
                base_pcd_name = base_pcd_name[0]

                with open(base_pcd_dir/base_pcd_name, 'r') as f:
                    target = o3d.io.read_point_cloud(f.name)

                name = splits[len(splits) - 1]

                name_npy = name.replace('pcd', 'npy')

                total_path = self.save_path/cat/self.folder/name
                total_path_npy = self.save_path/cat/self.folder/name_npy

                pcd_save = o3d.geometry.PointCloud()
                pcd_save.points = o3d.utility.Vector3dVector(pointcloud.pos)
                pcd_save.normals = o3d.utility.Vector3dVector(
                    pointcloud.normal)

                target.paint_uniform_color([0.5, 0.5, 0.5])
                source_pcd = copy.deepcopy(pcd_save)

                aabb_target = target.get_oriented_bounding_box()
                aabb_source = source_pcd.get_oriented_bounding_box()

                centroid_target = o3d.geometry.PointCloud.get_center(target)
                centroid_source = o3d.geometry.PointCloud.get_center(
                    source_pcd)

                source_pcd = source_pcd.rotate(aabb_source.R.T)

                source_pcd.translate(-centroid_source)

                dists_1 = target.compute_point_cloud_distance(source_pcd)
                dists_1 = np.sum(dists_1)

                source_pcd = rotate_pcd(source_pcd, np.pi, 2)

                dists_2 = target.compute_point_cloud_distance(source_pcd)

                dists_2 = np.sum(dists_2)

                if(dists_1 < dists_2):
                    source_pcd = rotate_pcd(source_pcd, -np.pi, 2)

                source_pcd.translate(centroid_target)
                source_pcd = source_pcd.rotate(aabb_target.R)

                aabb_2 = source_pcd.get_oriented_bounding_box()
                aabb_2.color = (0, 0, 1)

                cloud = copy.deepcopy(source_pcd)

                centroid = o3d.geometry.PointCloud.get_center(cloud)

                cloud = cloud.translate(-centroid)

                labels = 2*np.ones((np.asarray(cloud.points).shape[0], 1))

                label_indices = range(np.asarray(cloud.points).shape[0])

                label_indices = np.expand_dims(label_indices, axis=1)

                labels = np.concatenate((labels, label_indices), axis=1)
                labels = np.int_(labels)

                L_1 = 25
                l_1 = 21
                h_1 = 15

                angle_x_1 = 0
                angle_y_1 = 0
                angle_z_1 = -69

                centroid_x_1 = 14
                centroid_y_1 = -16
                centroid_z_1 = 106

                color_box_1 = [0.2, 0.2, 0.5]
                color_pass_cloud_1 = [0., 1., 1.]

                pf_filter1 = o3d.geometry.PointCloud()
                pf_filter2 = o3d.geometry.PointCloud()
                pf_filter3 = o3d.geometry.PointCloud()

                box_1 = o3d.geometry.PointCloud()
                box_2 = o3d.geometry.PointCloud()
                box_3 = o3d.geometry.PointCloud()

                label_nr_1 = 0

                pf_filter1, box_1, cloud, labels1, remaining_labels = pf_f.Passthrough_custom(L=L_1,
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

                # o3d.visualization.draw_geometries([cloud,box_1,pf_filter1])
                # o3d.visualization.draw_geometries([cloud])

                L_2 = 25
                l_2 = 21
                h_2 = 15

                angle_x_2 = 0
                angle_y_2 = 0
                angle_z_2 = -69

                centroid_x_2 = -12
                centroid_y_2 = -7
                centroid_z_2 = 112

                color_box_2 = [0.45, 0.1, 0.9]
                color_pass_cloud_2 = [1, 0.3, 0.5]

                label_nr_2 = 1

                pf_filter2, box_2, cloud, labels2, remaining_labels = pf_f.Passthrough_custom(L=L_2,
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

                # o3d.visualization.draw_geometries([cloud,box_2,pf_filter2])
                # o3d.visualization.draw_geometries([cloud])

                L_3 = 25
                l_3 = 21
                h_3 = 15

                angle_x_3 = -67
                angle_y_3 = -48
                angle_z_3 = 0

                centroid_x_3 = -5
                centroid_y_3 = -30
                centroid_z_3 = 124

                color_box_3 = [0.2, 0.2, 0.5]
                color_pass_cloud_3 = [0., 1., 1.]

                label_nr_3 = 0

                pf_filter3, box_3, cloud, labels3, remaining_labels = pf_f.Passthrough_custom(L=L_3,
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

                labels[labels1.T[1, :], 0] = label_nr_1
                labels[labels2.T[1, :], 0] = label_nr_2
                labels[labels3.T[1, :], 0] = label_nr_3

                labels_final = labels[:, 0]

                with open(total_path_npy, 'wb') as f_npy:
                    np.save(f_npy, labels_final)

                # o3d.visualization.draw_geometries([cloud,pf_filter1,pf_filter2,pf_filter3])

                #o3d.io.write_point_cloud(str(total_path), pcd_save, write_ascii=True)
                o3d.io.write_point_cloud(
                    str(total_path), source_pcd, write_ascii=True)

                #o3d.io.write_point_cloud(str(total_path), result, write_ascii=True)
        return pointcloud


def process_dataset(root, save_path, transform=None, num_points=512):
    dataset_train = PcdDataset(
        root, folder="train", transform=transform, save_path=save_path, points=num_points)
    dataset_test = PcdDataset(
        root, folder="test", transform=transform, save_path=save_path, points=num_points)
    print("Processing train data: ")
    for i in tqdm(range(len(dataset_train))):
        _ = dataset_train[i]

    print("Processing test data: ")
    for i in tqdm(range(len(dataset_test))):
        _ = dataset_test[i]


if __name__ == '__main__':
    # 3
    # New tests

    mu = 0
    sigma = 0.0

    rot_x = 1
    rot_y = 1
    rot_z = 1

    ceva = 4

    radius = 0.25
    percentage = 0.25

    random_rotate = Compose([
        RandomRotate(degrees=rot_x*ceva*10, axis=0),
        RandomRotate(degrees=rot_y*ceva*10, axis=1),
        RandomRotate(degrees=rot_z*ceva*10, axis=2),
    ])

    test_transform = Compose([
        # random_rotate,
        # GaussianNoiseTransform(mu=mu,sigma=sigma)
        #Sphere_Occlusion_Transform(radius=radius, percentage=percentage,num_points=1024)
    ])

    # 3

    num_points = 512
    root = Path("/home/victor/Desktop/Dataset_from_ROS/")

    root_noise_1 = Path("/home/victor/Desktop/Dataset_from_ROS_processed")

    # Processing the datasets
    process_dataset(root=root, save_path=root_noise_1,
                    num_points=num_points, transform=test_transform)
