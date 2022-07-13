import os
from turtle import Shape
from matplotlib import transforms
from matplotlib.colors import Normalize
import imports
from pathlib import Path
import open3d as o3d
import utils
from utils import BoundingBoxRotate
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import Compose, FixedPoints, NormalizeScale, RandomRotate
import copy
from FilteredShapenetDataset import FilteredShapeNet, ShapeNetCustom
import numpy as np
from tqdm import tqdm

def noise_transform(noise_levels):
    
    categories = sorted(["Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
                "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
                "Pistol", "Rocket", "Skateboard", "Table"])

    for level in noise_levels:
        pcds = []
        j=0
        k=0
        transforms = Compose([FixedPoints(2048), utils.GaussianNoiseTransform(0, level)])
        for i, category in enumerate(categories):
            dataset = ShapeNet(root=imports.dataset_path + "/ShapeNet", categories=category, transform=transforms)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dataset[0].pos))
            k += 1
            if i%4 == 0:
                j += 1
                k=0
            pcd = pcd.translate([k, j, 0])
            # pcd.paint_uniform_color([190 / 255, 95 / 255, 92 / 255])
            pcds.append(copy.deepcopy(pcd))
        o3d.visualization.draw_geometries(pcds)


def occlusion_transform(occlusion_levels):
    categories = sorted(["Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
                "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
                "Pistol", "Rocket", "Skateboard", "Table"])
        
    for level in occlusion_levels:
        pcds = []
        j=0
        k=0
        transforms = Compose([utils.Sphere_Occlusion_Transform(radius=level, num_points=2048)])
        for i, category in enumerate(categories):
            dataset = ShapeNet(root=imports.dataset_path + "/ShapeNet", categories=category, transform=transforms)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dataset[0].pos))
            k += 1
            if i%4 == 0:
                j += 1
                k=0
            pcd = pcd.translate([k, j, 0])
            # pcd.paint_uniform_color([190 / 255, 95 / 255, 92 / 255])
            pcds.append(copy.deepcopy(pcd))
        o3d.visualization.draw_geometries(pcds)


def save_dataset(root, transform, save_path, categories=None, split="train", include_normals=True):
        dataset = ShapeNet(root=root, categories=categories, split=split, include_normals=include_normals, transform=transform)
        
        all_categories = sorted(["Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
                "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
                "Pistol", "Rocket", "Skateboard", "Table"])
        
        if not os.path.isdir(str((save_path/split).resolve())):
            os.makedirs(str((save_path/split).resolve()))
        if categories is None:
            categories = range(16)
        for category in categories:
            if not os.path.isdir(str((save_path/split/all_categories[category]).resolve())):
                os.makedirs(str((save_path/split/all_categories[category]).resolve()))
        
        for i in tqdm(range(len(dataset))):
            name = f"{i}"
            data = dataset[i]
            lbl_name = f"{name}.npy"
            pcd_name = f"{name}.pcd"
            category_name = all_categories[int(data.category)]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
            pcd.normals = o3d.utility.Vector3dVector(data.x)
            
            o3d.io.write_point_cloud(str((save_path/split/category_name/pcd_name).resolve()), pcd)
            np.save(str((save_path/split/category_name/lbl_name).resolve()), data.y)
            
        # for i, data in enumerate(dataset):
        #     name = f"{i}"
        #     lbl_name = f"{name}.npy"
        #     pcd_name = f"{name}.pcd"
        #     category_name = all_categories[int(data.category)]
        #     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
        #     pcd.normals = o3d.utility.Vector3dVector(data.x)
            
        #     o3d.io.write_point_cloud(str((save_path/split/category_name/pcd_name).resolve()), pcd)
        #     np.save(str((save_path/split/category_name/lbl_name).resolve()), data.y)
            
            
            
def test():
    # noise_transform([0, 0.01, 0.02, 0.05, 0.1])
    # occlusion_transform([0.1,0.15,0.2])
    num_points = 2048
    splits = ['trainval', 'test']
    
    # transform = Compose([FixedPoints(2048)])
    # for split in splits:
    #     save_dataset(root=imports.dataset_path + "/ShapeNet", transform=transform,
    #             save_path=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Original_{num_points}/"), split=split)
    
    # dataset = ShapeNetCustom(root_dir=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Original_{num_points}/"), transform=None, folder="trainval")
    # data = dataset[0]
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    # colors = np.array([[1,0,0], [0,1,0], [0,0,1], [0.4, 0.3, 0.6]])
    # pcd.colors = o3d.utility.Vector3dVector(colors[data.y])
    
    # dataset = ShapeNetCustom(root_dir=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Original_{num_points}/"), transform=NormalizeScale(), folder="trainval")
    # data = dataset[5500]
    # print(data.y)
    # print(f"{min(data.y)} - {max(data.y)}")
    # pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    # colors = np.array([[1,0,0], [0,1,0], [0,0,1], [0.4, 0.3, 0.6]])
    # o3d.visualization.draw_geometries([pcd2])
    # print(data.category)
    
    # dataset = ShapeNet(root=imports.dataset_path + "/ShapeNet", categories='Car', transform=FixedPoints(2048))
    # data = dataset[0]
    # pcd3 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    # colors = np.array([[1,0,0], [0,1,0], [0,0,1], [0.4, 0.3, 0.6]])

    # print(data.y)
    # pcd3.colors = o3d.utility.Vector3dVector(colors[data.y])
    # o3d.visualization.draw_geometries([pcd, pcd2, pcd3])

    # print(data.category)
    
    sigma_levels = [0.1, 0.15, 0.2]
    for sigma in sigma_levels:
        transform = Compose([utils.Sphere_Occlusion_Transform(sigma, num_points=2048)])
        for split in splits:
            save_dataset(root=imports.dataset_path + "/ShapeNet", transform=transform, 
                    save_path=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Occlusion_{num_points}_{sigma}/"), split=split)

    # sigma_levels = [0.01, 0.02, 0.05]
    # splits = ['trainval', 'test']
    # for sigma in sigma_levels:
    #     transform = Compose([FixedPoints(2048), utils.GaussianNoiseTransform(mu=0, sigma=sigma, recompute_normals=True)])
    #     for split in splits:
    #         save_dataset(root=imports.dataset_path + "/ShapeNet", transform=transform, 
    #                 save_path=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Gaussian_Recomputed_{num_points}_{sigma}/"), split=split)


    # sigma_levels = [0.01, 0.02, 0.05]
    # splits = ['trainval', 'test']
    # for sigma in sigma_levels:
    #     transform = Compose([FixedPoints(2048), utils.GaussianNoiseTransform(mu=0, sigma=sigma, recompute_normals=False)])
    #     for split in splits:
    #         save_dataset(root=imports.dataset_path + "/ShapeNet", transform=transform, 
    #                 save_path=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Gaussian_Original_{num_points}_{sigma}/"), split=split)
    # dataset = FilteredShapeNet(Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Gaussian_{num_points}_{sigma}/"), folder="train")
    # colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,1]])
    # data = dataset[0]
    # print(data)
    # print(data.pos.shape)
    # print(data.y.shape)
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    # pcd.colors = o3d.utility.Vector3dVector(colors[data.y - min(data.y)])
    # o3d.visualization.draw_geometries([pcd])

def create_rotated_dataset(rotation_levels):
    num_points = 2048
    splits = ["trainval", "test"]
    for sigma in rotation_levels:
        transform = Compose([
            FixedPoints(num_points),
            RandomRotate(sigma, 0),
            RandomRotate(sigma, 1),
            RandomRotate(sigma, 2)
        ])

        for split in splits:
            print(f"Creating dataset for: {split} - {sigma}")
            save_dataset(root=imports.dataset_path + "/ShapeNet", transform=transform, 
                    save_path=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/RandomRotated_{num_points}_{sigma}/"), split=split)



def test_rotation():
    
    root = Path(imports.dataset_path) / "ShapeNet"
    transforms = Compose([FixedPoints(2048), BoundingBoxRotate()])
    transforms_rot = Compose([
        FixedPoints(2048),
        RandomRotate(185, axis=0),
        RandomRotate(185, axis=1),
        RandomRotate(185, axis=2),
        BoundingBoxRotate()])
    dataset = ShapeNet(root=root, categories=None, transform=transforms)
    
    dataset_normal = ShapeNet(root=root, categories=None, transform=transforms_rot)
    colors = np.array([[1,0,0], [0,1,0], [0,0,1], [0.4, 0.3, 0.6]])

    data = dataset_normal[5]
    pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    pcd2.normals = o3d.utility.Vector3dVector(data.x)
    # pcd2.colors = o3d.utility.Vector3dVector(colors[data.y - min(data.y)])
    pcd2.paint_uniform_color([0.4, 0.3, 0.6])
    pcd2 = pcd2.translate([1,0,0])
    
    colors = np.array([[1,0,0], [0,1,0], [0,0,1], [0.4, 0.3, 0.6]])

    data = dataset[5]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    pcd.normals = o3d.utility.Vector3dVector(data.x)
    pcd.colors = o3d.utility.Vector3dVector(colors[data.y - min(data.y)])

    o3d.visualization.draw_geometries([pcd, pcd2])
    # o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    occlusion_transform([0.2])
    # test()

    # test_rotation()

    # rotation_levels = [10, 20, 30, 40]
    # create_rotated_dataset(rotation_levels)