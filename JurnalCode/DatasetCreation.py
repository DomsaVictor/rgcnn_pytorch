import os
from matplotlib import transforms
import imports
from pathlib import Path
import open3d as o3d
import utils
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import Compose, FixedPoints
import copy
from FilteredShapenetDataset import FilteredShapeNet
import numpy as np

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


def create_dataset(dataset, transfrom, num_points=2048):
    for _ in dataset:
        pass
    
def save_dataset(root, transform, save_path, categories=None, split="train", include_normals=True):
        dataset = ShapeNet(root=root, categories=categories, split=split, include_normals=include_normals, transform=transform)
        
        all_categories = sorted(["Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
                "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
                "Pistol", "Rocket", "Skateboard", "Table"])
        
        if not os.path.isdir(str((save_path/split).resolve())):
            os.makedirs(str((save_path/split)).resolve())
        if categories is None:
            categories = range(16)
        for category in categories:
            if not os.path.isdir(str((save_path/split/all_categories[category]).resolve())):
                os.makedirs(str((save_path/split/all_categories[category]).resolve()))
        for i, data in enumerate(dataset):
            name = f"{i}"
            lbl_name = f"{name}.npy"
            pcd_name = f"{name}.pcd"
            category_name = all_categories[int(data.category)]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
            pcd.normals = o3d.utility.Vector3dVector(data.x)
            
            o3d.io.write_point_cloud(str((save_path/split/category_name/pcd_name).resolve()), pcd)
            np.save(str((save_path/split/category_name/lbl_name).resolve()), data.y)
            
if __name__ == '__main__':
    # noise_transform([0, 0.01, 0.02, 0.05, 0.1])
    # occlusion_transform([0.1,0.15,0.2])
    num_points = 2048
    sigma = 0.05
    transform = Compose([FixedPoints(num_points), utils.GaussianNoiseTransform(0, sigma)])
    category = None
    save_dataset(root=imports.dataset_path + "/ShapeNet", transform=transform,
                 save_path=Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Gaussian_{num_points}_{sigma}/"), 
                 categories=category, split="train")
        
    dataset = FilteredShapeNet(Path(f"{imports.dataset_path}/Journal/ShapeNetCustom/Gaussian_{num_points}_{sigma}/"), folder="train")
    colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,1]])
    data = dataset[0]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.pos))
    pcd.colors = o3d.utility.Vector3dVector(colors[data.y])
    o3d.visualization.draw_geometries([pcd])