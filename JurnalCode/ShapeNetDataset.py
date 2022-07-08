from numpy import save
import imports
import torch
from torch_geometric.datasets import ShapeNet
from torch_geometric.data.dataset import Dataset
import open3d as o3d
import numpy as np

class ShapeNetCustom(Dataset):
    def __init__(self, root, transform, categories=None, split="train", include_normals=True, save_path=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.save_path = save_path
        self.dataset = ShapeNet(root, categories=categories, include_normals=include_normals, transform=transform)

    
            