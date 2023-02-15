import open3d as o3d
import imports
from noise_visualization import get_noisy_pcd, show_pcds
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import NormalizeScale, Compose, FixedPoints, RandomRotate
import numpy as np
from utils import seg_classes


class VisualizerPCD():
    def __init__(self) -> None:
        self.pcds_o3d = []        
        self.colors = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1],[0.3, 0.4, 0.6]])
            
    def _prepare_pcds(self, pcds, y, row_nr=0):
        
        if not type(pcds) == list:
            pcds = [pcds]
            y = [y]
        
        pcds_o3d = [None] * len(pcds)
        for i, pcd in enumerate(pcds):
            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            pcd_o3d.colors = o3d.utility.Vector3dVector(self.colors[y[i]]) 
            pcd_o3d = pcd_o3d.translate([2.5 * i, row_nr*1.5, 0])
            pcds_o3d[i] = pcd_o3d
        
        return pcds_o3d
    
    
    def add_pcds(self, pcds, y, row_nr):
        self.pcds_o3d.append(self._prepare_pcds(pcds, y, row_nr=row_nr))
        
        
    def show(self):
        o3d.visualization.draw_geometries(self.pcds_o3d)


vis = VisualizerPCD()


# dataset_name = "Original_2048"
dataset = ShapeNet(root=f"{imports.dataset_path}/Journal/ShapeNet/", transform=FixedPoints(2048), categories=seg_classes)
pcd = dataset[0]
print(pcd.y - min(pcd.y))
vis.add_pcds(pcd.pos, pcd.y-min(pcd.y), 0)
