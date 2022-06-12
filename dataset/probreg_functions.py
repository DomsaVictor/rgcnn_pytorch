import copy
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import probreg

class pcd_registration():
    def __init__(self) -> None:
        self.source = None
        self.target = None
        self.result = None

    def set_source(self, pcd):
        self.source = pcd

    def set_target(self, pcd):
        self.target = pcd

    def set_clouds(self, source, target):
        self.source = source
        self.target = target

    def __get_transform_matirx(self, rot, t):
        T = np.eye(4, 4)
        T[:3, :3] = rot
        T[:3, 3] = t.T
        return T

    def draw_pcds(self):
        self.source.paint_uniform_color([1, 0.706, 0])
        self.target.paint_uniform_color([0, 0.651, 0.929])
        if self.result:
            self.result.paint_uniform_color([1, 0.235, 0.722])
            # o3d.visualization.draw_geometries(
            #     [self.source, self.target, self.result])
            o3d.visualization.draw_geometries(
                [self.target, self.result])
        else:
            o3d.visualization.draw_geometries([self.source, self.target])

    def __get_transformed_matrix(self, T):
        source_temp = copy.deepcopy(self.source)
        # target_temp = copy.deepcopy(target)
        source_temp.transform(T)
        return source_temp

    def register_pcds(self, source=None, target=None):
        if source is not None:
            self.source = source

        if target is not None:
            self.target = target

        tf_param = probreg.filterreg.registration_filterreg(
            self.source, self.target)
        T = self.__get_transform_matirx(
            tf_param.transformation.rot, tf_param.transformation.t)
        self.result = self.__get_transformed_matrix(T)
        return self.result

def align_pcds(pcd, registrator):
    registrator.set_source(pcd)
    result = registrator.register_pcds()
    result.paint_uniform_color([1,0,0])
    dist = pcd.compute_point_cloud_distance(registrator.target)
    dist = np.mean(dist)
    print(dist)
    r.draw_pcds()
    # o3d.visualization.draw_geometries([result, registrator.target])
    # if (pcd.compute_point_cloud_distance())

def align_all_pcds(root, folder='train'):
    aligner = pcd_registration()
    root_dir = Path(root)
    folders = [dir for dir in sorted(os.listdir(
        root_dir)) if os.path.isdir(root_dir/dir)]
    classes = {folder: i for i, folder in enumerate(folders)}

    for category in classes.keys():
        new_dir = root_dir/Path(category)/folder
        base_dir = root_dir/Path(category)
        # Only get one file (hopefully)
        base_pcd_name = [file for file in os.listdir(base_dir) if file.endswith('.pcd')]
        base_pcd_name = base_pcd_name[0]
        with open(base_dir/base_pcd_name, 'r') as f:
            target = o3d.io.read_point_cloud(f.name)
            aligner.set_target(target)

        for file in os.listdir(new_dir):
            if file.endswith('.pcd'):
                # Here we realign the pointclouds and apply filters.
                with open(new_dir/file) as f:
                    source_pcd = o3d.io.read_point_cloud(f.name)
                    aligner.set_source(source_pcd)
                    result = aligner.register_pcds()
                    # Do the rest here!!!

                    #o3d.visualization.draw_geometries([result])

                    #aligner.draw_pcds()


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

if __name__ == "__main__":
    # root = Path("/home/alex/Alex_documents/RGCNN_git/Git_folder/data/")

    # align_all_pcds(root)
    pcd1 = o3d.io.read_point_cloud("/home/victor/Desktop/ros_pcd_with_normals.pcd")
    pcd2 = o3d.io.read_point_cloud("/home/victor/Desktop/Dataset_pcd.pcd")
    
    r = pcd_registration()
    r.set_target(pcd2)
    # o3d.visualization.draw_geometries([pcd1, pcd2])

    align_pcds(pcd1, r)
    pcd1 = rotate_pcd(pcd1, 0, 90)
    
    align_pcds(pcd1, r)