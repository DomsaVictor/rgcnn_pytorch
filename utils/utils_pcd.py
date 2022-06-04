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
            o3d.visualization.draw_geometries(
                [self.source, self.target, self.result])
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
        base_pcd_name = [file for file in os.listdir(
            base_dir) if file.endswith('.pcd')]
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

                    # aligner.draw_pcds()

                    break
