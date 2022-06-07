import torch as t
import open3d as o3d
import torch_geometric
from torch_geometric.datasets import ShapeNet
from tkinter import *
from tkinter import ttk
from pathlib import Path
import numpy as np
import copy
import threading

class ModelChooser():
    def __init__(self, path:Path = Path(__file__).parent, start_index:int=0, save_file_name:str="ChosenPCDIndexes.txt", category:str="Airplane"):
        self.dataset = ShapeNet(str(path) + "/ShapeNet", categories=category, split="trainval")
        self.dataset_len = len(self.dataset)
        self.vis = o3d.visualization.Visualizer()
        self.save_file_name = save_file_name
        self.vis.create_window()
        self.i = start_index
        self.pcd = None 
        self.master = Tk()
        self.pcd_list = []
        self.index_list = []
        frame = ttk.Frame(self.master, padding=10)
        frame.grid()
        Button(frame, text="Yes", command=lambda:self.btn_callback(1)).grid(column=0, row=0)
        Button(frame, text="No", command=lambda:self.btn_callback(0)).grid(column=1, row=0)
        Button(frame, text="Back", command=lambda:self.btn_back_callback()).grid(column=0, row=1)
        Button(frame, text="Finish", command=lambda:self.btn_finish()).grid(column=1, row=1)
        self.update_pcd()
        self.vis.add_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.master.mainloop()
        
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
    
    def update_pcd(self):
        if(self.i <= self.dataset_len):
            self.pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.dataset[self.i].pos))
            self.pcd = self.rotate_pcd(self.pcd, axis=0, angle=45)
            self.pcd = self.rotate_pcd(self.pcd, axis=1, angle=-45)
            self.i += 1 
        else:
            self.btn_finish()
        
    def btn_callback(self, btn_input):
        if btn_input == 1:
            self.pcd_list.append(self.pcd)
            self.index_list.append(self.i)
        self.update_pcd()
        self.vis.clear_geometries()
        self.vis.add_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
           
    def btn_back_callback(self):
        self.pcd_list.pop()
        self.index_list.pop()
        self.i -= 1
        self.update_pcd()
        self.btn_callback(0)

        
    def btn_finish(self):
        print(self.index_list)
        file_path = (Path(__file__).parent / self.save_file_name).resolve()
        with open(str(file_path), 'w') as f:
            for index in self.index_list:
                f.write(str(index))
                f.write("\n")
        return
    
if __name__ == "__main__":
    # 693
    # 202
    ModelChooser(start_index=0, save_file_name="ChosenPCDIndexes_TEST.txt")