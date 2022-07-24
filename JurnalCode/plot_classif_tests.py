import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
curr_dir = Path(__file__).parent

class Plotter():
    def __init__(self,  save_path, show=True):
        self.save_path = save_path
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        self.show = show
    
    def __plot_one(self, x, y, label, line_type="*"):
        plt.plot(x, y, line_type, label=label, markersize=10)
    
    def __plot_fig(self):
        plt.figure()
        plt.title(self.title)
        for i in range(len(self.labels)):
            self.__plot_one(self.x_values, self.y_values[i], self.labels[i], line_type=self.line_types[i])
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.grid()
        plt.legend()
        plt.savefig(str(self.save_path)+f"/{self.title}.png")
        if self.show:
            plt.show()

    def __load_txt(self, path):
        with open(path, 'r') as f:
            data = np.loadtxt(f)
        return data
    
    def plot_from_file(self, path, x_values, labels, x_label, y_label, title, line_types=["-"]):
        self.x_values = x_values
        self.y_values = self.__load_txt(path)
        self.labels = labels
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.line_types = line_types
        self.__plot_fig()
        

def plot_all_data():
    txt_root_dir = str((curr_dir/"plot_txt").resolve())
    
    save_dir = str((curr_dir/"all_plots/classification").resolve())

    x_values = [[0,	    0.02,   0.05,     0.08,   0.1],
                [0,     10,     20,     30,     40],
                [0,     0.25],
                [0,	    0.02,   0.05,     0.08,   0.1],
                [0,     10,     20,     30,     40],
                [0,     10,     20,     30,     40],
                [0,     0.25]]
    
    titles = ["Additional Gaussian noise", "Rotation noise", "Occlusion noise",
              "Noisy model on dataset with additional Gaussian noise",
              "Noisy model on dataset with rotation noise",
              "Noisy model on dataset with rotation noise 2",
              "Noisy model on dataset with occlusion noise",]
    
    labels = ['RGCNN', 'RGCNN bounding box', 'RGCNN gram matrix', 'RGCNN multiview bounding box', 'RGCNN multiview eig']
    
    x_labels = ["Std deviation noise", "Random rotation degree limit", "Occlusion radius",
                "Std deviation noise", "Random rotation degree limit", "Random rotation degree limit", "Occlusion radius"]
    
    y_label = "Accuracy"
    
    plotter = Plotter(save_path=save_dir, show=False)
    line_types = ["o--", "X--", "P--", "*--", "^--"]
    for i, title in enumerate(titles):
        file = txt_root_dir + "/" + title + ".txt"
        plotter.plot_from_file(file, x_values[i], labels, x_labels[i], y_label, title, line_types=line_types)

def test():
    x_values = [0,	0.02,	0.05,	0.08,	0.1]
    
    labels = ['RGCNN', 'RGCNN bounding box', 'RGCNN gram matrix', 'RGCNN multiview bounding box lap recomp', 'RGCNN multiview eig lap recomp']
    title = "Test"
    x_label = "Gaussian Noise Sigma"
    y_label = "Accuracy"
    
    file_path = str((curr_dir/"plot_txt/test.txt").resolve())
    print(file_path)
    plotter = Plotter(save_path=(curr_dir/"all_plots").resolve())
    plotter.plot_from_file(file_path, x_values, labels, x_label, y_label, title)
    
if __name__ == "__main__":
    
    plot_all_data()
    