from distutils.log import warn
import imports
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
from pathlib import Path
import sys

def prepare_data_rot(raw_data):
    to_plot_data = np.empty((len(raw_data), len(raw_data[0])))
    for i in range(len(raw_data)):
        for j, acc in enumerate(raw_data[i]):
            to_plot_data[i, j] = acc
    return to_plot_data

def plot_values(data, model_names, x_values=[0, 10, 20, 30, 40], line_types=None, title="ShapeNet with random rotations", x_label="Rotations in degrees", y_label="Accuracy", show=True, folder_name=""):
    plt.figure()
    # plt.title(title)
    for i in range(len(model_names)):
        if model_names[i] is not None:
            plt.plot(x_values, data[i], line_types[i], label=model_names[i], markersize=10)
            with open(str((imports.curr_path/"results.txt").resolve()), "a") as f:
                text = f"{title} \n\t{model_names[i]} \n\t\t{str(x_values)} {str(data[i])}\n\n"
                print(text)
                f.write(text)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    if show == True:
        plt.show()
    else:
        path = Path(imports.curr_path)/"Plots/"/folder_name
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(str(path.resolve())+f"/{title}.png")
# for_model_acc = np.reshape(for_model_acc,(len(dataset_names), len(for_model_acc)))
# print(for_model_acc[0])

def open_file(file_name):
    with open(f"/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/{file_name}", "rb") as fp:
        data = pickle.load(fp)
    return data

def load_data(noise:str):
    acc, cat_iou, tot_iou = None, None, None
    if noise == "Raw":
        file_name = "raw"
        acc = open_file(f"{file_name}_acc")
        cat_iou = open_file(f"{file_name}_cat_iou")
        tot_iou = open_file(f"{file_name}_tot_iou")
    elif noise == "Rotation":
        file_name = "for_model"
        acc = open_file(f"{file_name}_acc")
        cat_iou = open_file(f"{file_name}_cat_iou")
        tot_iou = open_file(f"{file_name}_tot_iou")
    elif noise == "Gaussian Original":
        file_name = "gaussian_original"
        acc = open_file(f"{file_name}_acc")
        cat_iou = open_file(f"{file_name}_cat_iou")
        tot_iou = open_file(f"{file_name}_tot_iou")
    elif noise == "Gaussian Recomputed":
        file_name = "gaussian_recomputed"
        acc = open_file(f"{file_name}_acc")
        cat_iou = open_file(f"{file_name}_cat_iou")
        tot_iou = open_file(f"{file_name}_tot_iou")
    elif noise == "Occlusion":
        file_name = "occlusion"
        acc = open_file(f"{file_name}_acc")
        cat_iou = open_file(f"{file_name}_cat_iou")
        tot_iou = open_file(f"{file_name}_tot_iou")
    elif noise == "Gaussian and Rotations":
        file_name = "gaussian_recomputed_rotations"
        acc = open_file(f"{file_name}_acc")
        cat_iou = open_file(f"{file_name}_cat_iou")
        tot_iou = open_file(f"{file_name}_tot_iou")
    else:
        file_name = noise
        acc = open_file(f"{file_name}_acc")
        cat_iou = open_file(f"{file_name}_cat_iou")
        tot_iou = open_file(f"{file_name}_tot_iou")
    return acc, cat_iou, tot_iou

def get_gaussian_data():
    datasets = ["final_all_models_ordered_gaussian"]
    x_values = [[0, 0.01, 0.02, 0.05]]
    x_labels = ["Sigma"]
    titles   = ["Gaussian Recomputed Normals"]
    return datasets, x_values, x_labels, titles

def get_gaussian_original_data():
    datasets = ["final_all_models_ordered_gaussian_original"]
    x_values = [[0, 0.01, 0.02, 0.05]]
    x_labels = ["Sigma"]
    titles   = ["Gaussian Original Normals"]
    return datasets, x_values, x_labels, titles

def get_rotated_data():
    datasets = ["final_all_models_ordered_random_rotate"]
    x_values = [[0, 10, 20, 30, 40]]
    x_labels = ["Sigma"]
    titles   = ["Random Rotated"]
    return datasets, x_values, x_labels, titles

def get_occlusion_data():
    datasets = ["final_all_models_ordered_occlusion"]
    x_values = [[0, 0.1, 0.15, 0.2]]
    x_labels = ["Radius"]
    titles   = ["Occlusion"]
    return datasets, x_values, x_labels, titles

def get_real_approx_data():
    datasets = ["final_all_models_ordered_gaussian_random_rotations"]
    x_values = [[0, 0.01, 0.02, 0.05]]
    x_labels = ["Sigma"]
    titles   = ["Gaussian and Rotations"]
    return datasets, x_values, x_labels, titles

def plot_and_save_all():
    
    # datasets    = ["Rotation", "Gaussian Original", "Gaussian Recomputed", "Occlusion", "Gaussian and Rotations"]
    # datasets    = ["gaussian", "Rotation"]
    # datasets = ["final_all_models_ordered_gaussian"]
    # model_names = ["RGCNN", "RGCNN Bounding Box", "RGCNN Rotations and Bounding Box", "RGCNN Eig", "RGCNN Gaussian RR BB", "RGCNN Gaussian RR Eig"]
    # model_names = ["RGCNN", "RGCNN Bounding Box", "RGCNN Rotations and Bounding Box", "RGCNN Eig"]
    # model_names = ["RGCNN gauss", "RGCNN gauss rr bb", "RGCNN gauss rr eig"]
    # model_names = ["RGCNN", "RGCNN Bounding Box", "RGCNN Eig "]
    
    
    # x_values    = [[0, 10, 20, 30 ,40], [0, 0.01, 0.02, 0.05], [0, 0.01, 0.02, 0.05],
    #                [0, 0.1, 0.15, 0.2], [0, 0.01, 0.02, 0.05]]
    
    # x_values    = [[0, 10, 20, 30 ,40], [0, 0.01, 0.02, 0.05], [0, 0.01, 0.02, 0.05],
    #                [0, 0.1, 0.15, 0.2], [0, 0.01, 0.02, 0.05]]
    # x_values    = [[0, 0.01, 0.02, 0.05]]
    
    
    
    # x_labels    = ["Degrees", "Sigma", "Sigma", "Radius", "Sigma"]
    # x_labels    = ["Sigma"]
    
    datasets, x_values, x_labels, titles = get_gaussian_data()
    # datasets, x_values, x_labels, titles = get_gaussian_original_data()
    # datasets, x_values, x_labels, titles = get_rotated_data()
    # datasets, x_values, x_labels, titles = get_occlusion_data()
    datasets, x_values, x_labels, titles = get_real_approx_data()

    
    # titles      = ["Random Rotations", "Gaussian Noise with original normals", "Gaussian Noise with recomputed normals",
    #                "Occlusion Noise", "Gaussian and Rotations"]
    model_names = ["Original", "Bounding Box", "Eig", "RR and Bounding Box", "Gaussian RR and Bounding Box", "Gaussian RR and Eig"]
    # model_names = ["Original", "Bounding Box", "Eig", None, None, None,]
    model_names = ["Original", None, None, None, "Gaussian RR and Bounding Box", "Gaussian RR and Eig"]

    # model_names = ["Original", "Bounding Box", "Eig", None, "Gaussian RR and BB", "Gaussian RR and Eig"]

    
    # folder_name = input("Enter folder name: ")
    folder_name = "final/segmentation/enhanced_models"
    line_types  = ["o--", "X--", "P--", "*--", "^--", "v--"]
    y_label     = "mIoU"

    for i, dataset in enumerate(datasets):
        acc, cat_iou, tot_iou = load_data(dataset)
        x_value = x_values[i]
        title = titles[i]
        x_label = x_labels[i]
        data = prepare_data_rot(tot_iou)
        plot_values(data, model_names, x_value, title=title, y_label=y_label, x_label=x_label, show=False, folder_name=folder_name, line_types=line_types)
        

if __name__ == "__main__":
    model_names = ["RGCNN", "RGCNN Bounding Box", "RGCNN Rotations and Bounding Box", "RGCNN Eig", "RGCNN Gaussian RR BB"]

    print("Choose what to plot:")
    print("\t0. Raw Dataset")
    print("\t1. Rotation  Noise")
    print("\t2. Gaussian  Noise (original)")
    print("\t3. Gaussian  Noise (recomputed)")
    print("\t4. Occlusion Noise")
    print("\t5. Gaussian and Rotations")
    print("\t6. Save all plots")
    # noise_type = input("Waiting: ")
    noise_type = "6"
    if noise_type == "6":
        plot_and_save_all()
        sys.exit(0)
    print("Choose metric:")
    print("\t1. Accuracy")
    print("\t2. mIoU")
    metric = input("Waiting: ")
    if noise_type == "0":
        acc, cat_iou, tot_iou = load_data("Raw")
        raw_data = None
        y_data = None
        if metric == "1":
            raw_data = acc
            y_label = "Accuracy"
        else:
            raw_data = tot_iou
            y_label = "mIoU"
        data = prepare_data_rot(raw_data)
        plot_values(data, model_names, x_values=[0], line_type="x" , x_label="", y_label=y_label, title="Raw Dataset")
    elif noise_type == "1":
        acc, cat_iou, tot_iou = load_data("Rotation")
        raw_data = None
        y_data = None
        if metric == "1":
            raw_data = acc
            y_label = "Accuracy"
        else:
            raw_data = tot_iou
            y_label = "mIoU"
        data = prepare_data_rot(raw_data)
        plot_values(data, model_names,  x_values=[0, 10, 20, 30, 40], y_label=y_label)
    elif noise_type == "2":
        acc, cat_iou, tot_iou = load_data("Gaussian Original")
        x_label = "Sigma"
        if metric == "1":
            raw_data = acc
            y_label = "Accuracy"
        else:
            raw_data = tot_iou
            y_label = "mIoU"
        data = prepare_data_rot(raw_data)
        plot_values(data, x_values=[0, 0.01, 0.02, 0.05], model_names=model_names, x_label=x_label, y_label=y_label, title="Gaussian Noise with Original Normals")
    elif noise_type == "3":
        acc, cat_iou, tot_iou = load_data("Gaussian Recomputed")
        x_label = "Sigma"

        if metric == "1":
            raw_data = acc
            y_label = "Accuracy"
        else:
            raw_data = tot_iou
            y_label = "mIoU"
        data = prepare_data_rot(raw_data)
        plot_values(data, x_values=[0, 0.01, 0.02, 0.05], model_names=model_names, x_label=x_label, y_label=y_label, title="Gaussian Noise with Recomputed Normals")
    elif noise_type == "4":
        acc, cat_iou, tot_iou = load_data("Occlusion")
        x_label = "Radius"
        if metric == "1":
            raw_data = acc
            y_label = "Accuracy"
        else:
            raw_data = tot_iou
            y_label = "mIoU"
        data = prepare_data_rot(raw_data)
        data = np.array(data)
        data[:, [1, 2]] = data[:, [2, 1]]
        plot_values(data, x_values=[0, 0.1, 0.15, 0.2], model_names=model_names, x_label=x_label, y_label=y_label, title="Occlusion Noise")
    elif noise_type == "5":
        acc, cat_iou, tot_iou = load_data("Gaussian and Rotations")
        x_label = "Sigma"
        if metric == "1":
            raw_data = acc
            y_label = "Accuracy"
        else:
            raw_data = tot_iou
            y_label = "mIoU"
        data = prepare_data_rot(raw_data)
        plot_values(data, x_values=[0, 0.01, 0.02, 0.05], model_names=model_names, x_label=x_label, y_label=y_label, title="Gaussian and Rotations")
    else:
        raise Exception("Not Implemented.")