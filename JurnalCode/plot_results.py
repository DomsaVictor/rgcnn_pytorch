from distutils.log import warn
import imports
from matplotlib import pyplot as plt
import numpy as np
import pickle

def prepare_data_rot(raw_data):
    print(len(raw_data))
    print(len(raw_data[0]))
    to_plot_data = np.empty((len(raw_data), len(raw_data[0])))
    for i in range(len(raw_data)):
        for j, acc in enumerate(raw_data[i]):
            to_plot_data[i, j] = acc
    return to_plot_data

def plot_values(data, model_names, x_values=[0, 10, 20, 30, 40], line_type="-", title="ShapeNet with random rotations", x_label="Rotations in degrees", y_label="Accuracy"):
    plt.figure()
    plt.title(title)
    for i in range(len(model_names)):
        plt.plot(x_values, data[i], line_type, label=model_names[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend()
    plt.show()
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
        raise Exception("Not Implemented.")
    return acc, cat_iou, tot_iou

if __name__ == "__main__":
    model_names = ["RGCNN", "RGCNN Bounding Box", "RGCNN Rotations and Bounding Box", "RGCNN Eig", "RGCNN Gaussian RR BB"]

    print("Choose what to plot:")
    print("\t0. Raw Dataset")
    print("\t1. Rotation  Noise")
    print("\t2. Gaussian  Noise (original)")
    print("\t3. Gaussian  Noise (recomputed)")
    print("\t4. Occlusion Noise")
    print("\t5. Gaussian and Rotations")
    noise_type = input("Waiting: ")
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
