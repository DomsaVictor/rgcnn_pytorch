import imports
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open("/home/domsa/workspace/git/rgcnn_pytorch/for_model_acc", "rb") as fp:
    for_model_acc = pickle.load(fp)

model_names = ["2048_seg_clean.pt", "2048_seg_bb.pt", "2048_seg_rrbb.pt", "2048_seg_eig.pt"]
dataset_names = ["RandomRotated_2048_10"]

for i, name in enumerate(dataset_names):
    for data in for_model_acc[i]:
        plt.plot(data, label=model_names[i])
    plt.title(name)
    plt.show()

print(for_model_acc)
