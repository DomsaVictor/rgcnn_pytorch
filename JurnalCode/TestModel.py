from ast import Num
import time
import imports
import torch
import os
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DenseDataLoader
from torch_geometric.transforms import FixedPoints, Compose, NormalizeScale, NormalizeRotation, RandomRotate
from FilteredShapenetDataset import FilteredShapeNet, ShapeNetCustom
from RGCNNSegmentation import seg_model
import matplotlib.pyplot as plt
import numpy as np

import pickle

import copy
from pathlib import Path
from collections import defaultdict

from utils import BoundingBoxRotate, label_to_cat
from utils import seg_classes

class ModelTester():
    def __init__(self, model, dataset_path, transforms=NormalizeScale()):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.path = dataset_path
        self.transforms = transforms

        if not type(self.path) == Path:
            self.path = Path(self.path)
        self.dataset = ShapeNetCustom(root_dir=self.path, folder="test", transform=transforms)
        # print(self.dataset[0])
        # self.dataset = ShapeNet(root=f"{imports.dataset_path}/ShapeNet",split="test", transform=Compose([FixedPoints(2048), NormalizeScale()]))
        self.loader = DenseDataLoader(dataset=self.dataset, batch_size=32, pin_memory=True, num_workers=8, shuffle=True)
        self.all_categories = sorted(["Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
                "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
                "Pistol", "Rocket", "Skateboard", "Table"])

    def test_model(self):
        self.model.eval()
        size = len(self.loader.dataset)
        predictions = np.empty((size, self.model.vertice))
        labels = np.empty((size, self.model.vertice))
        total_correct = 0
        add_cat = True
        if (len(self.dataset.categories) == 1):
            add_cat = False
        
        for i, data in enumerate(self.loader):

            cat = None
            if add_cat:
                cat = data.category.to(self.device)

            x = torch.cat([data.pos.type(torch.float32),
                    data.x.type(torch.float32)], dim=2)
            y = (data.y).type(torch.LongTensor)
            logits, _, _ = self.model(x.to(self.device), cat.to(self.device))

            # print(logits.shape)
            # print(f"{min(logits)} - {max(logits)}")

            # logits = logits.to('cpu')

            pred = logits.argmax(dim=2)

            pred = pred.to('cpu')

            total_correct += int((pred == y).sum())
            start = i * self.loader.batch_size
            stop = start + self.loader.batch_size
            predictions[start:stop] = pred
            # lab = y
            # labels[start:stop] = lab.reshape([-1, num_points])
            labels[start:stop] = y

        tot_iou = []
        cat_iou = defaultdict(list)
        for i in range(predictions.shape[0]):
            segp = predictions[i, :]
            segl = labels[i, :]
            cat = label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]

            for l in seg_classes[cat]:
                # part is not present, no prediction as well
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                        np.sum((segl == l) | (segp == l)))
            cat_iou[cat].append(np.mean(part_ious))
            tot_iou.append(np.mean(part_ious))

        ncorrects = np.sum(predictions == labels)
        accuracy = ncorrects * 100 / (len(self.dataset) * self.model.vertice)

        return accuracy, cat_iou, tot_iou, ncorrects


def test_all_models(dataset_names:list, model_name="2048p_seg_all200.pt", transform=NormalizeScale()):
    num_points = 2048

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]

    input_dim = 22

    model = seg_model(num_points, F, K, M, input_dim, dropout=0.2, reg_prior=False)
    model.load_state_dict(torch.load(f"{imports.curr_path}/{model_name}"))
    model.eval()

    
    if not os.path.isdir(Path(imports.curr_path)/"results/"):
        os.makedirs(Path(imports.curr_path)/"results/")
    
    save_path = str((Path(imports.curr_path) / "results").resolve())
    all_tot_iou = []
    all_cat_iou = []
    all_acc = []
    for name in dataset_names:
        tester = ModelTester(model, f"{imports.dataset_path}/Journal/ShapeNetCustom/{name}", transforms=transform)

        acc, cat_iou, tot_iou, ncorrect = tester.test_model()
        print(f"\n!!!!!!! {name} !!!!!!!")
        print(f"Accuracy = {acc}")
        # for key, value in cat_iou.items():
        #         print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
        print(f"Tot IoU  = {np.mean(tot_iou)*100}")
        # print(f"Ncorrect = {ncorrect}")
        print("**"*20)
        if not os.path.isdir(Path(save_path)/model_name/name):
            os.makedirs(Path(save_path)/model_name/name)
        np.savetxt(f"{save_path}/{model_name}/{name}/acc.txt", np.expand_dims(acc, axis=0))
        file = open(f"{save_path}/{model_name}/{name}/cat_iou.txt", "w")
        file.write(str(cat_iou)) 
        file.close()
        # np.savetxt(f"{save_path}/{model_name}/{name}/cat_iou.txt",  cat_iou)
        np.savetxt(f"{save_path}/{model_name}/{name}/tot_iou.txt", np.expand_dims(np.mean(tot_iou)*100, axis=0))

        all_acc.append(acc)
        all_tot_iou.append(np.mean(tot_iou)*100)
        all_cat_iou.append(cat_iou)

    return all_acc, all_tot_iou, all_cat_iou

        # np.save(f"{save_path}/{name}/ncorrect.npy", ncorrect)
    # plt.plot(all_acc)
    # plt.plot(all_tot_iou)
    

def test():
    num_points = 2048

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]

    input_dim = 22

    model_name = "2048p_seg_all200.pt"

    model = seg_model(num_points, F, K, M, input_dim, dropout=0.2, reg_prior=False)
    model.load_state_dict(torch.load(f"{imports.curr_path}/{model_name}"))
    model.eval()

    dataset_name = "Original2_2048"

    tester = ModelTester(model, f"{imports.dataset_path}/Journal/ShapeNetCustom/{dataset_name}")

    acc, cat_iou, tot_iou, ncorrect = tester.test_model()

    print(f"Accuracy = {acc}")
    for key, value in cat_iou.items():
            print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
    print(f"Tot IoU  = {np.mean(tot_iou)*100}")
    print(f"Ncorrect = {ncorrect}")

if __name__ == '__main__':
    # dataset_names = [
    #     "Gaussian_Original_2048_0.01", "Gaussian_Original_2048_0.02",  "Gaussian_Original_2048_0.05",
    #     "Gaussian_Recomputed_2048_0.01", "Gaussian_Recomputed_2048_0.02", "Gaussian_Recomputed_2048_0.05",
    #     "Occlusion_2048_0.1", "Occlusion_2048_0.2", "Occlusion_2048_0.15",
    #     "RandomRotated_2048_10", "RandomRotated_2048_20", "RandomRotated_2048_30", "RandomRotated_2048_40"]
   
    # dataset_names = ["Original_2048", "RandomRotated_2048_10", "RandomRotated_2048_20", "RandomRotated_2048_30", "RandomRotated_2048_40"]
    # file_name = "for_model"

    # dataset_names = ["Original_2048", "Gaussian_Original_2048_0.01", "Gaussian_Original_2048_0.02",  "Gaussian_Original_2048_0.05"]
    # file_name = "gaussian_original"
    
    dataset_names = ["Original_2048", "Occlusion_2048_0.1", "Occlusion_2048_0.15", "Occlusion_2048_0.2"]
    file_name = "occlusion"
   
    # dataset_names = ["RandomRotated_2048_10"]
    # transforms = Compose([NormalizeScale(), BoundingBoxRotate()])
    # , model_name="2048_shapenet_bb.pt"
    # transforms = Compose([NormalizeScale(), BoundingBoxRotate()])

    # dataset_names = ["Original_2048", "Gaussian_Recomputed_2048_0.01", "Gaussian_Recomputed_2048_0.02", "Gaussian_Recomputed_2048_0.05"]
    # file_name = "gaussian_recomputed"

    # dataset_names = ["Original_2048"]
    # file_name = "raw"

    big_rotations = Compose([
                        RandomRotate(180, axis=0),
                        RandomRotate(180, axis=1),
                        RandomRotate(180, axis=2)])
    
    transforms = [Compose([NormalizeScale(), big_rotations]),
                Compose([NormalizeScale(), big_rotations, BoundingBoxRotate()]),
                Compose([NormalizeScale(), big_rotations, BoundingBoxRotate()]),
                Compose([NormalizeScale(), big_rotations, NormalizeRotation()])]

    model_names = ["2048_seg_clean.pt", "2048_seg_bb.pt", "2048_seg_rrbb.pt", "2048_seg_eig.pt", "2048_seg_gauss_rr_bb.pt"]

    transforms = [Compose([NormalizeScale()]),
                  Compose([NormalizeScale(), BoundingBoxRotate()]),
                  Compose([NormalizeScale(), BoundingBoxRotate()]),
                  Compose([NormalizeScale(), NormalizeRotation()]),
                  Compose([NormalizeScale(), BoundingBoxRotate()])]

    for_model_acc = []
    for_model_tot_iou = []
    for_model_cat_iou = []
    for i in range(len(model_names)):
        model = model_names[i]
        transform = transforms[i]
        all_acc, all_tot_iou, all_cat_iou = test_all_models(dataset_names=dataset_names, transform=transform, model_name=model)
        for_model_acc.append(all_acc)
        for_model_tot_iou.append(all_tot_iou)
        for_model_cat_iou.append(all_cat_iou)

    with open(f"{file_name}_acc", "wb") as fp:
        pickle.dump(for_model_acc, fp)
    with open(f"{file_name}_tot_iou", "wb") as fp:
        pickle.dump(for_model_tot_iou, fp)
    with open(f"{file_name}_cat_iou", "wb") as fp:
        pickle.dump(for_model_cat_iou, fp)

    # for i, name in enumerate(dataset_names):
    #     for data in for_model_acc[i]:
    #         plt.plot(data, label=model_names[i])
    #     plt.title(name)
    #     plt.show()

    # model_name = "2048_seg_rrbb.pt"
    # start_time = time.time()
    # test_all_models(dataset_names=dataset_names, transform=transforms, model_name=model_name)
    # end_time   = time.time()
    # print(f"Total Test Time: - {end_time - start_time}")