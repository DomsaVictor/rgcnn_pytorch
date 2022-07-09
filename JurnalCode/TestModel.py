from ast import Num
import imports
import torch

from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DenseDataLoader
from torch_geometric.transforms import FixedPoints
from FilteredShapenetDataset import FilteredShapeNet, ShapeNetCustom
from RGCNNSegmentation import seg_model

import numpy as np

import copy
from pathlib import Path
from collections import defaultdict

from utils import label_to_cat
from utils import seg_classes

class ModelTester():
    def __init__(self, model, dataset_path, transforms=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.path = dataset_path
        self.transforms = transforms

        if not type(self.path) == Path:
            self.path = Path(self.path)
        self.dataset = ShapeNetCustom(root_dir=self.path, folder="test", transform=transforms)
        # self.dataset = ShapeNet(root=f"{imports.dataset_path}/ShapeNet",split="test", transform=FixedPoints(2048))
        print(self.dataset)
        self.loader = DenseDataLoader(dataset=self.dataset, batch_size=16, pin_memory=True, num_workers=8)
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
            logits = logits.to('cpu')
            pred = logits.argmax(dim=2)

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


if __name__ == '__main__':
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
