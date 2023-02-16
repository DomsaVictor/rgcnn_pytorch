import torch
import torch_geometric
import imports 
from torch_geometric.loader import DenseDataLoader
from torch_geometric.datasets import ShapeNet
import numpy as np
from collections import defaultdict
from utils import BoundingBoxRotate, label_to_cat
from utils import seg_classes
from multiprocessing import Process
from torch_geometric.transforms import FixedPoints, Compose, NormalizeScale, NormalizeRotation, RandomRotate
from RGCNNSegmentation import seg_model
from FilteredShapenetDataset import FilteredShapeNet, ShapeNetCustom
from pathlib import Path




class ModelTester():
    def __init__(self, model_list: list, dataset_list: list, is_classification=False) -> None:
        self.model_list = model_list
        self.dataset_list = dataset_list
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("Warning: No CUDA device found. Running tests on CPU!")
        
        self.results = []
        for i, model in enumerate(self.model_list):
            self.results.append([])
            for j, dataset in enumerate(self.dataset_list):
                self.results[i].append(None)
        
        self.add_cat = True
        if is_classification:
            self.add_cat = False
        
    def testAll(self):
        for j, dataset in enumerate(self.dataset_list):
            loader = DenseDataLoader(dataset=dataset, batch_size=16, pin_memory=True,
                                     num_workers=16, shuffle=True)
            for model in self.model_list:
                self.testOne(model, loader)
                
    
    def testOne(self, model, loader):
        model = model.to(self.device)
        size = len(loader.dataset)
        predictions = np.empty((size, 2048))
        labels = np.empty((size, 2048))
        total_correct = 0

        for i, data in enumerate(loader):
            data = data.to(self.device)
            cat = None
            if self.add_cat:
                cat = data.category
            
            x = torch.cat([data.pos.type(torch.float32),
                           data.x.type(torch.float32)], dim=2)
            y = (data.y).type(torch.LongTensor)
            
            logits, _, _ = model(x, cat)
            
            pred = logits.argmax(dim=2)
            
            pred = pred.to('cpu')
            
            total_correct += int((pred == y).sum())
            start = i * loader.batch_size
            stop = start + loader.batch_size
            predictions[start:stop] = pred
            
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
        accuracy = ncorrects * 100 / (len(loader.dataset) * model.vertice)
        print(f"{model.__class__.__name__} - {accuracy}% - {ncorrects}")
        for key, value in cat_iou.items():
            print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
        # return accuracy, cat_iou, tot_iou, ncorrects

            
if __name__ == "__main__":    
    num_points = 2048
    input_dim  = 22 
    
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 50]
    model_name = "final_models/2048_seg_clean.pt"
    
    transforms = Compose([FixedPoints(2048), NormalizeScale()])
    dataset = ShapeNetCustom(root_dir=Path("/home/domsa/workspace/git/rgcnn_pytorch/dataset/Journal/ShapeNetCustom/Original_2048"), folder="test", transform=transforms)
    model = seg_model(num_points, F, K, M, input_dim, reg_prior=False)
    model.load_state_dict(torch.load(f"{imports.curr_path}/{model_name}"))

    mt = ModelTester([model], [dataset])
    
    mt.testAll()