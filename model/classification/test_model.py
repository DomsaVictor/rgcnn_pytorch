# PyTorch, PyG and other imports
from torch_geometric.transforms import RandomScale
from torch_geometric.transforms import RandomRotate
from torch_geometric.transforms import FixedPoints, SamplePoints, NormalizeScale
from torch_geometric.loader import DenseDataLoader
from torch_geometric.datasets import ModelNet
import torch
from datetime import datetime
from collections import defaultdict
import time
import os
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import Compose


# Imports related to this repo
from pathlib import Path
cwd = Path.cwd()
curr_path = Path(__file__).parent
dataset_path    = (curr_path / "../../dataset/").resolve()
# model_path      = (curr_path / "../model/classification/").resolve()
utils_path      = (curr_path / "../../utils").resolve()

import sys
# sys.path.append(str(model_path))
sys.path.append(str(utils_path))
sys.path.append(str(dataset_path))


from utils import GaussianNoiseTransform
from FilteredShapenetDataset import FilteredShapeNet, PcdDatasetNoise
from RGCNNClassification import cls_model
from utils import compute_loss_with_weights
from utils import label_to_cat
from utils import seg_classes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(model, loader):
    model.eval()
    total_correct = 0
    for i, data in enumerate(loader):
            # x = torch.cat([data.pos.type(torch.float32), data.x.type(torch.float32)], dim=2)
            x = torch.cat([data.pos.type(torch.float32), data.normal.type(torch.float32)], dim=2)
            y = data.y.type(torch.LongTensor).squeeze()
            
            logits, _, _ = model(x.to(device))
            logits = logits.to('cpu')
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == y).sum())
    
    return total_correct / len(loader.dataset), total_correct

if __name__ == '__main__':
    num_points = 512
    batch_size = 8
    num_epochs = 200
    learning_rate = 1e-3
    modelnet_num = 36
    dropout = 0.2
    gamma = 0.8
    one_layer = False
    reg_prior = False
    recompute_L = False
    
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orde
    M = [512, 128, modelnet_num]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")


    model_path = str((curr_path / "512p_model_v2_200.pt").resolve())

    model = cls_model(num_points, F, K, M, input_dim=6, dropout=dropout, one_layer=one_layer, reg_prior=reg_prior, recompute_L=recompute_L)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    print(model)

    name_list = ["Modelnet40_512_r_10","Modelnet40_512_r_20","Modelnet40_512_r_30","Modelnet40_512_r_40"]
    
    transforms = Compose([SamplePoints(2048, include_normals=True), NormalizeScale()])

    dataset = ModelNet(root=str((dataset_path/"Modelnet").resolve()), name=str(40), train=False, transform=transforms)
    loader  = DenseDataLoader(dataset, batch_size=1, num_workers=8)
    test_acc, ncorrect = test(model, loader)



    for name in name_list:
        dataset = PcdDatasetNoise(root_dir=dataset_path/"Test_rotation_invariant"/name, folder="test")
        loader  = DenseDataLoader(dataset, batch_size=1, num_workers=8)
        test_acc, ncorrect = test(model, loader)
        print(f"{name} - acc:   {test_acc*100}%")
