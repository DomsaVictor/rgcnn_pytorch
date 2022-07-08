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
dataset_path    = (curr_path / "../dataset/").resolve()
model_path      = (curr_path / "../model/classification/").resolve()
utils_path      = (curr_path / "../utils/").resolve()

import sys
sys.path.append(str(model_path))
sys.path.append(str(utils_path))
sys.path.append(str(dataset_path))


from utils import GaussianNoiseTransform
from FilteredShapenetDataset import FilteredShapeNet, PcdDataset
from RGCNNClassification import cls_model
from utils import compute_loss_with_weights
from utils import label_to_cat
from utils import seg_classes

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, optimizer, loader, regularization, criterion):
    model.train()
    total_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        
        # x = torch.cat([data.pos.type(torch.float32), data.x.type(torch.float32)], dim=2)
        x = torch.cat([data.pos.type(torch.float32), data.normal.type(torch.float32)], dim=2)

        y = data.y.type(torch.LongTensor)
        
        logits, out, L = model(x.to(device))
        
        loss = compute_loss_with_weights(logits, y, out, L, criterion=criterion, model=model,s=regularization)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i%100 == 0:
            print(f"{i}: curr loss: {loss}")
    return total_loss / len(loader)

@torch.no_grad()
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

def start_training(model, train_loader, test_loader, optimizer, criterion, writer, epochs=50, learning_rate=1e-3, regularization=1e-9, decay_rate=0.95):
    print(model)
    print(f"\nTraining on {device}")

    model.to(device)
    my_lr_scheduler = lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay_rate)

    for epoch in range(1, epochs+1):
        train_start_time = time.time()
        loss = train(model, optimizer, train_loader,
                     criterion=criterion, regularization=regularization)
        train_stop_time = time.time()

        writer.add_scalar('loss/train', loss, epoch)

        test_start_time = time.time()
        test_acc, ncorrects = test(model, test_loader)
        test_stop_time = time.time()

        writer.add_scalar("accuracy/test", test_acc, epoch)

        print(f'Epoch:    {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc*100:.4f}%')
        print(f'ncorrect: {ncorrects} / {len(test_loader.dataset)}')
        print(f'Train Time: \t{train_stop_time - train_start_time} \nTest Time: \t{test_stop_time - test_start_time }')
        print("~~~" * 30)

        my_lr_scheduler.step()

        # Save the model every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            if not os.path.isdir(str(model_path)):
                os.makedirs(str(model_path))
            torch.save(model.state_dict(), str(model_path) + '/' +
                       str(model.vertice) + 'p_model_v2_' + str(epoch) + '.pt')

    print(f"Training finished")


if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    model_path = (curr_path / f"models_cls/rotated/{directory}/").resolve()

    num_points = 1024
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-3
    modelnet_num = 40
    dropout = 0.2
    gamma = 0.9
    one_layer = False
    reg_prior = True
    recompute_L = False
    b2relu = False

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orde
    M = [512, 128, modelnet_num]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    transforms = Compose([SamplePoints(num_points, include_normals=True), NormalizeScale()])
    # transforms = Compose([SamplePoints(num_points, include_normals=True)])
    print(str((dataset_path/"Modelnet").resolve()))

    dataset_train = PcdDataset(root=str((dataset_path/"Modelnet").resolve()), name=str(modelnet_num), train=True, transform=transforms)
    dataset_test  = PcdDataset(root=str((dataset_path/"Modelnet").resolve()), name=str(modelnet_num), train=False, transform=transforms)

    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")


    train_loader = DenseDataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader  = DenseDataLoader(dataset_test,  batch_size=batch_size, shuffle=True, pin_memory=True)
    
    model = cls_model(num_points, F, K, M, modelnet_num, dropout=dropout, one_layer=False, reg_prior=False)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    regularization = 1e-9
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    
    log_dir_path = (curr_path / "tensorboard_cls/modelnet_normalized/").resolve()


    writer = SummaryWriter(log_dir=str(log_dir_path) + "/", comment='cls_' + str(num_points) +
            '_' + str(dropout), filename_suffix='_reg')
    start_training(model, train_loader, test_loader, optimizer, criterion, writer, epochs=num_epochs)