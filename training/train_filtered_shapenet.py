# PyTorch, PyG and other imports
from torch_geometric.transforms import RandomScale
from torch_geometric.transforms import RandomRotate
from torch_geometric.transforms import FixedPoints
from torch_geometric.loader import DenseDataLoader
from torch_geometric.datasets import ShapeNet
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
model_path      = (curr_path / "../model/segmentation/").resolve()
utils_path      = (curr_path / "../utils/").resolve()

import sys
sys.path.append(str(model_path))
sys.path.append(str(utils_path))
sys.path.append(str(dataset_path))

from utils import GaussianNoiseTransform
from FilteredShapenetDataset import FilteredShapeNet
from RGCNNSegmentation import seg_model
from utils import compute_loss_with_weights
from utils import label_to_cat
from utils import seg_classes


def train(model, optimizer, loader, regularization, criterion):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        # cat = data.category
        y = data.y.type(torch.LongTensor)
        # x = data.pos
        x = torch.cat([data.pos.type(torch.float32),
                  data.x.type(torch.float32)], dim=2)
        # out, L are for regularization
        logits, out, L = model(x.to(device), None)
        logits = logits.permute([0, 2, 1])

        loss = compute_loss_with_weights(logits, y, out, L, criterion, model, s=regularization)
        # y = y.to(device)
        # loss = criterion(logits, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"{i}: curr loss: {loss}")
    return total_loss * batch_size / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()
    size = len(loader.dataset)
    predictions = np.empty((size, num_points))
    labels = np.empty((size, num_points))
    total_correct = 0

    for i, data in enumerate(loader):
        # cat = data.category
        x = torch.cat([data.pos.type(torch.float32),
                  data.x.type(torch.float32)], dim=2)
        y = data.y
        logits, _, _ = model(x.to(device), None)
        logits = logits.to('cpu')
        pred = logits.argmax(dim=2)

        total_correct += int((pred == y).sum())
        start = i * batch_size
        stop = start + batch_size
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
    accuracy = ncorrects * 100 / (len(dataset_test) * num_points)

    return accuracy, cat_iou, tot_iou, ncorrects


def start_training(model, train_loader, test_loader, optimizer, criterion, epochs=50, learning_rate=1e-3, regularization=1e-9, decay_rate=0.95):
    print(model.parameters)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        test_acc, cat_iou, tot_iou, ncorrects = test(model, test_loader)
        test_stop_time = time.time()

        for key, value in cat_iou.items():
            print(key + ': {:.4f}, total: {:d}'.format(np.mean(value), len(value)))
            writer.add_scalar(key + '/test', np.mean(value), epoch)

        writer.add_scalar("IoU/test", np.mean(tot_iou) * 100, epoch)
        writer.add_scalar("accuracy/test", test_acc, epoch)

        print(
            f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}%, IoU: {np.mean(tot_iou)*100:.4f}%')
        print(f'ncorrect: {ncorrects} / {len(dataset_test) * num_points}')
        print(
            f'Train Time: \t{train_stop_time - train_start_time} \nTest Time: \t{test_stop_time - test_start_time }')
        print("~~~" * 30)

        my_lr_scheduler.step()

        # Save the model every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            if not os.path.isdir(str(model_path)):
                os.mkdir(str(model_path))
            torch.save(model.state_dict(), str(model_path) + '/' +
                       str(num_points) + 'p_model_v2_' + str(epoch) + '.pt')

    print(f"Training finished")


if __name__ == '__main__':
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    model_path = (curr_path / f"models_seg/{directory}/").resolve()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_points = 400
    batch_size = 4
    num_epochs = 200
    learning_rate =  1e-3 # 0.003111998
    decay_rate = 0.8
    weight_decay = 1e-9  # 1e-9
    dropout = 0.2 # 0.09170225
    regularization = 1e-9 # 5.295088673159155e-9

    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 3]

    
    # transforms = Compose([FixedPoints(num_points), GaussianNoiseTransform(
    #     mu=0, sigma=0, recompute_normals=False), RandomScale([0.8, 1.2]), RandomRotate(15, 0), RandomRotate(15, 1), RandomRotate(15, 2)])

    transforms = Compose([FixedPoints(num_points)])

    dataset_path = (dataset_path / "Plane").resolve()
    # dataset_path = (dataset_path / "ShapeNet").resolve()
    # dataset_train = ShapeNet(root=str(dataset_path), categories="Airplane", include_normals=True, split="train", transform=transforms)
    # dataset_test = ShapeNet(root=str(dataset_path), categories="Airplane", include_normals=True, split="test", transform=transforms)

    print(str(dataset_path))
    
    # root_dir MUST BE A Path(...) 
    dataset_train = FilteredShapeNet(root_dir=dataset_path, folder="train", transform=transforms)
    dataset_test = FilteredShapeNet(root_dir=dataset_path, folder="test", transform=transforms)

    # Define loss criterion.
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Training on {device}")

    # Verification...
    print(f"Train dataset shape: {dataset_train}")
    print(f"Test dataset shape:  {dataset_test}")

    train_loader = DenseDataLoader(
        dataset_train, batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=4)

    test_loader = DenseDataLoader(
        dataset_test, batch_size=batch_size,
        shuffle=True, num_workers=4)

    model = seg_model(num_points, F, K, M, input_dim=6,
                      dropout=dropout,
                      one_layer=False,
                      reg_prior=True,
                      recompute_L=False,
                      b2relu=True)

    model = model.to(device)
    
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log_dir_path = (curr_path / "tensorboard_seg/").resolve()
    
    writer = SummaryWriter(log_dir=str(log_dir_path)+"/", comment='seg_' + str(num_points) +
                           '_' + str(dropout), filename_suffix='_reg')

    start_training(model, train_loader, test_loader, optimizer,
                   epochs=num_epochs, criterion=criterion, regularization=regularization)
