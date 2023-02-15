import sys
from pathlib import Path

curr_path = (Path(__file__).parent).resolve()
dataset_path = str((curr_path/"../dataset/").resolve())
utils_path   = str((curr_path/"../utils/").resolve())
model_path   = str((curr_path/"../model/segmentation/").resolve())
cls_path     = str((curr_path/"../model/classification/").resolve())

sys.path.append(dataset_path)
sys.path.append(utils_path)
sys.path.append(model_path)
sys.path.append(cls_path)