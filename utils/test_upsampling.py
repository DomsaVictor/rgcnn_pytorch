import torch as t
import torch_geometric as tg
from torch_geometric.data import Data
import numpy as np

# pcd = Data(x=np.ones((233, 3)), pos=np.random.rand(233, 3))
# # print(pcd.pos)
# # print(pcd)
# choice = np.random.choice(233, 400, replace=True)
# for key, item in pcd:
#     pcd[key] = item[choice]
#     print(f"Key  =  {key}")
#     print(f"Item =  {item[0]}")

# print(pcd)
# print(pcd.pos)

choice = np.random.choice(233, 400, replace=True)
points = np.ones((233,3))
print(points.shape)
points = points[choice]
print(points.shape)