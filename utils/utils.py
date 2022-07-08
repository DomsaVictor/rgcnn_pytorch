import torch.nn as nn
import torch as t
from typing import Optional, Union
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
import torch
import numpy as np
import torch_geometric as tg
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
import open3d as o3d
import copy
from torch_geometric.nn import fps


def get_laplacian(adj_matrix, normalize=True):
    """ 
    Function to compute the Laplacian of an adjacency matrix

    Args:
        adj_matrix: tensor (batch_size, num_points, num_points)
        normlaize:  boolean
    Returns: 
        L:          tensor (batch_size, num_points, num_points)
    """

    if normalize:
        D = t.sum(adj_matrix, dim=1)
        eye = t.ones_like(D)
        eye = t.diag_embed(eye)
        D = 1 / t.sqrt(D)
        D = t.diag_embed(D)
        L = eye - t.matmul(t.matmul(D, adj_matrix), D)
    else:
        D = t.sum(adj_matrix, dim=1)
        D = t.diag_embed(D)
        L = D - adj_matrix
    return L

def pairwise_distance(point_cloud, normalize=False):
    """
    Compute the pairwise distance of a point cloud.

    Args: 
        point_cloud: tensor (batch_size, num_points, num_features)

    Returns: 
        pairwise distance: (batch_size, num_points, num_points)
    """

    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    point_cloud_inner = t.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = t.sum(t.mul(point_cloud, point_cloud), dim=2, keepdim=True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2, 1)
    adj_matrix = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

    if(normalize):
        nr_points=point_cloud.shape[1]
        maximum_values=torch.max(adj_matrix,dim=2)
        minimum_values=torch.min(adj_matrix,dim=2)

        interval=torch.subtract(maximum_values[0],minimum_values[0])

        interval=torch.tile(interval,[nr_points])

        interval=torch.reshape(interval,(point_cloud.shape[0],point_cloud.shape[1],point_cloud.shape[1]))

        adj_matrix=torch.div(adj_matrix,interval)

    adj_matrix = t.exp(-adj_matrix)
    return adj_matrix



class DenseChebConvV2(nn.Module):
    '''
    Convolutional Module implementing ChebConv. The input to the forward method needs to be
    a tensor 'x' of size (batch_size, num_points, num_features) and a tensor 'L' of size
    (batch_size, num_points, num_points).
    '''
    def __init__(self, in_channels: int, out_channels: int, K: int, normalization: Optional[bool]=True, bias: bool=False, **kwargs):
        assert K > 0
        super(DenseChebConvV2, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.lins = t.nn.ModuleList([
            Linear(in_channels, out_channels, bias=True, 
                weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(t.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):        
        for lin in self.lins:
            lin.weight = t.nn.init.trunc_normal_(lin.weight, mean=0, std=0.2)
            lin.bias   = t.nn.init.normal_(lin.bias, mean=0, std=0.2)


    def forward(self, x, L):
        x0 = x # N x M x Fin
        out = self.lins[0](x0)

        if self.K > 1:
            x1 = t.matmul(L, x0)
            out = out + self.lins[1](x1)

        for i in range(2, self.K):
            x2 = 2 * t.matmul(L, x1) - x0
            out += self.lins[i](x2)
            x0, x1 = x1, x2

        if self.bias is not None:
            out += self.bias
        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(in_features={self.in_channels}, '
                f'out_features={self.out_channels}, K={self.K}, '
                f'bias={self.bias is not None})'
                f'')


def IoU_accuracy(pred, target, n_classes=16):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[
            0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[
            0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            # If there is no ground truth, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)

def get_weights(dataset, num_points=2048, nr_classes=40):
    from sklearn.utils import class_weight

    '''
    If sk=True the weights are computed using Scikit-learn. Otherwise, a 'custom' implementation will
    be used. It is recommended to use the sk=True.
    '''

    weights = torch.zeros(nr_classes)
    
    y = np.empty(len(dataset)*dataset[0].y.shape[0])
    i = 0
    for data in dataset:
        y[i:num_points+i] = data.y
        i += num_points
    weights = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y)
    return weights


def get_one_matrix_knn(matrix, k,batch_size,nr_points):

    values,indices = torch.topk(matrix, k,sorted=False)
    
    batch_correction=torch.arange(0,batch_size,device='cuda')*nr_points
    batch_correction=torch.reshape(batch_correction,[batch_size,1])
    batch_correction=torch.tile(batch_correction,(1,nr_points*k))
    batch_correction=torch.reshape(batch_correction,(batch_size,nr_points,k))
       
    my_range=torch.unsqueeze(torch.arange(0,indices.shape[1],device='cuda'),1)
    my_range_repeated=torch.tile(my_range,[1,k])
    my_range_repeated=torch.unsqueeze(my_range_repeated,0)
    my_range_repeated_2=torch.tile(my_range_repeated,[batch_size,1,1])

    indices=indices+batch_correction
    my_range_repeated_2=my_range_repeated_2+batch_correction
    
    edge_indices=torch.cat((torch.unsqueeze(my_range_repeated_2,2),torch.unsqueeze(indices,2)),axis=2)
    edge_indices=torch.transpose(edge_indices,2,3)
    edge_indices=torch.reshape(edge_indices,(batch_size,nr_points*k,2))
    edge_indices=torch.reshape(edge_indices,(batch_size*nr_points*k,2))
    edge_indices=torch.transpose(edge_indices,0,1)
    edge_indices=edge_indices.long()

    edge_weights=torch.reshape(values,[-1])

    batch_indexes=torch.arange(0,batch_size,device='cuda')
    batch_indexes=torch.reshape(batch_indexes,[batch_size,1])
    batch_indexes=torch.tile(batch_indexes,(1,nr_points))
    batch_indexes=torch.reshape(batch_indexes,[batch_size*nr_points])
    batch_indexes=batch_indexes.long()

    knn_weight_matrix=tg.utils.to_dense_adj(edge_indices,batch_indexes,edge_weights)

    return knn_weight_matrix

def get_centroid(point_cloud,num_points):

    nr_coordinates=point_cloud.shape[2]
    batch_size=point_cloud.shape[0]
    
    centroid=torch.sum(point_cloud,1)
    centroid=centroid/num_points

    centroid=torch.tile(centroid,(1,num_points))
    centroid=torch.reshape(centroid,(batch_size,num_points,nr_coordinates))

    point_cloud_2=torch.subtract(point_cloud,centroid)

    Distances=torch.linalg.norm(point_cloud_2,dim=2)

    Distances=torch.unsqueeze(Distances,2)

    return Distances


def compute_loss_with_weights(logits, y, x, L, criterion, model, s=1e-9):
    if not logits.device == y.device:
        y = y.to(logits.device)

    l_norm = 0
    # for name, p in model.named_parameters():
    #      if 'bias_relus' not in name:
    #         l_norm += t.norm(p, p=2)

    loss = criterion(logits, y.squeeze())
    
    l = torch.tensor(0).to(torch.double).to(loss.device)
    if len(x) > 0:
        for i in range(len(x)):
            # l += (1/2) * t.linalg.norm(t.matmul(t.matmul(t.permute(x[i], (0, 2, 1)), L[i]), x[i]))**2
            l += (t.matmul(t.matmul(t.permute(x[i].to(torch.double), (0, 2, 1)), L[i].to(torch.double)), x[i].to(torch.double))).sum() / 2 
        l = (l_norm + l) * s
        loss += l
    return loss

def compute_loss(logits, y, x, L, criterion, s=1e-9):
    if not logits.device == y.device:
        y = y.to(logits.device)

    loss = criterion(logits, y)
    l=0
    for i in range(len(x)):
        l += (1/2) * t.linalg.norm(t.matmul(t.matmul(t.permute(x[i], (0, 2, 1)), L[i]), x[i]))**2
    l = l * s
    loss += l
    return loss

class GaussianNoiseTransform(BaseTransform):

    def __init__(self, mu: Optional[float] = 0, sigma: Optional[float] = 0.1, recompute_normals : bool = True):
        torch.manual_seed(0)
        np.random.seed(0)
        self.mu = mu
        self.sigma = sigma
        self.recompute_normals = recompute_normals

    def __call__(self, data: Union[Data, HeteroData]):
        noise = np.random.normal(self.mu, self.sigma, data.pos.shape)
        data.pos += noise
        data.pos = data.pos.float()
        if self.recompute_normals:
            pcd_o3d = o3d.geometry.PointCloud()
            p = data.pos.cpu().detach().numpy()
            pos = copy.deepcopy(p)
            pcd_o3d.points = o3d.utility.Vector3dVector(pos)
            pcd_o3d.estimate_normals(fast_normal_computation=True)
            pcd_o3d.normalize_normals()
            if hasattr(data, 'normal'):
                data.normal = np.asarray(pcd_o3d.normals)
                data.normal = torch.tensor(data.normal, dtype=torch.float32)
            else:
                data.x = np.asarray(pcd_o3d.normals)
                data.x = torch.tensor(data.x, dtype=torch.float32)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'



def count_categories(dataset):
    cat = dataset.categories
    count_cat = np.zeros_like(cat, dtype=int)
    for i, data in enumerate(dataset):
        c = data.category.item()
        count_cat[c] += 1
    return count_cat

def reduce_dataset(dataset, reduce_factor=0.6):
    count_cat = count_categories(dataset)
    new_count = np.zeros_like(count_cat, dtype=int)
    new_data = []
    for i, data in enumerate(dataset):
        c = data.category.item()
        new_count[c] += 1
        if(new_count[c] < int(count_cat[c] * reduce_factor) or count_cat[c] < 100):
            new_data.append(data)
    return new_data

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
               'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
               'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
               'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

label_to_cat = {}
for key in seg_classes.keys():
    for label in seg_classes[key]:
        label_to_cat[label] = key
        

class Sphere_Occlusion_Transform(BaseTransform):

    def __init__(self, radius: Optional[float] = 0.1,percentage:Optional[float] = 0.1, num_points: Optional[int]=1024):
        torch.manual_seed(0)
        np.random.seed(0)
        
        self.radius = radius
        self.percentage=percentage
        self.num_points=num_points

    def __call__(self, data: Union[Data, HeteroData]):
        chosen_center= np.random.randint(0, data.pos.shape[0])
       
        pcd_center=data.pos[chosen_center]

        nr_coordinates=pcd_center.shape[0]

        pcd_center=np.tile(pcd_center, data.pos.shape[0])
        pcd_center=pcd_center.reshape(data.pos.shape[0],nr_coordinates)

        points=data.pos
        points=points-pcd_center
        points=np.linalg.norm(points, axis=1)

        # sorted_points=np.sort(points)

        # selected_position=int(data.pos.shape[0]*self.percentage)

        # radius=sorted_points[selected_position]

        # remaining_index= np.squeeze(np.argwhere(points>=radius))
        
        remaining_index= np.squeeze(np.argwhere(points>=self.radius))

        

    
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(data.pos)

        pcd_o3d_remaining = o3d.geometry.PointCloud()
        pcd_o3d_remaining.points = o3d.utility.Vector3dVector(np.squeeze(data.pos[remaining_index]))
        aux_normals = data.normal[remaining_index] if "normal" in data else data.x[remaining_index]
        pcd_o3d_remaining.normals = o3d.utility.Vector3dVector(np.squeeze(aux_normals))
        pcd_o3d.paint_uniform_color([0, 1, 0])

        # pcd_o3d_sphere = o3d.geometry.PointCloud()
        # pcd_o3d_sphere.points=o3d.utility.Vector3dVector(np.squeeze(data.pos[index_pcd]))
        # pcd_o3d_sphere.normals=o3d.utility.Vector3dVector(np.squeeze(data.normal[index_pcd]))
        # pcd_o3d_sphere.paint_uniform_color([1, 0, 0])



        

        # o3d.visualization.draw_geometries([pcd_o3d_sphere]) 
        



        # o3d.visualization.draw_geometries([pcd_o3d_remaining]) 
        # o3d.visualization.draw_geometries([pcd_o3d])        

        # pcd_o3d_remaining.estimate_normals(fast_normal_computation=False)
        # pcd_o3d_remaining.normalize_normals()
        # pcd_o3d_remaining.orient_normals_consistent_tangent_plane(100)

        normals=np.asarray(pcd_o3d_remaining.normals)

        points_remaining=np.asarray(pcd_o3d_remaining.points)
        points_remaining=torch.tensor(points_remaining)

        if len(pcd_o3d_remaining.points) < self.num_points:
                alpha = 0.03
                rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd_o3d_remaining, alpha)

                num_points_sample = data.pos.shape[0]

                pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points_sample) 

                points = pcd_sampled.points

                pcd_o3d_remaining.points=points

                pcd_o3d_remaining.paint_uniform_color([0.5, 0.3, 0.2])

                # o3d.visualization.draw_geometries([pcd_o3d_remaining]) 
                # o3d.visualization.draw_geometries([pcd_o3d_remaining,pcd_o3d])


                points = torch.tensor(points)
                points=points.float()
                normals = np.asarray(pcd_sampled.normals)
                normals = torch.tensor(normals)
                normals=normals.float()

                data.pos = points
                if 'normal' in data:                
                    data.normal = normals
                else:
                    data.x = normals
        else:
            nr_points_fps=self.num_points
            nr_points=remaining_index.shape[0]

            index_fps = fps(points_remaining, ratio=float(nr_points_fps/nr_points) , random_start=True)

            index_fps=index_fps[0:nr_points_fps]

            fps_points=points_remaining[index_fps]
            fps_normals=normals[index_fps]

            points=fps_points
            normals = fps_normals

            data.pos=points
            data.normal=normals


        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'