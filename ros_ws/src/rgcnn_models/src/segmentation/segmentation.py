#!/home/victor/anaconda3/envs/thesis_env/bin/python3
from subprocess import call
import rospy
import std_msgs.msg as msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointField
import numpy as np
from numpy.random import default_rng
np.random.seed(0)
import open3d as o3d
import torch as t
from torch_geometric.transforms import Center

from pathlib import Path
cwd = Path.cwd()
curr_path  = Path(__file__).parent
model_path = (curr_path / "../../../../../model/segmentation").resolve()
utils_path = (curr_path / "../../../../../utils").resolve()
import sys
sys.path.append(str(model_path))
sys.path.append(str(utils_path))

from RGCNNSegmentation import seg_model
from utils_pcd import pcd_registration

weight_name = "400p_model_v2_200.pt"

weight_path = f"{str(curr_path)}/{weight_name}"

device = 'cuda' if t.cuda.is_available() else 'cpu'



def rotate_pcd(pcd, angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis == 0:
        """rot on x"""
        R = np.array([[1 ,0 ,0],[0 ,c ,-s],[0 ,s ,c]])
    elif axis == 1:
        """rot on y"""
        R = np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]])
    elif axis == 1:
        """rot on z"""
        R = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]])

    return pcd.rotate(R)

def callback(data, args):
    '''
        Reads a point cloud from the topic. The pointclouds also includes normals in the rgb fields.
        The data from the pointcloud is stored into a open3d.geometry.PointCloud data type in 
        order to process it further: center the pointcloud and rotate it ??? - TBD
    '''
    
    model = args[0]
    registrator = args[1]
    
    gen = pcl2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)
    init_data = list(gen)
    xyz = np.empty(shape=(len(init_data), 6))

    for i, x in enumerate(init_data):
        xyz[i] = np.array([x[0],x[1],x[2],x[3],x[4],x[5]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
    pcd.normals = o3d.utility.Vector3dVector(xyz[:, 3:])
    # pcd.normalize_normals()
    # center = pcd.get_center()
    # pcd = pcd.translate(-center, relative=True)
    # pcd = rotate_pcd(pcd, 26, 0)
    # registrator.set_source(pcd)
    # pcd = registrator.register_pcds()
    # pcd = rotate_pcd(pcd, 90, 1)
    points = t.tensor(np.asarray(pcd.points))
    scale = (1 / points.abs().max()) * 0.999999
    points = points * scale
    normals = t.tensor(np.asarray(pcd.normals))
    
    # points = t.tensor(registrator.target.points)
    # normals = t.tensor(registrator.target.normals)
    
    # print(f"Points : {points.abs().max()}")
    # print(f"Normals: {normals.abs().max()}")
    
    xyz = t.cat([points, normals], 1)

    xyz = xyz.unsqueeze(0)

    pred, _, _ = model(xyz.to(t.float32).to(device), None)
    labels = pred.argmax(dim=2).squeeze(0)
    labels = labels.to('cpu')

    aux_label = np.zeros([num_points, 3])
    for i in range(num_points):
        aux_label[i] = color[int(labels[i])]
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    aux_label = colors[labels]

    points = np.append(points, aux_label, axis=1)
    print(points) 

    message = pcl2.create_cloud(header, fields, points)
    pub.publish(message)

def listener(model, registrator):
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/reshaped_pointcloud", PointCloud2, callback=callback, callback_args=[model, registrator])
    rospy.spin()

if __name__ == "__main__":
    
    registrator = pcd_registration()
    target_pcd = o3d.io.read_point_cloud(str((curr_path / "plane_0_1.pcd").resolve()))
    registrator.set_target(target_pcd)
    
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 3]

    num_points = 400
    header = msg.Header()
    header.frame_id = 'camera_depth_optical_frame'
    fields = [PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)]

    rng = default_rng()
    pub = rospy.Publisher("/SegmentedPCD", PointCloud2, queue_size=10)

    color = [[1,0,0], [0,1,0], [0,0,1], [0.2, 0.5, 0.]]
    
    model = seg_model(num_points, F, K, M, input_dim=6)
    model.load_state_dict(t.load(weight_path))
    model.to(device)
    model.eval()
    print("OK")
    listener(model, registrator)