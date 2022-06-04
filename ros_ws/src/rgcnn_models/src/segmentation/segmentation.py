#!/home/victor/anaconda3/envs/thesis_env/bin/python3
import struct
from subprocess import call
import rospy
import std_msgs.msg as msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointField
import ctypes
import os
import numpy as np
from numpy.random import default_rng
import sys
np.random.seed(0)
import open3d as o3d
import torch as t
import GaussianNoiseTransform
from classification_model import cls_model
# from seg_model_rambo_v2 import seg_model
from train_filtered_shapenet import seg_model

model_path = "/home/victor/workspace/rgcnn_ws/src/pcl_tutorial/models/"
model_name = "1024p_model_v2_5.pt"
model_file = model_path + model_name
device = 'cuda'


def callback(data, model):
    gen = pcl2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)

    init_data = list(gen)
    xyz = np.empty(shape=(len(init_data), 6))

    for i, x in enumerate(init_data):
        xyz[i] = np.array([x[0],x[1],x[2],x[3],x[4],x[5]])

        # xyz = np.append(xyz, [[x[0],x[1],x[2]]], axis = 0)
   
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.estimate_normals(fast_normal_computation=False)
    # pcd.normalize_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, 0:3])
    pcd.normals = o3d.utility.Vector3dVector(xyz[:, 3:])

    center = pcd.get_center()
    pcd = pcd.translate(-center, relative=True)

    points = t.tensor(np.asarray(pcd.points))
    normals = t.tensor(np.asarray(pcd.points))
    



    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # diameter = np.linalg.norm(np.asarray(pcd.get_max_bound() - np.asarray(pcd.get_min_bound())))
    # print(diameter)
    # xyz = t.tensor(xyz)
    # xyz = t.cat([xyz, t.tensor(np.asarray(pcd.normals))], dim=1)
    

    # xyz = xyz[0:-1]
    
    xyz = t.cat([points, normals], 1)

    xyz = xyz.unsqueeze(0)

    pred, _, _ = model(xyz.to(t.float32).to(device), None)
    labels = pred.argmax(dim=2).squeeze(0)
    # labels = pred.argmax(dim=-1)
    labels = labels.to('cpu')
    # print(xyz)
    # rospy.loginfo(labels.shape)
    # print(label_to_names[labels.item()])
    # print(labels.shape)

    aux_label = np.zeros([num_points, 3])
    for i in range(num_points):
        aux_label[i] = color[int(labels[i])]

    # print(aux_label) 

    points = np.append(points, aux_label, axis=1)

    # print(points.shape)
    # print(color[0].shape)
    # for i in range(points.shape[0]):
    #     for j in range(3):
    #         points[i] = np.append(points[i], color[int(labels[i])][j])
    

    message = pcl2.create_cloud(header, fields, points)
    pub.publish(message)

    # else: 
    #     rospy.loginfo(xyz.shape)

def listener(model):
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/Segmented_Point_Cloud", PointCloud2, callback=callback, callback_args=model)
    rospy.spin()


if __name__ == "__main__":
    F = [128, 512, 1024]  # Outputs size of convolutional filter.
    K = [6, 5, 3]         # Polynomial orders.
    M = [512, 128, 4]

    num_points = 1024
    header = msg.Header()
    header.frame_id = 'camera_depth_optical_frame'
    fields = [PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)]
   
    color = []
    rng = default_rng()
    pub = rospy.Publisher("/Final_pcd", PointCloud2, queue_size=10)
    for i in range(50):
        color.append(rng.choice(254, size=3, replace=False).tolist())

    model = seg_model(num_points, F, K, M, input_dim=6)
    model.load_state_dict(t.load(model_file))
    model.to(device)
    model.eval()
    listener(model)