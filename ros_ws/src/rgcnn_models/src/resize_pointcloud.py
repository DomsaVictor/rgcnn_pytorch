#!/home/victor/anaconda3/envs/thesis_env/bin/python3
import ctypes
import os
import struct
import sys
from subprocess import call

import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg as msg
import torch as t
from matplotlib.pyplot import axis
from numpy.random import default_rng
from sensor_msgs.msg import PointCloud2, PointField
from torch_geometric.nn import fps

from classification_model_cam import cls_model
from seg_model_rambo_v2 import seg_model

np.random.seed(0)

model_path = "/home/victor/workspace/catkin_ws/src/RGCNN_demo_ws/src/pcl_tutorial/models/"
model_name = "model15.pt"
model_file = model_path + model_name
device = 'cuda'

label_to_names = {0: 'chair',
                  1: 'plane'}

counter = 0


def callback(data):
    global counter
    counter += 1
    if counter % 1 == 0:
        counter = 0
        # xyz = np.array([[0, 0, 0]])
        # rgb = np.array([[0, 0, 0]])

        gen = pcl2.read_points(data, skip_nans=True)

        init_data = list(gen)
        xyz = np.empty(shape=(len(init_data), 3))

        for i, x in enumerate(init_data):
            xyz[i] = np.array([x[0], x[1], x[2]])
        # xyz = xyz[1:-1]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        points = np.asarray(pcd.points)
        points = t.tensor(points)

        pcd.estimate_normals(fast_normal_computation=False)
        # pcd.normalize_normals()
        pcd.orient_normals_consistent_tangent_plane(100)

        normals = np.asarray(pcd.normals)

        if len(points) < 1024:
            alpha = 0.03
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha)

            #o3d.visualization.draw_geometries([pcd, rec_mesh])

            num_points_sample = 1024

            pcd_sampled = rec_mesh.sample_points_poisson_disk(
                num_points_sample)

            points = pcd_sampled.points
            normals = np.asarray(pcd_sampled.normals)
        else:
            nr_points_fps = float(1024)
            nr_points = points.shape[0]

            index_fps = fps(points, ratio=nr_points_fps/nr_points, random_start=True)

            fps_points = points[index_fps]
            fps_normals = normals[index_fps]

            points = fps_points
            normals = fps_normals

        test_pcd = np.concatenate((points, normals), axis=1)
        # print(test_pcd.shape)
        xyz = t.tensor(points)

        xyz = t.cat([xyz, t.tensor(np.asarray(normals))], dim=1)

        xyz = xyz.unsqueeze(0)

        # print(test_pcd.shape)

        # if xyz.shape[1] == num_points:
        # pred,_ = model(xyz.to(t.float32).to(device))
        # # labels = pred.argmax(dim=2).squeeze(0)
        # labels = pred.argmax(dim=-1)
        # labels = labels.to('cpu')
        # # rospy.loginfo(labels.shape)
        # print(f'{label_to_names[labels.item()]}, {xyz.shape[1]}')
        # print(labels.shape)

        #message = pcl2.create_cloud(header, fields, points)
        message = pcl2.create_cloud(header, fields, test_pcd)

        pub.publish(message)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/no_floor_out", PointCloud2, callback=callback)
    rospy.spin()


if __name__ == "__main__":
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
    pub = rospy.Publisher("/Segmented_Point_Cloud", PointCloud2, queue_size=1)
    for i in range(40):
        color.append(rng.choice(254, size=3, replace=False).tolist())

    listener()
