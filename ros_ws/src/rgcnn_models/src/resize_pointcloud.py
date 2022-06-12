#!/home/victor/anaconda3/envs/thesis_env/bin/python3
from subprocess import call

import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg as msg
import torch as t
from sensor_msgs.msg import PointCloud2, PointField
from torch_geometric.nn import fps

counter = 0 

def callback(data, num_points=1024):
    global counter

    gen = pcl2.read_points(data, skip_nans=True)

    init_data = list(gen)
    data_length = len(init_data)
    xyz = np.empty(shape=(data_length, 3))

    for i, x in enumerate(init_data):
        xyz[i] = np.array([x[0], x[1], x[2]])

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    pcd.estimate_normals(fast_normal_computation=False)
    # pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    points = t.tensor(np.asarray(pcd.points))
    if data_length < num_points:
        alpha = 0.03
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        #o3d.visualization.draw_geometries([pcd, rec_mesh])
        
        pcd_sampled = rec_mesh.sample_points_poisson_disk(num_points)

        points = np.asarray(pcd_sampled.points)
        normals = np.asarray(pcd_sampled.normals)
    else:
        index_fps = fps(points, ratio=float(num_points)/data_length, random_start=True)

        points = np.asarray(pcd.points)[index_fps]
        normals = np.asarray(pcd.normals)[index_fps]

    aux_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    aux_pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud('/home/victor/Desktop/Dataset_from_ROS/plane_' + str(counter) + '.pcd', aux_pcd)
    # o3d.visualization.draw_geometries([aux_pcd], point_show_normal=True)
    test_pcd = np.concatenate((points, normals), axis=1)
    message = pcl2.create_cloud(header, fields, test_pcd)
    counter += 1
    pub.publish(message)


def listener(num_points):
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/no_floor_out", PointCloud2, callback=callback, callback_args=num_points)
    rospy.spin()


if __name__ == "__main__":
    num_points = 400
    header = msg.Header()
    header.frame_id = 'camera_depth_optical_frame'

    fields = [PointField('x', 0,  PointField.FLOAT32, 1),
              PointField('y', 4,  PointField.FLOAT32, 1),
              PointField('z', 8,  PointField.FLOAT32, 1),
              PointField('r', 12, PointField.FLOAT32, 1),
              PointField('g', 16, PointField.FLOAT32, 1),
              PointField('b', 20, PointField.FLOAT32, 1)]
    
    pub = rospy.Publisher("/reshaped_pointcloud", PointCloud2, queue_size=1)

    listener(num_points)
