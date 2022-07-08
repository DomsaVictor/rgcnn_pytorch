from turtle import Shape
import open3d as o3d
from torch_geometric.datasets import ShapeNet
from torch_geometric.transforms import NormalizeScale

show_normals = False

root = "/home/victor/Desktop/"
pcd1_name = "Dataset_pcd.pcd"
pcd2_name = "ros_pcd_with_normals.pcd"

root_sh = "/home/victor/workspace/thesis_ws/github/rgcnn_pytorch/dataset/ShapeNet"
dataset = ShapeNet(root=root_sh, categories="Airplane")

# pcd1 = o3d.io.read_point_cloud(root+pcd1_name)
# pcd1.paint_uniform_color([1,0,0])

pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dataset[0].pos))
# pcd1.paint_uniform_color([1,0,0])
c = pcd1.get_center()
pcd1.translate(-c)
print(c)

pcd2 = o3d.io.read_point_cloud(root+pcd2_name)
c = pcd2.get_center()
pcd2.translate(-c)
bb1 = pcd1.get_oriented_bounding_box()
bb2 = pcd2.get_oriented_bounding_box()

pcd1 = pcd1.rotate(bb1.R.T)
bb22 = pcd1.get_oriented_bounding_box()
pcd2 = pcd2.rotate(bb2.R.T)

pcd2 = pcd2.translate([0,0.5,0])

pcd1.paint_uniform_color([0,1,0])
pcd2.paint_uniform_color([0,1,0])

o3d.visualization.draw_geometries([pcd1, pcd2], point_show_normal=show_normals)