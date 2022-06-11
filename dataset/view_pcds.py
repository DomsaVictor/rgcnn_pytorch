import open3d as o3d

root = "/home/victor/Desktop/"
pcd1_name = "Dataset_pcd.pcd"
pcd2_name = "ros_pcd_with_normals.pcd"

pcd1 = o3d.io.read_point_cloud(root+pcd1_name)
pcd1.paint_uniform_color([1,0,0])
pcd2 = o3d.io.read_point_cloud(root+pcd2_name)
pcd2 = pcd2.translate([0.5, 0, 0])
pcd2.paint_uniform_color([0,1,0])
o3d.visualization.draw_geometries([pcd1, pcd2], point_show_normal=True)