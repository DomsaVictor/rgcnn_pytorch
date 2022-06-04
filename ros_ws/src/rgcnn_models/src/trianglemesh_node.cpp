/*
 * voxel_filter.cpp
 *
 *  Created on: 06.09.2013
 *      Author: goa
 */

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <rgcnn_models/voxel_filter_nodeConfig.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/vtk_io.h>


#include <pcl/PolygonMesh.h>
#include <pcl_msgs/PolygonMesh.h>

#include <geometry_msgs/TransformStamped.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <shape_msgs/Mesh.h>

class TriangleMeshNode
{
public:
  typedef pcl::PointXYZRGB Point;
  typedef pcl::PointCloud<Point> PointCloud;
  //typedef pcl::PolygonMesh PMesh;


  
  
  TriangleMeshNode()
  {

    pub_ = nh_.advertise<pcl_msgs::PolygonMesh>("pcl_meshtriangle",1); // set tpoic name to publish
    // set topic name from receive, 'pf_out' - passthrough filter, 'cloud_pcd' - from file, not good, 'orig_cloud_pcd', 'point_cloud_in' - camera live feed
    sub_ = nh_.subscribe ("orig_cloud_pcd", 1,  &TriangleMeshNode::cloudCallback, this); 
    
           // "~" means, that the node hand is opened within the private namespace (to get the "own" paraemters)
    ros::NodeHandle private_nh("~");
    ros::spin();
  }

  ~TriangleMeshNode() {}

   void
  //cloudCallback(const pcl::PCLPointCloud2::ConstPtr& cloud_in)
  cloudCallback(const pcl::PCLPointCloud2& cloud_in)
  {
    // Load input file into a PointCloud<T> with an appropriate type
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    //pcl::PCLPointCloud2 cloud_blob;
    //pcl::io::loadPCDFile ("test.pcd", cloud_blob);
    //pcl::fromPCLPointCloud2 (cloud_blob, *cloud);
    pcl::fromPCLPointCloud2 (cloud_in, *cloud);
    //* the data should be available in cloud

    //pub_.publish(cloud_in);

    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    n.setInputCloud (cloud);
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.compute (*normals);
    //* normals should not contain the point normals + surface curvatures

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.025);

    // Set typical values for the parameters
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);
    pcl::io::saveVTKFile("/home/cuda/catkin_ws/src/rgcnn_models/src/mesh.vtk", triangles); 
    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();
    
    //publish PointCloud2
    /*sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL( triangles.cloud, output );
    pub_.publish(output);*/

    //publish PolygonMesh
    pcl_msgs::PolygonMesh pcl_msg_mesh;
    pcl_conversions::fromPCL(triangles, pcl_msg_mesh);
    pub_.publish(pcl_msg_mesh);

  }

  

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  
};

int main (int argc, char** argv)
{
  ros::init (argc, argv, "trianglemesh_node");

  TriangleMeshNode tm;
  //ros::spin();
}

