#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/project_inliers.h>

#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>

#include <sensor_msgs/PointCloud2.h>


#include <dynamic_reconfigure/server.h>
#include "std_msgs/String.h"

#include <pointcloud_preparation/compute_surface_normals_nodeConfig.h>



class ComputeSurfaceNormalNode
{
public:
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  ComputeSurfaceNormalNode()
  {

    bool ok2;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals;

   
   

    sub_ = nh_.subscribe("/pf_out", 1, &ComputeSurfaceNormalNode::cloudCallback, this);

    config_server_.setCallback(boost::bind(&ComputeSurfaceNormalNode::dynReconfCallback, this, _1, _2));

  
  }

  ~ComputeSurfaceNormalNode() {}


 void compute_surface_normals (pcl::PointCloud<pcl::PointXYZ>::Ptr points,
                               double normal_radius, 
                              pcl::PointCloud<pcl::PointNormal> &cloudnormals)
{
  
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
  pcl::PointCloud<pcl::Normal>::Ptr normals_out (new pcl::PointCloud<pcl::Normal>);

  // Use a FLANN-based KdTree to perform neighborhood searches
  norm_est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZ>::Ptr 
							(new pcl::search::KdTree<pcl::PointXYZ>));

   std::cout << "Input cloud size:" << points->size() << '\n';


  
  // Specify the size of the local neighborhood to use when computing the surface normals
  norm_est.setRadiusSearch (normal_radius);


  // Set the input points
  norm_est.setInputCloud (points);


  // Estimate the surface normals and store the result in "normals_out"
  norm_est.compute (*normals_out);

  
 
   std::cout << "Normal cloud size:" << normals_out->size() << '\n';
    std::cout <<'\n'; 


    pcl::PointNormal p;
    for (int i = 0; i < points->size(); i++)
    {

      if( (isnan(normals_out->points[i].normal_x)==0) && 
          (isnan(normals_out->points[i].normal_y)==0) && 
          (isnan(normals_out->points[i].normal_z)==0)) {
      p.x = points->points[i].x;
      p.y = points->points[i].y;
      p.z = points->points[i].z;
      p.normal_x = normals_out->points[i].normal_x;
      p.normal_y = normals_out->points[i].normal_y;
      p.normal_z = normals_out->points[i].normal_z;
      cloudnormals.points.push_back(p);


      }
      
    }
    
    cloudnormals.width = cloudnormals.points.size();
    std::cout << "Points without nan normal:" << cloudnormals.width << '\n';
     std::cout <<'\n'; 
    cloudnormals.height = 1;
    cloudnormals.points.resize(cloudnormals.width * cloudnormals.height);
    cloudnormals.is_dense = false;

 
  
}

  void
  dynReconfCallback(pointcloud_preparation::compute_surface_normals_nodeConfig &config, uint32_t level)
  {
    

    radius_norm=config.radius_norm;

  }

 

  void
  cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {

	pcl::PointCloud<pcl::PointXYZ> cloud_Test;
    pcl::fromROSMsg(*cloud_msg, cloud_Test);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPTR(new pcl::PointCloud<pcl::PointXYZ>);
    *cloudPTR = cloud_Test;
    
    
    
    pcl::PointCloud<pcl::PointNormal> cloudnormals;


    if(cloudPTR->size() >0){
    compute_surface_normals(cloudPTR,radius_norm,cloudnormals);

   

   
    }
  
	  
  }

private:
  bool ok2;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_floor;
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals;

  double  radius_norm=0.03;
  dynamic_reconfigure::Server<pointcloud_preparation::compute_surface_normals_nodeConfig> config_server_;
  
  

 

  ros::NodeHandle nh_;
  ros::Subscriber sub_;
 
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "compute_surface_normals");

  ComputeSurfaceNormalNode compute_surface;

  ros::spin();
}
