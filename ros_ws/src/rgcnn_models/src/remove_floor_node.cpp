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
//#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>

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


class RemoveFloorNode
{
public:
  typedef pcl::PointXYZRGB Point;
  typedef pcl::PointCloud<Point> PointCloud;

  RemoveFloorNode()
  {
    pub_ = nh_.advertise<PointCloud>("no_floor_out",1);
    sub_ = nh_.subscribe ("pf_out", 1,  &RemoveFloorNode::cloudCallback, this);
    // config_server_.setCallback(boost::bind(&RemoveFloorNode::dynReconfCallback, this, _1, _2));
  }

  ~RemoveFloorNode() {}

  void euclidean_segmenting(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f
                            )
  {

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());

    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);
    // std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl; //*

    // Create the segmentation object for the planar model and set all the parameters

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);

    int i = 0, nr_points = (int)cloud_filtered->points.size();

    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      //////
      ///////////
      //////////
      // TREBUIE FUNCTIE DE OPRIRE
      /////////////
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);

    // Write the planar inliers to disk
    extract.filter(*cloud_plane);
    //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative(true);
    extract.filter(*cloud_f);
    *cloud_filtered = *cloud_f; //     HERE IS THE CLOUD FILTERED EXTRACTED

    if (cloud_filtered->size() != 0)
    {
      // Creating the KdTree object for the search method of the extraction
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(cloud_filtered);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(0.02); // 2cm
      ec.setMinClusterSize(100);
      ec.setMaxClusterSize(25000);
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_filtered);
      ec.extract(cluster_indices);

      for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
          cloud_cluster->points.push_back(cloud_filtered->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        cloud = cloud_cluster;
      }

      ////////////////////////////////////
    }
    
  }


  void
  dynReconfCallback(rgcnn_models::voxel_filter_nodeConfig &config, uint32_t level)
  {

  }

  void
  
cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
  {

	  pcl::PointCloud<pcl::PointXYZ> cloud_Test;
    pcl::fromROSMsg(*cloud_msg, cloud_Test);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPTR(new pcl::PointCloud<pcl::PointXYZ>);
    *cloudPTR = cloud_Test;
    
    

    pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>);


    if(cloudPTR->size() >0){
    euclidean_segmenting(cloudPTR,segmented_cloud);

    pub_.publish(segmented_cloud);

   
    }
  
	  
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  // dynamic_reconfigure::Server<rgcnn_models::voxel_filter_nodeConfig> config_server_;



};

int main (int argc, char** argv)
{
  ros::init (argc, argv, "no_floor_node");

  RemoveFloorNode vf;

  ros::spin();
}

