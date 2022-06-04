
#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <rgcnn_models/passthrough_filter_nodeConfig.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/passthrough.h>

#include "std_msgs/String.h"


class PassthroughFilterNode
{
public:
  typedef pcl::PointXYZRGB Point;
  typedef pcl::PointCloud<Point> PointCloud;

  PassthroughFilterNode()
  {
    pub_ = nh_.advertise<PointCloud>("pf_out",1);
    pub_float_ = nh_.advertise<std_msgs::String>("Dimensions",1);
    sub_ = nh_.subscribe ("point_cloud_in", 1,  &PassthroughFilterNode::cloudCallback, this);
    config_server_.setCallback(boost::bind(&PassthroughFilterNode::dynReconfCallback, this, _1, _2));

    double z_upper_limit, z_lower_limit;
    double x_upper_limit, x_lower_limit;
    double y_upper_limit, y_lower_limit;

    // "~" means, that the node hand is opened within the private namespace (to get the "own" paraemters)
    ros::NodeHandle private_nh("~");

    //read parameters with default value
    private_nh.param("z_lower_limit", z_lower_limit, 2.);
    private_nh.param("z_upper_limit", z_upper_limit, 5.);

    pt1_.setFilterFieldName ("z");
    pt1_.setFilterLimits (z_lower_limit, z_upper_limit);
    
    // read parameters with default value
    private_nh.param("x_lower_limit", x_lower_limit, 2.);
    private_nh.param("x_upper_limit", x_upper_limit, 5.);

    pt2_.setFilterFieldName ("x");
    pt2_.setFilterLimits (x_lower_limit, x_upper_limit);
    
    //read parameters with default value
    private_nh.param("y_lower_limit", y_lower_limit, 2.);
    private_nh.param("y_upper_limit", y_upper_limit, 5.);

    pt3_.setFilterFieldName ("y");
    pt3_.setFilterLimits (y_lower_limit, y_upper_limit);
    
  }

  ~PassthroughFilterNode() {}

  void
  dynReconfCallback(rgcnn_models::passthrough_filter_nodeConfig &config, uint32_t level)
  {
    pt1_.setFilterLimits(config.z_lower_limit, config.z_upper_limit);
    pt2_.setFilterLimits(config.x_lower_limit, config.x_upper_limit);
    pt3_.setFilterLimits(config.y_lower_limit, config.y_upper_limit);

     z_lower_limit=config.z_lower_limit;
     z_upper_limit=config.z_upper_limit;
     x_lower_limit=config.x_lower_limit;
     x_upper_limit=config.x_upper_limit;
     y_lower_limit=config.y_lower_limit;
     y_upper_limit=config.y_upper_limit;
  }

  void
  cloudCallback(const PointCloud::ConstPtr& cloud_in)
  {
      std::stringstream ss;

    std_msgs::String msg;

    ss<<x_lower_limit<<" "<<x_upper_limit<<" "<<y_lower_limit<<" "<<y_lower_limit;
      

    msg.data=ss.str();

    pub_float_.publish(msg);
   
   
   
   
   
    pt1_.setInputCloud(cloud_in);
    PointCloud cloud_out1;
    pt1_.filter(cloud_out1);
    
    
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_2(new pcl::PointCloud<pcl::PointXYZRGB>);
  *cloud_in_2 = cloud_out1;
 
	pt2_.setInputCloud(cloud_in_2);
    PointCloud cloud_out2;
    pt2_.filter(cloud_out2);
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in_3(new pcl::PointCloud<pcl::PointXYZRGB>);
  *cloud_in_3 = cloud_out2;
 
	pt3_.setInputCloud(cloud_in_3);
    PointCloud cloud_out3;
    pt3_.filter(cloud_out3);
    
    
    
    pub_.publish(cloud_out3);

    

    
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::Publisher pub_float_;
  dynamic_reconfigure::Server<rgcnn_models::passthrough_filter_nodeConfig> config_server_;

  pcl::PassThrough<Point> pt1_,pt2_,pt3_;


  float z_lower_limit;
  float z_upper_limit;
  float y_lower_limit;
  float y_upper_limit;
  float x_lower_limit;
  float x_upper_limit;


};

int main (int argc, char** argv)
{
  ros::init (argc, argv, "voxel_filter_node");

  PassthroughFilterNode vf;

  ros::spin();
}

