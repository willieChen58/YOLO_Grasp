/*
    需求: 订阅人的信息

*/
//std
// #include <tuple>
// #include <map>
// #include <vector>
// #include <queue>
// #include <mutex>
// #include <string>


// ROS

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

// #include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Empty.h>
#include <std_srvs/SetBool.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Empty.h>
// this project (services)
//tf
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include "tf2_ros/transform_broadcaster.h"
#include <tf2_ros/transform_listener.h>
#include "geometry_msgs/TransformStamped.h"
#include "tf2/LinearMath/Quaternion.h"
//moveit  

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/dynamics_solver/dynamics_solver.h>




//action
#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>


#include <dynamic_reconfigure/server.h>




#include "ros/ros.h"
#include "yolo_grasp/yolo_msg.h"

void detect_grasps_callback(const yolo_grasp::yolo_msg::ConstPtr& l){
    

            static tf2_ros::TransformBroadcaster broadcaster;
            //  5-2.创建 广播的数据(通过 pose 设置)
            geometry_msgs::TransformStamped tfs;
            //  |----头设置
            // tfs.header.frame_id = "kinect2_ir_optical_frame";
            tfs.header.frame_id = "camera_color_frame";
            tfs.header.stamp = ros::Time::now();

            //  |----坐标系 ID
            tfs.child_frame_id = "object";

            //  |----坐标系相对信息设置
            tfs.transform.translation.x = l->x1; //向右为jia？
            tfs.transform.translation.y = l->y1;  //向我移动为
            tfs.transform.translation.z = l->z1; //向上为减  在grasp转换后的数据进行加减
            //  |--------- 四元数设置
            tfs.transform.rotation.x = 0.74;
            tfs.transform.rotation.y = 0.675;
            tfs.transform.rotation.z = 0;
            tfs.transform.rotation.w = 0;

            //  5-3.广播器发布数据
            broadcaster.sendTransform(tfs);
            ROS_INFO("订阅的信息:%.2f, %.2f, %.2f", l->x1, l->y1, l->z1);
    }

int main(int argc, char *argv[])
{   
    setlocale(LC_ALL,"");

    //1.初始化 ROS 节点
    ros::init(argc,argv,"yolo_grasp1");
    //2.创建 ROS 句柄
    ros::NodeHandle nh;
    //3.创建订阅对象
    ros::Subscriber sub = nh.subscribe<yolo_grasp::yolo_msg>("grasp_pub",10,detect_grasps_callback);

    //4.回调函数中处理 person

    //5.ros::spin();
    ros::spin();    
    return 0;
}
