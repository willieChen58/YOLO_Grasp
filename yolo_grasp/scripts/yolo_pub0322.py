#! /usr/bin/env python
"""
    发布方:
        循环发送消息

"""
import rospy
from yolo_grasp.msg import yolo_msg


if __name__ == "__main__":
    #1.初始化 ROS 节点
    rospy.init_node("yolo_grasp")
    #2.创建发布者对象
    pub = rospy.Publisher("grasp_pub",yolo_msg,queue_size=10)
    #3.组织消息
    Y = yolo_msg()
    Y.x1 = 0.2
    Y.y1 = 0.5
    Y.z1 = 0.2

    #4.编写消息发布逻辑
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub.publish(Y)  #发布消息
        rate.sleep()  #休眠
        rospy.loginfo("X = %.2f, Y = %.2f, Z = %.2f",Y.x1, Y.y1, Y.z1)
