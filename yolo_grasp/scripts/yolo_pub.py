
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir) # 添加当前文件夹，以便于调用同级目录下的realsense.py
# sys.path.append(os.path.dirname(os.path.dirname(current_dir)) + '/yolov5-D435')
sys.path.append('/home/dell/catkin_ws/src/yolo_grasp/scripts/yolov5-D435')
# 需要改
sys.path.append('/home/dell/anaconda3/envs/yolo/lib/python3.8/site-packages') # 加载所需要的python的环境变量
# 导入yolo5
from GraspDetect import detect
import rospy
from yolo_grasp.msg import yolo_msg

if __name__ == "__main__":
    #1.初始化 ROS 节点
    rospy.init_node("yolo_grasp")
    #2.创建发布者对象
    pub = rospy.Publisher("grasp_pub",yolo_msg,queue_size=10)
    #3.组织消息
    Y = yolo_msg()

    #4.编写消息发布逻辑
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        camera_coordinate = detect()
        if camera_coordinate != None:
            Y.x1, Y.y1, Y.z1 = camera_coordinate
            pub.publish(Y)  #发布消息
            rospy.loginfo("X = %.2f, Y = %.2f, Z = %.2f",Y.x1, Y.y1, Y.z1)

        rate.sleep()  #休眠
