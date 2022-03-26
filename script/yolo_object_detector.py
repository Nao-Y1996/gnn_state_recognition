#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes

class GetYoloObjectInfo():
    def __init__(self):
        self.bounding_boxes = None
        self.objects_info = []
        topic_name = "/darknet_ros/bounding_boxes"
        yolo_sub = rospy.Subscriber(topic_name, BoundingBoxes, self.callback)
        rospy.wait_for_message(topic_name, BoundingBoxes, timeout=5.0)

    def callback(self,data):
        self.bounding_boxes = data.bounding_boxes
        
    def get_objects(self):
        self.objects_info = []
        for obj in self.bounding_boxes:
            object_info = [obj.Class, int((obj.xmin+obj.xmax)/2), int((obj.ymin + obj.ymax)/2), obj.probability]
            self.objects_info.append(object_info)
        return self.objects_info

        
if __name__ == '__main__':
    rospy.init_node('YoloSubscriber', anonymous=True)
    yolodetector = GetYoloObjectInfo()
    while not rospy.is_shutdown():
        objects_info = yolodetector.get_objects()
        print(objects_info)