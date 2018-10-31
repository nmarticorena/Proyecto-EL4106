#!/usr/bin/env python
import numpy as np
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def show_point(punto):
    rospy.init_node('rviz_publisher')
    rate = rospy.Rate(5)
     #transform from x,y points to x,y,z points
    p = Point() 
    p.x = punto[0]
    p.y = punto[1]
    p.z = punto[2]
    iterations = 1
    points=[]
    points.append(p)
    frame='/goal'
    while not rospy.is_shutdown() and iterations <= 10:
        pub = rospy.Publisher(frame, Marker, queue_size = 100)
        marker = Marker()
        marker.header.frame_id = "/world"

        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.orientation.w = 1

        marker.points = points;
        marker.pose.position=p
        t = rospy.Duration()
        marker.lifetime = t
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0

        pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
	punto = np.array([0.5,0.5,0.5])
	show_point(punto)