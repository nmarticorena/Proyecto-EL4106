#!/usr/bin/env python

import sqlite3
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import math

con = sqlite3.connect("/home/nmarticorena/Proyecto-EL4106/db/energy.db")
cursor=con.cursor()

InsertQuery="INSERT INTO ENERGY(ID, E_0,E_1,E_2,E_3)\
                  VALUES (?,?,?,?,?)"


cursor.execute('''CREATE TABLE ENERGY 
   (ID STRING PRIMARY KEY NOT NULL,E_0 FLOAT NOT NULL,E_1 FLOAT NOT NULL,E_2 FLOAT NOT NULL,E_3 FLOAT NOT NULL)''' )



class Generator(object):
    global con
    global cursor
    def __init__(self):
        self.ready=False
        self.save=False
        self.sub = rospy.Subscriber('/SimpleArm/joint_states', JointState, self.save_info_callback)
        self.pub1 = rospy.Publisher('/SimpleArm/joint1_position_controller/command', Float64, queue_size=1)
        self.pub2 = rospy.Publisher('/SimpleArm/joint2_position_controller/command', Float64, queue_size=1)
        self.pub3 = rospy.Publisher('/SimpleArm/joint3_position_controller/command', Float64, queue_size=1)
        self.pub4 = rospy.Publisher('/SimpleArm/joint4_position_controller/command', Float64, queue_size=1)
        self.command1=Float64()
        self.command2=Float64()
        self.command3=Float64()
        self.command4=Float64()
        self.E_0=0
        self.E_1=0
        self.E_2=0
        self.E_3=0
        self.i=0
        self.j=0
        self.k=-90
        self.l=-92
        self.publisher()
    
    def save_info_callback(self,msg):
        global con
        global cursor
        global InsertQuery
        if self.ready:
            self.E_0=msg.effort[0]
            self.E_1=msg.effort[1]
            self.E_2=msg.effort[2]
            self.E_3=msg.effort[3]
            self.ready=False
            self.save=True

        #rospy.loginfo(msg)

    def publisher(self):
        self.command1.data=self.i
        self.command2.data=self.j
        self.command3.data=self.k
        self.command4.data=self.l
        self.pub1.publish(self.command1)
        self.pub2.publish(self.command2)
        self.pub3.publish(self.command3)
        self.pub4.publish(self.command4)
        rospy.sleep(1)
        self.pub1.publish(self.command1)
        self.pub2.publish(self.command2)
        self.pub3.publish(self.command3)
        self.pub4.publish(self.command4)
        rospy.sleep(10)
        rospy.loginfo("Inicio benchmark")
        delta_angulo=2.25
        delta_tiempo=5
        l_mul=1.0
        k_mul=1.0
        while self.j < 90:
            self.l+=delta_angulo*l_mul
            if self.l>90:
                self.k+=delta_angulo*k_mul
                l_mul=-1
            if self.l<-90:
                self.k+=delta_angulo*k_mul
                l_mul=1
            if self.k>90:
                self.j+=delta_angulo
                k_mul=-1
            if self.k<-90:
                self.j+=delta_angulo
                k_mul=1
            self.command1.data=self.i*(2.0*math.pi)/360.0
            self.command2.data=self.j*(2.0*math.pi)/360.0
            self.command3.data=self.k*(2.0*math.pi)/360.0
            self.command4.data=self.l*(2.0*math.pi)/360.0
            self.pub1.publish(self.command1)
            self.pub2.publish(self.command2)
            self.pub3.publish(self.command3)
            self.pub4.publish(self.command4)
            rospy.sleep(0.0277*delta_tiempo)
            rospy.loginfo("Seteo en true ready")
            self.ready=True
            while not self.save:
                rospy.sleep(0.00001)
            mac=("["+str(0)+","+str(self.j)+","+str(self.k)+","+str(self.l)+"]",)
            datos=(self.E_0,self.E_1,self.E_2,self.E_3)             
            cursor.execute(InsertQuery,mac+datos)
            rospy.loginfo("Guardo mensaje")
            self.save=False

            self.i+=2

            self.command1.data=self.i*(2.0*math.pi)/360.0
            self.command2.data=self.j*(2.0*math.pi)/360.0
            self.command3.data=self.k*(2.0*math.pi)/360.0
            self.command4.data=self.l*(2.0*math.pi)/360.0
            self.pub1.publish(self.command1)
            self.pub2.publish(self.command2)
            self.pub3.publish(self.command3)
            self.pub4.publish(self.command4)
            rospy.sleep(0.0277*delta_tiempo)
            rospy.loginfo("Seteo en true ready")
            self.ready=True
            while not self.save:
                rospy.sleep(0.00001)
            mac=("["+str(2)+","+str(self.j)+","+str(self.k)+","+str(self.l)+"]",)
            datos=(self.E_0,self.E_1,self.E_2,self.E_3)             
            cursor.execute(InsertQuery,mac+datos)
            rospy.loginfo("Guardo mensaje")
            self.save=False

        rospy.loginfo("all ready")
        con.commit()
        con.close()
        rospy.loginfo("file saved")

def main():
    rospy.init_node('base_controller')
    rospy.loginfo('Init base controller')
    try:
        base = Generator()   
    except:
        con.commit()
        con.close()
    rospy.spin()

if __name__ == '__main__':
    main()

# InsertQuery="INSERT INTO ENERGY(ID, E_0,E_1,E_2,E_3)\
#                   VALUES (?,?,?,?,?)"

# for i in range(10):
#   for j in range(10):
#       for k in range(10):
#           for l in range(10):
#               mac=("["+str(i)+","+str(j)+","+str(k)+","+str(l)+"]",)
#               datos=(i,j,k,l)             
#               cursor.execute(InsertQuery,mac+datos)
# con.commit()
# con.close()
