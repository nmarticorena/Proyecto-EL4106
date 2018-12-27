#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import fitness
from ctypes import POINTER,c_double
import matplotlib.pyplot as plt
import seaborn as sns
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
try:
    from tqdm import tqdm
except:
    print("Falle importando la libreria tdqm, si quieres visualizar mediante una barra de progreso favor instalarla")

import pandas as pd

# configuracion graficos
sns.set()  # estilo por defecto de seaborn
sns.set_context('poster', font_scale=0.8) # contexto notebook 
#plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = (7,7)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
#plt.style.use('seaborn-white')


#PARAMS
pi=np.pi
bounds=[(-5, 5), (-pi/2, pi/2), (-pi/2, pi/2), (-pi/2, pi/2)]
nk=20

dataP=POINTER(c_double)


class Generator(object):
    def __init__(self,nk,lista,velocidad):
        self.pub1 = rospy.Publisher('/SimpleArm/joint1_position_controller/command', Float64, queue_size=1)
        self.pub2 = rospy.Publisher('/SimpleArm/joint2_position_controller/command', Float64, queue_size=1)
        self.pub3 = rospy.Publisher('/SimpleArm/joint3_position_controller/command', Float64, queue_size=1)
        self.pub4 = rospy.Publisher('/SimpleArm/joint4_position_controller/command', Float64, queue_size=1)
        self.command1=Float64()
        self.command2=Float64()
        self.command3=Float64()
        self.command4=Float64()
        self.velocidad=velocidad
        print(self.velocidad)
        for i in lista:
            self.Iterator(nk=nk,lista=i)
    def Iterator(self,nk,lista):
        print("Voy a empezar a graficar")
        rospy.sleep(2)
        df=pd.read_csv("./trayectorias/file_path{}.csv".format(lista))
        a=df.values
        nk=a.shape[0]
        a=a[:,1:]
        self.Reset_Arm(a)
        self.Show(a,nk,lista)
    def Reset_Arm(self,lista):
        self.command1.data=lista[0,0]
        self.command2.data=lista[0,1]
        self.command3.data=lista[0,2]
        self.command4.data=lista[0,3]
        self.pub1.publish(self.command1)
        self.pub2.publish(self.command2)
        self.pub3.publish(self.command3)
        self.pub4.publish(self.command4)
        print("Reiniciando brazo...")
        rospy.sleep(10)
        return
    def Show(self,lista,nk,indice):
        print("Moviendo el mejor de la epoca {}".format(indice))
        for i in range(nk):
            self.command1.data=lista[i,0]
            self.command2.data=lista[i,1]
            self.command3.data=lista[i,2]
            self.command4.data=lista[i,3]
            self.pub1.publish(self.command1)
            self.pub2.publish(self.command2)
            self.pub3.publish(self.command3)
            self.pub4.publish(self.command4)
            rospy.sleep(self.velocidad/nk)
            print(self.velocidad/nk)
            print("Angulo {}".format(i))
        return




if __name__ == '__main__':
    document=sys.argv[1].split(',')
    velocidad=float(sys.argv[2])
    print("velocidad {}".format(velocidad))
    document_arr=np.array([int(i) for i in document])
    rospy.init_node('CGA')
    rospy.loginfo('CGA Arm controller')
    try:
        for i in document_arr:
            df=pd.read_csv("./trayectorias/file_path{}.csv".format(i))
    except:
        print("Existe un individuo que no existe, recuerda que deben ser divisibles por 10")
        exit(1)

    Generator(lista=document_arr,nk=nk,velocidad=velocidad)
    

    #except:

    point =''
    print("Debe entregar objetivo de la forma: x,y,z")
        
