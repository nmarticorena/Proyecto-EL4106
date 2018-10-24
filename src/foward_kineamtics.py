import numpy as np


def Rotaciony(thetha):
    rot=np.array([[np.cos(thetha),0,np.sin(thetha),0],[0,1,0,0],[-1*np.sin(thetha),0,np.cos(thetha),0],[0,0,0,1]])
    return rot

def Rotacionz(thetha):
    rot=np.array([[np.cos(thetha),-1*np.sin(thetha),0,0],[np.sin(thetha),np.cos(thetha),0,0],[0,0,1,0],[0,0,0,1]])
    return rot

def Traslacion(x,y,z):
    t=np.array([[0,0,0,x],[0,0,0,y],[0,0,0,z],[0,0,0,0]])
    return t


R=Rotaciony(12)
T=Traslacion(0,0,0.5+0.1)

T1=Rotacionz(1.5)+Traslacion(0,0,0.1)
T2=Rotaciony(0.25)+Traslacion(0,0,0.5+0.1)
T3=Rotaciony(0.5)+Traslacion(0,0,0.5+0.1)
T4=Rotaciony(0.5)+Traslacion(0,0,0.5+0.1)
T5=Rotaciony(0.0)+Traslacion(0,0,0.5)
T12=np.dot(T1,T2)
T123=np.dot(T12,T3)
T1234=np.dot(T123,T4)
T12345=np.dot(T1234,T5)
T34=np.dot(T3,T4)
T345=np.dot(T34,T5)
T_ref=np.dot(T12,T345)
T_ref