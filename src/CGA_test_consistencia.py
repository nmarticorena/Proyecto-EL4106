#!/usr/bin/env python
from __future__ import print_function
import sys
import matplotlib.pyplot as plt
from CGA_Trayectoria_Numba import *

#PARAMS
Masa=5
phi=0.00001
sigma=-1000
pi=np.pi
bounds=[(-5, 5), (-pi/2, pi/2), (-pi/2, pi/2), (-pi/2, pi/2)]
min_b, max_b = np.array(bounds).T
nk=50
n_tests=5

#Graph of the trayectory
def makeTGraph(ind,num):
    Joint0rad=ind[:,0]
    Joint1rad=ind[:,1]
    Joint2rad=ind[:,2]
    Joint3rad=ind[:,3]

    Joint0=np.rad2deg(Joint0rad)
    Joint1=np.rad2deg(Joint1rad)
    Joint2=np.rad2deg(Joint2rad)
    Joint3=np.rad2deg(Joint3rad)

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(np.arange(nk),Joint0)
    plt.xlabel("pasos")
    plt.title("Joint 0")
    plt.ylabel("Angulo del Joint Grad")
    plt.ylim([-181,181])
    
    plt.subplot(2,2,2)
    plt.plot(np.arange(nk),Joint1)
    plt.xlabel("pasos")
    plt.title("Joint 1")
    plt.ylabel("Angulo del Joint Grad")
    plt.ylim([-95,95])
    
    plt.subplot(2,2,3)
    plt.plot(np.arange(nk),Joint2)
    plt.xlabel("pasos")
    plt.title("Joint 2")
    plt.ylabel("Angulo del Joint Grad")
    plt.ylim([-95,95])

    plt.subplot(2,2,4)
    plt.plot(np.arange(nk),Joint3)
    plt.xlabel("pasos")
    plt.title("Joint 3")
    plt.ylabel("Angulo del Joint Grad")
    plt.ylim([-95,95])
    plt.savefig("./TrayectoriaNum{}.png".format(num),bbox_inches='tight', dpi=150)

if __name__ == '__main__':
    #Get the parameters
    try:
        point=sys.argv[1].split(',')
        angle=sys.argv[2].split(',')
        its=int(sys.argv[3])
    except:
        print("USO: {} <x,y,x> <t1,t2,t3,t4> <n_gen>".format(sys.argv[0]))
        exit()

    ob=np.array([float(i) for i in point])
    angle_in=np.array([float(i) for i in angle])
    angle_in=np.deg2rad(angle_in)
    print("Objetivo: {}".format(ob))
    print("Inicio: {}".format(angle_in))
    print("Iteraciones: {}".format(its))
    #Run tests
    energies=np.zeros(n_tests)
    errors=np.zeros(n_tests)
    for i in range(5):
        print("Prueba {}".format(i+1))
        l=list(cga(fitPar,ob,min_b,max_b,angle_in,nk=nk,its=its,popsize=1000))
        error,energy=fit_grafico(l[-1][0],ob,angle_in,nk)
        energies[i]=energy
        errors[i]=error
        makeTGraph(l[-1][0],i+1)
    #Get mean and std of error and energy cost
    meanEnergy=np.mean(energies)
    stdEnergy=np.std(energies)
    meanError=np.mean(errors)
    stdError=np.std(errors)
    
    print("Error= {} +- {}".format(meanError,stdError))
    print("Gasto Energia= {} +- {}".format(meanEnergy,stdEnergy))