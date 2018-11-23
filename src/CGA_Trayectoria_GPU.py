#!/usr/bin/env python
from __future__ import print_function
import sys
import cupy as cp
import numpy as np
import fitness
from ctypes import POINTER,c_double
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
try:
    from tqdm import tqdm
except:
    print("Falle importando la libreria tdqm, si quieres visualizar mediante una barra de progreso favor instalarla")

# configuracion graficos
sns.set()  # estilo por defecto de seaborn
sns.set_context('poster', font_scale=0.8) # contexto notebook 
#plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = (7,7)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
#plt.style.use('seaborn-white')


#PARAMS
pi=cp.pi
bounds=[(-5, 5), (-pi/2, pi/2), (-pi/2, pi/2), (-pi/2, pi/2)]
nk=20

dataP=POINTER(c_double)




#Diferential Evolution

def cga(fobj,obj, bounds,angle_in, mut=0.1, crossp=0.6,nk=50, popsize=1000, its=4000,pcj=0.5,pmj=1):
    global errores_mejor
    errores_mejor=cp.zeros(its)
    pop=initGauss(bounds,nk,popsize,angle_in)
    #pop=initanh(bounds,nk,popsize,angle_in)
    Pfitness = cp.asarray([fobj(ind,obj,angle_in,nk) for ind in pop])
    best_idx = cp.argmax(Pfitness)
    best = pop[best_idx]
    hall_of_fame=best
    termino=False
    elite_total=int(popsize/10)
    try:
        a=tqdm(range(its))
    except:
        a=range(its)

    for i in a:
        if termino:
            errores_mejor[i]=Pfitness[best_idx]
            yield best,Pfitness[best_idx]
        else:
            fitnessT=cp.zeros(popsize)
            childs=cp.zeros((popsize,nk,4))
            for j in range(popsize):
                fitnessT[j]=fobj(pop[j],obj,angle_in,nk)
            max_fitness=cp.sum(fitnessT)
            elite_index=cp.argpartition(fitnessT, -1*elite_total)[-1*elite_total:]
            fitnessRel=fitnessT/max_fitness
            aceptados=0
            aceptados_array=cp.random.choice(popsize, popsize, p=fitnessRel)
            
            childs[0:elite_total]=pop[elite_index]
            


            for j in range((popsize/2)-elite_total):
                indexP1=aceptados_array[j*2]
                indexP2=aceptados_array[j*2+1]
                while indexP1==indexP2:
                    indexP2=(indexP2+1)%(popsize-1)
                P1=pop[aceptados_array[indexP1]]
                P2=pop[aceptados_array[indexP2]]
                if cp.random.rand()<crossp:
                    C1,C2=cross(P1,P2,pcj)
                else:
                    C1 = P1.copy()
                    C2 = P2.copy()
                if cp.random.rand()<mut:
                    C1=cp.array(mutation(C1,bounds,pmj))
                if cp.random.rand()<mut:
                    C2=cp.array(mutation(C2,bounds,pmj))
                childs[j*2+elite_total]=C1.copy()
                childs[j*2+1+elite_total]=C2.copy()

            pop=childs.copy()
            Pfitness = cp.asarray([fobj(ind,obj,angle_in,nk) for ind in pop])
            best_idx=cp.argmax(Pfitness)
            best=pop[best_idx]
            if i%10==0:
                df = pd.DataFrame(cp.asnumpy(best))
                df.to_csv("trayectorias/file_path{}.csv".format(i))
            if fobj(hall_of_fame,obj,angle_in,nk)<Pfitness[best_idx]:
                print("Encontre uno mejor con fitness igual a {}".format(Pfitness[best_idx]))
                print("alcance un fitness de:{} con promedio {}".format(Pfitness[best_idx],cp.mean(Pfitness)))
                hall_of_fame=best.copy()

            if Pfitness[best_idx]>=2.0-0.001:
                print("alcance un fitness de:{}".format(Pfitness[best_idx]))
                termino=True
            #print(i)
            errores_mejor[i]=Pfitness[best_idx]
            yield best, Pfitness[best_idx]






        # aceptados_array=pop[cp.argmax(fitnessT)].reshape(1,nk,4)
        # print(aceptados_array.shape)
        
        # print(aceptados_array)
        # for i in range(3):
        #     aceptados_array=cp.concatenate((aceptados_array,pop[j].reshape(1,nk,4)),axis=1)
        #print(aceptados_array.shape)
        
        #print(aceptados_array)

        #while aceptados<popsize:
         #   a=2




def derangementN(n):
    v=cp.arange(n)
    num=cp.arange(n)
    while True:
        cp.random.shuffle(v)
        if cp.all((v-num)!=0):
            break
    return v
    
#Funcion de fitness
dataP=POINTER(c_double)


def Rotaciony(thetha):
    rot=cp.asarray([[np.cos(thetha),0,np.sin(thetha),0],[0,1,0,0],[-1*np.sin(thetha),0,np.cos(thetha),0],[0,0,0,1]])
    return rot

def Rotacionz(thetha):
    rot=cp.asarray([[np.cos(thetha),-1*np.sin(thetha),0,0],[np.sin(thetha),np.cos(thetha),0,0],[0,0,1,0],[0,0,0,1]])
    return rot

def Traslacion(x,y,z):
    t=cp.asarray([[0,0,0,x],[0,0,0,y],[0,0,0,z],[0,0,0,0]])
    return t



def Cinematica_directa(angle):
    T1=Rotacionz(angle[0])+Traslacion(0,0,0.1)
    T2=Rotaciony(angle[1])+Traslacion(0,0,0.5+0.1)
    T3=Rotaciony(angle[2])+Traslacion(0,0,0.5+0.1)
    T4=Rotaciony(angle[3])+Traslacion(0,0,0.5+0.1)
    T5=Rotaciony(angle[4])+Traslacion(0,0,0.5)
    T12=cp.dot(T1,T2)
    T123=cp.dot(T12,T3)
    T1234=cp.dot(T123,T4)
    T12345=cp.dot(T1234,T5)
    return T12345


def Error_cuadratico(angle,obj):
    A=Cinematica_directa(angle)
    error1=cp.abs(obj[0]-A[0,3])
    error2=cp.abs(obj[1]-A[1,3])
    error3=cp.abs(obj[2]-A[2,3])
    error_total=cp.sqrt(error1**2+error2**2+error3**2)
    return error_total

def fitC(ind,ob,angle_in,nk):
    res=0;
    ind=cp.asnumpy(ind)
    ob=cp.asnumpy(ob)
    res+=fitness._distance(ind[-1,:].ctypes.data_as(dataP),ob.ctypes.data_as(dataP))*(1000)
    #res+=Error_cuadratico(ind[-1,:],ob)
    res2=0
    for i in range(nk-1):
        res2+=(np.sum(np.abs(ind[i,:]-ind[i+1,:])))*5 #quizas ponerlo en 5

    res2=res2
    #res+=cp.sum(cp.abs(angle_in-ind[0,:]))
    #print("funcion de diferencia final {}".format(res))
    #print("Suma de errores {}".format(res2))
    return (1/(1+res))*(1/(1+res2))
    n=cp.random.rand()
    r#eturn (1/(1+res))*n + (1-n)*(1/(1+res2))


def initGauss(bounds,nk,popsize,angle_in):
    min_b, max_b = cp.asarray(bounds).T
    diff = cp.array([2*pi,pi,pi,pi])
    #ends=min_b+cp.random.rand(popsize,2,4)*diff
    A=(cp.ones((popsize,1,4)))*cp.array(angle_in)
    B=min_b+cp.random.rand(popsize,1,4)*diff
    ends=cp.concatenate((A,B),axis=1)
    #print(ends.shape)
    #print(ends)
    rank=cp.arange(nk)
    travel=ends[:,1]-ends[:,0]
    pop=(travel.reshape(popsize,1,4)*rank.reshape((1,nk,1)))/(nk-1)
    pop+=ends[:,0].reshape(popsize,1,4)
    A=(6*cp.abs(travel)*cp.random.rand(popsize,4)-3*cp.abs(travel)).reshape(popsize,1,4)
    u=cp.random.rand(popsize,4)*(nk-1)
    sigma=(1+cp.random.rand(popsize,4)*(nk/6-1)).reshape(popsize,1,4)
    pop+=A*cp.exp(-((rank.reshape(1,nk,1)-u.reshape(popsize,1,4))/(2*sigma))**2)
    pop=pop.clip(min_b,max_b)
    return pop


def initanh(bounds,nk,popsize,angle_in):
    min_b, max_b = cp.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    A=(cp.ones((popsize,1,4)))*cp.array(angle_in)
    B=min_b+cp.random.rand(popsize,1,4)*diff
    ends=cp.concatenate((A,B),axis=1)
    rank=cp.arange(nk)
    travel=ends[:,1]-ends[:,0]
    u=cp.random.rand(popsize,1,4)*(nk-1)
    sigma=(1+cp.random.rand(popsize,4)*(nk/6-1)).reshape(popsize,1,4)
    print("rank")
    print(rank.shape)
    print("u")
    print(u.shape)
    print("sigma")
    print(sigma.shape)
    print(nk)
    print((rank.reshape(1,nk,1)-u.reshape(popsize,1,4)).shape)
    print(travel.reshape(popsize,1,4).shape)
    A=cp.tanh(rank.reshape(1,nk,1)-u.reshape(popsize,1,4))/sigma
    print(A.shape)

    pop=(travel.reshape(popsize,1,4)*0.5*(1+cp.tanh((rank.reshape(1,nk,1)-u.reshape(popsize,1,4))/sigma).reshape(popsize,nk,4)))
    pop+=ends[:,0].reshape(popsize,1,4)
    #A=(6*cp.abs(travel)*cp.random.rand(popsize,4)-3*cp.abs(travel)).reshape(popsize,1,4)
    #pop=0.5*(1+cp.tanh((1,nk,1)-u.reshape(popsize,1,4))/(2*sigma))**2
    pop=pop.clip(min_b,max_b)
    return pop

def cross(P1,P2,pcj):
    nk=P1.shape[0]
    rank=cp.arange(nk)
    crossJ=np.random.rand(4) > pcj
    u=cp.random.rand(4)*(nk-1)
    sigma=1+cp.random.rand(4)*(nk/6-1)
    W=0.5*(1+cp.tanh((rank.reshape(nk,1)-u)/sigma))
    W=cp.asnumpy(W)
    mW=1-W
    C1=P1.copy()
    C2=P1.copy()
    C1=cp.asnumpy(C1)
    C2=cp.asnumpy(C2)
    P1=cp.asnumpy(P1)
    P2=cp.asnumpy(P2)
    C1[1:,crossJ]=(W*P1+mW*P2)[1:,crossJ]
    C2[1:,crossJ]=(mW*P1+W*P2)[1:,crossJ]
    return cp.array(C1),cp.array(C2)

def mutation(C,bounds,pmj):
    min_b, max_b = cp.asarray(bounds).T
    nk=C.shape[0]
    rank=cp.arange(nk)
    mutJ=cp.random.rand(4) < pmj
    mut=C.copy()
    u=cp.random.rand(4)*(nk-1)
    sigma=1+cp.random.rand(4)*(nk/6-1)
    travel=C.max(0)-C.min(0)
    d=cp.random.rand(4)*2*travel-travel
    M=cp.exp(-((rank.reshape(nk,1)-u)/(2*sigma))**2)
    mutJ=cp.asnumpy(mutJ)
    M=cp.asnumpy(M)
    d=cp.asnumpy(d)
    max_b=cp.asnumpy(max_b)
    min_b=cp.asnumpy(min_b)
    C=cp.asnumpy(C)
    mut=cp.asnumpy(mut)
    mut[1:,mutJ]=(C+d*M)[1:,mutJ]
    return mut.clip(min_b,max_b)


if __name__ == '__main__':

    #try:
        
    point=sys.argv[1].split(',')
    angle=sys.argv[2].split(',')
    ob=cp.array([float(i) for i in point])
    angle_in=cp.array([float(i) for i in angle])
    print(ob)
    print(angle_in)
    nk=50
    
    l=list(cga(fitC,ob,bounds,angle_in,nk=nk))
    print(l[-1])
    print(cp.sqrt(fitness._distance(l[-1][0][-1,:].ctypes.data_as(dataP),ob.ctypes.data_as(dataP))))
    plt.subplot(2,2,1)
    plt.plot(cp.arange(nk),l[-1][0][:,0])
    plt.xlabel("pasos")
    plt.title("Joint 0")
    plt.ylabel("Angulo del Joint Rad")
    plt.subplot(2,2,2)

    plt.plot(cp.arange(nk),l[-1][0][:,1])
    plt.xlabel("pasos")
    plt.title("Joint 1")
    plt.ylabel("Angulo del Joint Rad")

    plt.subplot(2,2,3)
    plt.plot(cp.arange(nk),l[-1][0][:,2])
    plt.xlabel("pasos")
    plt.title("Joint 2")
    plt.ylabel("Angulo del Joint Rad")

    plt.subplot(2,2,4)
    plt.plot(cp.arange(nk),l[-1][0][:,3])
    plt.xlabel("pasos")
    plt.title("Joint 3")
    plt.ylabel("Angulo del Joint Rad")
    plt.show()
    #print(errores_mejor)
    plt.plot(errores_mejor)
    plt.xlabel("iteraciones")
    plt.title("fitness mejor individuo")
    plt.ylabel("fitness")
    plt.show()

    #except:

    point =''
    print("Debe entregar objetivo de la forma: x,y,z")
        
