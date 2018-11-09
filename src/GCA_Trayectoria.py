#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import fitness
from ctypes import POINTER,c_double
import matplotlib.pyplot as plt
import seaborn as sns
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
pi=np.pi
bounds=[(-5, 5), (-pi/2, pi/2), (-pi/2, pi/2), (-pi/2, pi/2)]
nk=20

dataP=POINTER(c_double)


#Diferential Evolution
def cga(fobj,obj, bounds,angle_in, mut=0.1, crossp=0.6,nk=50, popsize=1000, its=1000,pcj=0.5,pmj=1):
    global errores_mejor
    errores_mejor=np.zeros(its)
    pop=initGauss(bounds,nk,popsize,angle_in)
    #pop=initanh(bounds,nk,popsize,angle_in)
    Pfitness = np.asarray([fobj(ind,obj,angle_in,nk) for ind in pop])
    best_idx = np.argmax(Pfitness)
    best = pop[best_idx]
    hall_of_fame=best
    termino=False
    elite_ind=popsize/10
    try:
        a=tqdm(range(its))
    except:
        a=range(its)

    for i in a:
        if termino:
            errores_mejor[i]=Pfitness[best_idx]
            yield best,Pfitness[best_idx]
        else:
            fitnessT=np.zeros(popsize)
            childs=np.zeros((popsize,nk,4))
            for j in range(popsize):
                fitnessT[j]=fobj(pop[j],obj,angle_in,nk)
            max_fitness=np.sum(fitnessT)
            fitnessRel=fitnessT/max_fitness
            aceptados=0
            aceptados_array=np.random.choice(popsize, popsize, p=fitnessRel)
            for j in range(popsize/2):
                indexP1=aceptados_array[j*2]
                indexP2=aceptados_array[j*2+1]
                while indexP1==indexP2:
                    indexP2=(indexP2+1)%(popsize-1)
                P1=pop[aceptados_array[indexP1]]
                P2=pop[aceptados_array[indexP2]]
                if np.random.rand()<crossp:
                    C1,C2=cross(P1,P2,pcj)
                else:
                    C1 = P1.copy()
                    C2 = P2.copy()
                if np.random.rand()<mut:
                    C1=mutation(C1,bounds,pmj)
                if np.random.rand()<mut:
                    C2=mutation(C2,bounds,pmj)
                childs[j*2]=C1.copy()
                childs[j*2+1]=C2.copy()

            pop=childs.copy()
            Pfitness = np.asarray([fobj(ind,obj,angle_in,nk) for ind in pop])
            best_idx=np.argmax(Pfitness)
            best=pop[best_idx]
            if fobj(hall_of_fame,obj,angle_in,nk)<Pfitness[best_idx]:
                print("Encontre uno mejor con fitness igual a {}".format(Pfitness[best_idx]))
                print("alcance un fitness de:{} con promedio {}".format(Pfitness[best_idx],np.mean(Pfitness)))
                hall_of_fame=best.copy()

            if Pfitness[best_idx]>=1.0-0.001:
                print("alcance un fitness de:{}".format(Pfitness[best_idx]))
                termino=True
            #print(i)
            errores_mejor[i]=Pfitness[best_idx]
            yield best, Pfitness[best_idx]






        # aceptados_array=pop[np.argmax(fitnessT)].reshape(1,nk,4)
        # print(aceptados_array.shape)
        
        # print(aceptados_array)
        # for i in range(3):
        #     aceptados_array=np.concatenate((aceptados_array,pop[j].reshape(1,nk,4)),axis=1)
        #print(aceptados_array.shape)
        
        #print(aceptados_array)

        #while aceptados<popsize:
         #   a=2




def derangementN(n):
    v=np.arange(n)
    num=np.arange(n)
    while True:
        np.random.shuffle(v)
        if np.all((v-num)!=0):
            break
    return v
    
#Funcion de fitness
dataP=POINTER(c_double)
def fitC(ind,ob,angle_in,nk):
    res=0;
    res+=fitness._distance(ind[-1,:].ctypes.data_as(dataP),ob.ctypes.data_as(dataP))*(200)
    res2=0
    for i in range(nk-1):
        res2+=(np.sum(np.abs(ind[i,:]-ind[i+1,:])))*5 #quizas ponerlo en 5

    res2=res2
    #res+=np.sum(np.abs(angle_in-ind[0,:]))
    #print("funcion de diferencia final {}".format(res))
    #print("Suma de errores {}".format(res2))
    return (1/(1+res))*(1/(1+res2))
    n=np.random.rand()
    r#eturn (1/(1+res))*n + (1-n)*(1/(1+res2))


def initGauss(bounds,nk,popsize,angle_in):
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    #ends=min_b+np.random.rand(popsize,2,4)*diff
    A=(np.ones((popsize,1,4)))*np.array(angle_in)
    B=min_b+np.random.rand(popsize,1,4)*diff
    ends=np.concatenate((A,B),axis=1)
    #print(ends.shape)
    #print(ends)
    rank=np.arange(nk)
    travel=ends[:,1]-ends[:,0]
    pop=(travel.reshape(popsize,1,4)*rank.reshape((1,nk,1)))/(nk-1)
    pop+=ends[:,0].reshape(popsize,1,4)
    A=(6*np.abs(travel)*np.random.rand(popsize,4)-3*np.abs(travel)).reshape(popsize,1,4)
    u=np.random.rand(popsize,4)*(nk-1)
    sigma=(1+np.random.rand(popsize,4)*(nk/6-1)).reshape(popsize,1,4)
    pop+=A*np.exp(-((rank.reshape(1,nk,1)-u.reshape(popsize,1,4))/(2*sigma))**2)
    pop=pop.clip(min_b,max_b)
    return pop


def initanh(bounds,nk,popsize,angle_in):
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    A=(np.ones((popsize,1,4)))*np.array(angle_in)
    B=min_b+np.random.rand(popsize,1,4)*diff
    ends=np.concatenate((A,B),axis=1)
    rank=np.arange(nk)
    travel=ends[:,1]-ends[:,0]
    u=np.random.rand(popsize,1,4)*(nk-1)
    sigma=(1+np.random.rand(popsize,4)*(nk/6-1)).reshape(popsize,1,4)
    print("rank")
    print(rank.shape)
    print("u")
    print(u.shape)
    print("sigma")
    print(sigma.shape)
    print(nk)
    print((rank.reshape(1,nk,1)-u.reshape(popsize,1,4)).shape)
    print(travel.reshape(popsize,1,4).shape)
    A=np.tanh(rank.reshape(1,nk,1)-u.reshape(popsize,1,4))/sigma
    print(A.shape)

    pop=(travel.reshape(popsize,1,4)*0.5*(1+np.tanh((rank.reshape(1,nk,1)-u.reshape(popsize,1,4))/sigma).reshape(popsize,nk,4)))
    pop+=ends[:,0].reshape(popsize,1,4)
    #A=(6*np.abs(travel)*np.random.rand(popsize,4)-3*np.abs(travel)).reshape(popsize,1,4)
    #pop=0.5*(1+np.tanh((1,nk,1)-u.reshape(popsize,1,4))/(2*sigma))**2
    pop=pop.clip(min_b,max_b)
    return pop

def cross(P1,P2,pcj):
    nk=P1.shape[0]
    rank=np.arange(nk)
    crossJ=np.random.rand(4) > pcj
    u=np.random.rand(4)*(nk-1)
    sigma=1+np.random.rand(4)*(nk/6-1)
    W=0.5*(1+np.tanh((rank.reshape(nk,1)-u)/sigma))
    mW=1-W
    C1=P1.copy()
    C2=P1.copy()
    C1[1:,crossJ]=(W*P1+mW*P2)[1:,crossJ]
    C2[1:,crossJ]=(mW*P1+W*P2)[1:,crossJ]
    return C1,C2

def mutation(C,bounds,pmj):
    min_b, max_b = np.asarray(bounds).T
    nk=C.shape[0]
    rank=np.arange(nk)
    mutJ=np.random.rand(4) < pmj
    mut=C.copy()
    u=np.random.rand(4)*(nk-1)
    sigma=1+np.random.rand(4)*(nk/6-1)
    travel=C.max(0)-C.min(0)
    d=np.random.rand(4)*2*travel-travel
    M=np.exp(-((rank.reshape(nk,1)-u)/(2*sigma))**2)
    mut[1:,mutJ]=(C+d*M)[1:,mutJ]
    return mut.clip(min_b,max_b)


if __name__ == '__main__':

    #try:
        
    point=sys.argv[1].split(',')
    angle=sys.argv[2].split(',')
    ob=np.array([float(i) for i in point])
    angle_in=np.array([float(i) for i in angle])
    print(ob)
    print(angle_in)
    nk=300
    l=list(cga(fitC,ob,bounds,angle_in,nk=nk))
    print(l[-1])
    print(np.sqrt(fitness._distance(l[-1][0][-1,:].ctypes.data_as(dataP),ob.ctypes.data_as(dataP))))
    plt.subplot(2,2,1)
    plt.plot(np.arange(nk),l[-1][0][:,0])
    plt.xlabel("pasos")
    plt.title("Joint 0")
    plt.ylabel("Angulo del Joint Rad")
    plt.subplot(2,2,2)

    plt.plot(np.arange(nk),l[-1][0][:,1])
    plt.xlabel("pasos")
    plt.title("Joint 1")
    plt.ylabel("Angulo del Joint Rad")

    plt.subplot(2,2,3)
    plt.plot(np.arange(nk),l[-1][0][:,2])
    plt.xlabel("pasos")
    plt.title("Joint 2")
    plt.ylabel("Angulo del Joint Rad")

    plt.subplot(2,2,4)
    plt.plot(np.arange(nk),l[-1][0][:,3])
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
        
