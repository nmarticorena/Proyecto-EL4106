#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
from numba import jit,vectorize,guvectorize, prange
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
Masa=5

#PARAMS
pi=np.pi
bounds=[(-5, 5), (-pi/2, pi/2), (-pi/2, pi/2), (-pi/2, pi/2)]
nk=20


#Diferential Evolution
def cga(fobj,obj, min_b,max_b,angle_in, mut=0.1, crossp=0.6,nk=50, popsize=1000, its=3000,pcj=0.5,pmj=1.0):
    global errores_mejor
    errores_mejor=np.zeros(its)
    pop=initGauss(min_b,max_b,nk,popsize,angle_in)
    #pop=initanh(bounds,nk,popsize,angle_in)
    Pfitness=np.zeros(popsize)
    fobj(pop,obj,angle_in,nk,Pfitness)
    best_idx = np.argmax(Pfitness)
    best = pop[best_idx]
    hall_of_fame=best
    hall_of_fame_f=Pfitness[best_idx];
    termino=False
    elite_total=int(popsize/10)
    childs=np.zeros((popsize,nk,4))
    try:
        a=tqdm(range(its))
    except:
        a=range(its)

    for i in a:
        if termino:
            errores_mejor[i]=Pfitness[best_idx]
            yield best,Pfitness[best_idx]
        else:
            fitnessRel=Pfitness/np.sum(Pfitness)

            aceptados=0
            aceptados_array=np.random.choice(popsize, popsize, p=fitnessRel)
            
            makeChilds(pop,popsize,Pfitness,aceptados_array,childs,min_b,max_b,elite_total,mut,crossp,pcj,pmj)

            pop=childs.copy()
            fobj(pop,obj,angle_in,nk,Pfitness)
            best_idx=np.argmax(Pfitness)
            best=pop[best_idx]
            if i%10==0:
                df = pd.DataFrame(best)
                df.to_csv("trayectorias/file_path{}.csv".format(i))
            if i==its-1:
                df = pd.DataFrame(best)
                df.to_csv("trayectorias/file_path{}.csv".format(its))
            if hall_of_fame_f<Pfitness[best_idx]:
                #print("Encontre uno mejor con fitness igual a {}".format(Pfitness[best_idx]))
                #print("alcance un fitness de:{} con promedio {}".format(Pfitness[best_idx],np.mean(Pfitness)))
                hall_of_fame=best.copy()
                hall_of_fame_f=Pfitness[best_idx]

            if Pfitness[best_idx]>=2.0-0.001:
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

@jit(['void(float64[:,:], float64[:], float64[:], float64[:,:])'],nopython=True)
def clip2(a,minv,maxv,out):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i,j] > maxv[j]:
                out[i,j] = maxv[j]
            elif a[i,j] <minv[j]:
                out[i,j] = minv[j]
            else :
                out[i,j]= a[i,j]
    
@jit(['void(float64[:,:,:], float64[:], float64[:], float64[:,:,:])'],nopython=True)
def clip3(a,minv,maxv,out):
    for i in range(a.shape[0]):
        for t in range(a.shape[1]):
            for j in range(a.shape[2]):
                if a[i,t,j] > maxv[j]:
                    out[i,t,j] = maxv[j]
                elif a[i,t,j] <minv[j]:
                    out[i,t,j] = minv[j]
                else :
                    out[i,t,j]= a[i,t,j]


#Funcion de fitness

@jit(['float64(float64[:], float64[:])'],nopython=True)
def fitDistance(individual,objective):
    cosT=np.cos(individual)
    sinT=np.sin(individual)
    T=np.array([[cosT[0],-sinT[0], 0, 0   ],
                   [sinT[0],cosT[0] , 0, 0   ],
                   [0      ,0       , 1, 0.1]])
    for j in range(1,4):
        for i in range(3):
            a=T[i,0]
            c=T[i,2]
            T[i,0]=a*cosT[j]-c*sinT[j]
            T[i][2]=a*sinT[j]+c*cosT[j]
            T[i][3]+=0.6*c
    T[:,3]+=0.5*T[:,2]
    return np.sum(np.square(T[:,3]-objective))

@jit(nopython=True)
def fitC(ind,ob,angle_in,nk):
    global Masa
    res=0;
    0.25
    res+=fitDistance(ind[-1,:],ob)*(1000)
    res2=0
    for i in range(nk-2):
        res2+=np.abs((np.abs(ind[i,0]-ind[i+1,0]))-((np.abs(ind[i+1,0]-ind[i+2,0]))))*(2.0) #quizas ponerlo en 5
        res2+=np.abs((np.abs(ind[i,1]-ind[i+1,1]))-((np.abs(ind[i+1,1]-ind[i+2,1]))))*(3.0/2.0)
        res2+=np.abs((np.abs(ind[i,2]-ind[i+1,2]))-((np.abs(ind[i+1,2]-ind[i+2,2]))))*(1)
        res2+=np.abs((np.abs(ind[i,3]-ind[i+1,3]))-((np.abs(ind[i+1,3]-ind[i+2,3]))))*(1.0/2.0)
    #print(res,res2)
    res2=res2
    #res+=np.sum(np.abs(angle_in-ind[0,:]))
    #print("funcion de diferencia final {}".format(res))
    #print("Suma de errores {}".format(res2))
    return (1/(1+res))*(1/(1+res2))

@guvectorize(['void(float64[:,:,:], float64[:],float64[:],int64, float64[:])'], '(n,k,m),(p),(m),()->(n)',target='cpu')
def fitPar(pop,ob,angle_in,nk,fit):
    for j in range(pop.shape[0]):
            fit[j]=fitC(pop[j],ob,angle_in,nk)

@jit(['float64[:,:,:](float64[:],float64[:], int64, int64, float64[:])'],nopython=True)
def initGauss(min_b,max_b,nk,popsize,angle_in):
    diff = np.fabs(min_b - max_b)
    #ends=min_b+np.random.rand(popsize,2,4)*diff
    start=(np.ones((popsize,1,4)))*angle_in
    end=min_b+np.random.rand(popsize,1,4)*diff
    #print(ends.shape)
    #print(ends)
    rank=np.arange(nk)
    travel=start-end
    pop=(travel.reshape(popsize,1,4)*rank.reshape((1,nk,1)))/(nk-1)
    pop+=start
    A=(6*np.abs(travel)*np.random.rand(popsize,1,4)-3*np.abs(travel))
    u=np.random.rand(popsize,1,4)*(nk-1)
    sigma=(1+np.random.rand(popsize,1,4)*(nk/6-1))
    pop[:,1:,:]+=(A*np.exp(-((rank.reshape(1,nk,1)-u)/(2*sigma))**2))[:,1:,:]
    clip3(pop,min_b,max_b,pop)
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

@jit(nopython=True,fastmath=True)
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

@jit(['void(float64[:,:], float64[:])'],nopython=True)
def getTravel(a,out):
    for i in range(a.shape[1]):
        out[i]=a[:,i].max()-a[:,i].min()

@jit(['float64[:,:](float64[:,:],float64[:],float64[:],float64)'],nopython=True,fastmath=True)
def mutation(C,min_b,max_b,pmj):
    nk=C.shape[0]
    rank=np.arange(nk)
    mutJ=np.random.rand(4) < pmj
    mut=C.copy()
    u=np.random.rand(4)*(nk-1)
    sigma=1+np.random.rand(4)*(nk/6-1)
    travel=np.zeros(4)
    getTravel(C,travel)
    d=np.random.rand(4)*2*travel-travel
    M=np.exp(-((rank.reshape(nk,1)-u)/(2*sigma))**2)
    mut[1:,mutJ]=(C+d*M)[1:,mutJ]
    clip2(mut,min_b,max_b,mut)
    return mut

@jit(['void(float64[:,:,:],int64,float64[:], int64[:], float64[:,:,:],float64[:],float64[:], int64, float64,float64,float64,float64)'],nopython=True,fastmath=True,parallel=True)
def makeChilds(pop,popsize,Pfitness,aceptados_array,childs,min_b,max_b,elite_total,mut, crossp, pcj,pmj):
    
    elite_index=np.argsort(Pfitness)[-1*elite_total:]
    childs[0:elite_total]=pop[elite_index]

    for j in range(int((popsize/2)-elite_total)):
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
            C1=mutation(C1,min_b,max_b,pmj)
        if np.random.rand()<mut:
            C2=mutation(C2,min_b,max_b,pmj)
        childs[j*2+elite_total]=C1.copy()
        childs[j*2+1+elite_total]=C2.copy()


if __name__ == '__main__':

    #try:
        
    point=sys.argv[1].split(',')
    angle=sys.argv[2].split(',')
    ob=np.array([float(i) for i in point])
    angle_in=np.array([float(i) for i in angle])
    print(ob)
    print(angle_in)
    nk=50
    min_b, max_b = np.array(bounds).T
    l=list(cga(fitPar,ob,min_b,max_b,angle_in,nk=nk))
    print(l[-1])
    print(np.sqrt(fitDistance(l[-1][0][-1,:],ob)))
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
        
