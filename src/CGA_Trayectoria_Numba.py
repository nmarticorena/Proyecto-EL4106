#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
from numba import jit,vectorize,guvectorize, prange
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import Energy
try:
    from tqdm import tqdm
except:
    print("Falle importando la libreria tdqm, si quieres visualizar mediante una barra de progreso favor instalarla")

# configuracion graficos
sns.set()  # estilo por defecto de seaborn
sns.set_context('talk', font_scale=0.8) # contexto notebook 
sns.set_style("whitegrid")
#plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = (7,7)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
#plt.style.use('seaborn-white')
Masa=5

phi=0.001
sigma=-1000
#PARAMS
pi=np.pi
bounds=[(-5, 5), (-pi/2, pi/2), (-pi/2, pi/2), (-pi/2, pi/2)]
nk=20

best_error_np=np.array([])
best_fError_np=np.array([])
best_fEnergia_np=np.array([])
best_E_np=np.array([])


#Diferential Evolution
def cga(fobj,obj, min_b,max_b,angle_in, mut=0.1, crossp=0.6,nk=50, popsize=2000, its=4000,pcj=0.5,pmj=1.0):
    global cambio_fitness
    global errores_mejor
    global best_error_np
    global best_fEnergia_np
    global best_fError_np
    global best_E_np

    errores_mejor=np.zeros(its)
    cambio_fitness=np.zeros(its)
    dfitness=0;
    pop=initGauss(min_b,max_b,nk,popsize,angle_in)
    #pop=initanh(bounds,nk,popsize,angle_in)
    Pfitness=np.zeros(popsize)
    fDistancia=np.zeros(popsize)
    fEnergia=np.zeros(popsize)
    fobj(pop,obj,angle_in,nk,Pfitness,fDistancia,fEnergia)
    best_idx = np.argmax(Pfitness)
    best = pop[best_idx].copy()
    hall_of_fame=best.copy()
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
            fobj(pop,obj,angle_in,nk,Pfitness,fDistancia,fEnergia)
            best_idx=np.argmax(Pfitness)
            best=pop[best_idx].copy()
            best_error,best_energia=fit_grafico(best,ob,angle_in,nk)
            best_error_np=np.append(best_error_np,best_error)
            best_fError_np=np.append(best_fError_np,fDistancia[best_idx])
            best_fEnergia_np=np.append(best_fEnergia_np,fEnergia[best_idx])
            best_E_np=np.append(best_E_np,best_energia)

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

            #if Pfitness[best_idx]>=2.0-0.001:
            #    print("alcance un fitness de:{}".format(Pfitness[best_idx]))
            #    termino=True
            #print(i)
            errores_mejor[i]=Pfitness[best_idx]
            dfitness+=(its*(errores_mejor[i]-errores_mejor[i-1])/1-dfitness)/8
            dfitness_ang=np.arctan(dfitness)
            #mut=0.4-0.6/pi*dfitness_ang
            cambio_fitness[i]=dfitness_ang
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
    T[:,3]+=0.6*T[:,2]
    return np.sum(np.square(T[:,3]-objective))

# @jit(nopython=True)
# def fitC(ind,ob,angle_in,nk,a,b,c):
#     global Masa
#     global gamma
#     res=0;
#     0.25
#     res+=fitDistance(ind[-1,:],ob)*(1000)
#     if (np.sqrt(res)<0.005):
#         res=res
#     else:
#         res=res

#     res=1/(1+res)
    
#     Energia=Energy.fitness(ind)
#     if (Energia<0.0001):
#         res2=0    
#     else:
#         res2=1/(1+Energia/(125*5))
#     #Energia=gamma*Energia
    


#     return res*res2
#     #return (res*f[0]+Energia*f[1])*(res*Energia)

@jit(nopython=True)
def fit_grafico(ind,ob,angle_in,nk):
    global Masa
    distancia=np.sqrt(fitDistance(ind[-1,:],ob))
    Energia=Energy.fitness(ind)
    #res+=np.sum(np.abs(angle_in-ind[0,:]))
    #print("funcion de diferencia final {}".format(res))
    #print("Suma de errores {}".format(res2))
    return distancia,Energia

@jit(['void(float64[:,:,:], float64[:],float64[:],int64,float64[:],float64[:],float64[:])'],nopython=True)
def fitPar(pop,ob,angle_in,nk,fit,fDistancia,fEnergia):
    global Masa,phi,sigma
    popsize=pop.shape[0]
    distancia=np.zeros(popsize)
    energia=np.zeros(popsize)
    for j in range(popsize):
        energia[j]=Energy.fitness(pop[j])
        #print(energia[j])
        distancia[j]=fitDistance(pop[j][-1,:],ob)
    Emin=np.min(energia)
    Emean=np.mean(energia)
    logPhi=np.log(phi)
    fEnergia[:]=np.exp((logPhi/(Emean- Emin))*(energia-Emin))
    #print(fEnergia)
    fDistancia[:]=np.exp(sigma*distancia)
    fit[:]=fDistancia*(1+fEnergia)


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
    C2=P2.copy()
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
    angle_in=np.deg2rad(angle_in)
    print(ob)
    print(angle_in)
    nk=50
    min_b, max_b = np.array(bounds).T
    l=list(cga(fitPar,ob,min_b,max_b,angle_in,nk=nk,its=10000,popsize=1000))
    print(l[-1])
    for i in range(100):
        energia=Energy.fitness(l[i][0])
        #print(energia)
    print(np.sqrt(fitDistance(l[-1][0][-1,:],ob)))

    plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(best_error_np)
    #plt.ylim([0,0.01])
    plt.ylim([0,1.1])
    plt.title("Error")
    plt.ylabel("Error")

    plt.subplot(2,2,2)
    plt.plot(best_E_np)
    #plt.ylim([0,1.1])
    plt.xlabel("pasos")
    plt.title("Energia")
    plt.ylabel("Unidad de energia")

    plt.subplot(2,2,3)
    plt.plot(best_fError_np)
    plt.ylim([0,1.1])
    plt.xlabel("interaciones")
    plt.title("fError")
    plt.ylabel("fitness Error")
    
    plt.subplot(2,2,4)
    plt.plot(best_fEnergia_np)
    plt.ylim([0,1])
    plt.xlabel("interaciones")
    plt.title("fEnergia")
    plt.ylabel("fitness Energia")

    



    Joint0rad=l[-1][0][:,0]
    Joint1rad=l[-1][0][:,1]
    Joint2rad=l[-1][0][:,2]
    Joint3rad=l[-1][0][:,3]

    Joint0=np.rad2deg(Joint0rad)
    Joint1=np.rad2deg(Joint1rad)
    Joint2=np.rad2deg(Joint2rad)
    Joint3=np.rad2deg(Joint3rad)

    plt.figure(2)
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

    plt.figure(3)
    plt.plot(errores_mejor)
    plt.xlabel("iteraciones")
    plt.title("fitness mejor individuo")
    plt.ylabel("fitness")

    plt.figure(4)
    plt.plot(cambio_fitness)
    plt.xlabel("iteraciones")
    plt.title("Derivada fitness")
    plt.ylabel("dfitness/di")
    plt.show()

    #except:

    point =''
    print("Debe entregar objetivo de la forma: x,y,z")
        
