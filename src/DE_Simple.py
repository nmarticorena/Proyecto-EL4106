#!/usr/bin/env python
from __future__ import print_function
import sys
import numpy as np
import fitness
from ctypes import POINTER,c_double
import matplotlib.pyplot as plt
import seaborn as sns

# configuracion graficos
sns.set()  # estilo por defecto de seaborn
sns.set_context('notebook', font_scale=1.5) # contexto notebook 
plt.rcParams['figure.figsize'] = (6,6)

dataP=POINTER(c_double)



#Diferential Evolution
def de(fobj,obj, bounds, mut=0.8, crossp=0.7, popsize=20, its=100):
    global errores_mejor
    errores_mejor=np.zeros(its)
    pop = np.random.rand(popsize, 4)
    min_b, max_b = np.asarray(bound).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    Pfitness = np.asarray([fobj(ind,obj) for ind in pop_denorm])
    best_idx = np.argmin(Pfitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        #Elegir padres
        ina=derangementN(popsize)
        inb=derangementN(popsize)
        inc=derangementN(popsize)
        #crear mutantes
        mut=np.clip((pop[ina]+mut*(pop[inb]-pop[inc])),0,1)
        cross_points = np.random.rand(popsize,4) < crossp
        #Ver cross
        f=np.nonzero(cross_points.sum(1)==0)
        cross_points[f,np.random.randint(4,size=len(f))]=1
        trial=np.where(cross_points,mut,pop)
        #Probar nueo fitness
        trial_denorm=min_b + trial * diff
        #fitnessT=np.apply_along_axis(fobjv,1,trial_denorm)
        fitnessT=np.zeros(popsize)
        for j in range(popsize):
            fitnessT[j]=fobj(trial_denorm[j],obj)
        #Actualizar generacion
        test=fitnessT<Pfitness
        Pfitness=np.where(test,fitnessT,Pfitness)
        indn=test.nonzero()
        pop[indn]=trial[indn]
        best_idx = np.argmin(Pfitness)
        pop_denorm = min_b + pop * diff
        best = pop_denorm[best_idx]
        #print(best)
        errores_mejor[i]=Pfitness[best_idx]
        yield best, Pfitness[best_idx]

def derangementN(n):
    v=np.arange(n)
    num=np.arange(n)
    while True:
        np.random.shuffle(v)
        if np.all((v-num)!=0):
            break
    return v
    
#Funcion de fitness
def fitC(ind,ob):
    return fitness._distance(ind.ctypes.data_as(dataP),ob.ctypes.data_as(dataP))

ob=np.array([0.7,0.7,1])
pi=np.pi
bound=[(-5, 5), (-pi/2, pi/2), (-pi/2, pi/2), (-pi/2, pi/2)]

if __name__ == '__main__':
    try:
        point = sys.argv[1].split(',')
        ob=np.array([float(i) for i in point])
        print(ob)
        l=list(de(fitC,ob,bound))
        print(l[-1])
        #print(errores_mejor)
        plt.plot(errores_mejor)
        plt.xlabel("iteraciones")
        plt.title("Error cartesiano de posicion de mejor individuo")
        plt.ylabel("Error cartesiano")
        plt.show()

    except:
        point =''
        print("Debe entregar objetivo de la forma: x,y,z")
    
