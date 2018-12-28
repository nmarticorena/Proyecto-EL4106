from __future__ import print_function
import numpy as np
from numba import jit,vectorize,guvectorize, prange
M=5
l=0.5
g=9.8

@jit(['float64[:](float64[:],float64[:])'],nopython=True)
def sin_vector(vector,sin):
	sin=np.sin(vector)
	return sin
	
@jit(['void(float64[:],float64[:])'],nopython=True)
def cos_vector(vector,cos):
	cos[:3]=np.cos(vector[:3])
	cos[3]=np.cos(vector[5])

@jit(['float64(float64[:],float64[:],float64[:],float64[:],float64[:])'],nopython=True)
def torque1(theta,vel,acc,sin,cos):
	global M,l,g
	res1=-1.*g*(2*sin[3]+sin[4]+3*sin[0])
	res2=(vel[2]+vel[3])*sin[5]*(-2*vel[1]-vel[2]-vel[3])
	res3=-1*(sin[1]*(4*vel[1]*vel[2]+2*vel[2]**2)+sin[2]*(2*vel[1]*vel[3]+2*vel[2]*vel[3]+vel[3]**2))
	res4=acc[1]*(2*cos[5]+4*cos[1]+2*cos[2]+6)+acc[2]*(cos[5]+2*cos[1]+2*cos[2]+3)+acc[3]*(cos[5]+cos[2]+1)

	res=M*l*(res1+res2+res3+res4)
	return res

@jit(['float64(float64[:],float64[:],float64[:],float64[:],float64[:])'],nopython=True)
def torque2(theta,vel,acc,sin,cos):
	global M,l,g
	t1=-2*g*sin[3]-g*sin[4]+l*sin[5]*(vel[1])**2+2*l*sin[1]*(vel[1])**2-sin[2]*l*vel[3]*(vel[1]+vel[2]+vel[3])
	t2=l*cos[5]*acc[1]+2*l*cos[1]*acc[1]+2*l*cos[2]*(acc[1]+acc[2]+0.5*acc[3])+l*(3*acc[1]+3*acc[2]+acc[3])
	res=M*l*(t1+t2)
	return res


@jit(['float64(float64[:],float64[:],float64[:],float64[:],float64[:])'],nopython=True)
def torque3(theta,vel,acc,sin,cos):
	global M,l,g
	res=M*l*(-1.*g*sin[4]+l*sin[5]*acc[1]**2+sin[2]*l*(vel[1]+vel[2])**2+l*cos[5]*acc[1]+l*cos[2]*(acc[1]+acc[2])+l*(acc[1]+acc[2]+acc[3]))
	return res

@jit(['float64(float64[:,:])'],nopython=True)
def fitness(ind):
	global M
	nk=ind.shape[0]
	sin=np.zeros(6)
	cos=np.zeros(4)
	theta=np.zeros(6) #12,123,23
	acc=np.zeros(4) # acc 1 acc 2 acc 3
	vel=ind[1,:]-ind[0,:]
	t1=0.0
	t2=0.0
	t3=0.0

	t0=0.0
	energia=0

	for i in range(1,nk-1):
		vel_2=ind[i+1,:]-ind[i,:]
		acc=vel_2-vel
		theta[:3]=ind[i,1:]
		theta[3]=theta[0]+theta[1]
		theta[4]=theta[3]+theta[2]
		theta[5]=theta[1]+theta[2]
		sin=sin_vector(theta,sin)
		cos_vector(theta,cos)
		t1=np.abs(torque1(theta,vel,acc,sin,cos)*vel[1])
		t2=np.abs(torque2(theta,vel,acc,sin,cos)*vel[2])
		t3=np.abs(torque3(theta,vel,acc,sin,cos)*vel[3])
		
		t0+=3*M*vel[0]**2
		#print("===================")
		#print(t0)
		#print(t1)
		#print(t2)
		#print(t3)		
		energia+=np.abs(t1+t2+t3+t0)
		vel=vel_2

	#print(energia)
	return np.abs(energia)



if __name__ == '__main__':
    import pandas as pd
    df=pd.read_csv("./file_path{}.csv".format(10000))
    ind=df.values
    print(ind[1:,1:])
    res=fitness(ind[1:,1:])
    print(res)