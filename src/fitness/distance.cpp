#include <cmath>
#include <stdio.h>
#define JOINTL 0.6

void roty(double TA[4][4],double theta) {
    double cosT=cos(theta);
    double sinT=sin(theta);
    for (int i = 0; i < 3; ++i)
    {
    	double a=TA[i][0];
    	double c=TA[i][2];
    	TA[i][0]=a*cosT-c*sinT;
    	TA[i][2]=a*sinT+c*cosT;
    	TA[i][3]+=JOINTL*c;
    }
}

void prinT(double TT[4][4]) {
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			printf("%f ",TT[i][j] );
		}
		printf("%s\n","|" );
	}
	printf("%s\n","**************" );
	
}

extern "C" {

//Fitness: Calculates the fitness of the individual according to the objective distance
double distance(double *individual,double *objective) {
	double cosT=cos(individual[0]);
	double sinT=sin(individual[0]);
	double T[4][4]={
		{cosT  ,-sinT ,0 ,0},
		{sinT  ,cosT  ,0 ,0},
		{0     ,0     ,1 ,0.1},
		{0     ,0     ,0 ,1}};
	//Multiplicar 3 links iguales
	roty(T,individual[1]);
	roty(T,individual[2]);
	roty(T,individual[3]);
	//joint final
	T[0][3]+=0.5*T[0][2];
	T[1][3]+=0.5*T[1][2];
	T[2][3]+=0.5*T[2][2];
	//Calcular distancia
	double dx=T[0][3]-objective[0];
	double dy=T[1][3]-objective[1];
	double dz=T[2][3]-objective[2];
	return dx*dx+dy*dy+dz*dz;
}

}