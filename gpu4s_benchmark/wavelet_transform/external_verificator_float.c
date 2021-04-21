#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define NUMBER_BASE 1

#define INT


# ifdef FLOAT
const float  LowPassFilter[] = {	
0.037828455507,
-0.023849465020,
-0.110624404418,
0.377402855613, 
0.852698679009,
0.377402855613, 
-0.110624404418,
-0.023849465020,
0.037828455507};

const float  HighPassFilter[] = {
-0.064538882629,
0.040689417609,
0.418092273222,
-0.788485616406,
0.418092273222,
0.040689417609,
-0.064538882629};

static float *x_alloc = NULL;	//** temp work
#define F_EXTPAD 4


void forwardf97f(float *x_in, int N)
{
	int i;
	int n; 
	int half;
	float *x; 
	float *r;
	float *d;

	x = x_alloc + F_EXTPAD;

	memcpy(x,x_in,sizeof(float)*N);

	
	half = (N >> 1);  
	d = malloc(sizeof(float)*(half + 3));
	r = malloc(sizeof(float)*(half + 2));
	
	for(i = 1; i <= F_EXTPAD; i++) 
	{
		x[-i] = x[i];
		x[(N-1) + i] = x[(N-1) - i];
	}
	if( 1) 
	{
		float const *LPF = LowPassFilter + 4;
		float const *HPF = HighPassFilter + 3;

		for (n = 0; n < half; n++)
		{
			d[n] = 0;
			r[n] = 0;

			d[n]= (float)
				(  ( LPF[0] * x[2 * n] )
				+ ( LPF[1] * (x[2 * n -1] + x[2 * n + 1] ))
				+ ( LPF[2] * (x[2 * n -2] + x[2 * n + 2]) )
				+ ( LPF[3] * (x[2 * n -3] + x[2 * n + 3]) )
				+ ( LPF[4] * (x[2 * n -4] + x[2 * n + 4])) );
			
			r[n] = (float)(
				(HPF[0] * x[2 * n + 1] )
			+ ( HPF[1]  * ( x[2 * n ] + x[2 * n + 2]))
			+ ( HPF[2] * (x[2 * n -1] + x[2 * n + 3]))
			+ ( HPF[3] * (x[2 * n -2] + x[2 * n + 4]) ));

		}

	} 

	memcpy(x_in,d,sizeof(float)*half);
	memcpy(x_in+half,r,sizeof(float)*half);

}
#endif
#ifdef INT
#define F_EXTPAD 4
#define D_EXTPAD 2

static int *x_alloc = NULL;	//** temp work

void forwardf97M(int *x_in, 
				 int N)
{
	int i; 
	int n;
	int half;
	int *x;
	int	*r;
	int *d;
	int *temp_0;
	int *temp_1;
	double temp; 

	x = x_alloc + F_EXTPAD;
	memcpy(x,x_in,sizeof(int)*N);
	for(i=1;i<=F_EXTPAD;i++) 
	{
		x[-i] = x[i];
		x[(N-1) + i] = x[(N-1) - i];
	}

	half = (N>>1);  	
	d = malloc(sizeof(int)*(half + 1));
	temp_0 = d;

	for (n=half + 1;n > 0; n--) 
	{
		temp = (-1.0/16 * (x[-4] + x[2]) + 9.0/16 * (x[-2] + x[0]) + 0.5);
		if (temp > 0) 
			temp = (int)temp;
		else 
		{
			if (temp != (int)temp) 
				temp = (int)(temp - 1);
		}

		*d++ = x[-1] - (int)temp;
		x += 2;
	}
	d = temp_0;
	r = malloc(sizeof(int)*(half));
	temp_1 = r;
	x = x_alloc + F_EXTPAD;
	for (n=half;n> 0; n--) 
	{
		temp = -.25 * (d[0] + d[1]) +.5;
		if (temp > 0) 
			temp = (int)temp;
		else 
		{
			if (temp != (int)temp) 
				temp = (int)(temp - 1);
		}

		*r++ = x[0] - (int)(temp);
		x+=2;
		d++;
	}
	d = temp_0;
	r = temp_1;
	memcpy(x_in,r, sizeof(int)*half);
	memcpy(x_in + half,d+1, sizeof(int)*half);	
	
	free(temp_0);
	free(temp_1);
}

#endif
int main(int argc, char *argv[]){
    srand (21121993);
    unsigned int size = 64;
	// A input matrix
	unsigned int size_A = size;
	#ifdef FLOAT
    unsigned int mem_size_A = sizeof(float) * size_A;
	float* A = (float*) malloc(mem_size_A);
	
    
	for (int i=0; i<size; i++){
	    A[i] = (float)rand()/(float)(RAND_MAX/NUMBER_BASE);
        printf("%f,", A[i]);
	}
	printf("\n");
	
	x_alloc = malloc(sizeof(float)*(size+F_EXTPAD+F_EXTPAD));
    forwardf97f(A, size);
    // print the output
	for (int i=0; i<size; i++){
        printf("%f,", A[i]);
	}
	#endif
	#ifdef INT
	unsigned int mem_size_A = sizeof(int) * size_A;
	int* A = (int*) malloc(mem_size_A);
	
	for (int i=0; i<size; i++){
	    A[i] = rand() % (NUMBER_BASE * 100);
        printf("%d,", A[i]);
	}
	printf("\n");
	
	x_alloc = malloc(sizeof(int)*(size+F_EXTPAD+F_EXTPAD));
    forwardf97M(A, size);
    // print the output
	for (int i=0; i<size; i++){
        printf("%d,", A[i]);
	}
	#endif
	printf("\n");


}