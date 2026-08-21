#include "../benchmark_library.h"
#include <cstring>
#include <cmath>


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	const unsigned int squared_size = n*n;
	
	#pragma omp parallel for
	for (unsigned int i = 0; i < squared_size; ++i)
	{
		deviceObj->d_B[i] = deviceObj->d_A[i]/pow((K+ALPHA*pow(deviceObj->d_A[i],2)),BETA);
	}

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}
