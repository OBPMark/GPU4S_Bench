#include "../benchmark_library.h"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// Compute traditional matrix multiplication approach 
	#pragma omp parallel for
	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < w; j++)
		{
			for (unsigned int k = 0; k < m; k++)
			{   
				deviceObj->d_C[i*n+j] = deviceObj->d_C[i*n+j] + deviceObj->d_A[i*n+k] * deviceObj->d_B[k*w+j];
			}
		}
	}

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


