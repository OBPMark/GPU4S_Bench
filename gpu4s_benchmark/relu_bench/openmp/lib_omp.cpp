#include "../benchmark_library.h"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// Compute traditional relu approach 
	#pragma omp parallel for
	for (unsigned int i = 0; i < n; ++i)
	{
		for (unsigned int j = 0; j < n; ++j)
		{
			if (deviceObj->d_A[i*n+j] > 0)
			{
				deviceObj->d_B[i*n+j] = deviceObj->d_A[i*n+j];
			}
			else 
			{
				deviceObj->d_B[i*n+j] = 0;
			}
		}
	}

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


