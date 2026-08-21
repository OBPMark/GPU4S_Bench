#include "../benchmark_library.h"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// Compute optimized relu
	#pragma omp parallel for
	for (unsigned int i = 0; i < n*n; ++i)
	{
		deviceObj->d_B[i] = deviceObj->d_A[i] > 0 ? deviceObj->d_A[i] : 0;
	}

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


