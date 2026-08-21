#include "../benchmark_library.h"


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	int kernel_rad = kernel_size / 2;

	#pragma omp parallel for
	for (unsigned int block = 0; block < n*n; ++block)
	{
		const unsigned int x = block/n;
		const unsigned int y = block%n;
		bench_t sum = 0;
		for(int i = -kernel_rad; i <= kernel_rad; ++i)
		{
			for(int j = -kernel_rad; j <= kernel_rad; ++j){
				bench_t value = 0;
				if (i + x < 0 || j + y < 0)
				{
					value = 0;
				}
				else if ( i + x > n - 1 || j + y > n - 1)
				{
					value = 0;
				}
				else
				{
					value = deviceObj->d_A[(x + i)*n+(y + j)];
				}
				sum += value * deviceObj->kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
			}
		}			
		deviceObj->d_B[x * n + y] = sum;
	}

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}

