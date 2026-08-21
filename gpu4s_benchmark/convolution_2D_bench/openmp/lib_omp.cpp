#include "../benchmark_library.h"


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	int kernel_rad = kernel_size / 2;
	int x, y, kx, ky = 0;
	bench_t sum = 0;
	bench_t value = 0;

	const unsigned int squared_kernel_size = kernel_size * kernel_size;
	
	#pragma omp parallel for private(x, y, kx, ky, sum, value)
	for (unsigned int block = 0; block < n*n; ++block)
	{
		x = block/n;
		y = block%n;
		sum = 0;
		for(unsigned int k = 0; k < squared_kernel_size; ++k)
		{
			value = 0;
			kx = (k/kernel_size) - kernel_rad; 
			ky = (k%kernel_size) - kernel_rad;
			if(!(kx + x < 0 || ky + y < 0) && !( kx + x > n - 1 || ky + y > n - 1))
			{
				value = deviceObj->d_A[(x + kx)*n+(y + ky)];
			}
			sum += value * deviceObj->kernel[(kx+kernel_rad)* kernel_size + (ky+kernel_rad)];
		}
		deviceObj->d_B[x*n+y] = sum;
	}

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}

