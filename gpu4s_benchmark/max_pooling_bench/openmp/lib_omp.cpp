#include "../benchmark_library.h"
#include <cstring>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })



void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int lateral_stride)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	bench_t max_value = 0;
	const unsigned int block_size = n/stride;
	const unsigned int stride_squared = stride*stride;
	unsigned int blockx, blocky, block_zero, x, y = 0;

	#pragma omp parallel for private(max_value,blockx, blocky, block_zero, x, y)
	for (unsigned int block = 0; block < block_size*block_size; ++block)
	{
		{
			blockx = block%block_size;
			blocky = block/block_size;
			block_zero = blockx*stride + blocky*stride*n;
			max_value = deviceObj->d_A[block_zero];		
			for(unsigned int i = 0; i < stride_squared; ++i)
			{
				x = i%stride;
				y = i/stride; 
				max_value = max(max_value, deviceObj->d_A[(block_zero+x) + y*n]);
			}
			deviceObj->d_B[block] = max_value;	
		}
	}

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


