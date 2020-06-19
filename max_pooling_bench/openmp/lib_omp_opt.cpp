#include "../benchmark_library.h"
#include <cstring>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a)
{
	device_object->d_A = h_A;
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int lateral_stride)
{
		// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	bench_t max_value = 0;
	const unsigned int block_size = n/stride;

	#pragma omp parallel for private(max_value)
	for (unsigned int block = 0; block < block_size*block_size; ++block)
	{
		{
			const unsigned int blockx = block%block_size;
			const unsigned int blocky =	block/block_size;
			const unsigned int block_zero = blockx*stride + blocky*stride*n;
			max_value = device_object->d_A[block_zero];		
			for(unsigned int x = 0; x < stride; ++x)
			{
				for(unsigned int y = 0; y < stride; ++y)
				{
					max_value = max(max_value, device_object->d_A[(block_zero+x) + y*n]);
				}
			}
			device_object->d_B[block] = max_value;	
		}
	}

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;

}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format)
{
	if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f miliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f miliseconds\n", device_object->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f miliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_B);
}