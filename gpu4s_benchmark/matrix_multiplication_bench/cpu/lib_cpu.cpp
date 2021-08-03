#include "../benchmark_library.h"
#include <cstring>

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix) 
{
	device_object->d_C = (bench_t*) malloc ( size_c_matrix * sizeof(bench_t*));
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b)
{
	device_object->d_A = h_A;
	device_object->d_B = h_B;
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w)
{   
    struct timespec start, end;
	// Start compute timer
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	// Compute traditional matrix multiplication approach 
	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < w; j++)
		{
			for (unsigned int k = 0; k < m; k++)
			{   
				device_object->d_C[i*n+j] = device_object->d_C[i*n+j] + device_object->d_A[i*n+k] * device_object->d_B[k*w+j];
			}
		}
	}

	// End compute timer
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    device_object->elapsed_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	memcpy(h_C, &device_object->d_C[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",(bench_t) 0, device_object->elapsed_time , (bench_t) 0, current_time);
    }
    else if (csv_format){
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time );
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
    return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_C);
}