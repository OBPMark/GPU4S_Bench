#include "../benchmark_library.h"
#include <cstring>

void init(GraficCommon* device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t));
   	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_A = h_A;
	deviceObj->kernel = kernel;
}


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	const unsigned int kernel_rad = kernel_size / 2;
	const unsigned int output_size = n + kernel_size - 1;

	#pragma omp parallel for
	for(unsigned int i = 0; i < output_size; ++i)
	{
		for (unsigned int j = 0; j < kernel_size; ++j)
		{		 
			if (i +(j - kernel_size + 1) >= 0 && i +(j - kernel_size +1)<  n)
    		{	
    			deviceObj->d_B[i] += deviceObj->kernel[kernel_size - j - 1] * deviceObj->d_A[i +(j - kernel_size + 1) ];
    		}
		}
	}
	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);	     
	memcpy(h_C, &deviceObj->d_B[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",(bench_t) 0, deviceObj->elapsed_time , (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return deviceObj->elapsed_time * 1000.f;
}


void clean(GraficCommon* device_object)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	free(deviceObj->d_B);
}