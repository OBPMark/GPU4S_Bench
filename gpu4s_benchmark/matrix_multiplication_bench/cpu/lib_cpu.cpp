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
	deviceObj->d_C = (bench_t*) malloc ( size_c_matrix * sizeof(bench_t*));
   	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_A = h_A;
	deviceObj->d_B = h_B;
}


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w)
{
	//Fix: cast to merge all the prototype in on single file 
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);   
	// Start compute timer
	Clock kernelCLK;
	kernelCLK.start();

	// Compute traditional matrix multiplication approach 
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
	kernelCLK.end();
	deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);	
	memcpy(h_C, &deviceObj->d_C[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",(bench_t) 0, deviceObj->elapsed_time , (bench_t) 0, current_time);
    }
    else if (csv_format){
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
    return deviceObj->elapsed_time;
}


void clean(GraficCommon* device_object)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	free(deviceObj->d_C);
}