#include "../benchmark_library.h"
#include <cstring>


void init(GraficCommon* device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform, int device, char* device_name)
{
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

	
	deviceObj->d_A = (bench_t*) malloc ( size_a_matrix * sizeof(bench_t*));
	deviceObj->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	
   	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_A = h_A;
}


void execute_kernel(GraficCommon* device_object, unsigned int n)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	Clock kernelCLK;

	//Start compute timer
	kernelCLK.start();
	//copy data from d_A to d_B
	memcpy(deviceObj->d_B, deviceObj->d_A, n * sizeof(bench_t)); 
    // End compute timer
    kernelCLK.end();
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	//Must have memcpy otherwise, we need to modify the prototype of the function with &
	memcpy(h_C, deviceObj->d_B, size * sizeof(bench_t));	     
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
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
	free(deviceObj->d_B);
}