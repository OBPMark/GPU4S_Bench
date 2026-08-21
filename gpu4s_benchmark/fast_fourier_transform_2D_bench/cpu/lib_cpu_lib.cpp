// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "../cpu_functions/cpu_functions.h"
#include <cstring>


//#define BLOCK_SIZE 32
void init(GraficCommon* device_object, char* device_name){
    init(device_object, 0,0, device_name);
}
void init(GraficCommon* device_object, int platform ,int device, char* device_name){
    strcpy(device_name,"Generic device");
}

bool device_memory_init(GraficCommon* device_object, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    deviceObj->d_B = (COMPLEX**) malloc(size * sizeof(COMPLEX*));
    // Allocate the actual 2D data block
    COMPLEX* d_B_data = (COMPLEX*) malloc(size * size * sizeof(COMPLEX));
    
    // Link the pointers to the data block
    for (int64_t i = 0; i < size; ++i) {
        deviceObj->d_B[i] = d_B_data + (i * size);
    }
	
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficCommon *device_object, COMPLEX **h_B,int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->d_A= h_B;
    
}

void execute_kernel(GraficCommon* device_object, int64_t size) {
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock kernelCLK;

    kernelCLK.start(); // Start clock
    FFT2D(deviceObj->d_A,size,size,deviceObj->d_B);
    kernelCLK.end(); // End clock
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}

void copy_memory_to_host(GraficCommon* device_object, COMPLEX **h_B, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    for (int64_t i = 0; i < size; ++i) {
        memcpy(h_B[i], deviceObj->d_B[i], size * sizeof(COMPLEX));
    }
}

float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0);
    } 
	else
	{
		//--- FIX: print te time in milliseconds
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time );
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return deviceObj->elapsed_time;
}

void clean(GraficCommon* device_object)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	free(deviceObj->d_B);
}