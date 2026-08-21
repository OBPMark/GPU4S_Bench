/** * ====================================================================
 * @file        omp_common.cpp (./fast_fourier_transform_bench)
 * @brief       Common OpenMP platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"
#include <cstring>

void init(GraficCommon* device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}

__attribute__((weak))
bool device_memory_init(GraficCommon* device_object, int64_t size)
{
   	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
   	deviceObj->d_Br = (bench_t*) malloc ( size * sizeof(bench_t*));
	return true;
}

__attribute__((weak))
void copy_memory_to_device(GraficCommon* device_object, bench_t* h_B,int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_Br = h_B;
}

__attribute__((weak))
void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);	     
	memcpy(h_B, &deviceObj->d_Br[0], sizeof(bench_t)*size);
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

__attribute__((weak))
void clean(GraficCommon* device_object)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	free(deviceObj->d_Br);
}