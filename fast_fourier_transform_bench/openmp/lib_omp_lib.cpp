#include "../benchmark_library.h"
#include <cmath>
#include <cstring>
#include <fftw3.h>

void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, int64_t size)
{
   	device_object->d_Br = (bench_t*) malloc ( size * sizeof(bench_t*));
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size)
{
	device_object->d_B = h_B;
}


void execute_kernel(GraficObject *device_object, int64_t size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

    // FFTW implementation - First draft.
	fftw_plan plan;
	fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);
	fftw_complex *out_data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*size);
	for (int i=0; i<size; ++i)
	{
		in[i] = device_object->d_B[i];
	}
	plan = fftw_plan_dft_1d(size,in,out_data,FFTW_FORWARD,FFTW_ESTIMATE);
	fftw_execute(plan);
	for (int i=0; i<size; ++i)
	{
		device_object->d_Br[i] = out_data[i];
	}

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size)
{	     
	memcpy(h_B, &device_object->d_Br[0], sizeof(bench_t)*size);
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
	free(device_object->d_Br);
}