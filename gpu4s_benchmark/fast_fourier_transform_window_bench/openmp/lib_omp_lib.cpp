
#include "../benchmark_library.h"
#include <cmath>
#include <cstring>
#include <fftw3.h>
#include <unistd.h>

void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object,  int64_t size_a_array, int64_t size_b_array)
{
	device_object->d_Br = (bench_t*) malloc ((size_b_array/2) * sizeof(fftw_complex*));
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size)
{
	device_object->d_B = h_B;
}


void execute_kernel(GraficObject *device_object, int64_t window, int64_t size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// FFTW implementation
	fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_num_threads());

	fftw_plan plan;
	fftw_complex *in, *out;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (size));
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (((size * 2 - window) + 1) * window));
		
		
	for(int i = 0; i < size; ++i) 
	{
		in[i][0] = device_object->d_B[i*2];
		in[i][1] = device_object->d_B[i*2+1];
	}
	
	fftw_complex *aux_in = in;
	fftw_complex *aux_out = out;
	
	for (unsigned int i = 0; i < (size * 2  - window + 1); i+=1)
	{
		plan = fftw_plan_dft_1d((window/2),aux_in,aux_out, FFTW_FORWARD, FFTW_ESTIMATE);
		fftw_execute(plan);
		aux_out += window;
		++aux_in;
   	}


	for (int64_t i=0; i< (((size * 2 - window) + 1) * window)/2; i++)
	{
		device_object->d_Br[i*2] = out[i][0];
		device_object->d_Br[i*2+1] = out[i][1];
	}
	

	fftw_destroy_plan(plan);	
	fftw_free(in); fftw_free(out);

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size)
{
	memcpy(h_B, &device_object->d_Br[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_Br);
}

