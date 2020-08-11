#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b)
{
	device_object->d_A = h_A;
	device_object->d_B = h_B;
}


result_bench_t get_mean_matrix(const bench_t* A,const int size){
	
	bench_t sum_val = 0;
	
	#pragma omp parallel for reduction(+:sum_val)
	for (unsigned int i=0; i < size*size; ++i)
	{
		sum_val += A[i];
	}

	return result_bench_t(sum_val) / result_bench_t(size*size);
}


void execute_kernel(GraficObject *device_object, unsigned int size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	result_bench_t mean_a_matrix =  get_mean_matrix(device_object->d_A, size);
	result_bench_t mean_b_matrix =  get_mean_matrix(device_object->d_B, size);

	result_bench_t acumulate_value_a_b = 0;
	result_bench_t acumulate_value_a_a = 0;
	result_bench_t acumulate_value_b_b = 0;
	
	result_bench_t result_mean_a = 0;
	result_bench_t result_mean_b = 0;
	
	#pragma parallel for reduction(+:acumulate_value_a_b,acumulate_value_a_a,acumulate_value_b_b)
	for (unsigned int i=0; i<size*size; i++){
		result_mean_a = device_object->d_A[i] - mean_a_matrix;
		result_mean_b = device_object->d_B[i] - mean_b_matrix;
		acumulate_value_a_b += result_mean_a * result_mean_b;
		acumulate_value_a_a += result_mean_a * result_mean_a;
		acumulate_value_b_b += result_mean_b * result_mean_b;
	}

	
	device_object->acumulate_value_a_b = acumulate_value_a_b;
	device_object->acumulate_value_a_a = acumulate_value_a_a;
	device_object->acumulate_value_b_b = acumulate_value_b_b;

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R)
{	     
    *h_R = (result_bench_t)(device_object->acumulate_value_a_b / (result_bench_t)(sqrt(device_object->acumulate_value_a_a * device_object->acumulate_value_b_b)));
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
	return;
}
