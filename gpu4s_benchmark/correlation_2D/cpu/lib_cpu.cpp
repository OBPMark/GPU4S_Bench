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
double suma_valores = 0;
double final_value = 0;
	
	// Be aware of precision errors.

	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			suma_valores += A[i*size+j];
		}
	}

final_value = result_bench_t(suma_valores) / result_bench_t(size*size);
return final_value;
}


void execute_kernel(GraficObject *device_object, unsigned int size){
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	result_bench_t mean_a_matrix =  get_mean_matrix(device_object->d_A, size);
	result_bench_t mean_b_matrix =  get_mean_matrix(device_object->d_B, size);

	result_bench_t acumulate_value_a_b = 0;
	result_bench_t acumulate_value_a_a = 0;
	result_bench_t acumulate_value_b_b = 0;
	
	result_bench_t result_mean_a = 0;
	result_bench_t result_mean_b = 0;
	
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			result_mean_a = device_object->d_A[i*size+j] - mean_a_matrix;
			result_mean_b = device_object->d_B[i*size+j] - mean_b_matrix;
			acumulate_value_a_b += result_mean_a * result_mean_b;
			acumulate_value_a_a += result_mean_a * result_mean_a;
			acumulate_value_b_b += result_mean_b * result_mean_b;
		}
	}

	
	device_object->acumulate_value_a_b = acumulate_value_a_b;
	device_object->acumulate_value_a_a = acumulate_value_a_a;
	device_object->acumulate_value_b_b = acumulate_value_b_b;
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    device_object->elapsed_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
}


void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R)
{	     
    *h_R = (result_bench_t)(device_object->acumulate_value_a_b / (result_bench_t)(sqrt(device_object->acumulate_value_a_a * device_object->acumulate_value_b_b)));
}


float get_elapsed_time(GraficObject *device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, device_object->elapsed_time, (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	return;
}
