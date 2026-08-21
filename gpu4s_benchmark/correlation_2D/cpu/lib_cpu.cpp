#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

void init(GraficCommon* device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_A = h_A;
	deviceObj->d_B = h_B;
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


void execute_kernel(GraficCommon* device_object, unsigned int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	Clock kernelCLK;
	kernelCLK.start();

	result_bench_t mean_a_matrix =  get_mean_matrix(deviceObj->d_A, size);
	result_bench_t mean_b_matrix =  get_mean_matrix(deviceObj->d_B, size);

	result_bench_t acumulate_value_a_b = 0;
	result_bench_t acumulate_value_a_a = 0;
	result_bench_t acumulate_value_b_b = 0;
	
	result_bench_t result_mean_a = 0;
	result_bench_t result_mean_b = 0;
	
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			result_mean_a = deviceObj->d_A[i*size+j] - mean_a_matrix;
			result_mean_b = deviceObj->d_B[i*size+j] - mean_b_matrix;
			acumulate_value_a_b += result_mean_a * result_mean_b;
			acumulate_value_a_a += result_mean_a * result_mean_a;
			acumulate_value_b_b += result_mean_b * result_mean_b;
		}
	}

	
	deviceObj->acumulate_value_a_b = acumulate_value_a_b;
	deviceObj->acumulate_value_a_a = acumulate_value_a_a;
	deviceObj->acumulate_value_b_b = acumulate_value_b_b;
	kernelCLK.end();
	//FIX: add float division
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, result_bench_t* h_R)
{
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);	     
    *h_R = (result_bench_t)(deviceObj->acumulate_value_a_b / (result_bench_t)(sqrt(deviceObj->acumulate_value_a_a * deviceObj->acumulate_value_b_b)));
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return deviceObj->elapsed_time * 1000.f;
}


void clean(GraficCommon* device_object)
{
	return;
}
