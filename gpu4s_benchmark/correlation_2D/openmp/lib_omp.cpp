#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

result_bench_t get_mean_matrix(const bench_t* A,const int size){
	
	bench_t sum_val = 0;

	#pragma omp parallel for reduction(+:sum_val)
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			sum_val += A[i*size+j];
		}
	}

	return result_bench_t(sum_val) / result_bench_t(size*size);
}

void execute_kernel(GraficCommon* device_object, unsigned int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	result_bench_t mean_a_matrix =  get_mean_matrix(deviceObj->d_A, size);
	result_bench_t mean_b_matrix =  get_mean_matrix(deviceObj->d_B, size);

	result_bench_t acumulate_value_a_b = 0;
	result_bench_t acumulate_value_a_a = 0;
	result_bench_t acumulate_value_b_b = 0;
	
	result_bench_t result_mean_a = 0;
	result_bench_t result_mean_b = 0;
	
	#pragma parallel for reduction(+:acumulate_value_a_b,acumulate_value_a_a,acumulate_value_b_b)
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

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


