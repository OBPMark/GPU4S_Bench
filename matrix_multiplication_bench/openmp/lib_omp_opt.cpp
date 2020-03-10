#include "../benchmark_library.h"
#include <cstring>

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix) 
{
	device_object->d_A = (bench_t*) malloc ( size_a_matrix * sizeof(bench_t*));
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	device_object->d_C = (bench_t*) malloc ( size_c_matrix * sizeof(bench_t*));
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b)
{
	memcpy(&device_object->d_A[0], h_A, sizeof(bench_t)*size_a);
	memcpy(&device_object->d_B[0], h_B, sizeof(bench_t)*size_b);
}


void transpose(bench_t *A, bench_t *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	// Transpose B to then compute matrix multiply
	bench_t *B_transposed;
    B_transposed = (bench_t*)malloc( sizeof(bench_t) * n * n);
    transpose(device_object->d_B, B_transposed, n);
	unsigned int i, j, k;
	
	#pragma omp parallel for
	for (i = 0; i < n; i++) { 
		for (j = 0; j < n; j++) {
			bench_t dot  = 0;
			for (k = 0; k < n; k++) {
				dot += device_object->d_A[i*n+k]*B_transposed[j*n+k];
			} 
			device_object->d_C[i*n+j ] = dot;
		}
	}

    free(B_transposed);

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	memcpy(h_C, &device_object->d_C[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format)
{
	// TODO: TBD Time scoping.
	printf("Elapsed time: %.10f miliseconds\n", device_object->elapsed_time * 1000.f);
	return device_object->elapsed_time;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_A);
	free(device_object->d_B);
	free(device_object->d_C);
}