#include "../benchmark_library.h"
#include <cstring>


void transpose(bench_t *A, bench_t *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	// Transpose B to then compute matrix multiply
	bench_t *B_transposed;
    B_transposed = (bench_t*)malloc( sizeof(bench_t) * n * n);
    transpose(deviceObj->d_B, B_transposed, n);
	
	// FIX: declare the variable inside so OpenMP treats them as private per thread
	#pragma omp parallel for
    for (unsigned int i = 0; i < n; i++) { 
        for (unsigned int j = 0; j < n; j++) {
            bench_t dot = 0;
            for (unsigned int k = 0; k < n; k++) {
                dot += deviceObj->d_A[i*n+k]*B_transposed[j*n+k];
            } 
            deviceObj->d_C[i*n+j ] = dot;
   		}
	}

    free(B_transposed);

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}





