#include "../benchmark_library.h"

#ifdef OPENBLAS
	#include <cblas.h>
#elif ATLAS
	extern "C" {
        #include <cblas.h>
	}
#endif




void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

    #ifdef FLOAT
    cblas_sgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, n, m, w, 1, deviceObj->d_A, w, deviceObj->d_B, m, 1, deviceObj->d_C, m);
    #elif DOUBLE
    cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans, n, m, w, 1, deviceObj->d_A, w, deviceObj->d_B, m, 1, deviceObj->d_C, m);
    #else
    printf("Error: OpenBlas doesn't support the specified operand type.\n");
    #endif
                
	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}

