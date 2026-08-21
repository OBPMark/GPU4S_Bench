#include <cublas_v2.h>
#include "../benchmark_library.h"

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // cublas settings
    int lda=m,ldb=m,ldc=m;
    const bench_t_gpu alf = 1;
    const bench_t_gpu bet = 0;
    const bench_t_gpu *alpha = &alf;
    const bench_t_gpu *beta = &bet;
    cublasHandle_t handle;
    cublasCreate(&handle);
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    #ifdef INT
    printf("CUBLAS NOT SUPPORT INT OPERATIOS\n");
    #elif FLOAT16
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->d_half_B, lda, deviceObj->d_half_A, ldb, beta, deviceObj->d_half_C, ldc);
    #elif FLOAT 
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->d_B, lda, deviceObj->d_A, ldb, beta, deviceObj->d_C, ldc);
    #else // DOUBLE
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->d_B, lda, deviceObj->d_A, ldb, beta, deviceObj->d_C, ldc);
    #endif
    
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();

    // destroy cublas
    cublasDestroy(handle);
}

