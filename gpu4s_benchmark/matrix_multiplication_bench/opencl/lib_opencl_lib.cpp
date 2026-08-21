// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <clblast.h>


void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    const bench_t alpha = 1.0f;
    const bench_t beta = 1.0f;
    const unsigned int a_ld = n;
    const unsigned int b_ld = n;
    const unsigned int c_ld = n;
    // kernel time execution
    Clock kernelCLK;

    

    #ifdef INT
        printf("CLBLAST NOT SUPPORT INT OPERATIOS\n");
    #else
        // Clock profilling start 
        kernelCLK.start();

        auto status = clblast::Gemm(clblast::Layout::kRowMajor,clblast::Transpose::kNo, clblast::Transpose::kNo, n, n, n, alpha, (*deviceObj->d_A)() , 0, a_ld, (*deviceObj->d_B)(), 0, b_ld, beta, (*deviceObj->d_C)(), 0, c_ld,&(*deviceObj->queue)(), &(*deviceObj->evt)());

        // Wait for completion before stopping the clock
        deviceObj->queue->finish();

        // Clock profilling end 
        kernelCLK.end();

        // store the kernel time
        deviceObj->elapsed_time = kernelCLK.getElapsedNS();
    #endif
}
