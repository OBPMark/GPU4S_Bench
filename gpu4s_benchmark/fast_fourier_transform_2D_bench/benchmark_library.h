/** * ====================================================================
 * @file        benchmark_library.h (./fast_fourier_transform_2D_bench)
 * @brief       Specific memory structures and function overloads 
 *              for the Fast Fourier Transform 2D benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
 // Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"

// ======= Benchmark local variable =======
// --- CUDA Data types  ---
#ifdef CUDA
	// CUDA lib
	#include <cufft.h>
	#ifdef FLOAT
		typedef cufftComplex bench_cuda_complex;
	#elif DOUBLE 
		typedef cufftDoubleComplex bench_cuda_complex;
	#endif
#endif

// --- 2D specific struct  ---
struct COMPLEX{
	bench_t x;
	bench_t y;
};

struct GraficObject : public GraficCommon {
	#ifdef CUDA 
		// CUDA PART
		#ifdef LIB
			bench_cuda_complex* d_A;
			bench_cuda_complex* d_B;
		#else
			bench_t* d_A;
			bench_t* d_B;
		#endif
   	#elif OPENCL
		// OpenCL PART
		cl::Event *evt_copyA;
		cl::Event *evt_copyB;
		cl::Event *evt;
		cl::Buffer *d_A;
		cl::Buffer *d_B;
	
	#else
		//CPU PART
		COMPLEX** d_A;
		COMPLEX** d_B;
	#endif
};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficCommon* device_object, int64_t size_b_matrix);
void copy_memory_to_device(GraficCommon* device_object, COMPLEX **h_A,int64_t size);
void execute_kernel(GraficCommon* device_object, int64_t n);
void copy_memory_to_host(GraficCommon* device_object, COMPLEX **h_B, int64_t size);


#ifdef UMA_COMPATIBILITY
    /**
     * @brief Maps the flat d_A/d_B device buffers into host-visible memory, then reconnects
     *        A[i]/B[i] row pointers into that single contiguous mapped block (2D COMPLEX**
     *        to 1D flat buffer).
	 * 
     * @param device_object Pointer to the device common structure
     * @param A Reference to the row-pointer array to reconnect over the mapped input buffer
     * @param B Reference to the row-pointer array to reconnect over the mapped output buffer
     * @param memSize N - the FFT side length (rows == cols), NOT a byte size
     */
	
    void get_unified_memory_pointers(GraficCommon* device_object, COMPLEX** &A, COMPLEX** &B, int64_t memSize);
    /**
     * @brief Unmaps d_A/d_B, blocked for host until the device give aigain ownership  
     *       
     * @param device_object Pointer to the device common structure
     * @param A Reference to the row-pointer array whose A[0] gives the mapped pointer to unmap
     * @param B Reference to the row-pointer array whose B[0] gives the mapped pointer to unmap
     */
    void sync_unified_memory_to_device(GraficCommon* device_object, COMPLEX** &A, COMPLEX** &B);
    /**
     * @brief Maps d_B back to a host-readable pointer, then reconnects d_output[i] row
     *        pointers into that mapped block, same N-based reasoning as
     *        get_unified_memory_pointers above.
	 * 
     * @param device_object Pointer to the device common structure
     * @param d_output Reference to the row-pointer array to reconnect over the mapped output buffer
     * @param memSize N - the FFT side length (rows == cols), NOT a byte size
     */
    void sync_unified_memory_to_host(GraficCommon* device_object, COMPLEX** &d_output, int64_t memSize);
#endif