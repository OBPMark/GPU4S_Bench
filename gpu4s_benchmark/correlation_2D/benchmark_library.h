#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>




#ifdef INT
typedef int bench_t;
typedef float result_bench_t;
static const char type_kernel[] = "typedef int bench_t;\ntypedef float result_bench_t;\n";
#elif FLOAT
typedef float bench_t;
typedef float result_bench_t;
static const char type_kernel[] = "typedef float bench_t;\ntypedef float result_bench_t;\n";
#elif DOUBLE
typedef double bench_t;
typedef double result_bench_t;
static const char type_kernel[] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\ntypedef double result_bench_t;\n";
#endif

#ifdef OPENCL
// OpenCL lib
//#include <CL/opencl.h>
#include <CL/cl.hpp>
#elif OPENMP
// OpenMP lib
#include <omp.h>
#elif HIP
// HIP part
#include <hip/hip_runtime.h>
#else
// CUDA lib
#include <cuda_runtime.h>
#endif

#ifdef INT
	typedef int bench_t;
	#define __ptype "%d"
#elif FLOAT
	typedef float bench_t;
	#define __ptype "%f"
#elif DOUBLE 
	typedef double bench_t;
	#define __ptype "%f"
#else 
	// printf type helper, will resolve to %d or %f given the computed type
	#define __ptype "%f"
#endif

#ifndef BENCHMARK_H
#define BENCHMARK_H

struct GraficObject{
   	#ifdef OPENCL
   	// OpenCL PART
	cl::Context *context;
	cl::CommandQueue *queue;
	cl::Device default_device;
	cl::Event *evt_copyA;
	cl::Event *evt_copyB;
	cl::Event *evt_copyAB;
	cl::Event *evt_copyAA;
	cl::Event *evt_copyBB;
	cl::Event *evt;
	cl::Event *evt_mean;
	cl::Buffer *d_A;
	cl::Buffer *d_B;
	cl::Buffer *d_R;
	cl::Buffer *mean_A; // axuliar values for the mean of matrix A
	cl::Buffer *mean_B; // axuliar values for the mean of matrix B
	cl::Buffer *acumulate_value_a_b; // auxiliar values for the acumulation
	cl::Buffer *acumulate_value_a_a; // auxiliar values for the acumulation
	cl::Buffer *acumulate_value_b_b; // auxiliar values for the acumulation

	#elif OPENMP
	// OpenMP part
	bench_t* d_A;
	bench_t* d_B;
	result_bench_t d_R;
	result_bench_t mean_A; // axuliar values for the mean of matrix A
	result_bench_t mean_B; // axuliar values for the mean of matrix B
	result_bench_t acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t acumulate_value_b_b; // auxiliar values for the acumulation
	#elif HIP
	// Hip part --
	bench_t* d_A;
	bench_t* d_B;
	result_bench_t* d_R;
	result_bench_t* mean_A; // axuliar values for the mean of matrix A
	result_bench_t* mean_B; // axuliar values for the mean of matrix B
	result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
	hipEvent_t *start_memory_copy_device;
	hipEvent_t *stop_memory_copy_device;
	hipEvent_t *start_memory_copy_host;
	hipEvent_t *stop_memory_copy_host;
	hipEvent_t *start;
	hipEvent_t *stop;
	#else
	// CUDA PART
	bench_t* d_A;
	bench_t* d_B;
	result_bench_t* d_R;
	result_bench_t* mean_A; // axuliar values for the mean of matrix A
	result_bench_t* mean_B; // axuliar values for the mean of matrix B
	result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
	cudaEvent_t *start;
	cudaEvent_t *stop;
	#endif
	float elapsed_time;
};

void init(GraficObject *device_object, char* device_name);
void init(GraficObject *device_object, int platform, int device, char* device_name);
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b);
void execute_kernel(GraficObject *device_object, unsigned int n);
void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R);
float get_elapsed_time(GraficObject *device_object, bool csv_format);
void clean(GraficObject *device_object);


#endif