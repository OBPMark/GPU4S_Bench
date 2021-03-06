#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


#ifdef INT
typedef int bench_t;
typedef int bench_t_gpu;
static const std::string type_kernel = "typedef int bench_t;\n";
#elif FLOAT
#include <cuda_fp16.h>
typedef float bench_t;
typedef half bench_t_gpu;
static const std::string type_kernel = "typedef float bench_t;\n";
#elif FLOAT16

typedef float bench_t;
#ifdef OPENCL
// OpenCL lib
#else
#include <cuda_fp16.h>
typedef half bench_t_gpu;
// CUDA lib
#endif

#else 
typedef double bench_t;
typedef double bench_t_gpu;
static const std::string type_kernel = "typedef double bench_t;\n";
#endif

#ifdef OPENCL
// OpenCL lib
//#include <CL/opencl.h>
#include <CL/cl.hpp>
#else
// CUDA lib
#include <cuda_runtime.h>
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
	cl::Event *evt_copyC;
	cl::Event *evt;
	cl::Buffer *d_A;
	cl::Buffer *d_B;
	cl::Buffer *d_C;
	#else
	// CUDA PART
	bench_t* d_A;
	bench_t* d_B;
	bench_t_gpu* d_half_A;
	bench_t_gpu* d_half_B;
	bench_t_gpu* d_half_C;
	bench_t* d_C;
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
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b);
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
float get_elapsed_time(GraficObject *device_object, bool csv_format);
void clean(GraficObject *device_object);


#endif