#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


#define HIGHPASSFILTERSIZE 7
#define LOWPASSFILTERSIZE 9


#ifdef INT
typedef int bench_t;
static const std::string type_kernel = "typedef int bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {1,1,1,1,1,1,1,1,1};
static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {1,1,1,1,1,1,1};
#elif FLOAT
typedef float bench_t;
static const std::string type_kernel = "typedef float bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};
#elif DOUBLE
typedef double bench_t;
static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};
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
	cl::Event *evt_copyC;
	cl::Event *evt;
	cl::Event *evt_int;
	cl::Buffer *d_A;
	cl::Buffer *d_B;
	cl::Buffer *low_filter;
	cl::Buffer *high_filter;
	#elif OPENMP
	// OpenMP part
	bench_t* d_A;
	bench_t* d_B;
	bench_t* low_filter;
	bench_t* high_filter;
	#elif HIP
	// Hip part --
	bench_t* d_A;
	bench_t* d_B;
	bench_t* low_filter;
	bench_t* high_filter;
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
	bench_t* low_filter;
	bench_t* high_filter;
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
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a);
void execute_kernel(GraficObject *device_object, unsigned int n);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
float get_elapsed_time(GraficObject *device_object, bool csv_format);
void clean(GraficObject *device_object);


#endif