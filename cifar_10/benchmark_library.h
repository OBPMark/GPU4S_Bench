#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


#ifdef INT
typedef int bench_t;
#define __ptype "%d"
static const std::string type_kernel = "typedef int bench_t;\n";
#elif FLOAT
typedef float bench_t;
#define __ptype "%f"
static const std::string type_kernel = "typedef float bench_t;\n";
const float K = 2;
const float ALPHA = 10e-4;
const float BETA = 0.75;
#elif DOUBLE
typedef double bench_t;
#define __ptype "%f"
static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n";
const double K = 2;
const double ALPHA = 10e-4;
const double BETA = 0.75;
#else 
	// printf type helper, will resolve to %d or %f given the computed type
	#define __ptype "%f"
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


#ifndef BENCHMARK_H
#define BENCHMARK_H

struct GraficObject{
   	#ifdef OPENCL
   	// OpenCL PART
	cl::Context *context;
	cl::CommandQueue *queue;
	cl::Device default_device;
	cl::Event *evt_copyIN;
	cl::Event *evt_copyK1;
	cl::Event *evt_copyK2;
	cl::Event *evt_copyW1;
	cl::Event *evt_copyW2;
	cl::Event *evt_copyOut;
	cl::Event *evt1_1;
	cl::Event *evt1_2;
	cl::Event *evt1_3;
	cl::Event *evt1_4;
	cl::Event *evt2_1;
	cl::Event *evt2_2;
	cl::Event *evt2_3;
	cl::Event *evt2_4;
	cl::Event *evtd_1;
	cl::Event *evtd_1_a;
	cl::Event *evtd_2;
	cl::Event *evtd_2_a;
	cl::Event *evt_softmax;
	cl::Event *evt_softmax_fin;

	cl::Buffer *input_data;
	cl::Buffer *kernel_1;
	cl::Buffer *conv_1_output;
	cl::Buffer *pooling_1_output;
	cl::Buffer *kernel_2;
	cl::Buffer *conv_2_output;
	cl::Buffer *pooling_2_output;
	cl::Buffer *dense_layer_1_weights;
	cl::Buffer *dense_layer_1_output;
	cl::Buffer *dense_layer_2_weights;
	cl::Buffer *dense_layer_2_output;
	cl::Buffer *output_data;
	cl::Buffer *sum_ouput;

	#elif OPENMP
	// OpenMP part
	bench_t* input_data;
	bench_t* kernel_1;
	bench_t* conv_1_output;
	bench_t* pooling_1_output;
	bench_t* kernel_2;
	bench_t* conv_2_output;
	bench_t* pooling_2_output;
	bench_t* dense_layer_1_weights;
	bench_t* dense_layer_1_output;
	bench_t* dense_layer_2_weights;
	bench_t* dense_layer_2_output;
	bench_t* output_data;

	#elif HIP
	bench_t* input_data;
	bench_t* kernel_1;
	bench_t* conv_1_output;
	bench_t* pooling_1_output;
	bench_t* kernel_2;
	bench_t* conv_2_output;
	bench_t* pooling_2_output;
	bench_t* dense_layer_1_weights;
	bench_t* dense_layer_1_output;
	bench_t* dense_layer_2_weights;
	bench_t* dense_layer_2_output;
	bench_t* output_data;
	bench_t* sum_ouput;
	hipEvent_t *start_memory_copy_device;
	hipEvent_t *stop_memory_copy_device;
	hipEvent_t *start_memory_copy_host;
	hipEvent_t *stop_memory_copy_host;
	hipEvent_t *start;
	hipEvent_t *stop;

	#else
	// CUDA PART
	bench_t* input_data;
	bench_t* kernel_1;
	bench_t* conv_1_output;
	bench_t* pooling_1_output;
	bench_t* kernel_2;
	bench_t* conv_2_output;
	bench_t* pooling_2_output;
	bench_t* dense_layer_1_weights;
	bench_t* dense_layer_1_output;
	bench_t* dense_layer_2_weights;
	bench_t* dense_layer_2_output;
	bench_t* output_data;
	bench_t* sum_ouput;
	
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
bool device_memory_init(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2);
void copy_memory_to_device(GraficObject *device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size);
void execute_kernel(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
float get_elapsed_time(GraficObject *device_object, bool csv_format);
void clean(GraficObject *device_object);


#endif
