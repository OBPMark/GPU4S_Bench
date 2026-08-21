/** * ====================================================================
 * @file        benchmark_library.h (./cifar_10)
 * @brief       Specific memory structures and function overloads 
 *              for the Cifar 10 benchmark.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#pragma once
// Include all the benchmark common variable, struct, prototype, lib
#include "benchmark_common.h"
#include "Clock.h"

// ======= Benchmark local variable =======
// --- Compute ---
const bench_t K = 2;
const bench_t ALPHA = 10e-4;
const bench_t BETA = 0.75;


struct GraficObject : public GraficCommon {
	#ifdef CUDA
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
   	#elif OPENCL
		// OpenCL PART
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
	#elif HIP
		//HIP part
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
	#else
		// CPU part
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
	#endif
};

// --- Specefic overload of benchmarking function ---
bool device_memory_init(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2);
void copy_memory_to_device(GraficCommon* device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size);
void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2);

#ifdef UMA_COMPATIBILITY

// --- 5 buffer, mixed sizes (cifar_10_multiple) ---
    /**
     * @brief Maps cifar's five input buffers and its output buffer into host-visible memory.
     *        Unlike the equal-sized overloads, each buffer here has its own byte size -
     *        there's no single shared memSize. 
     * @param device_object Pointer to the device common structure
     * @param input_data Reference to receive the mapped input host pointer
     * @param input_mem_size Size of input_data, in bytes
     * @param kernel_1 Reference to receive the mapped first conv kernel host pointer
     * @param kernel_2 Reference to receive the mapped second conv kernel host pointer
     * @param kernel_mem_size Size of EACH kernel buffer, in bytes - kernel_1 and kernel_2 share this one size
     * @param weights_1 Reference to receive the mapped dense-layer-1 weights host pointer
     * @param weights_1_mem_size Size of weights_1, in bytes
     * @param weights_2 Reference to receive the mapped dense-layer-2 weights host pointer
     * @param weights_2_mem_size Size of weights_2, in bytes
     * @param d_output Reference to receive the mapped output host pointer
     * @param output_mem_size Size of d_output, in bytes
     */
    void get_unified_memory_pointers(GraficCommon* device_object, bench_t* &input_data, unsigned int input_mem_size, bench_t* &kernel_1, bench_t* &kernel_2, unsigned int kernel_mem_size, bench_t* &weights_1, unsigned int weights_1_mem_size, bench_t* &weights_2, unsigned int weights_2_mem_size, bench_t* &d_output, unsigned int output_mem_size);
    /**
     * @brief Unmaps all six cifar buffers, blocked for host until the device give aigain ownership .
     * @param device_object Pointer to the device common structure
     * @param input_data Reference to the mapped input host pointer to unmap
     * @param kernel_1 Reference to the mapped first conv kernel host pointer to unmap
     * @param kernel_2 Reference to the mapped second conv kernel host pointer to unmap
     * @param weights_1 Reference to the mapped dense-layer-1 weights host pointer to unmap
     * @param weights_2 Reference to the mapped dense-layer-2 weights host pointer to unmap
     * @param d_output Reference to the mapped output host pointer to unmap
     */
    void sync_unified_memory_to_device(GraficCommon* device_object, bench_t* &input_data, bench_t* &kernel_1, bench_t* &kernel_2, bench_t* &weights_1, bench_t* &weights_2, bench_t* &d_output);
#endif
