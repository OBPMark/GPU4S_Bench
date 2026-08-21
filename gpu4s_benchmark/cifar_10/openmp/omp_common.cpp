/** * ====================================================================
 * @file        omp_common.cpp (./cifar_10)
 * @brief       Common OpenMP platform initialization, device setup, 
 *              profiling timer evaluation, and generic cleanup routines.
 * @paragraph   License
 * ESA-PL Strong Copyleft – v2.5
 * ======================================================================= */
#include "../benchmark_library.h"
#include <cstring>

void init(GraficCommon* device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	const unsigned int size_pooling_1 = input_data / stride_1;
    const unsigned int size_pooling_2 = size_pooling_1 / stride_2;
    const unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;
    const unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2; 

	// Convolution 1
   	deviceObj->conv_1_output = (bench_t*) malloc ( input_data * input_data * sizeof(bench_t*));
	// Pooling 1
	deviceObj->pooling_1_output = (bench_t*) malloc ( size_pooling_1 * size_pooling_1 * sizeof(bench_t));
	// Convolution 2
   	deviceObj->conv_2_output = (bench_t*) malloc ( size_pooling_1 * size_pooling_1 * sizeof(bench_t*));
	// Pooling 2
	deviceObj->pooling_2_output = (bench_t*) malloc ( size_pooling_2 * size_pooling_2 * sizeof(bench_t));
	// Dense 1
   	deviceObj->dense_layer_1_output = (bench_t*) malloc ( neurons_dense_1 * sizeof(bench_t));
   	// Dense 2
   	deviceObj->dense_layer_2_output = (bench_t*) malloc ( neurons_dense_2 * sizeof(bench_t));
   	// Output data
   	deviceObj->output_data = (bench_t*) malloc ( neurons_dense_2 * sizeof(bench_t));
	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Input data
	deviceObj->input_data = input_data;
	deviceObj->kernel_1 = kernel_1_data;
	deviceObj->kernel_2 = kernel_2_data;
	deviceObj->dense_layer_1_weights = weights_1;
	deviceObj->dense_layer_2_weights = weights_2;
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);	     
	memcpy(h_C, &deviceObj->output_data[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, deviceObj->elapsed_time * 1000.f, (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return deviceObj->elapsed_time * 1000.f;
}


void clean(GraficCommon* device_object)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	free(deviceObj->conv_1_output);
	free(deviceObj->pooling_1_output);
	free(deviceObj->conv_2_output);
	free(deviceObj->pooling_2_output);
	free(deviceObj->dense_layer_1_output);
	free(deviceObj->dense_layer_2_output);
	free(deviceObj->output_data);
}