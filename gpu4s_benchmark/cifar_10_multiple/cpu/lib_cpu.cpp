#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


void convolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
{
	const int kernel_rad = kernel_size / 2;
	int x, y, kx, ky = 0;
	bench_t sum = 0;
	bench_t value = 0;

	const unsigned int squared_kernel_size = kernel_size * kernel_size;
	
	for (unsigned int block = 0; block < n*n; ++block)
	{
		x = block/n;
		y = block%n;
		sum = 0;
		for(unsigned int k = 0; k < squared_kernel_size; ++k)
		{
			value = 0;
			kx = (k/kernel_size) - kernel_rad; 
			ky = (k%kernel_size) - kernel_rad;
			if(!(kx + x < 0 || ky + y < 0) && !( kx + x > n - 1 || ky + y > n - 1))
			{
				value = A[(x + kx)*n+(y + ky)];
			}
			sum += value * kernel[(kx+kernel_rad)* kernel_size + (ky+kernel_rad)];
		}
		B[x*n+y] = sum;
	}
}


void relu_kernel(const bench_t *A, bench_t *B, const int size)
{
	// Compute traditional relu approach 
	for (unsigned int i = 0; i < size; ++i)
	{
		for (unsigned int j = 0; j < size; ++j)
		{
			if (A[i*size+j] > 0)
			{
				B[i*size+j] = A[i*size+j];
			}
			else 
			{
				B[i*size+j] = 0;
			}
		}
	}
}


void relu_linear_kernel(const bench_t *A, bench_t *B, const int size)
{
	// Compute traditional relu approach 
	for (unsigned int i = 0; i < size; ++i)
	{
		if (A[i] > 0)
		{
			B[i] = A[i];
		}
		else
		{
			// FIX: reset to 0 negative values
			B[i] = 0;
		}
	}
}


void max_pooling_kernel(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride)
{	
	bench_t max_value = 0;
	for (unsigned int i = 0; i < size; i+= stride)
	{
		for (unsigned int j = 0; j < size; j+= stride)
		{
			max_value = A[i*size+j];
			for(unsigned int x = 0; x < stride; ++x)
			{
				for(unsigned int y = 0; y < stride; ++y)
				{
					max_value = max(max_value, A[(i + x) * size + (j +y)]);
				}
			}
			B[(i / stride)* lateral_stride + (j/stride)] = max_value;
		}
	}
}


void lrn_kernel(const bench_t *A, bench_t *B, const int size)
{
	for (unsigned int i = 0; i < size; ++i)
	{
		for (unsigned int j = 0; j < size; ++j)
		{
			B[i*size+j] = A[i*size+j]/pow((K+ALPHA*pow(A[i*size+j],2)),BETA);
		}
	}
}


void matrix_multiplication_kernel(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w)
{
	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < m; j++)
		{
			bench_t acumulated = 0;
			for (unsigned int k = 0; k < w; k++)
			{   
				acumulated += A[i*w+k] * B[k*m+j];
			}
			C[i*m+j] = acumulated;
		}
	}
}


void softmax_kernel(const bench_t *A, bench_t *B, const int size)
{	
	bench_t sum_values = 0;
	bench_t value = 0;

	
	for (unsigned int i = 0; i < size; i++)
	{
		value = expf (A[i]);
		sum_values += value;
		B[i] = value;
	}

	for (unsigned int i = 0; i < size; i++)
	{
		B[i] = (B[i]/sum_values);
	}

}


void init(GraficCommon* device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images)
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
   	deviceObj->output_data = (bench_t*) malloc ( number_of_images * neurons_dense_2 * sizeof(bench_t));
	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size, unsigned int number_of_images)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->input_data = input_data;
	deviceObj->kernel_1 = kernel_1_data;
	deviceObj->kernel_2 = kernel_2_data;
	deviceObj->dense_layer_1_weights = weights_1;
	deviceObj->dense_layer_2_weights = weights_2;
}


void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

	bench_t* aux_output_data = deviceObj->output_data;
    bench_t* aux_input_data = deviceObj->input_data;


	// Start compute timer
	Clock kernelCLK;
	kernelCLK.start();	

	for(unsigned int position = 0; position < number_of_images; ++position)
    {
		aux_input_data = deviceObj->input_data + position * input_data * input_data;
        aux_output_data = deviceObj->output_data + position * output_data;
	
		// 1-1 Step convolution
		convolution_kernel(aux_input_data, deviceObj->conv_1_output, deviceObj->kernel_1, input_data, input_data, input_data, kernel_1);

		// 1-2 Step activation
		relu_kernel(deviceObj->conv_1_output, deviceObj->conv_1_output, input_data);
		
		// 1-3 Step pooling
		const unsigned int size_lateral_1 = input_data / stride_1;
		max_pooling_kernel(deviceObj->conv_1_output, deviceObj->pooling_1_output, input_data, stride_1, size_lateral_1);

		// 1-4 Normalization
		lrn_kernel(deviceObj->pooling_1_output, deviceObj->pooling_1_output, size_lateral_1);
		
		// 2-1 Step convolution
		convolution_kernel(deviceObj->pooling_1_output, deviceObj->conv_2_output, deviceObj->kernel_2, size_lateral_1, size_lateral_1, size_lateral_1, kernel_2);

		// 2-2 Step activation
		relu_kernel(deviceObj->conv_2_output, deviceObj->conv_2_output, size_lateral_1);

		// 2-3 Normalization
		lrn_kernel(deviceObj->conv_2_output, deviceObj->conv_2_output, size_lateral_1);

		// 2-4 Step pooling
		const unsigned int size_lateral_2 = size_lateral_1 / stride_2;
		max_pooling_kernel(deviceObj->conv_2_output, deviceObj->pooling_2_output, size_lateral_1, stride_2, size_lateral_2);

		// Dense layer 1
		matrix_multiplication_kernel(deviceObj->dense_layer_1_weights, deviceObj->pooling_2_output,deviceObj->dense_layer_1_output,neurons_dense_1, 1, size_lateral_2*size_lateral_2);

		// Activation layer dense 1
		relu_linear_kernel(deviceObj->dense_layer_1_output, deviceObj->dense_layer_1_output, neurons_dense_1);
		
		// Dense layer 2
		matrix_multiplication_kernel(deviceObj->dense_layer_2_weights, deviceObj->dense_layer_1_output, deviceObj->dense_layer_2_output, neurons_dense_2, 1, neurons_dense_1);

		// Activation layer dense 2
		relu_linear_kernel(deviceObj->dense_layer_2_output, deviceObj->dense_layer_2_output, neurons_dense_2);

		// Softmax - Output
		softmax_kernel(deviceObj->dense_layer_2_output, aux_output_data, neurons_dense_2);
	}

	// End compute timer
	kernelCLK.end();
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size, unsigned int number_of_images)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);	     
	memcpy(h_C, &deviceObj->output_data[0], sizeof(bench_t)*size*number_of_images);
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0, current_time);
    } 
	else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0);
    } 
	else
	{
		//--- FIX: print te time in milliseconds
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return deviceObj->elapsed_time;
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