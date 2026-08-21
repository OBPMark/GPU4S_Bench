#include "../benchmark_library.h"
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
	
	#pragma omp parallel for private(x, y, kx, ky, sum, value)
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
	#pragma omp parallel for
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
	#pragma omp parallel for
	for (unsigned int i = 0; i < size; ++i)
	{
		if (A[i] > 0)
		{
			B[i] = A[i];
		}
		else 
		{
			B[i] = 0;
		}
	}
}


void max_pooling_kernel(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride)
{	
	bench_t max_value = 0;
	const unsigned int block_size = size/stride;
	const unsigned int stride_squared = stride*stride;
	unsigned int blockx, blocky, block_zero, x, y = 0;

	#pragma omp parallel for private(max_value,blockx, blocky, block_zero, x, y)
	for (unsigned int block = 0; block < block_size*block_size; ++block)
	{
		{
			blockx = block%block_size;
			blocky = block/block_size;
			block_zero = blockx*stride + blocky*stride*size;
			max_value = A[block_zero];		
			for(unsigned int i = 0; i < stride_squared; ++i)
			{
				x = i%stride;
				y = i/stride; 
				max_value = max(max_value, A[(block_zero+x) + y*size]);
			}
			B[block] = max_value;	
		}
	}
}


void lrn_kernel(const bench_t *A, bench_t *B, const int size)
{
	#pragma omp parallel for
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
	#pragma omp parallel for
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

	
	#pragma omp parallel for reduction(+:sum_values)
	for (unsigned int i = 0; i < size; i++)
	{
		value = expf (A[i]);
		sum_values += value;
		B[i] = value;
	}

	#pragma omp parallel for
	for (unsigned int i = 0; i < size; i++)
	{
		B[i] = (B[i]/sum_values);
	}

}



void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	// 1-1 Step convolution
	convolution_kernel(deviceObj->input_data, deviceObj->conv_1_output, deviceObj->kernel_1, input_data, input_data, input_data, kernel_1);

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
	softmax_kernel(deviceObj->dense_layer_2_output, deviceObj->output_data, neurons_dense_2);

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}

