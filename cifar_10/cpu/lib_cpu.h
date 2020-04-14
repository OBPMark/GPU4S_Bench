#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string.h>

#ifndef CPU_LIB_H
#define CPU_LIB_H



#ifdef INT
typedef int bench_t;
#elif FLOAT
typedef float bench_t;
#elif DOUBLE
typedef double bench_t;
#endif

#ifdef BIGENDIAN
// bigendian version
union
	{
		double f;
		struct
		{
			unsigned char a,b,c,d,e,f,g,h;
		}binary_values;
	} binary_float;
#else
// littelendian version
union
	{
		double f;
		struct
		{
			unsigned char h,g,f,e,d,c,b,a;
		}binary_values;
	} binary_float;
#endif

void cifar10(bench_t* output_data, bench_t* conv_1_output, bench_t* pooling_1_output, bench_t* conv_2_output, bench_t* pooling_2_output, bench_t* dense_layer_1_output, bench_t* dense_layer_2_output, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2, 
unsigned int input_data_size, unsigned int output_data_size, unsigned int kernel_1_size, unsigned int kernel_2_size, unsigned int stride_1_size, unsigned int stride_2_size, unsigned int neurons_dense_1_size, unsigned int neurons_dense_2_size);
void convolution(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size);
void relu(const bench_t *A, bench_t *B, const int size);
void relu_linear(const bench_t *A, bench_t *B, const int size);
void max_pooling(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride);
void lrn(const bench_t *A, bench_t *B, const int size);
void matrix_multiplication(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w);
void softmax(const bench_t *A, bench_t *B, const int size);
bool compare_vectors(const bench_t* host,const bench_t* device, const int size);
void print_double_hexadecimal_values(const char* filename, bench_t* float_vector,  unsigned int size);
void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size);


#endif