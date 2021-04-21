#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <string.h>

#ifndef CPU_LIB_H
#define CPU_LIB_H

#ifdef INT
typedef int bench_t;
typedef float result_bench_t;
#elif FLOAT
typedef float bench_t;
typedef float result_bench_t;
#else 
typedef double bench_t;
typedef double result_bench_t;
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

void correlation_2D(const bench_t* A,const bench_t* B, result_bench_t* R ,const int size);
bool compare_values(const result_bench_t* host,const result_bench_t* device);
bool compare_vectors(const bench_t* host,const bench_t* device, const int size);
void print_double_hexadecimal_values(const char* filename, result_bench_t* float_vector,  unsigned int size);
void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size);
#endif