#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string.h>
#include "../shared_variables.h"

#ifndef CPU_LIB_H
#define CPU_LIB_H

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
bool FFT2D(COMPLEX **c,int n,int dir, COMPLEX **exit);
bool compare_vectors(COMPLEX **host, COMPLEX **device,  int64_t size);
bool compare_vectors(const bench_t* host,const bench_t* device, const int64_t size);
void print_double_hexadecimal_values(const char* filename, bench_t* float_vector,  unsigned int size);
void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size);


#endif