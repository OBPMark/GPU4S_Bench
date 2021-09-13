#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <sys/time.h>
#include <ctime>
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


struct BenchmarkParameters{
	int size = 0;
	unsigned int gpu = 0;
	bool verification = false;
	bool export_results = false;
	bool export_results_gpu = false;
	bool print_output = false;
	bool print_input = false;
	bool print_timing = false;
	bool csv_format = false;
	bool mute_messages = false;
	bool csv_format_timestamp = false;
	char input_file[100] = "";
	char output_file[100] = "";
};

bool FFT2D(COMPLEX **c,int n,int dir, COMPLEX **exit);
bool compare_vectors(COMPLEX **host, COMPLEX **device,  int64_t size);
bool compare_vectors(const bench_t* host,const bench_t* device, const int64_t size);
void print_double_hexadecimal_values(const char* filename, bench_t* float_vector,  unsigned int size);
void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size);
long int get_timestamp();


#endif