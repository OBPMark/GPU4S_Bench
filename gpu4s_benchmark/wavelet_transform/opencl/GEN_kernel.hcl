
std::string kernel_code = 
"void kernel wavelet_transform(global const bench_t* A,  global bench_t* B , const int n, global const bench_t* lowpass_filter, global const bench_t* highpass_filter ){\n"
"int i = get_global_id(0);\n"
"unsigned int size = n;\n"
"unsigned int full_size = size * 2;\n"
"int hi_start = -(LOWPASSFILTERSIZE / 2);\n"
"int hi_end = LOWPASSFILTERSIZE / 2;\n"
"int gi_start = -(HIGHPASSFILTERSIZE / 2 );\n"
"int gi_end = HIGHPASSFILTERSIZE / 2;\n"
"if (i < size){\n"
"bench_t sum_value_low = 0;\n"
"for (int hi = hi_start; hi < hi_end + 1; ++hi){\n"
"int x_position = (2 * i) + hi;\n"
"if (x_position < 0) {\n"
"// turn negative to positive\n"
"x_position = x_position * -1;\n"
"}\n"
"else if (x_position > full_size - 1)\n"
"{\n"
"x_position = full_size - 1 - (x_position - (full_size -1 ));;\n"
"}\n"
"// now I need to restore the hi value to work with the array\n"
"sum_value_low += lowpass_filter[hi + hi_end] * A[x_position];\n"
"}\n"
"// store the value\n"
"B[i] = sum_value_low;\n"
"bench_t sum_value_high = 0;\n"
"// second process the Highpass filter\n"
"for (int gi = gi_start; gi < gi_end + 1; ++gi){\n"
"int x_position = (2 * i) + gi + 1;\n"
"if (x_position < 0) {\n"
"// turn negative to positive\n"
"x_position = x_position * -1;\n"
"}\n"
"else if (x_position >  full_size - 1)\n"
"{\n"
"x_position = full_size - 1 - (x_position - (full_size -1 ));\n"
"}\n"
"sum_value_high += highpass_filter[gi + gi_end] * A[x_position];\n"
"}\n"
"// store the value\n"
"B[i+size] = sum_value_high;\n"
"}\n"
"}\n"
;
