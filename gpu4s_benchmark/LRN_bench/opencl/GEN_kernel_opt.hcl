std::string kernel_code = 
"void kernel kernel_lrn(global const bench_t* A, global bench_t* B, const int size, const bench_t K, const bench_t ALPHA, const bench_t BETA ){\n"
"long i = (long)get_global_id(0) * size + get_global_id(1);\n"
"if (i < (long)size * size){\n"
"B[i] = A[i]/pow((K+ALPHA*pow(A[i], (bench_t)2.0)), BETA);\n"
"}\n"
"}\n"
;