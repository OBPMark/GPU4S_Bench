
std::string kernel_code = 
"void kernel kernel_lrn(global const bench_t* A, global bench_t* B, const int size, const bench_t K, const bench_t ALPHA, const bench_t BETA ){\n"
"int i = get_global_id(0);\n"
"int j = get_global_id(1);\n"
"if (i < size && j < size){\n"
"B[i*size+j] = A[i*size+j]/pow((K+ALPHA*pow(A[i*size+j],2)),BETA);\n"
"}\n"
"}\n"
;
