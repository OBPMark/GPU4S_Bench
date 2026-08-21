
std::string kernel_code = 
"void kernel kernel_softmax(global const bench_t* A, global bench_t* B, global bench_t* sum_d_B, const int size ){\n"
"int i = get_global_id(0);\n"
"int j = get_global_id(1);\n"
"if (i < size && j < size){\n"
"B[i*size+j] = exp(A[i*size+j]);\n"
"atomic_add_global(sum_d_B, B[i*size+j]);\n"
"}\n"
"}\n"
"void kernel kernel_softmax_end(global  bench_t* B, global bench_t* sum_d_B, const int size ){\n"
"int i = get_global_id(0);\n"
"int j = get_global_id(1);\n"
"if (i < size && j < size){\n"
"B[i*size+j] = (B[i*size+j]/(*sum_d_B));\n"
"}\n"
"}\n"
;
