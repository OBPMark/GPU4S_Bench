std::string kernel_code = 
"void kernel kernel_relu(global const bench_t* A, global bench_t* B, const int size ){\n"
"int i = get_global_id(0);\n"
"int j = get_global_id(1);\n"
"if (i < size && j < size){\n"
"bench_t threshold = 0;\n"
"B[i*size+j] = max(threshold, A[i*size+j]);\n"
"}\n"
"}\n"
;
