std::string kernel_code = 
"void kernel kernel_relu(global const bench_t* A, global bench_t* B, const int size ){\n"
"int i = get_global_id(0);\n"
"if (i < (size * size) ){\n"
"bench_t threshold = 0;\n"
"B[i] = max(threshold, A[i]);\n"
"}\n"
"}\n"
;
