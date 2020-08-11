#htvar kernel_code

void kernel mean_matrices (global const bench_t *A, global const bench_t *B,global result_bench_t *mean_A ,global result_bench_t *mean_B ,const int n){
    unsigned int size = n;
    int i = get_global_id(0);
    int j = get_global_id(1);																														                                    																				    								                                    
    if (i < size && j < size){
        atomic_add_global(mean_A, A[i*size+j]);
        atomic_add_global(mean_B, B[i*size+j]);
    }
}

void kernel correlation_2D(global const bench_t *A,global const bench_t *B,global result_bench_t *R, global result_bench_t *mean_A ,global result_bench_t *mean_B, global result_bench_t *acumulate_value_a_b, global result_bench_t *acumulate_value_a_a, global result_bench_t *acumulate_value_b_b,const int n){
    unsigned int size = n;
    int i = get_global_id(0);
    int j = get_global_id(1);

   result_bench_t mean_a_matrix =  *mean_A / (n * n);
	result_bench_t mean_b_matrix =  *mean_B / (n * n);
    if (i < size && j < size){
        // first get the final value  in A (A - mean(a)) and in B (B - mean(b))
        result_bench_t result_mean_a = 0;
        result_bench_t result_mean_b = 0;
        result_mean_a = A[i*size+j] - mean_a_matrix;
        result_mean_b = B[i*size+j] - mean_b_matrix;
        atomic_add_global(acumulate_value_a_b, result_mean_a * result_mean_b);
        atomic_add_global(acumulate_value_a_a, result_mean_a * result_mean_a);
        atomic_add_global(acumulate_value_b_b, result_mean_b * result_mean_b);
    }

}
#htendvar