#htvar kernel_code

void kernel mean_matrices (global const bench_t *A, global const bench_t *B,global result_bench_t *mean_A ,global result_bench_t *mean_B ,const int n){
    unsigned int size = n;
    int i = get_global_id(0);
    int j = get_global_id(1);																														                                    																				    								                                    
    unsigned int tid_x = get_local_id(0);
    unsigned int tid_y = get_local_id(1);
  
   __local  bench_t shared_data_A[BLOCK_SIZE * BLOCK_SIZE];
   __local  bench_t shared_data_B[BLOCK_SIZE * BLOCK_SIZE];


    if (i < size && j < size){

        shared_data_A[tid_x*get_local_size(1) + tid_y] = A[i*size + j];
        shared_data_B[tid_x*get_local_size(1) + tid_y] = B[i*size + j];
        
        // sinc theads
        barrier(CLK_LOCAL_MEM_FENCE); 
        
        for(unsigned int s_y = get_local_size(1)/2; s_y > 0; s_y >>= 1)
        {
            if (tid_y < s_y)
            {
                shared_data_A[tid_x * get_local_size(1) + tid_y] += shared_data_A[tid_x * get_local_size(1) + tid_y + s_y];
                shared_data_B[tid_x * get_local_size(1) + tid_y] += shared_data_B[tid_x * get_local_size(1) + tid_y + s_y];
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        for(unsigned int s_x = get_local_size(0)/2; s_x > 0; s_x >>= 1 )
        {
            if(tid_x < s_x)
            {
                shared_data_A[tid_x * get_local_size(1)] += shared_data_A[(tid_x + s_x) * get_local_size(1)];
                shared_data_B[tid_x * get_local_size(1)] += shared_data_B[(tid_x + s_x) * get_local_size(1)];
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
            
        }

        if( tid_x == 0 && tid_y == 0)
        { 
            atomic_add_global(mean_A, shared_data_A[0]);
            atomic_add_global(mean_B, shared_data_B[0]);
        }
    }
}

void kernel correlation_2D(global const bench_t *A,global const bench_t *B,global result_bench_t *R, global result_bench_t *mean_A ,global result_bench_t *mean_B, global result_bench_t *acumulate_value_a_b, global result_bench_t *acumulate_value_a_a, global result_bench_t *acumulate_value_b_b,const int n){
    unsigned int size = n;
    int i = get_global_id(0);
    int j = get_global_id(1);

    unsigned int tid_x = get_local_id(0);
    unsigned int tid_y = get_local_id(1);

    result_bench_t mean_a_matrix =  *mean_A / (n * n);
    result_bench_t mean_b_matrix =  *mean_B / (n * n);
    
    __local bench_t shared_data_A_B[BLOCK_SIZE * BLOCK_SIZE];
    __local bench_t shared_data_A_A[BLOCK_SIZE * BLOCK_SIZE];
    __local bench_t shared_data_B_B[BLOCK_SIZE * BLOCK_SIZE];


    if (i < size && j < size){
        result_bench_t result_mean_a = 0;
        result_bench_t result_mean_b = 0;
        result_mean_a = A[i*size+j] - mean_a_matrix;
        result_mean_b = B[i*size+j] - mean_b_matrix;
        shared_data_A_B[tid_x*get_local_size(1) + tid_y] = result_mean_a * result_mean_b;
        shared_data_A_A[tid_x*get_local_size(1) + tid_y] = result_mean_a * result_mean_a;
        shared_data_B_B[tid_x*get_local_size(1) + tid_y] = result_mean_b * result_mean_b;

        // first get the final value  in A (A - mean(a)) and in B (B - mean(b))
        __syncthreads();
        
        for(unsigned int s_y = get_local_size(1)/2; s_y > 0; s_y >>= 1)
        {
            if (tid_y < s_y)
            {
                shared_data_A_B[tid_x * get_local_size(1) + tid_y] += shared_data_A_B[tid_x * get_local_size(1) + tid_y + s_y];
                shared_data_A_A[tid_x * get_local_size(1) + tid_y] += shared_data_A_A[tid_x * get_local_size(1) + tid_y + s_y];
                shared_data_B_B[tid_x * get_local_size(1) + tid_y] += shared_data_B_B[tid_x * get_local_size(1) + tid_y + s_y];
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
        }
        for(unsigned int s_x = get_local_size(0)/2; s_x > 0; s_x >>= 1 )
        {
            if(tid_x < s_x)
            {
                shared_data_A_B[tid_x * get_local_size(1)] += shared_data_A_B[(tid_x + s_x) * get_local_size(1)];
                shared_data_A_A[tid_x * get_local_size(1)] += shared_data_A_A[(tid_x + s_x) * get_local_size(1)];
                shared_data_B_B[tid_x * get_local_size(1)] += shared_data_B_B[(tid_x + s_x) * get_local_size(1)];
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
            
        }

        if( tid_x == 0 && tid_y == 0)
        { 
            atomic_add_global(acumulate_value_a_b, shared_data_A_B[0]);
            atomic_add_global(acumulate_value_a_a, shared_data_A_A[0]);
            atomic_add_global(acumulate_value_b_b, shared_data_B_B[0]);
        }
 
    }

}
#htendvar