#htvar kernel_code
void kernel kernel_lrn(global const bench_t* A, global bench_t* B, const int size, const bench_t K, const bench_t ALPHA, const bench_t BETA ){
    int i = get_global_id(0);
    if (i  < (size * size)){
        B[i] = A[i]/powf((K+ALPHA*powf(A[i],2)),BETA);
    }
}
#htendvar