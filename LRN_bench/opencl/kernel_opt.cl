#htvar kernel_code
void kernel kernel_relu(global const bench_t* A, global bench_t* B, const int size ){                      
    int i = get_global_id(0);                                                                                                                                                                                 
    if (i < (size * size) ){                                                                                              
        bench_t threshold = 0;                                                                                          
        B[i] = max(threshold, A[i]);                                                                                                                                                                     
    }                                                                                                                                                                                                                                        
}
#htendvar