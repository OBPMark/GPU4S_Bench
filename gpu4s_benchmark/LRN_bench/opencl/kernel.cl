#htvar kernel_code
void kernel kernel_lrn(global const bench_t* A, global bench_t* B, const int size, const bench_t K, const bench_t ALPHA, const bench_t BETA ){	                    
	int i = get_global_id(0);																							
	int j = get_global_id(1);																				    		
	if (i < size && j < size){						
		B[i*size+j] = A[i*size+j]/pow((K+ALPHA*pow(A[i*size+j],2)),BETA);															
	}             																				
}
#htendvar