#htvar kernel_code
void kernel kernel_relu(global const bench_t* A, global bench_t* B, const int size ){	                    
	int i = get_global_id(0);																							
	int j = get_global_id(1);																				    		
	if (i < size && j < size){																	 							
	bench_t threshold = 0;																							
		B[i*size+j] = max(threshold, A[i*size+j]);										 																								
	}										                             													
                                           																				
}																				 											
#htendvar