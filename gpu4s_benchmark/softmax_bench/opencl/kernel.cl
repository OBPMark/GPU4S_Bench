#htvar kernel_code
void kernel kernel_softmax(global const bench_t* A, global bench_t* B, global bench_t* sum_d_B, const int size ){	                                
		int i = get_global_id(0);																													
 		int j = get_global_id(1);																				    								
		if (i < size && j < size){																	 													
       																														
			B[i*size+j] = exp(A[i*size+j]);										 														
			atomic_add_global(sum_d_B, B[i*size+j]);											             																				
			    										             										 
			    									                             																		
																																
		}										                             																			
                                           																										
}																				 																	
void kernel kernel_softmax_end(global  bench_t* B, global bench_t* sum_d_B, const int size ){                                              
       int i = get_global_id(0);                                                                                                                   
       int j = get_global_id(1);                                                                                                                   
       if (i < size && j < size){                                                                                                                      
                                                                                                                             
           B[i*size+j] = (B[i*size+j]/(*sum_d_B));                                                                                              
                                                                                                                                                   
                                                                                                            
                                                                                                                                                      
                                                                                                                               
       }                                                                                                                                              
                                                                                                                                                   
}
#htendvar