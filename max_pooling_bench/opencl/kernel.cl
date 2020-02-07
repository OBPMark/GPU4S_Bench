#htvar kernel_code
void kernel kernel_max(global const bench_t* A, global bench_t* B, const int size, const  int stride,  const  int lateral_stride ){	                    
		int i = get_global_id(0);																							
 		int j = get_global_id(1);																				    		
		if (i < size && j < size){																	 							
       bench_t max_value = A[((i * stride)) * size + ((j*stride))];                                                                                                                  
       	for(unsigned int x = 0; x < stride; ++x)																							
			{										 								
				for(unsigned int y = 0; y < stride; ++y)										             														
			    {										             				 
			    	max_value = max(max_value, A[((i * stride) + x) * size + ((j*stride) +y)]);								                             												
               }                                                                                       
           }                                                                                           
           B[i * lateral_stride + j ] = max_value;                                                                                            
																										
		}										                             													
                                           																				
}
#htendvar