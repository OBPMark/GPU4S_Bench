#htvar kernel_code

void kernel kernel_matrix_convolution(global const bench_t* A,  global bench_t* B, global const bench_t* kernel_data, const int n, const int m, const int w, const int kernel_size ) {  
    int x = get_global_id(0);																								    
    int y = get_global_id(1);																				    				    
    unsigned int size = n;                                                                                                                                                          
    int kernel_rad = kernel_size / 2;                                                                                                                                               
    bench_t sum = 0;                                                                                                                                                                
                                                                                                                                                                                       
    if (x < size && y < size){																	 									    
       	for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3												    
        {									 														                            
            for(int j = -kernel_rad; j <= kernel_rad; ++j)										             							    
            {											             										                      
                bench_t value = 0;								                             											    
                if (i + x < 0 || j + y < 0)                                                                                                                                        
                {                                                                                                                                                                  
                    value = 0;                                                                                                                                                      
                }                                                                                                                                                                  
                else if ( i + x > size - 1 || j + y > size - 1)                                                                                                                    
                {                                                                                                                                                                  
                    value = 0;                                                                                                                                                     
                }                                                                                                                                                                  
                else                                                                                                                                                               
                {                                                                                                                                                                  
                    value = A[(x + i)*size+(y + j)];                                                                                                                                
                }                                                                                                                                                                  
                sum += value * kernel_data[(i+kernel_rad)* kernel_size + (j+kernel_rad)];                                                                                          
            }                                                                                                                                                                     
        }                                                                                                                                                                         
        B[x*size+y ] = sum;                                                                                                                                                         
    }										                             											                                                             																				                      
}																				 									          
                                                                                                                                                                                       
void kernel kernel_relu(global const bench_t* A, global bench_t* B, const int size ){                                                                                                  
       int i = get_global_id(0);                                                                                                                                                       
       int j = get_global_id(1);                                                                                                                                                       
       if (i < size && j < size){                                                                                                                                                      
           bench_t threshold = 0;                                                                                                                                                      
           B[i*size+j] = max(threshold, A[i*size+j]);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
       }                                                                                                                                                                                                                                                                                                                                                                         
}                                                                                                                                                                                      

void kernel kernel_max(global const bench_t* A, global bench_t* B, const int size, const  int stride,  const  int lateral_stride ){                                                    
    int i = get_global_id(0);                                                                                                                                                       
    int j = get_global_id(1);                                                                                                                                                       
    if (i < size && j < size) {                                                                                                                                                      
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

void kernel kernel_lrn(global const bench_t* A, global bench_t* B, const int size, const bench_t K, const bench_t ALPHA, const bench_t BETA ){                                         
    int i = get_global_id(0);                                                                                                                                                       
    int j = get_global_id(1);                                                                                                                                                       
    if (i < size && j < size){                                                                                                                                                                                                                                                                                                                                             
        B[i*size+j] = A[i*size+j]/pow((K+ALPHA*pow(A[i*size+j],2)),BETA);                                                                                                                                                                                                                                                                                   
    }                                                                                                                                                                                                                                                                                                                                                                      
}

void kernel kernel_matrix_multiplication(global const bench_t* A, const global bench_t* B, global bench_t* C, const int n, const int m, const int w ){                                 
    int i = get_global_id(0);                                                                                                                                                       
    int j = get_global_id(1);                                                                                                                                                       
    if (i < n && j < m){                                                                                                                                                            
        bench_t acumulated = 0;                                                                                                                                                     
        for (unsigned int k_d = 0; k_d < w; ++k_d )                                                                                                                                 
        {                                                                                                                                                                       
            acumulated += A[i*w+k_d] * B[k_d*m +j];                                                                                                                              
        }                                                                                                                                                                       
        C[i*m+j] =  acumulated;                                                                                                                                                 
    }                                                                                                                                                                                                                                                                                                                                                                  
}                                                                                                                                                                                   

void kernel kernel_relu_linear(global const bench_t* A, global bench_t* B, const int size ){                                                                                                                                                                                       
    int i = get_global_id(0);                                                                                                                                                                              
    if (i < size){                                                                                                                                                                             
        bench_t threshold = 0;                                                                                                                                                                              
        B[i] = max(threshold, A[i]);                                                                                                                                                                              
    }                                                                                                                                                                                       
}                                                                                                                                                                                       

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