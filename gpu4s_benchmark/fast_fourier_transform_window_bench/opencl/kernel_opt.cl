#htvar kernel_code
void kernel binary_reverse_kernel(global const bench_t* A, global bench_t* B, const long int size, const unsigned int group, const unsigned long int position_off){    
       int id = get_global_id(0);                                                                                                                                      
       unsigned int position = 0;                                                                                                                                      
                                                                                                                                                                       
       if (id < size)                                                                                                                                                  
       {                                                                                                                                                               
                                                                                                                                                                       
           unsigned int j = id;                                                                                                                                        
           j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;                                                                                                          
           j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;                                                                                                          
           j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;                                                                                                          
           j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;                                                                                                          
           j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;                                                                                                        
           j >>= (32-group);                                                                                                                                           
           position = j * 2;                                                                                                                                           
                                                                                                                                                                       
           B[position + (size * 2 * position_off)] = A[(id *2) + position_off];                                                                                        
           B[position + 1 +  (size * 2 * position_off)] = A[(id *2 + 1) + position_off];                                                                               
       }                                                                                                                                                               
                                                                                                                                                                       
                                                                                                                                                                       
                                                                                                                                                                       
}                                                                                                                                                                      
void kernel fft_kernel(global bench_t* B, const int loop,const int theads, const bench_t wpr, const bench_t wpi, const long int size, const long int position_off){    
                                                                                                                                                                       
           bench_t tempr, tempi;                                                                                                                                       
           unsigned int i = get_global_id(0);                                                                                                                          
           unsigned int j;                                                                                                                                             
           unsigned int inner_loop;                                                                                                                                    
           unsigned int subset;                                                                                                                                        
           unsigned int id;                                                                                                                                               
                                                                                                                                                                       
           bench_t wr = 1.0;                                                                                                                                           
           bench_t wi = 0.0;                                                                                                                                           
           bench_t wtemp = 0.0;                                                                                                                                        
                                                                                                                                                                       
           subset = theads / loop;                                                                                                                                     
           id = i % subset;                                                                                                                                            
           inner_loop = i / subset;                                                                                                                                    
           //get wr and wi                                                                                                                                             
           for(unsigned int z = 0; z < inner_loop ; ++z){                                                                                                              
                  wtemp=wr;                                                                                                                                            
                  wr += wr*wpr - wi*wpi;                                                                                                                               
                  wi += wi*wpr + wtemp*wpi;                                                                                                                            
           }                                                                                                                                                           
           // get I                                                                                                                                                    
           i = id *(loop * 2 * 2) + 1 + (inner_loop * 2);                                                                                                              
           j=i+(loop * 2 );                                                                                                                                            
           tempr = wr*B[j-1 + (size * 2 * position_off)] - wi*B[j+ (size * 2 * position_off)];                                                                         
           tempi = wr * B[j+ (size * 2 * position_off)] + wi*B[j-1+ (size * 2 * position_off)];                                                                        
           B[j-1+ (size * 2 * position_off)] = B[i-1+ (size * 2 * position_off)] - tempr;                                                                                 
           B[j+ (size * 2 * position_off)] = B[i+ (size * 2 * position_off)] - tempi;                                                                                  
           B[i-1+ (size * 2 * position_off)] += tempr;                                                                                                                 
           B[i+ (size * 2 * position_off)] += tempi;                                                                                                                   
}
#htendvar                                                                                                                                                                    