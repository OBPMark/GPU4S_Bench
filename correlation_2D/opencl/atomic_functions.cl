#htifdef FLOAT

#htvar atomic_code
void atomic_add_global(volatile global float *source, const float operand) {                                                                       
           union {                                                                                                                                        
                 unsigned int intVal;                                                                                                                                  
                 float floatVal;                                                                                                                                  
                 } newVal;                                                                                                                                  
           union {                                                                                                                                        
                  unsigned int intVal;                                                                                                                                 
                  float floatVal;                                                                                                                                 
                  } prevVal;                                                                                                                                 
                                                                                                                                                   
                                                                                                                                                   
            do {                                                                                                                                       
                 prevVal.floatVal = *source;                                                                                                                                  
                 newVal.floatVal = prevVal.floatVal + operand;                                                                                                                                  
                } while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);                                                                                                                                   
}
#htendvar                                                                                                                                          

#htelse

#htvar string atomic_code
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable                                                                                                                                                       
void atomic_add_global(volatile global double *source, const double operand) {                                                                       
           union {                                                                                                                                        
                 unsigned long int intVal;                                                                                                                                  
                 double floatVal;                                                                                                                                  
                 } newVal;                                                                                                                                  
           union {                                                                                                                                        
                  unsigned long int intVal;                                                                                                                                 
                  double floatVal;                                                                                                                                 
                  } prevVal;                                                                                                                                 
                                                                                                                                                   
                                                                                                                                                   
            do {                                                                                                                                       
                 prevVal.floatVal = *source;                                                                                                                                  
                 newVal.floatVal = prevVal.floatVal + operand;                                                                                                                                  
                } while (atomic_cmpxchg((volatile global unsigned long int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);                                                                                                                                   
 }
#htendvar

#htendif                        