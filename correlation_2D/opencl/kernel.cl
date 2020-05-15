#htvar kernel_code
void kernel wavelet_transform(global const bench_t* A,  global bench_t* B , const int n, global const bench_t* lowpass_filter, global const bench_t* highpass_filter ){	    
		int i = get_global_id(0);																													                                    																				    								                                    
      unsigned int size = n;

      unsigned int full_size = size * 2;
      int hi_start = -(LOWPASSFILTERSIZE / 2);
      int hi_end = LOWPASSFILTERSIZE / 2;
      int gi_start = -(HIGHPASSFILTERSIZE / 2 );
      int gi_end = HIGHPASSFILTERSIZE / 2;
      if (i < size){
         bench_t sum_value_low = 0;
         for (int hi = hi_start; hi < hi_end + 1; ++hi){
            int x_position = (2 * i) + hi;
            if (x_position < 0) {
               // turn negative to positive
               x_position = x_position * -1;
            }
            else if (x_position > full_size - 1)
            {
               x_position = full_size - 1 - (x_position - (full_size -1 ));;
            }
            // now I need to restore the hi value to work with the array
            sum_value_low += lowpass_filter[hi + hi_end] * A[x_position];
            
         }
         // store the value
         B[i] = sum_value_low;
         bench_t sum_value_high = 0;
         // second process the Highpass filter
         for (int gi = gi_start; gi < gi_end + 1; ++gi){
            int x_position = (2 * i) + gi + 1;
            if (x_position < 0) {
               // turn negative to positive
               x_position = x_position * -1;
            }
            else if (x_position >  full_size - 1)
            {
               x_position = full_size - 1 - (x_position - (full_size -1 ));
            }
            sum_value_high += highpass_filter[gi + gi_end] * A[x_position];
         }
         // store the value

         B[i+size] = sum_value_high;

      }									                             																		                                    
                                           																										                                    
}
#htendvar