#include "../benchmark_library.h"

void execute_kernel(GraficCommon* device_object, unsigned int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// the output will be in the B array the lower half will be the lowpass filter and the half_up will be the high pass filter
	#ifdef INT

	// high part
	deviceObj->d_B[size] = deviceObj->d_A[1] - (int)( ((9.0/16.0) * (deviceObj->d_A[0] + deviceObj->d_A[2])) - ((1.0/16.0) * (deviceObj->d_A[2] + deviceObj->d_A[4])) + (1.0/2.0)); 
	deviceObj->d_B[2*size-2] = deviceObj->d_A[2*size - 3] - (int)( ((9.0/16.0) * (deviceObj->d_A[2*size -4] + deviceObj->d_A[2*size -2])) - ((1.0/16.0) * (deviceObj->d_A[2*size - 6] + deviceObj->d_A[2*size - 2])) + (1.0/2.0));
	deviceObj->d_B[2*size-1] = deviceObj->d_A[2*size - 1] - (int)( ((9.0/8.0) * (deviceObj->d_A[2*size -2])) -  ((1.0/8.0) * (deviceObj->d_A[2*size - 4])) + (1.0/2.0));
	#pragma omp parallel for
	for (unsigned int i = 1; i < size-2; ++i){
		//store
		deviceObj->d_B[i+size] = deviceObj->d_A[2*i+1] - (int)( ((9.0/16.0) * (deviceObj->d_A[2*i] + deviceObj->d_A[2*i+2])) - ((1.0/16.0) * (deviceObj->d_A[2*i - 2] + deviceObj->d_A[2*i + 4])) + (1.0/2.0));
	}
	
	// low_part
	deviceObj->d_B[0] = deviceObj->d_A[0] - (int)(- (deviceObj->d_B[size]/2.0) + (1.0/2.0));
	#pragma omp parallel for
	for (unsigned int i = 1; i < size; ++i){	
		deviceObj->d_B[i] = deviceObj->d_A[2*i] - (int)( - (( deviceObj->d_B[i + size -1] +  deviceObj->d_B[i + size])/ 4.0) + (1.0/2.0) );;
	}

	
	#else
	// flotating part
	unsigned int full_size = size * 2;
	int hi_start = -(LOWPASSFILTERSIZE / 2);
	int hi_end = LOWPASSFILTERSIZE / 2;
	int gi_start = -(HIGHPASSFILTERSIZE / 2 );
	int gi_end = HIGHPASSFILTERSIZE / 2;

	#pragma omp parallel for 
	for (unsigned int i = 0; i < size; ++i){
		// loop over N elements of the input vector.
		bench_t sum_value_low = 0;
		int x_position = 0;
		// first process the lowpass filter
		for (int hi = hi_start; hi < hi_end + 1; ++hi){
			x_position = (2 * i) + hi;
			if (x_position < 0) {
				// turn negative to positive
				x_position = x_position * -1;
			}
			else if (x_position > full_size - 1)
			{
				x_position = full_size - 1 - (x_position - (full_size -1 ));
			}
			// now I need to restore the hi value to work with the array
			sum_value_low += deviceObj->low_filter[hi + hi_end] * deviceObj->d_A[x_position];
			
		}
		// store the value
		deviceObj->d_B[i] = sum_value_low;
		bench_t sum_value_high = 0;
		// second process the Highpass filter
		for (int gi = gi_start; gi < gi_end + 1; ++gi){
			x_position = (2 * i) + gi + 1;
			if (x_position < 0) {
				// turn negative to positive
				x_position = x_position * -1;
			}
			else if (x_position >  full_size - 1)
			{
				x_position = full_size - 1 - (x_position - (full_size -1 ));
			}
			sum_value_high += deviceObj->high_filter[gi + gi_end] * deviceObj->d_A[x_position];
		}
		// store the value
		deviceObj->d_B[i+size] = sum_value_high;
	}

	#endif

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}

