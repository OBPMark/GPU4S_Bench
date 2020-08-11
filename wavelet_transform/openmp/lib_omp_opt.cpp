#include "../benchmark_library.h"
#include <cstring>

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	#ifdef FLOAT
	device_object->low_filter = (bench_t*) malloc (LOWPASSFILTERSIZE * sizeof(bench_t));
	device_object->high_filter = (bench_t*) malloc (HIGHPASSFILTERSIZE * sizeof(bench_t));
	#endif
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a)
{
	device_object->d_A = h_A;
	#ifdef FLOAT
	memcpy(&device_object->low_filter[0], lowpass_filter, sizeof(bench_t)*LOWPASSFILTERSIZE);
	memcpy(&device_object->high_filter[0], highpass_filter, sizeof(bench_t)*HIGHPASSFILTERSIZE);
	#endif
}


void execute_kernel(GraficObject *device_object, unsigned int size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// the output will be in the B array the lower half will be the lowpass filter and the half_up will be the high pass filter
	#ifdef INT

	// high part
	device_object->d_B[size] = device_object->d_A[1] - (int)( ((9.0/16.0) * (device_object->d_A[0] + device_object->d_A[2])) - ((1.0/16.0) * (device_object->d_A[2] + device_object->d_A[4])) + (1.0/2.0)); 
	device_object->d_B[2*size-2] = device_object->d_A[2*size - 3] - (int)( ((9.0/16.0) * (device_object->d_A[2*size -4] + device_object->d_A[2*size -2])) - ((1.0/16.0) * (device_object->d_A[2*size - 6] + device_object->d_A[2*size - 2])) + (1.0/2.0));
	device_object->d_B[2*size-1] = device_object->d_A[2*size - 1] - (int)( ((9.0/8.0) * (device_object->d_A[2*size -2])) -  ((1.0/8.0) * (device_object->d_A[2*size - 4])) + (1.0/2.0));
	#pragma omp parallel for
	for (unsigned int i = 1; i < size-2; ++i){
		//store
		device_object->d_B[i+size] = device_object->d_A[2*i+1] - (int)( ((9.0/16.0) * (device_object->d_A[2*i] + device_object->d_A[2*i+2])) - ((1.0/16.0) * (device_object->d_A[2*i - 2] + device_object->d_A[2*i + 4])) + (1.0/2.0));
	}
	
	// low_part
	device_object->d_B[0] = device_object->d_A[0] - (int)(- (device_object->d_B[size]/2.0) + (1.0/2.0));
	#pragma omp parallel for
	for (unsigned int i = 1; i < size; ++i){	
		device_object->d_B[i] = device_object->d_A[2*i] - (int)( - (( device_object->d_B[i + size -1] +  device_object->d_B[i + size])/ 4.0) + (1.0/2.0) );;
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
			sum_value_low += device_object->low_filter[hi + hi_end] * device_object->d_A[x_position];
			
		}
		// store the value
		device_object->d_B[i] = sum_value_low;
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
			sum_value_high += device_object->high_filter[gi + gi_end] * device_object->d_A[x_position];
		}
		// store the value
		device_object->d_B[i+size] = sum_value_high;
	}

	#endif

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format)
{
	if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f miliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f miliseconds\n", device_object->elapsed_time * 1000.f);
		setvbuf(stdout, NULL, _IONBF, 0); 
		printf("Elapsed time Device->Host: %.10f miliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_B);
	free(device_object->low_filter);
	free(device_object->high_filter);
}