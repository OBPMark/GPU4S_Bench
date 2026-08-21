#include "../benchmark_library.h"
#include <cstring>

void init(GraficCommon* device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficCommon* device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	#ifdef FLOAT
	deviceObj->low_filter = (bench_t*) malloc (LOWPASSFILTERSIZE * sizeof(bench_t));
	deviceObj->high_filter = (bench_t*) malloc (HIGHPASSFILTERSIZE * sizeof(bench_t));
	#endif
   	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_A, unsigned int size_a)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_A = h_A;
	#ifdef FLOAT
	memcpy(&deviceObj->low_filter[0], lowpass_filter, sizeof(bench_t)*LOWPASSFILTERSIZE);
	memcpy(&deviceObj->high_filter[0], highpass_filter, sizeof(bench_t)*HIGHPASSFILTERSIZE);
	#endif
}


void execute_kernel(GraficCommon* device_object, unsigned int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	Clock kernelCLK;
	kernelCLK.start();

	// the output will be in the B array the lower half will be the lowpass filter and the half_up will be the high pass filter
	#ifdef INT
	unsigned int full_size = size * 2;
	// integer computation
	// high part
	for (unsigned int i = 0; i < size; ++i){
		bench_t sum_value_high = 0;
		// specific cases
		if(i == 0){
			sum_value_high = deviceObj->d_A[1] - (int)( ((9.0/16.0) * (deviceObj->d_A[0] + deviceObj->d_A[2])) - ((1.0/16.0) * (deviceObj->d_A[2] + deviceObj->d_A[4])) + (1.0/2.0));
		}
		else if(i == size -2){
			sum_value_high = deviceObj->d_A[2*size - 3] - (int)( ((9.0/16.0) * (deviceObj->d_A[2*size -4] + deviceObj->d_A[2*size -2])) - ((1.0/16.0) * (deviceObj->d_A[2*size - 6] + deviceObj->d_A[2*size - 2])) + (1.0/2.0));
		}
		else if(i == size - 1){
			sum_value_high = deviceObj->d_A[2*size - 1] - (int)( ((9.0/8.0) * (deviceObj->d_A[2*size -2])) -  ((1.0/8.0) * (deviceObj->d_A[2*size - 4])) + (1.0/2.0));
		}
		else{
			// generic case
			sum_value_high = deviceObj->d_A[2*i+1] - (int)( ((9.0/16.0) * (deviceObj->d_A[2*i] + deviceObj->d_A[2*i+2])) - ((1.0/16.0) * (deviceObj->d_A[2*i - 2] + deviceObj->d_A[2*i + 4])) + (1.0/2.0));
		}
		
		//store
		deviceObj->d_B[i+size] = sum_value_high;

	

	}
	// low_part
	for (unsigned int i = 0; i < size; ++i){
		bench_t sum_value_low = 0;
		if(i == 0){
			sum_value_low = deviceObj->d_A[0] - (int)(- (deviceObj->d_B[size]/2.0) + (1.0/2.0));
		}
		else
		{
			sum_value_low = deviceObj->d_A[2*i] - (int)( - (( deviceObj->d_B[i + size -1] +  deviceObj->d_B[i + size])/ 4.0) + (1.0/2.0) );
		}
		
		deviceObj->d_B[i] = sum_value_low;
	}

	
	#else
	// flotating part
	unsigned int full_size = size * 2;
	int hi_start = -(LOWPASSFILTERSIZE / 2);
	int hi_end = LOWPASSFILTERSIZE / 2;
	int gi_start = -(HIGHPASSFILTERSIZE / 2 );
	int gi_end = HIGHPASSFILTERSIZE / 2;

	for (unsigned int i = 0; i < size; ++i){
		// loop over N elements of the input vector.
		bench_t sum_value_low = 0;
		// first process the lowpass filter
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
			sum_value_low += deviceObj->low_filter[hi + hi_end] * deviceObj->d_A[x_position];
			
		}
		// store the value
		deviceObj->d_B[i] = sum_value_low;
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
			sum_value_high += deviceObj->high_filter[gi + gi_end] * deviceObj->d_A[x_position];
		}
		// store the value
		deviceObj->d_B[i+size] = sum_value_high;
	}

	#endif

	// End compute timer
	kernelCLK.end();
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);	     
	memcpy(h_C, &deviceObj->d_B[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0,current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0);
    } 
	else
	{
		//--- FIX: print te time in milliseconds
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time );
		setvbuf(stdout, NULL, _IONBF, 0); 
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return deviceObj->elapsed_time;
}


void clean(GraficCommon* device_object)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	free(deviceObj->d_B);
	free(deviceObj->low_filter);
	free(deviceObj->high_filter);
}