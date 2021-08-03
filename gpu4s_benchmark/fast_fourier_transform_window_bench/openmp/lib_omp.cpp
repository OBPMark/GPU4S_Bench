#include "../benchmark_library.h"
#include <cstring>
#include <cmath>


void init(GraficObject *device_object, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}

void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	init(device_object, device_name);
}

bool device_memory_init(GraficObject *device_object,  int64_t size_a_array, int64_t size_b_array)
{
	device_object->d_B = (bench_t*) malloc ( size_b_array * sizeof(bench_t*));
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A,int64_t size)
{
	device_object->d_A = h_A;
}


void aux_fft_function(GraficObject* device_object, int64_t nn, int64_t start_pos){
    
	bench_t Br[nn];
	// copy values of the  window to output
	for(unsigned int j = 0; j < nn ; ++j){
		Br[j] = device_object->d_A[start_pos+j];
	}
	
    int64_t n, mmax, m, j, istep, i , window = nn;
    
    // reverse-binary reindexing for all data 
    nn = nn>>1;

	const unsigned int mode = (unsigned int)log2(nn);
	unsigned int position = 0;
	for(i = 0; i < nn; ++i)
	{
		j = i;                                                                                                    
		j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;                                                                      
		j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;                                                                      
		j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;                                                                      
		j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;                                                                      
		j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;                                                                    
		j >>= (32-mode);                                                                                                       
		position = j * 2;                                                                                                       																											
		device_object->d_B[(start_pos * window) + position] = Br[i *2];
		device_object->d_B[(start_pos * window) + position + 1] = Br[i *2 + 1];  
	}

    
	bench_t wtemp, wpr, wpi, wi, theta, tempr, tempi, wr = 0.f;
    mmax=2;
	n = nn<<1;

    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
                j=i+mmax;
                tempr = wr*device_object->d_B[(start_pos * window) + j-1] - wi*device_object->d_B[(start_pos * window) +j];
                tempi = wr * device_object->d_B[(start_pos * window) + j] + wi*device_object->d_B[(start_pos * window) + j-1];
                device_object->d_B[(start_pos * window) + j-1] = device_object->d_B[(start_pos * window) + i-1] - tempr;
                device_object->d_B[(start_pos * window) +j] = device_object->d_B[(start_pos * window) + i] - tempi;
                device_object->d_B[(start_pos * window) + i-1] += tempr;
                device_object->d_B[(start_pos * window) +i] += tempi;
            }
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
        }
        mmax=istep;
    }
}


void execute_kernel(GraficObject *device_object, int64_t window, int64_t size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	#pragma omp parallel for
	for (unsigned int i = 0; i < (size * 2 - window + 1); i+=2){
        aux_fft_function(device_object, window, i);
    }

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size)
{	     
	memcpy(h_B, &device_object->d_B[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time * 1000.f);
		setvbuf(stdout, NULL, _IONBF, 0); 
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_B);
}