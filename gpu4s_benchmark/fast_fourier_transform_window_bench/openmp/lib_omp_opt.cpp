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
    
	// copy values of the  window to output
	for(unsigned int j = 0; j < nn ; ++j){
		device_object->d_B[start_pos * nn + j] = device_object->d_A[start_pos+j];
	}
	
	
	int64_t loop_w = 0, loop_for_1 = 0, loop_for_2 = 0; 
    int64_t n, mmax, m, j, istep, i , window = nn;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
    // reverse-binary reindexing for all data 
    nn = nn>>1;

    n = nn<<1;
    //printf(" nn %ld n %ld window %ld start_pos %ld,\n",nn, n, window, start_pos);
    j=1;

    for (i=1; i<n; i+=2) {
        if (j>i) {
            std::swap(device_object->d_B[(start_pos * window) + (j-1)], device_object->d_B[(start_pos * window) + (i-1)]);
            std::swap(device_object->d_B[(start_pos * window) + j], device_object->d_B[(start_pos * window) + i]);
            //printf("i %lu j %lu data %f \n",i ,j, data[(start_pos * window) + (j-1)] );
        }
        m = nn;
        while (m>=2 && j>m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    };
    
    // here begins the Danielson-Lanczos section for each window
    mmax=2;
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
                ++loop_for_1;
                //printf("wr %f wi %f\n", wr, wi);
            }
            loop_for_1 = 0;
            
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            ++loop_for_2;

        }
        loop_for_2 = 0;
        mmax=istep;
    ++loop_w;    
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