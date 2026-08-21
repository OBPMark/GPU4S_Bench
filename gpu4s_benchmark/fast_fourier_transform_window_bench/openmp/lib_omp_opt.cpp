#include "../benchmark_library.h"
#include <cstring>
#include <cmath>


void aux_fft_function(GraficCommon* device_object, int64_t nn, int64_t start_pos){
    
GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    
    unsigned int window_idx = start_pos;
    bench_t* b_out = &deviceObj->d_B[window_idx * nn];

	// copy values of the  window to output
	for(unsigned int j = 0; j < nn ; ++j){
		b_out[j] = deviceObj->d_A[start_pos+j];
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
            std::swap(b_out[j-1], b_out[i-1]); // Use b_out!
            std::swap(b_out[j], b_out[i]);     // Use b_out!
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
                tempr = wr * b_out[j-1] - wi * b_out[j];
                tempi = wr * b_out[j]   + wi * b_out[j-1];

                b_out[j-1]  = b_out[i-1] - tempr;
                b_out[j]    = b_out[i]   - tempi;
                b_out[i-1] += tempr;
                b_out[i]   += tempi;
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


void execute_kernel(GraficCommon* device_object, int64_t window, int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	#pragma omp parallel for
	for (int64_t i = 0; i < (size * 2 - window + 1); i+=2){
        aux_fft_function(device_object, window, i);
    }

	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}
