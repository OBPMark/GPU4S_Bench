#include "../benchmark_library.h"
#include <cstring>
#include <cmath>
#include <vector>


void aux_fft_function(GraficCommon* device_object, int64_t nn, int64_t start_pos){
    
GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    
	std::vector<bench_t> Br(nn);
	// copy values of the  window to output
	for(unsigned int j = 0; j < nn ; ++j){
		Br[j] = deviceObj->d_A[start_pos+j];
	}
	
    int64_t n, mmax, m, j, istep, i , window = nn;
	unsigned int window_idx = start_pos;
    
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
		deviceObj->d_B[(window_idx * window) + position] = Br[i *2];
		deviceObj->d_B[(window_idx * window) + position + 1] = Br[i *2 + 1];  
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
                tempr = wr * deviceObj->d_B[(window_idx * window) + j-1] - wi * deviceObj->d_B[(window_idx * window) + j];
				tempi = wr * deviceObj->d_B[(window_idx * window) + j]   + wi * deviceObj->d_B[(window_idx * window) + j-1]; 

                deviceObj->d_B[(window_idx * window) + j-1]  = deviceObj->d_B[(window_idx * window) + i-1] - tempr;
                deviceObj->d_B[(window_idx * window) +j] 	 = deviceObj->d_B[(window_idx * window) + i] - tempi;
                deviceObj->d_B[(window_idx * window) + i-1] += tempr;
                deviceObj->d_B[(window_idx * window) +i] 	+= tempi;
            }
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
        }
        mmax=istep;
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

