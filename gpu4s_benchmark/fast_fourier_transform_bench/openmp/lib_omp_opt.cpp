#include "../benchmark_library.h"
#include <cmath>
#include <cstring>


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_B,int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_B = h_B;
}


void execute_kernel(GraficCommon* device_object, int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	const unsigned int mode = (unsigned int)log2(size);
	unsigned int position = 0;
	unsigned int j, i, z, l, istep, mmax, n, m = 0;

	#pragma omp parallel for private(j,i,position)
	for(i = 0; i < size; ++i)
	{
		j = i;                                                                                                    
		j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;                                                                      
		j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;                                                                      
		j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;                                                                      
		j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;                                                                      
		j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;                                                                    
		j >>= (32-mode);                                                                                                       
		position = j * 2;                                                                                                       																											
		deviceObj->d_Br[position] = deviceObj->d_B[i *2];                                                                                                
		deviceObj->d_Br[position + 1] = deviceObj->d_B[i *2 + 1];  
	}

	bench_t wpr, wpi, theta, wi, tempr, tempi, wtemp, wr = 0.f;
    mmax=2;
	n = size << 1;

	bench_t* a = deviceObj->d_Br;

    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
		#pragma omp task untied
        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
				j=i+mmax;
				tempr = wr*a[j-1] - wi*a[j];
				tempi = wr * a[j] + wi*a[j-1];
                a[j-1] = a[i-1] - tempr;
                a[j] = a[i] - tempi;
                a[i-1] += tempr;
                a[i] += tempi;
            }
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
        }
        mmax=istep;
    }
	
	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}



