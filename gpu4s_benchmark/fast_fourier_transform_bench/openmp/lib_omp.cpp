#include "../benchmark_library.h"
#include <cmath>
#include <cstring>

bool device_memory_init(GraficCommon* device_object, int64_t size)
{
	return true;
}

void execute_kernel(GraficCommon* device_object, int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	int64_t loop_w = 0, loop_for_1 = 0, loop_for_2 = 0; 
	int64_t n, mmax, m, j, istep, i;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
 
    // reverse-binary reindexing
    n = size<<1;
    j=1;

	#pragma parallel for schedule(static)
    for (i=1; i<n; i+=2) {
        if (j>i) {
            std::swap(deviceObj->d_Br[j-1], deviceObj->d_Br[i-1]);
            std::swap(deviceObj->d_Br[j], deviceObj->d_Br[i]);
        }
        m = size;
        while (m>=2 && j>m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    };

    // here begins the Danielson-Lanczos section
    mmax=2;
    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

		#pragma parallel for collapse(2)
        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
                j=i+mmax;
                tempr = wr*deviceObj->d_Br[j-1] - wi*deviceObj->d_Br[j];
                tempi = wr * deviceObj->d_Br[j] + wi*deviceObj->d_Br[j-1];
 				
                deviceObj->d_Br[j-1] = deviceObj->d_Br[i-1] - tempr;
                deviceObj->d_Br[j] = deviceObj->d_Br[i] - tempi;
                deviceObj->d_Br[i-1] += tempr;
                deviceObj->d_Br[i] += tempi;
                ++loop_for_1;
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
	
	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size)
{
    return;
}


void clean(GraficCommon* device_object)
{
    return;
}