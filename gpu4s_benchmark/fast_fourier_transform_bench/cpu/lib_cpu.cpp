#include "../benchmark_library.h"
#include <cmath>
#include <cstring>

void init(GraficCommon* device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficCommon* device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficCommon* device_object, int64_t size)
{
	return true;
}


void copy_memory_to_device(GraficCommon* device_object, bench_t* h_B,int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	deviceObj->d_Br = h_B;
}


void execute_kernel(GraficCommon* device_object, int64_t size)
{
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    Clock kernelCLK;
	kernelCLK.start();

	int64_t loop_w = 0, loop_for_1 = 0, loop_for_2 = 0; 
	int64_t n, mmax, m, j, istep, i;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
 
    // reverse-binary reindexing
    n = size<<1;
    j=1;

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
    kernelCLK.end();
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_B, int64_t size)
{
    return;
}


float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",(bench_t) 0, deviceObj->elapsed_time , (bench_t) 0, current_time);
    }
    else if (csv_format){
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, deviceObj->elapsed_time, (bench_t) 0);
    } 
	else
	{
        //--- FIX: print te time in milliseconds
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", deviceObj->elapsed_time );
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
    return deviceObj->elapsed_time;
}


void clean(GraficCommon* device_object)
{
    return;
}