
#include "../benchmark_library.h"
#include <cmath>
#include <cstring>
#include <fftw3.h>


void execute_kernel(GraficCommon* device_object, int64_t size)
{
	GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// FFTW implementation
	fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_num_threads());
	fftw_plan plan;
	fftw_complex *in, *out;

	in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
	
	for(int i = 0; i < size; ++i) {
		in[i][0] = deviceObj->d_B[i*2];
		in[i][1] = deviceObj->d_B[i*2+1];
	}

	plan = fftw_plan_dft_1d(size,in,out,FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(plan);
	
	for (int64_t i=0; i<size; i++)
	{
		deviceObj->d_Br[i*2] = out[i][0];
		deviceObj->d_Br[i*2+1] = out[i][1];

	}
	
	fftw_destroy_plan(plan);
	fftw_free(in); fftw_free(out);
	
	// End compute timer
	deviceObj->elapsed_time = omp_get_wtime() - start_wtime;
}


