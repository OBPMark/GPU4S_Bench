#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32

#ifdef INT
__global__ void
wavelet_transform_low(const bench_t *A, bench_t *B, const int n){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        bench_t sum_value_low = 0;
        if(i == 0){
            sum_value_low = A[0] - (int)(- (B[size]/2.0) + (1.0/2.0));
        }
        else
        {
            sum_value_low = A[2*i] - (int)( - (( B[i + size -1] +  B[i + size])/ 4.0) + (1.0/2.0) );
        }
        
        B[i] = sum_value_low;
    }
}
__global__ void
wavelet_transform(const bench_t *A, bench_t *B, const int n){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        bench_t sum_value_high = 0;
        // specific cases
        if(i == 0){
            sum_value_high = A[1] - (int)( ((9.0/16.0) * (A[0] + A[2])) - ((1.0/16.0) * (A[2] + A[4])) + (1.0/2.0));
        }
        else if(i == size -2){
            sum_value_high = A[2*size - 3] - (int)( ((9.0/16.0) * (A[2*size -4] + A[2*size -2])) - ((1.0/16.0) * (A[2*size - 6] + A[2*size - 2])) + (1.0/2.0));
        }
        else if(i == size - 1){
            sum_value_high = A[2*size - 1] - (int)( ((9.0/8.0) * (A[2*size -2])) -  ((1.0/8.0) * (A[2*size - 4])) + (1.0/2.0));
        }
        else{
            // generic case
            sum_value_high = A[2*i+1] - (int)( ((9.0/16.0) * (A[2*i] + A[2*i+2])) - ((1.0/16.0) * (A[2*i - 2] + A[2*i + 4])) + (1.0/2.0));
        }
        
        //store
        B[i+size] = sum_value_high;

        //__syncthreads();
        // low_part
        //for (unsigned int i = 0; i < size; ++i){
        
        //}
    }

}
#else
__global__ void
wavelet_transform(const bench_t *A, bench_t *B, const int n, const bench_t *lowpass_filter,const bench_t *highpass_filter){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int full_size = size * 2;
	int hi_start = -(LOWPASSFILTERSIZE / 2);
	int hi_end = LOWPASSFILTERSIZE / 2;
	int gi_start = -(HIGHPASSFILTERSIZE / 2 );
    int gi_end = HIGHPASSFILTERSIZE / 2;
    
    if (i < size){
        bench_t sum_value_low = 0;
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
			sum_value_low += lowpass_filter[hi + hi_end] * A[x_position];
			
        }
		// store the value
		B[i] = sum_value_low;
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
			sum_value_high += highpass_filter[gi + gi_end] * A[x_position];
		}
		// store the value
		B[i+size] = sum_value_high;

    }
}
#endif

void execute_kernel(GraficCommon* device_object, unsigned int n){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x));
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    #ifdef INT
    wavelet_transform<<<dimGrid,dimBlock>>>(deviceObj->d_A, deviceObj->d_B, n);
    wavelet_transform_low<<<dimGrid,dimBlock>>>(deviceObj->d_A, deviceObj->d_B, n);
    #else
    wavelet_transform<<<dimGrid,dimBlock>>>(deviceObj->d_A, deviceObj->d_B, n, deviceObj->low_filter, deviceObj->high_filter);
    #endif

    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}
