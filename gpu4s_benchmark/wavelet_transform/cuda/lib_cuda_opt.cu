#include "../benchmark_library.h"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32

#ifdef INT
#define NUMBERSUBDIVISIONS  4
__global__ void
wavelet_transform_low(const bench_t *A, bench_t *B, const int n, unsigned int iter, unsigned int grid_size){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + ( grid_size *blockDim.x * iter); 

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
wavelet_transform(const bench_t *A, bench_t *B, const int n, unsigned int iter, unsigned int grid_size){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + ( grid_size *blockDim.x * iter);

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
/*__global__ void
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

        __syncthreads();
        // low_part
        for (unsigned int i = 0; i < size; ++i){
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

}*/
#else
__global__ void
wavelet_transform_high(const bench_t *A, bench_t *B, const int n, const bench_t *lowpass_filter,const bench_t *highpass_filter){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int full_size = size * 2;
	//int hi_start = -(LOWPASSFILTERSIZE / 2);
	//int hi_end = LOWPASSFILTERSIZE / 2;
	//int gi_start = -(HIGHPASSFILTERSIZE / 2 );
    const int gi_end = HIGHPASSFILTERSIZE / 2;
    
    if (i < size){
		bench_t sum_value_high = 0;
        // second process the Highpass filter
        //#pragma unroll
		//for (int gi = gi_start; gi < gi_end + 1; ++gi){
            //int x_position = (2 * i) + gi + 1;
            int x_position = (2 * i) + 1;
            x_position = x_position < 0 ? x_position * -1 : x_position > full_size - 1 ? full_size - 1 - (x_position - (full_size -1 )) : x_position;

            sum_value_high = (highpass_filter[-3 + gi_end] * A[ (x_position - 3) < 0 ? (x_position - 3) * -1 : (x_position - 3) > full_size - 1 ? full_size - 1 - ((x_position- 3) - (full_size -1 )) : (x_position- 3)]) + (highpass_filter[-2 + gi_end] * A[ (x_position - 2) < 0 ? (x_position - 2) * -1 : (x_position - 2) > full_size - 1 ? full_size - 1 - ((x_position- 2) - (full_size -1 )) : (x_position- 2)]) + (highpass_filter[-1 + gi_end] * A[ (x_position - 1) < 0 ? (x_position - 1) * -1 : (x_position - 1) > full_size - 1 ? full_size - 1 - ((x_position- 1) - (full_size -1 )) : (x_position- 1)]) + (highpass_filter[gi_end] * A[ (x_position) < 0 ? (x_position) * -1 : (x_position) > full_size - 1 ? full_size - 1 - ((x_position) - (full_size -1 )) : (x_position)]) + (highpass_filter[1 + gi_end] * A[ (x_position  + 1) < 0 ? (x_position + 1) * -1 : (x_position + 1) > full_size - 1 ? full_size - 1 - ((x_position + 1) - (full_size -1 )) : (x_position + 1)]) + (highpass_filter[2 + gi_end] * A[ (x_position + 2) < 0 ? (x_position + 2) * -1 : (x_position + 2) > full_size - 1 ? full_size - 1 - ((x_position + 2) - (full_size -1 )) : (x_position + 2)]) + (highpass_filter[3 + gi_end] * A[ (x_position + 3) < 0 ? (x_position + 3) * -1 : (x_position + 3) > full_size - 1 ? full_size - 1 - ((x_position + 3) - (full_size -1 )) : (x_position + 3)]);

			//sum_value_high += highpass_filter[gi + gi_end] * A[x_position];
		//}
		// store the value
		B[i+size] = sum_value_high;

    }
}
__global__ void
wavelet_transform(const bench_t *A, bench_t *B, const int n, const bench_t *lowpass_filter,const bench_t *highpass_filter){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int full_size = size * 2;
	//int hi_start = -(LOWPASSFILTERSIZE / 2);
	const int hi_end = LOWPASSFILTERSIZE / 2;
	//int gi_start = -(HIGHPASSFILTERSIZE / 2 );
    //int gi_end = HIGHPASSFILTERSIZE / 2;
    
    if (i < size){
        bench_t sum_value_low = 0;
        int x_position = (2 * i);
        sum_value_low = (lowpass_filter[-4 + hi_end] * A[(x_position -4) < 0 ? (x_position-4) * -1 : (x_position-4) > full_size - 1 ? full_size - 1 - ((x_position-4) - (full_size -1 )) : (x_position-4)]) + (lowpass_filter[-3 + hi_end] * A[(x_position -3) < 0 ? (x_position-3) * -1 : (x_position-3) > full_size - 1 ? full_size - 1 - ((x_position-3) - (full_size -1 )) : (x_position-3)]) + (lowpass_filter[-2 + hi_end] * A[(x_position -2) < 0 ? (x_position-2) * -1 : (x_position-2) > full_size - 1 ? full_size - 1 - ((x_position-2) - (full_size -1 )) : (x_position-2)]) + (lowpass_filter[-1 + hi_end] * A[(x_position -1) < 0 ? (x_position-1) * -1 : (x_position-1) > full_size - 1 ? full_size - 1 - ((x_position-1) - (full_size -1 )) : (x_position-1)]) + (lowpass_filter[hi_end] * A[(x_position) < 0 ? (x_position) * -1 : (x_position) > full_size - 1 ? full_size - 1 - ((x_position) - (full_size -1 )) : (x_position)]) + (lowpass_filter[1 + hi_end] * A[(x_position + 1) < 0 ? (x_position + 1) * -1 : (x_position + 1) > full_size - 1 ? full_size - 1 - ((x_position + 1) - (full_size -1 )) : (x_position+1)]) + (lowpass_filter[+2 + hi_end] * A[(x_position +2) < 0 ? (x_position+2) * -1 : (x_position+2) > full_size - 1 ? full_size - 1 - ((x_position+2) - (full_size -1 )) : (x_position+2)]) + (lowpass_filter[3 + hi_end] * A[(x_position +3) < 0 ? (x_position + 3) * -1 : (x_position + 3) > full_size - 1 ? full_size - 1 - ((x_position+3) - (full_size -1 )) : (x_position+3)]) + (lowpass_filter[4 + hi_end] * A[(x_position +4) < 0 ? (x_position+4) * -1 : (x_position+4) > full_size - 1 ? full_size - 1 - ((x_position+4) - (full_size -1 )) : (x_position+4)]);
		B[i] = sum_value_low;

    }
}
#endif

void execute_kernel(GraficCommon* device_object, unsigned int n){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    #ifdef INT
    dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n/NUMBERSUBDIVISIONS)/dimBlock.x));
    
    cudaStream_t cuda_streams[NUMBERSUBDIVISIONS];
    for (unsigned int streams = 0; streams < NUMBERSUBDIVISIONS; ++streams)
    {
        cudaStreamCreate(&cuda_streams[streams]);
    }
    
    for (unsigned int iter = 0; iter < NUMBERSUBDIVISIONS; ++iter)
    {   
        wavelet_transform<<<dimGrid,dimBlock,0,cuda_streams[iter]>>>(deviceObj->d_A, deviceObj->d_B, n, iter,dimGrid.x);
        wavelet_transform_low<<<dimGrid,dimBlock,0,cuda_streams[iter]>>>(deviceObj->d_A, deviceObj->d_B, n, iter,dimGrid.x);
    }
    #else
        cudaStream_t cuda_streams[2];
        dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid(ceil(float(n)/dimBlock.x));
        for (unsigned int streams = 0; streams < 2; ++streams)
        {
            cudaStreamCreate(&cuda_streams[streams]);
        }
        wavelet_transform<<<dimGrid,dimBlock,0,cuda_streams[0]>>>(deviceObj->d_A, deviceObj->d_B, n, deviceObj->low_filter, deviceObj->high_filter);
        wavelet_transform_high<<<dimGrid,dimBlock,0,cuda_streams[1]>>>(deviceObj->d_A, deviceObj->d_B, n, deviceObj->low_filter, deviceObj->high_filter);
    #endif

    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}
