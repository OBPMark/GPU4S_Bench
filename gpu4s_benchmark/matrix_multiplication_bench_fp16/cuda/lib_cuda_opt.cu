#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

 #include <mma.h>
 using namespace nvcuda;

 #define WMMA_M 16
 #define WMMA_N 16
 #define WMMA_K 16

 #ifdef FLOAT16
     __global__ void matrix_multiplication_kernel_tensor(bench_t_gpu *A,bench_t_gpu *B,  bench_t *C, const int n, const int m, const int w) {
    // Leading dimensions. Packed with no transpositions.
    int lda = m;
    int ldb = w;
    int ldc = w;
 
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
 
    wmma::fill_fragment(acc_frag, 0.0f);
 
    // Loop over k
    for (int i = 0; i < m; i += WMMA_K) {
       int aRow = warpM * WMMA_M;
       int aCol = i;
 
       int bRow = i;
       int bCol = warpN * WMMA_N;

        // Bounds checking
       if (aRow < n && aCol < m && bRow < m && bCol < w) {
        // Load the inputs
          wmma::load_matrix_sync(a_frag, A + (aRow * lda) + aCol, lda);
          wmma::load_matrix_sync(b_frag, B + (bRow * ldb) + bCol, ldb);
  
          // Perform the matrix multiplication
          wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
       }
    }
 
    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
 
    if (cRow < n && cCol < w) {
        wmma::load_matrix_sync(c_frag, C + (cRow * ldc) + cCol, ldc, wmma::mem_row_major); 
 
       for(int i=0; i < c_frag.num_elements; i++) {
          c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
       }
 
       // Store the output
       wmma::store_matrix_sync(C + (cRow * ldc) + cCol, c_frag, ldc, wmma::mem_row_major);
    }
 }

#endif

__global__ void 
matrix_multiplication_kernel(const bench_t *A, const bench_t *B, bench_t *C, const int n, const int m, const int w)
{
    __shared__ bench_t A_tile[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ bench_t B_tile[BLOCK_SIZE*BLOCK_SIZE];

    
    unsigned int i = blockIdx.x *  BLOCK_SIZE + threadIdx.x;
    unsigned int j = blockIdx.y *  BLOCK_SIZE + threadIdx.y;
    
    
    bench_t acumulated = 0;
    unsigned int idx = 0;
    // load memory
    for (unsigned int sub = 0; sub < gridDim.x; ++sub)
    {
        
        idx = i * n + sub * BLOCK_SIZE + threadIdx.y;

        if(idx >= m*n)
        {
            A_tile[threadIdx.x * BLOCK_SIZE+ threadIdx.y] = 0;
        }
        else
        {   
            A_tile[threadIdx.x * BLOCK_SIZE + threadIdx.y] = A[idx];
        }
        idx = (sub * BLOCK_SIZE + threadIdx.x) * w + j;

        if (idx >= m*w)
        {
            B_tile[threadIdx.x * BLOCK_SIZE +  threadIdx.y] = 0;
        }
        else
        {   
            B_tile[threadIdx.x* BLOCK_SIZE + threadIdx.y] = B[idx];
        }
        __syncthreads();
        for (unsigned int k = 0; k < BLOCK_SIZE; ++k)
        {
            acumulated +=  A_tile[threadIdx.x*BLOCK_SIZE + k] * B_tile[k*BLOCK_SIZE + threadIdx.y];
        }
        __syncthreads();

    }
    if (i < n && j < w)
    {
        
        C[i *n + j] = acumulated;
    }
}

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);
    
    #ifdef FLOAT16
        dim3 dimBlock(128, 4);
        dim3 dimGrid((n + (WMMA_M * dimBlock.x / 32 - 1)) / (WMMA_M * dimBlock.x / 32), (w + WMMA_N * dimBlock.y - 1) / (WMMA_N * dimBlock.y));
        matrix_multiplication_kernel_tensor<<<dimGrid, dimBlock>>> (deviceObj->d_half_A, deviceObj->d_half_B, deviceObj->d_C,  n, m, w);
        #else
        // Default tiled layout fallback for non-FP16 modes
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(ceil(float(w)/dimBlock.x), ceil(float(n)/dimBlock.y));
        matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(deviceObj->d_A, deviceObj->d_B, deviceObj->d_C, n, m, w);
    #endif

    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}


void copy_memory_to_host(GraficCommon* device_object, bench_t* h_C, int size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // device -> host
    Clock d2hCLK;

    // profilling start 
    d2hCLK.start();
    cudaEventRecord(*deviceObj->start_memory_copy_host);

    cudaError_t err = cudaMemcpy(h_C, deviceObj->d_C, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    // profilling end
    cudaEventRecord(*deviceObj->stop_memory_copy_host);
    d2hCLK.end();

    // store the hd2h time
    deviceObj->d2h_elapsed_time = d2hCLK.getElapsedMS();
}
