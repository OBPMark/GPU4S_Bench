#include "../benchmark_library.h"



/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
covolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
{
    unsigned int size = n;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int kernel_rad = kernel_size / 2;

    bench_t sum = 0;

    if (x < size && y < size)
    {
        for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
            {
                for(int j = -kernel_rad; j <= kernel_rad; ++j){
                    // get value
                    bench_t value = 0;
                    
                    if (i + x < 0 || j + y < 0)
                    {
                        value = 0;
                        //printf("ENTRO %d %d\n", i + x , j + y);
                    }
                    else if ( i + x > size - 1 || j + y > size - 1)
                    {
                        value = 0;
                        //printf("ENTRO UPPER%d %d\n", i + x , j + y);
                    }
                    else
                    {
                        value = A[(x + i)*size+(y + j)];
                    }
                    //printf("ACHIVED position  %d %d value %f\n", (x + i) , (y + j), value);
                    sum += value * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
                }
            }
            
    B[x*size+y ] = sum;
    }
    
}
__global__ void
relu_kernel(const bench_t *A, bench_t *B, const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    bench_t threshold = 0;
    if (i < size && j < size){
        #ifdef INT
        B[i*size+j] = max(threshold, A[i*size+j]);
        #elif FLOAT
        B[i*size+j] = max(threshold, A[i*size+j]);
        #else
        B[i*size+j] = fmax(threshold, A[i*size+j]);
        #endif
    }
}
__global__ void
relu_linear_kernel(const bench_t *A, bench_t *B, const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    bench_t threshold = 0;
    if (i  < size){
        
        #ifdef INT
        B[i] = max(threshold, A[i]);
        #elif FLOAT
        B[i] = max(threshold, A[i]);
        #else
        B[i] = fmax(threshold, A[i]);
        
        #endif
    }
}
__global__ void
max_pooling_kernel(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (i < size && j < size){
        bench_t max_value = A[((i * stride)) * size + ((j*stride))];
        for(unsigned int x = 0; x < stride; ++x)
        {
            for(unsigned int y = 0; y < stride; ++y)
            {
                //printf("max %f, value %f, pos x %d, pos y %d \n", max_value, A[(i + x) * size + (j +y)],i + x , j +y);
                max_value = max(max_value, A[((i * stride) + x) * size + ((j*stride) +y)]);
                
            }
        }
        //printf("value %f, position %d, lateral_stride %d\n", max_value,i * lateral_stride + j, lateral_stride );
        B[i * lateral_stride + j ] = max_value;
    }
}
__global__ void
lrn_kernel(const bench_t *A, bench_t *B, const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        #ifdef INT
        B[i*size+j] = A[i*size+j]/powf((K+ALPHA*powf(A[i*size+j],2)),BETA);
        #elif FLOAT
        B[i*size+j] = A[i*size+j]/powf((K+ALPHA*powf(A[i*size+j],2)),BETA);
        #else
        B[i*size+j] = A[i*size+j]/powf((K+ALPHA*powf(A[i*size+j],2)),BETA);
        #endif
    }
}

__global__ void
matrix_multiplication_kernel(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m){
        bench_t acumulated = 0;
        for (unsigned int k_d = 0; k_d < w; ++k_d )
        {   
            //printf("position %d valor %f, k_d %d , i %d, j %d, n %d\n ", i*n+k_d, acumulated,k_d,w,i,n);
            acumulated += A[i*w+k_d] * B[k_d*m +j];
        }
       
        //printf("value %f position %d \n", acumulated,i *m + j);
        C[i*m+j] =  acumulated;
    }
}

__global__ void
softmax_kernel(const bench_t *A, bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        #ifdef INT
        B[i*size+j] = exp(A[i*size+j]);
        #elif FLOAT
        B[i*size+j] = expf(A[i*size+j]);
        #else
        B[i*size+j] = exp(A[i*size+j]);
        #endif

        atomicAdd(sum_d_B, B[i*size+j]);
    }
}
__global__ void
softmax_finish_kernel(bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        B[i*size+j] = (B[i*size+j]/(*sum_d_B));
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// End CUDA part
//////////////////////////////////////////////////////////////////////////////////////

void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock, dimGrid;
    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dimGrid = dim3(ceil(float(input_data)/dimBlock.x), ceil(float(input_data)/dimBlock.y));
    
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    // 1-1 step convolution
    covolution_kernel<<<dimGrid, dimBlock>>>(deviceObj->input_data, deviceObj->conv_1_output, deviceObj->kernel_1, input_data, input_data, input_data, kernel_1);
    
    // 1-2 step activation
    relu_kernel<<<dimGrid, dimBlock>>>(deviceObj->conv_1_output, deviceObj->conv_1_output, input_data);
    // 1-3 step pooling
    unsigned int size_lateral_1 = input_data / stride_1;
    if(size_lateral_1 <= BLOCK_SIZE)
    {
        dimBlock = dim3(size_lateral_1, size_lateral_1);
        dimGrid = dim3(1, 1);
    }
    else
    {
        dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        dimGrid = dim3(ceil(((float(size_lateral_1)))/dimBlock.x), ceil(((float(size_lateral_1) ))/dimBlock.y));
    }
    max_pooling_kernel<<<dimGrid, dimBlock>>>(deviceObj->conv_1_output, deviceObj->pooling_1_output, input_data, stride_1, size_lateral_1);
    
    // 1-4 normalization
    lrn_kernel<<<dimGrid, dimBlock>>>(deviceObj->pooling_1_output, deviceObj->pooling_1_output, size_lateral_1);
    
    // 2-1 step convolution
    covolution_kernel<<<dimGrid, dimBlock>>>(deviceObj->pooling_1_output, deviceObj->conv_2_output, deviceObj->kernel_2, size_lateral_1, size_lateral_1, size_lateral_1, kernel_2);
    
    // 2-2 step activation
    relu_kernel<<<dimGrid, dimBlock>>>(deviceObj->conv_2_output, deviceObj->conv_2_output, size_lateral_1);
    // 2-3 normalization
    lrn_kernel<<<dimGrid, dimBlock>>>(deviceObj->conv_2_output, deviceObj->conv_2_output, size_lateral_1);
    // 2-4 step pooling
    
    unsigned int size_lateral_2 = size_lateral_1 / stride_2;
    if(size_lateral_2 < BLOCK_SIZE)
    {
        dimBlock = dim3(size_lateral_2, size_lateral_2);
        dimGrid = dim3(1, 1);
    }
    else
    {
        dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        dimGrid = dim3(ceil(((float(size_lateral_2) ))/dimBlock.x), ceil(((float(size_lateral_2) ))/dimBlock.y));
    }
    max_pooling_kernel<<<dimGrid, dimBlock>>>(deviceObj->conv_2_output, deviceObj->pooling_2_output, size_lateral_1, stride_2, size_lateral_2);
    // dense layer 1
    dimBlock = dim3(BLOCK_SIZE, 1);
    dimGrid = dim3(ceil(float(neurons_dense_1)/dimBlock.x), 1);
    matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(deviceObj->dense_layer_1_weights, deviceObj->pooling_2_output,deviceObj->dense_layer_1_output,neurons_dense_1, 1, size_lateral_2*size_lateral_2);
    //activation layer dense 1
    dimBlock = dim3(BLOCK_SIZE);
    dimGrid = dim3(ceil(float(neurons_dense_1)/dimBlock.x));
    relu_linear_kernel<<<dimGrid, dimBlock>>>(deviceObj->dense_layer_1_output, deviceObj->dense_layer_1_output, neurons_dense_1);
    // dense layer 2
    dimBlock = dim3(BLOCK_SIZE, 1);
    dimGrid = dim3(ceil(float(neurons_dense_2)/dimBlock.x), 1);

    matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(deviceObj->dense_layer_2_weights, deviceObj->dense_layer_1_output, deviceObj->dense_layer_2_output, neurons_dense_2, 1, neurons_dense_1);
    // activation layer dense 2
    dimBlock = dim3(BLOCK_SIZE*BLOCK_SIZE);
    dimGrid = dim3(ceil(float(neurons_dense_2)/dimBlock.x));
    relu_linear_kernel<<<dimGrid, dimBlock>>>(deviceObj->dense_layer_2_output, deviceObj->dense_layer_2_output, neurons_dense_2);

    // softmax 
    dimBlock = dim3(1, BLOCK_SIZE);
    dimGrid = dim3(1, ceil(float(neurons_dense_2)/dimBlock.x));

    softmax_kernel<<<dimGrid, dimBlock>>>(deviceObj->dense_layer_2_output, deviceObj->output_data, deviceObj->sum_ouput, neurons_dense_2);
    softmax_finish_kernel<<<dimGrid, dimBlock>>>(deviceObj->output_data, deviceObj->sum_ouput, neurons_dense_2);
    
    // profilling end
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}

