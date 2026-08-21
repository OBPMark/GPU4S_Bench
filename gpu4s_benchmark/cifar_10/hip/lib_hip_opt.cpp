#include "../benchmark_library.h"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
#define BLOCK_SIZE_PLANE (BLOCK_SIZE * BLOCK_SIZE)

__global__ void
covolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size, const int shared_size, const int kernel_rad)
{
    unsigned int size = n;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x0, y0;
    HIP_DYNAMIC_SHARED( bench_t, data)
    if (x < size && y < size)
    {
        // each thread load 4 values ,the corners
        //TOP right corner
        x0 = x - kernel_rad;
        y0 = y - kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 < 0 || y0 < 0 )
        {
            data[threadIdx.x * shared_size + threadIdx.y] = 0;
        }
        else
        {
            data[threadIdx.x * shared_size + threadIdx.y] = A[x0 *size+y0];
        } 
            
        //BOTTOM right corner
        x0 = x + kernel_rad;
        y0 = y - kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 > size-1  || y0 < 0 )
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + threadIdx.y] = 0;
        }
        else
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + threadIdx.y] = A[x0 *size+y0];
        } 

        //TOP left corner
        x0 = x - kernel_rad;
        y0 = y + kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 < 0  || y0 > size-1 )
        {
            data[threadIdx.x * shared_size + (threadIdx.y + kernel_rad * 2)] = 0;
        }
        else
        {
            data[threadIdx.x * shared_size + (threadIdx.y + kernel_rad * 2)] = A[x0 *size+y0];
        } 

        //BOTTOM left corner
        x0 = x + kernel_rad;
        y0 = y + kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 > size-1  || y0 > size-1 )
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + (threadIdx.y + kernel_rad * 2)] = 0;
        }
        else
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + (threadIdx.y + kernel_rad * 2)] = A[x0 *size+y0];
        } 

        __syncthreads();
        bench_t sum = 0;
        unsigned int xa = kernel_rad + threadIdx.x;
        unsigned int ya = kernel_rad + threadIdx.y;
        #pragma unroll 3
        for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
            {
                #pragma unroll 3
                for(int j = -kernel_rad; j <= kernel_rad; ++j)
                {
                    //printf("ACHIVED position  %d %d value %f\n", (xa + i) , (ya + j), data[(xa + i)][(ya + j)]);
                    sum += data[(xa + i) * shared_size +  (ya + j)] * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
                }
            }
                
        B[x*size+y ] = sum;
    }
    
}
__global__ void
covolution_kernel_base(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
{
    unsigned int size = n;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
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
    bench_t threshold = 0;
    if (i  < (size * size)){
        
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
relu_linear_kernel(const bench_t *A, bench_t *B, const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    bench_t threshold = 0;
    if (i  < size ){
        
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
    //unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (i  < lateral_stride*lateral_stride){
        
        bench_t max_value = A[(i * stride + ((i/lateral_stride)*size))];
        for(unsigned int x = 0; x < stride; ++x)
        {
            for(unsigned int y = 0; y < stride; ++y)
            {
                //printf("max %f,value %f, pos x %d, pos y %d i position %d, final position %d\n", max_value,  A[((i * stride + ((i/stride)*size)) + x)  + ( y * size)], x ,y, i, ((i * stride + ((i/stride)*size)) + x)  + ( y * size));
                max_value = max(max_value, A[((i * stride + ((i/lateral_stride)*size)) + x)  + ( y * size)]);
                
            }
        }
        //printf("i position %d value %f \n", i, max_value);
        B[i] = max_value;
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
    __shared__ bench_t A_tile[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ bench_t B_tile[BLOCK_SIZE*BLOCK_SIZE];

    
    unsigned int i = blockIdx.x *  BLOCK_SIZE + threadIdx.x;
    unsigned int j = blockIdx.y *  BLOCK_SIZE + threadIdx.y;

    unsigned int size_shared = 0;
    if (n < BLOCK_SIZE)
    {
        size_shared = n;
    }
    else
    {
        size_shared = BLOCK_SIZE;
    }
    
    
    bench_t acumulated = 0;
    unsigned int idx = 0;
    // load memory
    for (unsigned int sub = 0; sub < gridDim.x; ++sub)
    {
        
        idx = i * w + sub * size_shared + threadIdx.y;

        if(idx >= w*n)
        {
            A_tile[threadIdx.x * size_shared+ threadIdx.y] = 0;
        }
        else
        {   
            A_tile[threadIdx.x * size_shared + threadIdx.y] = A[idx];
        }
        idx = (sub * size_shared + threadIdx.x) * m + j;

        if (idx >= m*w)
        {
            B_tile[threadIdx.x * size_shared +  threadIdx.y] = 0;
        }
        else
        {   
            B_tile[threadIdx.x* size_shared + threadIdx.y] = B[idx];
        }
        __syncthreads();
        for (unsigned int k = 0; k < size_shared; ++k)
        {
            acumulated +=  A_tile[threadIdx.x*size_shared + k] * B_tile[k*size_shared + threadIdx.y];
        }
        __syncthreads();

    }
    if (i < n && j < m)
    {
        
        C[i *m + j] = acumulated;
    }
}

__global__ void
matrix_multiplication_kernel_other(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w)
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
{   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    bench_t value = 0;
    __shared__ bench_t shared_data[BLOCK_SIZE_PLANE];
    if (i  < (size)){
        
        #ifdef INT
        value = exp(A[i]);
        #elif FLOAT
        value = expf(A[i]);
        #else
        value = exp(A[i]);
        #endif

        shared_data[tid] = value;
        B[i] = value;
        // sync threads
        __syncthreads();
        for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
        {
            if (tid < s)  
            {
                shared_data[tid] += shared_data[tid + s];
            }
        __syncthreads();
        }
        if (tid == 0){
            atomicAdd(sum_d_B, shared_data[0]);
        }    
    }
}
__global__ void
softmax_finish_kernel(bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i  < (size)){
        B[i] = (B[i]/(*sum_d_B));
    }
}

//////////////////////////////////////////////////////////////////////////////////////
// End CUDA part
//////////////////////////////////////////////////////////////////////////////////////

void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    dim3 dimBlock, dimGrid,dimBlock_act, dimGrid_act;
    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dimGrid = dim3(ceil(float(input_data)/dimBlock.x), ceil(float(input_data)/dimBlock.y));
    unsigned int kernel_rad =  kernel_1 / 2;
    unsigned int size_shared = (BLOCK_SIZE + kernel_rad *2 ) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad *2) * sizeof(bench_t);
    unsigned int size_shared_position = (BLOCK_SIZE + kernel_rad *2);
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    (void)hipEventRecord(*deviceObj->start);

    // 1-1 step convolution
    hipLaunchKernelGGL((covolution_kernel), dim3(dimGrid), dim3(dimBlock), size_shared, 0, deviceObj->input_data, deviceObj->conv_1_output, deviceObj->kernel_1, input_data, input_data, input_data, kernel_1, size_shared_position, kernel_rad);
    // 1-2 step activation
    dimBlock = dim3(BLOCK_SIZE_PLANE);
    dimGrid = dim3(ceil(float(input_data)/dimBlock.x));
    hipLaunchKernelGGL((relu_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->conv_1_output, deviceObj->conv_1_output, input_data*input_data);
    // 1-3 step pooling
    unsigned int size_lateral_1 = input_data / stride_1;
    if(size_lateral_1*size_lateral_1 < BLOCK_SIZE_PLANE)
    {
        dimBlock = dim3(size_lateral_1*size_lateral_1);
        dimGrid = dim3(1);
    }
    else
    {
        dimBlock = dim3(BLOCK_SIZE_PLANE);
        dimGrid = dim3(ceil((size_lateral_1*size_lateral_1)/dimBlock.x));
    }
    hipLaunchKernelGGL((max_pooling_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->conv_1_output, deviceObj->pooling_1_output, input_data, stride_1, size_lateral_1);
    // 1-4 normalization
    if(size_lateral_1 < BLOCK_SIZE)
    {
        dimBlock = dim3(size_lateral_1, size_lateral_1);
        dimGrid = dim3(1, 1);
    }
    else
    {
        dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        dimGrid = dim3(ceil(((float(size_lateral_1) ))/dimBlock.x), ceil(((float(size_lateral_1) ))/dimBlock.y));

    }
    hipLaunchKernelGGL((lrn_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->pooling_1_output, deviceObj->pooling_1_output, size_lateral_1);
    

    // 2-1 step convolution
    //kernel_rad =  kernel_2 / 2;
    //size_shared = (BLOCK_SIZE + kernel_rad *2 ) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad *2) * sizeof(bench_t);
    //size_shared_position = (BLOCK_SIZE + kernel_rad *2);
    //hipLaunchKernelGGL((covolution_kernel), dim3(dimGrid), dim3(dimBlock), size_shared, 0, deviceObj->pooling_1_output, deviceObj->conv_2_output, deviceObj->kernel_2, size_lateral_1, size_lateral_1, size_lateral_1, kernel_2,size_shared_position, kernel_rad);
    hipLaunchKernelGGL((covolution_kernel_base), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->pooling_1_output, deviceObj->conv_2_output, deviceObj->kernel_2, size_lateral_1, size_lateral_1, size_lateral_1, kernel_2);
    // 2-2 step activation
    dimBlock_act = dim3(BLOCK_SIZE_PLANE);
    dimGrid_act = dim3(ceil(float(size_lateral_1*size_lateral_1)/dimBlock.x));
    hipLaunchKernelGGL((relu_kernel), dim3(dimGrid_act), dim3(dimBlock_act), 0, 0, deviceObj->conv_2_output, deviceObj->conv_2_output, size_lateral_1);
    // 2-3 normalization
    hipLaunchKernelGGL((lrn_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->conv_2_output, deviceObj->conv_2_output, size_lateral_1);
    // 2-4 step pooling
    unsigned int size_lateral_2 = size_lateral_1 / stride_2;
    if(size_lateral_2*size_lateral_2 <= BLOCK_SIZE_PLANE)
    {
        dimBlock = dim3(size_lateral_2 *size_lateral_2);
        dimGrid = dim3(1);
    }
    else
    {
        dimBlock = dim3(BLOCK_SIZE_PLANE);
        dimGrid = dim3(ceil((size_lateral_2*size_lateral_2)/dimBlock.x));
    }
    hipLaunchKernelGGL((max_pooling_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->conv_2_output, deviceObj->pooling_2_output, size_lateral_1, stride_2, size_lateral_2);
    // dense layer 1
    dimBlock = dim3(BLOCK_SIZE, 1);
    dimGrid = dim3(ceil(float(neurons_dense_1)/dimBlock.x), 1);
    hipLaunchKernelGGL((matrix_multiplication_kernel_other), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->dense_layer_1_weights, deviceObj->pooling_2_output,deviceObj->dense_layer_1_output,neurons_dense_1, 1, size_lateral_2*size_lateral_2);
    //activation layer dense 1
    dimBlock_act = dim3(BLOCK_SIZE_PLANE);
    dimGrid_act = dim3(ceil(float(neurons_dense_1)/dimBlock.x));
    hipLaunchKernelGGL((relu_linear_kernel), dim3(dimGrid_act), dim3(dimBlock_act), 0, 0, deviceObj->dense_layer_1_output, deviceObj->dense_layer_1_output, neurons_dense_1);
    
    // dense layer 2
    dimBlock = dim3(BLOCK_SIZE, 1);
    dimGrid = dim3(ceil(float(neurons_dense_2)/dimBlock.x), 1);

    hipLaunchKernelGGL((matrix_multiplication_kernel_other), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->dense_layer_2_weights, deviceObj->dense_layer_1_output, deviceObj->dense_layer_2_output, neurons_dense_2, 1, neurons_dense_1);
    // activation layer dense 2
    dimBlock = dim3(BLOCK_SIZE_PLANE);
    dimGrid = dim3(ceil(float(neurons_dense_2)/dimBlock.x));
    hipLaunchKernelGGL((relu_linear_kernel), dim3(dimGrid_act), dim3(dimBlock_act), 0, 0, deviceObj->dense_layer_2_output, deviceObj->dense_layer_2_output, neurons_dense_2);

    // softmax 
    dimBlock = dim3(BLOCK_SIZE);
    dimGrid = dim3(ceil(float(neurons_dense_2)/dimBlock.x));

    hipLaunchKernelGGL((softmax_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->dense_layer_2_output, deviceObj->output_data, deviceObj->sum_ouput, neurons_dense_2);
    hipLaunchKernelGGL((softmax_finish_kernel), dim3(dimGrid), dim3(dimBlock), 0, 0, deviceObj->output_data, deviceObj->sum_ouput, neurons_dense_2);
    
    // profilling end 
    (void)hipEventRecord(*deviceObj->stop);
    hipDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();
}
