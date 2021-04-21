#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32
__global__ void
covolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
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
                    else if ( i + x > size - 1 || j + y > size -1 )
                    {
                        value = 0;
                        //printf("ENTRO UPPER%d %d\n", i + x , j + y);
                    }
                    else
                    {
                        value = A[(x + i)*size+(y + j)];
                    }
                    
                   
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


void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}

void init(GraficObject *device_object, int platform ,int device, char* device_name){
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	//printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    device_object->start = new cudaEvent_t;
    device_object->stop = new cudaEvent_t;
    device_object->start_memory_copy_device = new cudaEvent_t;
    device_object->stop_memory_copy_device = new cudaEvent_t;
    device_object->start_memory_copy_host = new cudaEvent_t;
    device_object->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(device_object->start);
    cudaEventCreate(device_object->stop);
    cudaEventCreate(device_object->start_memory_copy_device);
    cudaEventCreate(device_object->stop_memory_copy_device);
    cudaEventCreate(device_object->start_memory_copy_host);
    cudaEventCreate(device_object->stop_memory_copy_host);
}


bool device_memory_init(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images){
   // Allocate input
	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&(device_object->input_data), number_of_images * input_data * input_data * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate kernel
    err = cudaMalloc((void **)&(device_object->kernel_1), kernel_1 * kernel_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate conv 1 output
    err = cudaMalloc((void **)&(device_object->conv_1_output), input_data * input_data * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate pooling output
    unsigned int size_pooling_1 = input_data / stride_1;
    err = cudaMalloc((void **)&(device_object->pooling_1_output), size_pooling_1 * size_pooling_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate kernel 2
    err = cudaMalloc((void **)&(device_object->kernel_2), kernel_2 * kernel_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate conv 1 output
    err = cudaMalloc((void **)&(device_object->conv_2_output), size_pooling_1 * size_pooling_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate pooling output
    unsigned int size_pooling_2 = size_pooling_1 / stride_2;
    err = cudaMalloc((void **)&(device_object->pooling_2_output), size_pooling_2 * size_pooling_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    //dense layer 1 weights 
    unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;

    err = cudaMalloc((void **)&(device_object->dense_layer_1_weights), weights_layer_1* sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // dense layer output 1
    err = cudaMalloc((void **)&(device_object->dense_layer_1_output), neurons_dense_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    //dense layer 2 weights 
    unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2;
    err = cudaMalloc((void **)&(device_object->dense_layer_2_weights), weights_layer_2  * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // dense layer output 2
    err = cudaMalloc((void **)&(device_object->dense_layer_2_output), neurons_dense_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
     // sum data
    err = cudaMalloc((void **)&(device_object->sum_ouput), sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // output data
    err = cudaMalloc((void **)&(device_object->output_data), number_of_images * neurons_dense_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
 }

void copy_memory_to_device(GraficObject *device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size, unsigned int number_of_images){
    cudaEventRecord(*device_object->start_memory_copy_device);
	cudaError_t err = cudaMemcpy(device_object->input_data, input_data, sizeof(bench_t) * input * input * number_of_images, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->kernel_1, kernel_1_data, sizeof(bench_t) * kernel_size_1 * kernel_size_1, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel_1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->kernel_2, kernel_2_data, sizeof(bench_t) * kernel_size_2 * kernel_size_2, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel_2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->dense_layer_1_weights, weights_1, sizeof(bench_t) * weights_1_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector weights_layer_1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->dense_layer_2_weights, weights_2, sizeof(bench_t) * weights_2_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector weights_layer_2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaMemset(device_object->sum_ouput, 0,  sizeof(bench_t));
    cudaEventRecord(*device_object->stop_memory_copy_device);
    
}
void execute_kernel(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images){
    // execute net 
    
    cudaEventRecord(*device_object->start);
    bench_t* aux_output_data = device_object->output_data;
    bench_t* aux_input_data = device_object->input_data;
    
    for(unsigned int position = 0; position < number_of_images; ++position)
    {
        aux_input_data = device_object->input_data + position * input_data * input_data;
        aux_output_data = device_object->output_data + position * output_data;
        // 1-1 step convolution
        dim3 dimBlock, dimGrid;
        dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        dimGrid = dim3(ceil(float(input_data)/dimBlock.x), ceil(float(input_data)/dimBlock.y));
        covolution_kernel<<<dimGrid, dimBlock>>>(aux_input_data, device_object->conv_1_output, device_object->kernel_1, input_data, input_data, input_data, kernel_1);

        // 1-2 step activation
        relu_kernel<<<dimGrid, dimBlock>>>(device_object->conv_1_output, device_object->conv_1_output, input_data);
        // 1-3 step pooling
        unsigned int size_lateral_1 = input_data / stride_1;
        if(size_lateral_1 < BLOCK_SIZE)
        {
            dimBlock = dim3(size_lateral_1, size_lateral_1);
            dimGrid = dim3(1, 1);
        }
        else
        {
            dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
            dimGrid = dim3(ceil(((float(size_lateral_1) / stride_1 ))/dimBlock.x), ceil(((float(size_lateral_1) / stride_1 ))/dimBlock.y));
        }
        max_pooling_kernel<<<dimGrid, dimBlock>>>(device_object->conv_1_output, device_object->pooling_1_output, input_data, stride_1, size_lateral_1);

        // 1-4 normalization
        lrn_kernel<<<dimGrid, dimBlock>>>(device_object->pooling_1_output, device_object->pooling_1_output, size_lateral_1);

        // 2-1 step convolution
        covolution_kernel<<<dimGrid, dimBlock>>>(device_object->pooling_1_output, device_object->conv_2_output, device_object->kernel_2, size_lateral_1, size_lateral_1, size_lateral_1, kernel_2);

        // 2-2 step activation
        relu_kernel<<<dimGrid, dimBlock>>>(device_object->conv_2_output, device_object->conv_2_output, size_lateral_1);
        // 2-3 normalization
        lrn_kernel<<<dimGrid, dimBlock>>>(device_object->conv_2_output, device_object->conv_2_output, size_lateral_1);
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
            dimGrid = dim3(ceil(((float(size_lateral_2) / stride_2 ))/dimBlock.x), ceil(((float(size_lateral_2) / stride_2 ))/dimBlock.y));
        }
        max_pooling_kernel<<<dimGrid, dimBlock>>>(device_object->conv_2_output, device_object->pooling_2_output, size_lateral_1, stride_2, size_lateral_2);
        // dense layer 1
        dimBlock = dim3(BLOCK_SIZE, 1);
        dimGrid = dim3(ceil(float(neurons_dense_1)/dimBlock.x), 1);
        matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(device_object->dense_layer_1_weights, device_object->pooling_2_output,device_object->dense_layer_1_output,neurons_dense_1, 1, size_lateral_2*size_lateral_2);
        //activation layer dense 1
        dimBlock = dim3(BLOCK_SIZE);
        dimGrid = dim3(ceil(float(neurons_dense_1)/dimBlock.x));
        relu_linear_kernel<<<dimGrid, dimBlock>>>(device_object->dense_layer_1_output, device_object->dense_layer_1_output, neurons_dense_1);
        // dense layer 2
        dimBlock = dim3(BLOCK_SIZE, 1);
        dimGrid = dim3(ceil(float(neurons_dense_2)/dimBlock.x), 1);

        matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(device_object->dense_layer_2_weights, device_object->dense_layer_1_output, device_object->dense_layer_2_output, neurons_dense_2, 1, neurons_dense_1);
        // activation layer dense 2
        dimBlock = dim3(BLOCK_SIZE);
        dimGrid = dim3(ceil(float(neurons_dense_2)/dimBlock.x));
        relu_linear_kernel<<<dimGrid, dimBlock>>>(device_object->dense_layer_2_output, device_object->dense_layer_2_output, neurons_dense_2);

        // softmax 
        dimBlock = dim3(1, BLOCK_SIZE);
        dimGrid = dim3(1, ceil(float(neurons_dense_2)/dimBlock.x));

        softmax_kernel<<<dimGrid, dimBlock>>>(device_object->dense_layer_2_output, aux_output_data, device_object->sum_ouput, neurons_dense_2);
        softmax_finish_kernel<<<dimGrid, dimBlock>>>(aux_output_data, device_object->sum_ouput, neurons_dense_2);
        cudaMemset(device_object->sum_ouput, 0, sizeof(bench_t));
    }
    cudaEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size, unsigned int number_of_images){
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_C, device_object->output_data, number_of_images * size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_C, device_object->dense_layer_2_output, 10 * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
}

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f miliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficObject *device_object){
    cudaError_t err = cudaSuccess;

    err = cudaFree(device_object->input_data);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector input_data (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->kernel_1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector kernel_1 (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->conv_1_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector conv_1_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->pooling_1_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector pooling_1_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->kernel_2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector kernel_2 (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->conv_2_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector conv_2_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->pooling_2_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector pooling_2_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->dense_layer_1_weights);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_1_weights (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->dense_layer_2_weights);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_2_weights (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->dense_layer_1_output);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_1_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->dense_layer_2_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector dense_layer_2_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->output_data);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector output_data (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->sum_ouput);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector sum_ouput (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    // delete events
    delete device_object->start;
    delete device_object->stop;
    delete device_object->start_memory_copy_device;
    delete device_object->stop_memory_copy_device;
    delete device_object->start_memory_copy_host;
    delete device_object->stop_memory_copy_host;
}
