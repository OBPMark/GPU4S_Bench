#include <cudnn.h>
#include "../benchmark_library.h"

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#ifdef INT
  #define CUDNNTYPE CUDNN_DATA_INT32
#elif FLOAT
  #define CUDNNTYPE CUDNN_DATA_FLOAT
#elif DOUBLE
  #define CUDNNTYPE CUDNN_DATA_DOUBLE
#endif

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    if (kernel_size % 2 == 0){
        printf ("-k args must be an odd number\n");
        exit(1);
    }

    // cublas settings
    const bench_t alf = 1;
    const bench_t bet = 0;
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    // create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/m,
                                      /*image_width=*/m));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/m,
                                      /*image_width=*/m));
    // create kernel tensor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNNTYPE,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/1,
                                      /*in_channels=*/1,
                                      /*kernel_height=*/kernel_size,
                                      /*kernel_width=*/kernel_size));
    // create kernel descriptor
    // --- FIX: Change the size of pad depending of kernel_size --- 
    int pad = kernel_size / 2;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/pad,
                                           /*pad_width=*/pad,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNNTYPE));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing convolution
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_algo_count;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        /*requestedAlgoCount=*/1,
                                        &returned_algo_count,
                                        &algo_perf));
    cudnnConvolutionFwdAlgo_t convolution_algorithm = algo_perf.algo;
    // get memory needed for the convolution
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
    // alocate memory for workspace
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);
    // perform the convolution
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alf,
                                   input_descriptor,
                                   deviceObj->d_A,
                                   kernel_descriptor,
                                   deviceObj->kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &bet,
                                   output_descriptor,
                                   deviceObj->d_B));
    
   
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();

    // destroy cuDNN
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}

