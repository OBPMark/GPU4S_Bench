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
#else
  #define CUDNNTYPE CUDNN_DATA_DOUBLE
#endif

void execute_kernel(GraficCommon* device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int stride, unsigned int size_lateral){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // CUDNN settings
    const bench_t alf = 1;
    const bench_t bet = 0;
    cudnnHandle_t cudnn;
    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);
    checkCUDNN(cudnnCreate(&cudnn));
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
                                      /*image_height=*/size_lateral,
                                      /*image_width=*/size_lateral));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing activation function
    cudnnPoolingDescriptor_t poolingDesc;
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolingDesc,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           stride,
                                           stride,
                                           0,
                                           0,
                                           stride,
                                           stride))

    
    checkCUDNN(cudnnPoolingForward(cudnn,
                                   poolingDesc,
                                   &alf,
                                   input_descriptor,
                                   deviceObj->d_A,
                                   &bet,
                                   output_descriptor,
                                   deviceObj->d_B))
     
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();

    // destroy cuDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(poolingDesc);

    cudnnDestroy(cudnn);
}
