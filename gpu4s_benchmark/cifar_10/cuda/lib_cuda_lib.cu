#include <cudnn.h>
#include <cublas_v2.h>
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

const bench_t alf = 1;
const bench_t bet = 0;

///////////////////////////////////////////////////////////////////////////////////
// START CUDNN
///////////////////////////////////////////////////////////////////////////////////

void convolution_1_1(GraficObject *device_object ,cudnnHandle_t cudnn, unsigned int input_data_size, unsigned int kernel_size){
 
    // create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data_size,
                                      /*image_width=*/input_data_size));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data_size,
                                      /*image_width=*/input_data_size));
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
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNNTYPE));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing convolution
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));
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
                                   device_object->input_data,
                                   kernel_descriptor,
                                   device_object->kernel_1,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &bet,
                                   output_descriptor,
                                   device_object->conv_1_output));
    
   
    
    // destroy data
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

}
void activation_1_2(GraficObject *device_object, cudnnHandle_t cudnn, unsigned int input_data){

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data,
                                      /*image_width=*/input_data));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data,
                                      /*image_width=*/input_data));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing activation function
    cudnnActivationDescriptor_t activation_algorithm;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_algorithm));
    checkCUDNN(cudnnSetActivationDescriptor(activation_algorithm,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            0)); // ???????? it sopuse that is only needed in ELU and CLIPPED_RELU
    checkCUDNN(cudnnActivationForward(cudnn, 
                                    activation_algorithm,
                                    &alf,
                                    input_descriptor,
                                    device_object->conv_1_output,
                                    &bet,
                                    output_descriptor,
                                    device_object->conv_1_output));
     
    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}
void pooling_1_3(GraficObject *device_object ,cudnnHandle_t cudnn, unsigned int  input_data,unsigned int size_lateral, unsigned int stride){
  // create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data,
                                      /*image_width=*/input_data));
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
                                   device_object->conv_1_output,
                                   &bet,
                                   output_descriptor,
                                   device_object->pooling_1_output))
     
    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(poolingDesc);

}
void normalization_1_4(GraficObject *device_object, cudnnHandle_t cudnn,unsigned int size_lateral_1){
cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/size_lateral_1,
                                      /*image_width=*/size_lateral_1));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/size_lateral_1,
                                      /*image_width=*/size_lateral_1));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing pooling
    cudnnLRNDescriptor_t lrn_descriptor;
    checkCUDNN(cudnnCreateLRNDescriptor(&lrn_descriptor));
    checkCUDNN(cudnnSetLRNDescriptor(lrn_descriptor, 
                                     5,
                                     ALPHA,
                                     BETA,
                                     K));
    checkCUDNN(cudnnLRNCrossChannelForward(cudnn, 
                                           lrn_descriptor,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &alf,
                                           input_descriptor,
                                           device_object->pooling_1_output,
                                           &bet,
                                           output_descriptor,
                                           device_object->pooling_1_output));
     
    // destroy cuDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyLRNDescriptor(lrn_descriptor);

}

void convolution_2_1(GraficObject *device_object, cudnnHandle_t cudnn, unsigned int input_data_size, unsigned int kernel_size){
 
    // create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data_size,
                                      /*image_width=*/input_data_size));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data_size,
                                      /*image_width=*/input_data_size));
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
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNNTYPE));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing convolution
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));
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
                                   device_object->pooling_1_output,
                                   kernel_descriptor,
                                   device_object->kernel_2,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &bet,
                                   output_descriptor,
                                   device_object->conv_2_output));
    
   
    
    // destroy data
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

}
void activation_2_2(GraficObject *device_object, cudnnHandle_t cudnn, unsigned int input_data){

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data,
                                      /*image_width=*/input_data));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data,
                                      /*image_width=*/input_data));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing activation function
    cudnnActivationDescriptor_t activation_algorithm;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_algorithm));
    checkCUDNN(cudnnSetActivationDescriptor(activation_algorithm,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            0)); // ???????? it sopuse that is only needed in ELU and CLIPPED_RELU
    checkCUDNN(cudnnActivationForward(cudnn, 
                                    activation_algorithm,
                                    &alf,
                                    input_descriptor,
                                    device_object->conv_2_output,
                                    &bet,
                                    output_descriptor,
                                    device_object->conv_2_output));
     
    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}
void normalization_2_3(GraficObject *device_object, cudnnHandle_t cudnn,unsigned int size_lateral_1){
cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/size_lateral_1,
                                      /*image_width=*/size_lateral_1));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/size_lateral_1,
                                      /*image_width=*/size_lateral_1));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing pooling
    cudnnLRNDescriptor_t lrn_descriptor;
    checkCUDNN(cudnnCreateLRNDescriptor(&lrn_descriptor));
    checkCUDNN(cudnnSetLRNDescriptor(lrn_descriptor, 
                                     5,
                                     ALPHA,
                                     BETA,
                                     K));
    checkCUDNN(cudnnLRNCrossChannelForward(cudnn, 
                                           lrn_descriptor,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &alf,
                                           input_descriptor,
                                           device_object->conv_2_output,
                                           &bet,
                                           output_descriptor,
                                           device_object->conv_2_output));
     
    // destroy cuDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyLRNDescriptor(lrn_descriptor);

}
void pooling_2_4(GraficObject *device_object, cudnnHandle_t cudnn, unsigned int  input_data,unsigned int size_lateral, unsigned int stride){
  // create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data,
                                      /*image_width=*/input_data));
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
                                   device_object->conv_2_output,
                                   &bet,
                                   output_descriptor,
                                   device_object->pooling_2_output))
     
    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(poolingDesc);

}

void dense_1(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w){
  int lda=m,ldb=n,ldc=w;
  const bench_t *alpha = &alf;
  const bench_t *beta = &bet;
  cublasHandle_t handle;
  cublasCreate(&handle);

  #ifdef INT
  printf("CUBLAS NOT SUPPORT INT OPERATIOS\n");
  #elif FLOAT
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, device_object->pooling_2_output, m, device_object->dense_layer_1_weights, w, beta, device_object->dense_layer_1_output, m);
  #else
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, device_object->pooling_2_output, m, device_object->dense_layer_1_weights, w, beta, device_object->dense_layer_1_output, m);
  #endif

  cublasDestroy(handle);
}

void activation_d_1(GraficObject *device_object, cudnnHandle_t cudnn, unsigned int input_data){

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/input_data));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/input_data));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing activation function
    cudnnActivationDescriptor_t activation_algorithm;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_algorithm));
    checkCUDNN(cudnnSetActivationDescriptor(activation_algorithm,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            0)); // ???????? it sopuse that is only needed in ELU and CLIPPED_RELU
    checkCUDNN(cudnnActivationForward(cudnn, 
                                    activation_algorithm,
                                    &alf,
                                    input_descriptor,
                                    device_object->dense_layer_1_output,
                                    &bet,
                                    output_descriptor,
                                    device_object->dense_layer_1_output));
     
    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}

void dense_2(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w){
  int lda=m,ldb=n,ldc=w;
  const bench_t *alpha = &alf;
  const bench_t *beta = &bet;
  cublasHandle_t handle;
  cublasCreate(&handle);
  
  #ifdef INT
  printf("CUBLAS NOT SUPPORT INT OPERATIOS\n");
  #elif FLOAT
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, device_object->dense_layer_1_output, m, device_object->dense_layer_2_weights, w, beta, device_object->dense_layer_2_output, m);
  #else
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, device_object->dense_layer_1_output, m, device_object->dense_layer_2_weights, w, beta, device_object->dense_layer_2_output, m);
  #endif

  cublasDestroy(handle);
}

void activation_d_2(GraficObject *device_object, cudnnHandle_t cudnn, unsigned int input_data){

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/input_data));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/input_data));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing activation function
    cudnnActivationDescriptor_t activation_algorithm;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_algorithm));
    checkCUDNN(cudnnSetActivationDescriptor(activation_algorithm,
                                            CUDNN_ACTIVATION_RELU,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            0)); // ???????? it sopuse that is only needed in ELU and CLIPPED_RELU
    checkCUDNN(cudnnActivationForward(cudnn, 
                                    activation_algorithm,
                                    &alf,
                                    input_descriptor,
                                    device_object->dense_layer_2_output,
                                    &bet,
                                    output_descriptor,
                                    device_object->dense_layer_2_output));
     
    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}


void softmax(GraficObject *device_object, cudnnHandle_t cudnn, unsigned int input_data){
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/input_data));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/input_data));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    checkCUDNN(cudnnSoftmaxForward(cudnn, 
                                   CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_INSTANCE,
                                   &alf,
                                   input_descriptor,
                                   device_object->dense_layer_2_output,
                                   &bet,
                                   output_descriptor,
                                   device_object->output_data));
     
    // destroy cuDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
}

///////////////////////////////////////////////////////////////////////////////////
// END CUDNN
///////////////////////////////////////////////////////////////////////////////////

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


bool device_memory_init(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){
   // Allocate input
  cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&device_object->input_data, input_data * input_data * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate kernel
    err = cudaMalloc((void **)&device_object->kernel_1, kernel_1 * kernel_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate conv 1 output
    err = cudaMalloc((void **)&device_object->conv_1_output, input_data * input_data * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate pooling output
    unsigned int size_pooling_1 = input_data / stride_1;
    err = cudaMalloc((void **)&device_object->pooling_1_output, size_pooling_1 * size_pooling_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate kernel 2
    err = cudaMalloc((void **)&device_object->kernel_2, kernel_2 * kernel_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate conv 1 output
    err = cudaMalloc((void **)&device_object->conv_2_output, size_pooling_1 * size_pooling_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate pooling output
    unsigned int size_pooling_2 = size_pooling_1 / stride_2;
    err = cudaMalloc((void **)&device_object->pooling_2_output, size_pooling_2 * size_pooling_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    //dense layer 1 weights 
    unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;

    err = cudaMalloc((void **)&device_object->dense_layer_1_weights, weights_layer_1* sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // dense layer output 1
    err = cudaMalloc((void **)&device_object->dense_layer_1_output, neurons_dense_1 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    //dense layer 2 weights 
    unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2;
    err = cudaMalloc((void **)&device_object->dense_layer_2_weights, weights_layer_2  * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // dense layer output 2
    err = cudaMalloc((void **)&device_object->dense_layer_2_output, neurons_dense_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
     // sum data
    err = cudaMalloc((void **)&device_object->sum_ouput, sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // output data
    err = cudaMalloc((void **)&device_object->output_data, neurons_dense_2 * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
 }

void copy_memory_to_device(GraficObject *device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size){
    cudaEventRecord(*device_object->start_memory_copy_device);
  cudaError_t err = cudaMemcpy(device_object->input_data, input_data, sizeof(bench_t) * input * input, cudaMemcpyHostToDevice);
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
    cudaEventRecord(*device_object->stop_memory_copy_device);
    
}
void execute_kernel(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){
     // cublas settings
    
    cudnnHandle_t cudnn;

    cudaEventRecord(*device_object->start);
    checkCUDNN(cudnnCreate(&cudnn));
    
    // 1-1 step convolution
    convolution_1_1(device_object, cudnn, input_data, kernel_1);
    // 1-2 step activation
    activation_1_2(device_object,cudnn, input_data);
    // 1-3 step pooling
    unsigned int size_lateral_1 = input_data / stride_1;
    pooling_1_3(device_object,cudnn, input_data, size_lateral_1, stride_1);
    // 1-4 step normalitation
    normalization_1_4(device_object,cudnn, size_lateral_1);

    // 2-1 step convolution
   convolution_2_1(device_object,cudnn, size_lateral_1, kernel_2);
    // 2-2 step activation
    activation_2_2(device_object,cudnn, size_lateral_1);
    // 2-3 step normalitation
    normalization_2_3(device_object,cudnn, size_lateral_1);
    // 2-4 step pooling
    unsigned int size_lateral_2 = size_lateral_1 / stride_2;
    pooling_2_4(device_object,cudnn, size_lateral_1,size_lateral_2, stride_2);

   
    // dense layer 1
    dense_1(device_object, neurons_dense_1, 1, size_lateral_2*size_lateral_2);
    // dense activation 1
    activation_d_1(device_object,cudnn, neurons_dense_1);


    // dense layer 2
    dense_2(device_object, neurons_dense_2, 1, neurons_dense_1);
    // dense activation 2
    activation_d_2(device_object,cudnn, neurons_dense_2);
    //softmax
    softmax(device_object,cudnn, neurons_dense_2);
    cudaEventRecord(*device_object->stop);
    cudnnDestroy(cudnn);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_C, device_object->output_data, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
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