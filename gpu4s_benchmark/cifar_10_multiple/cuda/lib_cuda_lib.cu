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
#else
  #define CUDNNTYPE CUDNN_DATA_DOUBLE
#endif

const bench_t alf = 1;
const bench_t bet = 0;
///////////////////////////////////////////////////////////////////////////////////
// START CUDNN DEFINITION
///////////////////////////////////////////////////////////////////////////////////

 struct cuddObject{
      cudnnTensorDescriptor_t input_descriptor_1_1;
      cudnnTensorDescriptor_t output_descriptor_1_1;
      cudnnFilterDescriptor_t kernel_descriptor_1_1;
      cudnnConvolutionDescriptor_t convolution_descriptor_1_1;
    };

///////////////////////////////////////////////////////////////////////////////////
// END CUDNN DEFINITION
///////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////
// START CUDNN
///////////////////////////////////////////////////////////////////////////////////

void convolution_1_1_init(cuddObject *cudd_object, unsigned int input_data_size, unsigned int kernel_size){
  // create input tensor
  checkCUDNN(cudnnCreateTensorDescriptor(&cudd_object->input_descriptor_1_1));
  checkCUDNN(cudnnSetTensor4dDescriptor(cudd_object->input_descriptor_1_1,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data_size,
                                      /*image_width=*/input_data_size));
  // create output tensor
  checkCUDNN(cudnnCreateTensorDescriptor(&cudd_object->output_descriptor_1_1));
  checkCUDNN(cudnnSetTensor4dDescriptor(cudd_object->output_descriptor_1_1,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/input_data_size,
                                      /*image_width=*/input_data_size));
  // create kernel tensor
    checkCUDNN(cudnnCreateFilterDescriptor(&cudd_object->kernel_descriptor_1_1));
    checkCUDNN(cudnnSetFilter4dDescriptor(cudd_object->kernel_descriptor_1_1,
                                      /*dataType=*/CUDNNTYPE,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/1,
                                      /*in_channels=*/1,
                                      /*kernel_height=*/kernel_size,
                                      /*kernel_width=*/kernel_size));
  // create kernel descriptor
    checkCUDNN(cudnnCreateConvolutionDescriptor(&cudd_object->convolution_descriptor_1_1));
    checkCUDNN(cudnnSetConvolution2dDescriptor(cudd_object->convolution_descriptor_1_1,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNNTYPE));
}
void convolution_1_1_clear(cuddObject *cudd_object){
    cudnnDestroyTensorDescriptor(cudd_object->input_descriptor_1_1);
    cudnnDestroyTensorDescriptor(cudd_object->output_descriptor_1_1);
    cudnnDestroyFilterDescriptor(cudd_object->kernel_descriptor_1_1);
    cudnnDestroyConvolutionDescriptor(cudd_object->convolution_descriptor_1_1);
}

void convolution_1_1(cuddObject *cudd_object, GraficCommon* device_object ,cudnnHandle_t cudnn, unsigned int  input_data, unsigned int  kernel_1, unsigned int offset){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing convolution
    // --- FIX: New code for cuDNN 8+ ---
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_algo_count;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                        cudd_object->input_descriptor_1_1,
                                        cudd_object->kernel_descriptor_1_1,
                                        cudd_object->convolution_descriptor_1_1,
                                        cudd_object->output_descriptor_1_1,
                                        /*requestedAlgoCount=*/1,
                                        &returned_algo_count,
                                        &algo_perf));
    cudnnConvolutionFwdAlgo_t convolution_algorithm = algo_perf.algo;
    // get memory needed for the convolution
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   cudd_object->input_descriptor_1_1,
                                                   cudd_object->kernel_descriptor_1_1,
                                                   cudd_object->convolution_descriptor_1_1,
                                                   cudd_object->output_descriptor_1_1,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
    // alocate memory for workspace
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);
    // perform the convolution
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alf,
                                   cudd_object->input_descriptor_1_1,
                                   deviceObj->input_data + offset,
                                   cudd_object->kernel_descriptor_1_1,
                                   deviceObj->kernel_1,
                                   cudd_object->convolution_descriptor_1_1,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &bet,
                                   cudd_object->output_descriptor_1_1,
                                   deviceObj->conv_1_output));

    // destroy data
    cudaFree(d_workspace);

}
void activation_1_2(GraficCommon* device_object, cudnnHandle_t cudnn, unsigned int input_data){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

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
                                    deviceObj->conv_1_output,
                                    &bet,
                                    output_descriptor,
                                    deviceObj->conv_1_output));

    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}
void pooling_1_3(GraficCommon* device_object ,cudnnHandle_t cudnn, unsigned int  input_data,unsigned int size_lateral, unsigned int stride){
  GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
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
                                   deviceObj->conv_1_output,
                                   &bet,
                                   output_descriptor,
                                   deviceObj->pooling_1_output))

    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(poolingDesc);

}
void normalization_1_4(GraficCommon* device_object, cudnnHandle_t cudnn,unsigned int size_lateral_1){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
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
                                           deviceObj->pooling_1_output,
                                           &bet,
                                           output_descriptor,
                                           deviceObj->pooling_1_output));

    // destroy cuDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyLRNDescriptor(lrn_descriptor);

}

void convolution_2_1(GraficCommon* device_object, cudnnHandle_t cudnn, unsigned int input_data_size, unsigned int kernel_size){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

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
    // --- FIX: New code for cuDNN 8+ ---
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
                                   deviceObj->pooling_1_output,
                                   kernel_descriptor,
                                   deviceObj->kernel_2,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &bet,
                                   output_descriptor,
                                   deviceObj->conv_2_output));

    // destroy data
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

}
void activation_2_2(GraficCommon* device_object, cudnnHandle_t cudnn, unsigned int input_data){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

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
                                    deviceObj->conv_2_output,
                                    &bet,
                                    output_descriptor,
                                    deviceObj->conv_2_output));

    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}
void normalization_2_3(GraficCommon* device_object, cudnnHandle_t cudnn,unsigned int size_lateral_1){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
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
                                           deviceObj->conv_2_output,
                                           &bet,
                                           output_descriptor,
                                           deviceObj->conv_2_output));

    // destroy cuDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyLRNDescriptor(lrn_descriptor);

}
void pooling_2_4(GraficCommon* device_object, cudnnHandle_t cudnn, unsigned int  input_data,unsigned int size_lateral, unsigned int stride){
  GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
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
                                   deviceObj->conv_2_output,
                                   &bet,
                                   output_descriptor,
                                   deviceObj->pooling_2_output))

    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyPoolingDescriptor(poolingDesc);

}

void dense_1(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w){
  GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
  const bench_t *alpha = &alf;
  const bench_t *beta = &bet;
  cublasHandle_t handle;
  cublasCreate(&handle);

  #ifdef INT
  printf("CUBLAS NOT SUPPORT INT OPERATIOS\n");
  #elif FLOAT
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->pooling_2_output, m, deviceObj->dense_layer_1_weights, w, beta, deviceObj->dense_layer_1_output, m);
  #else
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->pooling_2_output, m, deviceObj->dense_layer_1_weights, w, beta, deviceObj->dense_layer_1_output, m);
  #endif

  cublasDestroy(handle);
}

void activation_d_1(GraficCommon* device_object, cudnnHandle_t cudnn, unsigned int input_data){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

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
                                    deviceObj->dense_layer_1_output,
                                    &bet,
                                    output_descriptor,
                                    deviceObj->dense_layer_1_output));

    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}

void dense_2(GraficCommon* device_object, unsigned int n, unsigned int m, unsigned int w){
  GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
  const bench_t *alpha = &alf;
  const bench_t *beta = &bet;
  cublasHandle_t handle;
  cublasCreate(&handle);

  #ifdef INT
  printf("CUBLAS NOT SUPPORT INT OPERATIOS\n");
  #elif FLOAT
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->dense_layer_1_output, m, deviceObj->dense_layer_2_weights, w, beta, deviceObj->dense_layer_2_output, m);
  #else
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, w, alpha, deviceObj->dense_layer_1_output, m, deviceObj->dense_layer_2_weights, w, beta, deviceObj->dense_layer_2_output, m);
  #endif

  cublasDestroy(handle);
}

void activation_d_2(GraficCommon* device_object, cudnnHandle_t cudnn, unsigned int input_data){

GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

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
                                    deviceObj->dense_layer_2_output,
                                    &bet,
                                    output_descriptor,
                                    deviceObj->dense_layer_2_output));

    // destroy data
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyActivationDescriptor(activation_algorithm);
}

void softmax(GraficCommon* device_object, cudnnHandle_t cudnn, unsigned int input_data, unsigned int offset){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
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
                                   deviceObj->dense_layer_2_output,
                                   &bet,
                                   output_descriptor,
                                   deviceObj->output_data+offset));

    // destroy cuDNN
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
}

void clean_cudnn(){

}

///////////////////////////////////////////////////////////////////////////////////
// END CUDNN
///////////////////////////////////////////////////////////////////////////////////

void execute_kernel(GraficCommon* device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    // cublas settings
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    // init structures
    cuddObject *cudd_object = (cuddObject *)malloc(sizeof(cuddObject));

    // kernel time execution
    Clock kernelCLK;

    // profilling start 
    kernelCLK.start();
    cudaEventRecord(*deviceObj->start);

    // init comvolution
    convolution_1_1_init(cudd_object, input_data, kernel_1);
    for(unsigned int position = 0; position < number_of_images; ++position)
    {

      // 1-1 step convolution
      convolution_1_1(cudd_object, device_object, cudnn, input_data, kernel_1, position * input_data * input_data);
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
      softmax(device_object,cudnn, neurons_dense_2, position * output_data);

    }
   
    // profilling end 
    cudaEventRecord(*deviceObj->stop);
    cudaDeviceSynchronize(); 
    kernelCLK.end();

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedMS();

    convolution_1_1_clear(cudd_object);
    clean_cudnn();
    // delete struct
    free(cudd_object);
    cudnnDestroy(cudnn);
}
