cmake_minimum_required(VERSION 3.24)
project(gpu4s_benchmark_global NONE)

# add module
list(APPEND CMAKE_MODULE_PATH 
    "${CMAKE_CURRENT_SOURCE_DIR}/common/cmake/"
)
include(showConfig)

# create list of all the bench folder
set(GPU4S_SUBPROJECTS
    cifar_10
    cifar_10_multiple
    convolution_2D_bench
    correlation_2D
    fast_fourier_transform_2D_bench
    fast_fourier_transform_bench
    fast_fourier_transform_window_bench
    finite_impulse_response_filter
    LRN_bench
    matrix_multiplication_bench
    matrix_multiplication_bench_fp16
    matrix_multiplication_tensor_bench
    max_pooling_bench
    memory_bandwidth_bench
    relu_bench
    softmax_bench
    wavelet_transform
)


foreach(bench ${GPU4S_SUBPROJECTS})
    add_subdirectory(${bench})
endforeach()

# ====== Global aggregator targets ======
# --- project name array ---
set(GPU4S_PROJECT_NAME
    cifar_10
    cifar_10_multiple
    convolution_2D
    correlation_2D
    FFT_2D
    FFT
    FFT_window
    FIR_filter
    LRN
    matrix_mult
    matrix_mult_fp16
    matrix_mult_tensor
    max_pooling
    memory_bandwidth
    relu
    softmax
    wavelet_transform
)


set(GPU4S_ALL_TARGETS cpu all-opencl all-openmp all-cuda all-hip)

foreach(all ${GPU4S_ALL_TARGETS})
    add_custom_target(${all})
    foreach(bench ${GPU4S_PROJECT_NAME})
        if(TARGET ${bench}-${all})
            add_dependencies(${all} ${bench}-${all})
        endif()
    endforeach()
endforeach()

#show global config
showConfig()