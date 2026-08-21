// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include "vkFFT.h"

void execute_kernel(GraficCommon* device_object, int64_t window, int64_t size){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);

    // --- convert c++ pointer to raw c ---
    cl_context          raw_context = (*deviceObj->context)();
    cl_device_id        raw_device  = deviceObj->default_device();
    cl_command_queue    raw_queue   = (*deviceObj->queue)();
    cl_mem              raw_input   = (*deviceObj->d_A)();
    cl_mem              raw_output  = (*deviceObj->d_B)();

    // --- VkFFT configuration ---
    VkFFTConfiguration config = {};
    config.FFTdim           = 1;            // number of dim of FFT
    config.size[0]          = window/2;         // size of D1
    config.device           = &raw_device;  // select the device 
    config.context          = &raw_context; // memory space
    config.buffer           = &raw_output;  // output buff
    config.inputBuffer      = &raw_input;   // input buff  
    config.isInputFormatted = 1;            // different buffer for input/output
    config.specifyOffsetsAtLaunch = 1;      // enables per-launch offsets
    #ifdef DOUBLE
    config.doublePrecision  = 1;
    #endif

    // kernel time execution
    Clock kernelCLK;

    // --- init ---
    kernelCLK.start(); // Start clock
    VkFFTApplication app = {};
    initializeVkFFT(&app, config);

    // --- launch ---
    VkFFTLaunchParams launchParams = {};
    launchParams.commandQueue  = &raw_queue; //select the queue

    uint64_t input_offset  = 0;
    uint64_t output_offset = 0;

    for (unsigned int i = 0; i < (size * 2  - window + 1); i+=2){
        // update byte offsets into the same buffer for each window
        launchParams.inputBufferOffset  = input_offset;
        launchParams.bufferOffset       = output_offset; 

        VkFFTAppend(&app, -1, &launchParams);

        input_offset  += sizeof(bench_t) * 2;          // advance by 1 real element
        output_offset += sizeof(bench_t) * window *2; // advance by one window of output
    }

    deviceObj->queue->finish();
    kernelCLK.end(); // End clock

    // store the kernel time
    deviceObj->elapsed_time = kernelCLK.getElapsedNS();

    // --- cleanup ---
    deleteVkFFT(&app);
}



float get_elapsed_time(GraficCommon* device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    GraficObject* deviceObj = static_cast<GraficObject*>(device_object);
    deviceObj->evt_copyB->wait();
    
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    
    if (deviceObj->profiling_clock)
    {
        // --- FIX: Use <chrono> instead of CLBlast event profiling (unreliable on PROFILING_CLOCK) ---
        elapsed_h_d  = deviceObj->h2d_elapsed_time;
        elapsed      = deviceObj->elapsed_time;
        elapsed_d_h  = deviceObj->d2h_elapsed_time;
    }else{
        elapsed_h_d = deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
        
        elapsed      = deviceObj->elapsed_time;
        //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
        
        elapsed_d_h = deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - deviceObj->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        //printf("Elapsed time Device->Host: %.10f \n", );
    }


    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", elapsed_h_d / 1000000.0, elapsed / 1000000.0, elapsed_d_h / 1000000.0, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0, elapsed / 1000000.0,elapsed_d_h / 1000000.0);
    }else{
         printf("profiling mode: %s\n", deviceObj->profiling_clock ? "CLOCK" : "FALSE");   
         printf("Elapsed time Host->Device: %.10f milliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f milliseconds\n", elapsed / 1000000.0);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

