// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include "GEN_kernel.hcl"
#include "GEN_atomic_functions.hcl"


//#define BLOCK_SIZE 16
void init(GraficObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}
void init(GraficObject *device_object, int platform ,int device, char* device_name){
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[platform];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
   //get default device of the default platform
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[device];
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    strcpy(device_name,default_device.getInfo<CL_DEVICE_NAME>().c_str() );
    // context
    device_object->context = new cl::Context(default_device);
    device_object->queue = new cl::CommandQueue(*device_object->context,default_device,CL_QUEUE_PROFILING_ENABLE);
    device_object->default_device = default_device;
    
    // events
    device_object->evt_copyIN = new cl::Event;
    device_object->evt_copyK1 = new cl::Event;
    device_object->evt_copyK2 = new cl::Event;
    device_object->evt_copyW1 = new cl::Event;
    device_object->evt_copyW2 = new cl::Event;
    device_object->evt_copyOut = new cl::Event;

    device_object->evt1_1 = new cl::Event;
    device_object->evt1_2 = new cl::Event; 
    device_object->evt1_3 = new cl::Event; 
    device_object->evt1_4 = new cl::Event; 
    device_object->evt2_1 = new cl::Event;  
    device_object->evt2_2 = new cl::Event; 
    device_object->evt2_3 = new cl::Event;
    device_object->evt2_1 = new cl::Event;  
    device_object->evt2_2 = new cl::Event; 
    device_object->evt2_3 = new cl::Event; 
    device_object->evt2_4 = new cl::Event;  
    device_object->evtd_1 = new cl::Event; 
    device_object->evtd_1_a = new cl::Event;
    device_object->evtd_2 = new cl::Event; 
    device_object->evtd_2_a = new cl::Event;  
    device_object->evt_softmax = new cl::Event;
    device_object->evt_softmax_fin = new cl::Event;   

    
}

bool device_memory_init(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){

   unsigned int size_pooling_1 = input_data / stride_1;
   unsigned int size_pooling_2 = size_pooling_1 / stride_2;
   unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;
   unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2; 

   // input
   device_object->input_data = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,input_data * input_data * sizeof(bench_t));
   // convolution 1
   device_object->kernel_1 = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,kernel_1 * kernel_1 * sizeof(bench_t));
   device_object->conv_1_output = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,input_data * input_data * sizeof(bench_t));
   // pooling 1
   device_object->pooling_1_output = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,size_pooling_1 * size_pooling_1 * sizeof(bench_t));
   // convolution 1
   device_object->kernel_2 = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,kernel_2 * kernel_2 * sizeof(bench_t));
   device_object->conv_2_output = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,size_pooling_1 * size_pooling_1 * sizeof(bench_t));
   // pooling 2 
   device_object->pooling_2_output = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,size_pooling_2 * size_pooling_2 * sizeof(bench_t));
   // dense 1
   device_object->dense_layer_1_weights = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,weights_layer_1 * sizeof(bench_t));
   device_object->dense_layer_1_output = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,neurons_dense_1 * sizeof(bench_t));
   // dense 2
   device_object->dense_layer_2_weights = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,weights_layer_2 * sizeof(bench_t));
   device_object->dense_layer_2_output = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,neurons_dense_2 * sizeof(bench_t));
   // out
   device_object->sum_ouput = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(bench_t));
   device_object->output_data = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,neurons_dense_2 * sizeof(bench_t));
   return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size){
    // copy memory host -> device
    // input data
    device_object->queue->enqueueWriteBuffer(*device_object->input_data,CL_TRUE,0,sizeof(bench_t)* input * input, input_data, NULL, device_object->evt_copyIN);
    // kernels
    device_object->queue->enqueueWriteBuffer(*device_object->kernel_1,CL_TRUE,0,sizeof(bench_t)* kernel_size_1 * kernel_size_1, kernel_1_data, NULL, device_object->evt_copyK1);
    device_object->queue->enqueueWriteBuffer(*device_object->kernel_2,CL_TRUE,0,sizeof(bench_t)* kernel_size_2 * kernel_size_2, kernel_2_data, NULL, device_object->evt_copyK2);
    // dense layer
    device_object->queue->enqueueWriteBuffer(*device_object->dense_layer_1_weights,CL_TRUE,0,sizeof(bench_t)* weights_1_size, weights_1, NULL, device_object->evt_copyW1);
    device_object->queue->enqueueWriteBuffer(*device_object->dense_layer_2_weights,CL_TRUE,0,sizeof(bench_t)* weights_2_size, weights_2, NULL, device_object->evt_copyW2);
}


void execute_kernel(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2){
    unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local;
    cl::NDRange global;

    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_kernel + atomic_code + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    // build
    cl::Program program(*device_object->context,sources);
    if(program.build({device_object->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_object->default_device)<<"\n";
        exit(1);
    }
    // 1-1 step convolution
    if (input_data <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(input_data, input_data);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(input_data, input_data);
    }
    cl::Kernel  kernel_conv=cl::Kernel(program,"kernel_matrix_convolution");
    kernel_conv.setArg(0,*device_object->input_data);
    kernel_conv.setArg(1,*device_object->conv_1_output);
    kernel_conv.setArg(2,*device_object->kernel_1);
    kernel_conv.setArg(3,input_data);
    kernel_conv.setArg(4,input_data);
    kernel_conv.setArg(5,input_data);
    kernel_conv.setArg(6,kernel_1);

    device_object->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, device_object->evt1_1);

    // 1-2 step activation
    cl::Kernel  kernel_add=cl::Kernel(program,"kernel_relu");
    kernel_add.setArg(0,*device_object->conv_1_output);
    kernel_add.setArg(1,*device_object->conv_1_output);
    kernel_add.setArg(2,input_data);
    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evt1_2);

    // 1-3 step pooling
    unsigned int size_lateral_1 = input_data / stride_1;
    if(size_lateral_1 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (size_lateral_1, size_lateral_1);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(size_lateral_1, size_lateral_1);
    }
    kernel_add=cl::Kernel(program,"kernel_max");
    kernel_add.setArg(0,*device_object->conv_1_output);
    kernel_add.setArg(1,*device_object->pooling_1_output);
    kernel_add.setArg(2,input_data);
    kernel_add.setArg(3,stride_1);
    kernel_add.setArg(4,size_lateral_1);
    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evt1_3);
    
    // 1-4 step normalitation
    kernel_add=cl::Kernel(program,"kernel_lrn");
    kernel_add.setArg(0,*device_object->pooling_1_output);
    kernel_add.setArg(1,*device_object->pooling_1_output);
    kernel_add.setArg(2,size_lateral_1);
    kernel_add.setArg(3,K);
    kernel_add.setArg(4,ALPHA);
    kernel_add.setArg(5,BETA);

   device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evt1_4);

    // 2-1 step convolution
    kernel_conv=cl::Kernel(program,"kernel_matrix_convolution");
    kernel_conv.setArg(0,*device_object->pooling_1_output);
    kernel_conv.setArg(1,*device_object->conv_2_output);
    kernel_conv.setArg(2,*device_object->kernel_2);
    kernel_conv.setArg(3,size_lateral_1);
    kernel_conv.setArg(4,size_lateral_1);
    kernel_conv.setArg(5,size_lateral_1);
    kernel_conv.setArg(6,kernel_2);

    device_object->queue->enqueueNDRangeKernel(kernel_conv,cl::NullRange,global,local, NULL, device_object->evt2_1);

    // 2-2 step activation
    kernel_add=cl::Kernel(program,"kernel_relu");
    kernel_add.setArg(0,*device_object->conv_2_output);
    kernel_add.setArg(1,*device_object->conv_2_output);
    kernel_add.setArg(2,size_lateral_1);
    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evt2_2);
    
    // 2-3 normalization
    kernel_add=cl::Kernel(program,"kernel_lrn");
    kernel_add.setArg(0,*device_object->conv_2_output);
    kernel_add.setArg(1,*device_object->conv_2_output);
    kernel_add.setArg(2,size_lateral_1);
    kernel_add.setArg(3,K);
    kernel_add.setArg(4,ALPHA);
    kernel_add.setArg(5,BETA);

    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evt2_3);

    // 2-4 step pooling
    unsigned int size_lateral_2 = size_lateral_1 / stride_2;
    if(size_lateral_2 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (size_lateral_2, size_lateral_2);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(size_lateral_2, size_lateral_2);
    }
    kernel_add=cl::Kernel(program,"kernel_max");
    kernel_add.setArg(0,*device_object->conv_2_output);
    kernel_add.setArg(1,*device_object->pooling_2_output);
    kernel_add.setArg(2,size_lateral_1);
    kernel_add.setArg(3,stride_2);
    kernel_add.setArg(4,size_lateral_2);
    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evt2_4);
    // dense layer 1
    if(neurons_dense_1 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (neurons_dense_1, 1);
    }
    else
    {
        local = cl::NDRange(x_local, 1);
        global = cl::NDRange(neurons_dense_1, 1);
    }
    kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
    kernel_add.setArg(0,*device_object->dense_layer_1_weights);
    kernel_add.setArg(1,*device_object->pooling_2_output);
    kernel_add.setArg(2,*device_object->dense_layer_1_output);
    kernel_add.setArg(3,neurons_dense_1);
    kernel_add.setArg(4,1);
    kernel_add.setArg(5,size_lateral_2*size_lateral_2);

    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evtd_1);

    //activation layer dense 1
    /*if(neurons_dense_1 > BLOCK_SIZE * 32)
    {
        local = cl::NullRange;
        global = cl::NDRange (neurons_dense_1);
    }
    else
    {
        local = cl::NDRange(x_local*y_local);
        global = cl::NDRange(neurons_dense_1);
    }*/
    local = cl::NullRange;
    global = cl::NDRange (neurons_dense_1);
    kernel_add=cl::Kernel(program,"kernel_relu_linear");
    kernel_add.setArg(0,*device_object->dense_layer_1_output);
    kernel_add.setArg(1,*device_object->dense_layer_1_output);
    kernel_add.setArg(2,neurons_dense_1);
    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evtd_1_a);

    // dense layer 2
    if(neurons_dense_2 <= BLOCK_SIZE)
    {
        local = cl::NDRange(1, 1);
        global = cl::NDRange (neurons_dense_2, 1);
    }
    else
    {
        local = cl::NullRange;
        global = cl::NDRange(neurons_dense_2, 1);
    }
    kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
    kernel_add.setArg(0,*device_object->dense_layer_2_weights);
    kernel_add.setArg(1,*device_object->dense_layer_1_output);
    kernel_add.setArg(2,*device_object->dense_layer_2_output);
    kernel_add.setArg(3,neurons_dense_2);
    kernel_add.setArg(4,1);
    kernel_add.setArg(5,neurons_dense_1);

    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evtd_2);

    //activation layer dense 2
    /*if(neurons_dense_2 < BLOCK_SIZE * BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (neurons_dense_2);
    }
    else
    {
        local = cl::NDRange(x_local*y_local);
        global = cl::NDRange(neurons_dense_2);
    }*/
    local = cl::NullRange;
    global = cl::NDRange (neurons_dense_2);
    kernel_add=cl::Kernel(program,"kernel_relu_linear");
    kernel_add.setArg(0,*device_object->dense_layer_2_output);
    kernel_add.setArg(1,*device_object->dense_layer_2_output);
    kernel_add.setArg(2,neurons_dense_2);
    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evtd_2_a);

    //soft max
    /*if(neurons_dense_2 < BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange (1, neurons_dense_2);
    }
    else
    {
        local = cl::NDRange(1, x_local);
        global = cl::NDRange(1, neurons_dense_2);
    }*/
    local = cl::NullRange;
    global = cl::NDRange (1, neurons_dense_2);
    cl::Kernel softmax_kernel=cl::Kernel(program,"kernel_softmax");
    softmax_kernel.setArg(0,*device_object->dense_layer_2_output);
    softmax_kernel.setArg(1,*device_object->output_data);
    softmax_kernel.setArg(2,*device_object->sum_ouput);
    softmax_kernel.setArg(3,neurons_dense_2);

    device_object->queue->enqueueNDRangeKernel(softmax_kernel,cl::NullRange,global,local, NULL, device_object->evt_softmax);

    cl::Kernel softmax_end_kernel=cl::Kernel(program,"kernel_softmax_end");
    softmax_end_kernel.setArg(0,*device_object->output_data);
    softmax_end_kernel.setArg(1,*device_object->sum_ouput);
    softmax_end_kernel.setArg(2,neurons_dense_2);

    device_object->queue->enqueueNDRangeKernel(softmax_end_kernel,cl::NullRange,global,local, NULL, device_object->evt_softmax_fin);
    // end 
    device_object->queue->finish();

}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    device_object->queue->enqueueReadBuffer(*device_object->output_data,CL_TRUE,0,sizeof(bench_t)*size,h_C, NULL, device_object->evt_copyOut);
    //device_object->queue->enqueueReadBuffer(*device_object->conv_2_output,CL_TRUE,0,sizeof(bench_t)*16*16,h_C, NULL, device_object->evt_copyOut);
}

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    device_object->evt_copyOut->wait();

    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;

    // copy memory H -> D
    elapsed_h_d = device_object->evt_copyIN->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyIN->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyK1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyK1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyK2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyK2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyW1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyW1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyW2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyW2->getProfilingInfo<CL_PROFILING_COMMAND_START>();

    // kernel time

    elapsed = device_object->evt1_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt1_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt1_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt1_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt1_3->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt1_3->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt1_4->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt1_4->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt2_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt2_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt2_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt2_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt2_3->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt2_3->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt2_4->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt2_4->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evtd_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evtd_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evtd_1_a->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evtd_1_a->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evtd_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evtd_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evtd_2_a->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evtd_2_a->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt_softmax->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_softmax->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt_softmax_fin->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_softmax_fin->getProfilingInfo<CL_PROFILING_COMMAND_START>();

    // copy memory D -> H
    elapsed_d_h = device_object->evt_copyOut->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyOut->getProfilingInfo<CL_PROFILING_COMMAND_START>();



    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f miliseconds\n", elapsed / 1000000.0);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

void clean(GraficObject *device_object){

    // pointers clean
    delete device_object->context;
    delete device_object->queue;
    // pointer to memory

    delete device_object->evt_copyIN;
    delete device_object->evt_copyK1;
    delete device_object->evt_copyK2;
    delete device_object->evt_copyW1;
    delete device_object->evt_copyW2;
    delete device_object->evt_copyOut;
    delete device_object->evt1_1;
    delete device_object->evt1_2;
    delete device_object->evt1_3;
    delete device_object->evt1_4;
    delete device_object->evt2_1;
    delete device_object->evt2_2;
    delete device_object->evt2_3;
    delete device_object->evt2_4;
    delete device_object->evtd_1;
    delete device_object->evtd_1_a;
    delete device_object->evtd_2;
    delete device_object->evtd_2_a;
    delete device_object->evt_softmax;
    delete device_object->evt_softmax_fin;

    delete device_object->input_data;
    delete device_object->kernel_1;
    delete device_object->conv_1_output;
    delete device_object->pooling_1_output;
    delete device_object->kernel_2;
    delete device_object->conv_2_output;
    delete device_object->pooling_2_output;
    delete device_object->dense_layer_1_weights;
    delete device_object->dense_layer_1_output;
    delete device_object->dense_layer_2_weights;
    delete device_object->dense_layer_2_output;
    delete device_object->output_data;
    delete device_object->sum_ouput;
}