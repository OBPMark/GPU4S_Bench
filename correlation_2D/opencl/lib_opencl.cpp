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
   //get default device of the default platformB
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
    device_object->evt = new cl::Event;
    device_object->evt_mean = new cl::Event; 
    device_object->evt_copyA = new cl::Event;
    device_object->evt_copyB = new cl::Event;
    device_object->evt_copyAB = new cl::Event;
    device_object->evt_copyAA = new cl::Event;
    device_object->evt_copyBB = new cl::Event;
    
}

bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){
   device_object->d_A = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_a_matrix);
   device_object->d_B = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_b_matrix);
   device_object->d_R = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(result_bench_t));
   device_object->mean_A = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(result_bench_t));
   device_object->mean_B = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(result_bench_t));
   device_object->acumulate_value_a_b = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(result_bench_t));
   device_object->acumulate_value_a_a = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(result_bench_t));
   device_object->acumulate_value_b_b = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(result_bench_t));
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b){
	// copy memory host -> device
	//TODO Errors check
    device_object->queue->enqueueWriteBuffer(*device_object->d_A,CL_TRUE,0,sizeof(bench_t)*size_a, h_A, NULL, device_object->evt_copyA);
    device_object->queue->enqueueWriteBuffer(*device_object->d_B,CL_TRUE,0,sizeof(bench_t)*size_b, h_B, NULL, device_object->evt_copyB);
    
}


void execute_kernel(GraficObject *device_object, unsigned int n){
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local;
    cl::NDRange global;
    if (n <= BLOCK_SIZE )
    {
        local = cl::NullRange;
        global = cl::NDRange(n,n);
    }
    else
    {
        local = cl::NDRange(x_local,y_local);
        global = cl::NDRange(n,n);
    }
    
    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_kernel + atomic_code + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    cl::Program program(*device_object->context,sources);
    if(program.build({device_object->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_object->default_device)<<"\n";
        exit(1);
    }
    
    cl::Kernel kernel_mean=cl::Kernel(program,"mean_matrices");
    kernel_mean.setArg(0,*device_object->d_A);
    kernel_mean.setArg(1,*device_object->d_B);
    kernel_mean.setArg(2,*device_object->mean_A);
    kernel_mean.setArg(3,*device_object->mean_B);
    kernel_mean.setArg(4,n);
    device_object->queue->enqueueNDRangeKernel(kernel_mean,cl::NullRange,global,local, NULL, device_object->evt_mean);

    cl::Kernel kernel=cl::Kernel(program,"correlation_2D");
    kernel.setArg(0,*device_object->d_A);
    kernel.setArg(1,*device_object->d_B);
    kernel.setArg(2,*device_object->d_R);
    kernel.setArg(3,*device_object->mean_A);
    kernel.setArg(4,*device_object->mean_B);
    kernel.setArg(5,*device_object->acumulate_value_a_b);
    kernel.setArg(6,*device_object->acumulate_value_a_a);
    kernel.setArg(7,*device_object->acumulate_value_b_b);
    kernel.setArg(8,n);
    
    device_object->queue->enqueueNDRangeKernel(kernel,cl::NullRange,global,local, NULL, device_object->evt);

    device_object->queue->finish();

}

void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R){
    result_bench_t acumulate_value_a_a;
    result_bench_t acumulate_value_a_b;
    result_bench_t acumulate_value_b_b;
    device_object->queue->enqueueReadBuffer(*device_object->acumulate_value_a_a,CL_TRUE,0,sizeof(result_bench_t),&acumulate_value_a_a, NULL, device_object->evt_copyAA);
    device_object->queue->enqueueReadBuffer(*device_object->acumulate_value_a_b,CL_TRUE,0,sizeof(result_bench_t),&acumulate_value_a_b, NULL, device_object->evt_copyAB);
    device_object->queue->enqueueReadBuffer(*device_object->acumulate_value_b_b,CL_TRUE,0,sizeof(result_bench_t),&acumulate_value_b_b, NULL, device_object->evt_copyBB);
    device_object->evt_copyBB->wait();
    *h_R = (result_bench_t)(acumulate_value_a_b / (result_bench_t)(sqrt(acumulate_value_a_a * acumulate_value_b_b)));
}

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    device_object->evt_copyBB->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = device_object->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    
    elapsed = device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed += device_object->evt_mean->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_mean->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    
    elapsed_d_h = device_object->evt_copyAA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyAA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_d_h += device_object->evt_copyAB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyAB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_d_h += device_object->evt_copyBB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyBB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    

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
    delete device_object->d_A;
    delete device_object->d_B;
    delete device_object->d_R;
    delete device_object->acumulate_value_a_a;
    delete device_object->acumulate_value_a_b;
    delete device_object->acumulate_value_b_b;
    delete device_object->evt;
    delete device_object->evt_copyA;
    delete device_object->evt_copyB;
    delete device_object->evt_copyBB;
    delete device_object->evt_copyAB;
    delete device_object->evt_copyAA;
}
