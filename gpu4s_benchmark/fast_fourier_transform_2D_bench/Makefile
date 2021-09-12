# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
NVCC = /usr/local/cuda/bin/nvcc
# the build target executable:
TARGET = fft
# FLAGS 
# CC compiler flags:
CFLAGS  = -g  -lfftw3 
# NVCC compiler flags
NVCCFLAGS = -arch compute_72 -code sm_72 
# CUDA FLAGS
CUFLAGS = -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart 
# OPENCL FLAGS
OPFLAGS = -I/usr/local/cuda/include/   -L/oldroot/root/usr/lib/x86_64-linux-gnu/ -lOpenCL 
# Littelendian and Bigendian flags, by default if value is not set is Littelendian if value is set to -DBIGENDIAN is Bigendian
# -DBIGENDIAN
ENDIANFLAGS = 
# Data type can be DOUBLE and FLOAT
DATATYPE =
# FOLDERS
# CUDA FOLDER
CUFOLDER = ./cuda/
# OPENCL FOLDER
OPFOLDER = ./opencl/
# CPU FOLDER
CPUFUNCTIONFOLDER = ./cpu_functions/
# CPU FOLDER
CPUFOLDER = ./cpu/
# OUTPUT FOLDER
OUTPUTFOLDER = ./bin/

# COMPILER MAIN
all:
	@echo "YOU NEED TO SELECT CUDA OR OPENCL -> make CUDA / make OpenCL / make CUDA-opt / make OpenCL-opt / make CUDA-lib / make OpenCL-lib"
# End Main
# Shortcuts
.PHONY: all-bin
all-bin:cuda-lib opencl-lib
.PHONY: all-cuda
all-cuda:  cuda-lib
.PHONY: all-opencl
all-opencl: opencl-lib
.PHONY: OpenCL-lib
OpenCL-opt: opencl-lib
.PHONY: CUDA-lib
CUDA-opt: cuda-lib
# End Shortcuts
# CPU part
cpu_functions.o: $(CPUFUNCTIONFOLDER)cpu_functions.cpp
	$(CC) $(ENDIANFLAGS) -D$(DATATYPE) -c $(CPUFUNCTIONFOLDER)cpu_functions.cpp -o $(CPUFUNCTIONFOLDER)cpu_functions.o $(CFLAGS) 
# End CPU 

# CUDA part library
.PHONY: cuda
cuda-lib: main_cuda_lib

lib_cuda_lib.o: $(CUFOLDER)lib_cuda_lib.cu
	$(NVCC) -DLIB -D$(DATATYPE) -DCUDA -c $(CUFOLDER)lib_cuda_lib.cu -o $(CUFOLDER)lib_cuda_lib.o $(NVCCFLAGS) 


main_cuda_lib: main.cpp lib_cuda_lib.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -DLIB -D$(DATATYPE) -DCUDA main.cpp $(CUFOLDER)lib_cuda_lib.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CUFLAGS) $(CFLAGS) -lstdc++ -lcufft

# End CUDA library

# OpenCL Part library
opencl-lib:  main_opencl_lib

lib_opencl_lib.o: $(OPFOLDER)lib_opencl_lib.cpp 
	$(CC) -D$(DATATYPE) -DOPENCL -c $(OPFOLDER)lib_opencl_lib.cpp -o $(OPFOLDER)lib_opencl_lib.o $(CFLAGS) $(OPFLAGS) -I/home/irodrig/clfft/include/ -L/home/irodrig/clfft/library/ -lclFFT

main_opencl_lib: main.cpp lib_opencl_lib.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_lib.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OPFLAGS)  -I/home/irodrig/clfft/include/ -L/home/irodrig/clfft/library/ -lclFFT 

# End OpenCL library


# CPU part
.PHONY: cpu
cpu:  main_cpu

lib_cpu.o: $(CPUFOLDER)lib_cpu.cpp 
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CPUFOLDER)lib_cpu.cpp -o $(CPUFOLDER)lib_cpu.o $(CFLAGS)

main_cpu: main.cpp lib_cpu.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE)  main.cpp $(CPUFOLDER)lib_cpu.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cpu_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS)


# End part

# Clean
.PHONY: clean
clean:
	rm -rf *.o
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(CPUFUNCTIONFOLDER)*.o
	rm -rf $(OPFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)_opencl_lib
	rm -rf $(OUTPUTFOLDER)$(TARGET)_cuda_lib