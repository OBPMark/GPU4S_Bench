# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
NVCC = /usr/local/cuda/bin/nvcc
# the build target executable:
TARGET = fft
# FLAGS 
# CC compiler flags:
CFLAGS  = -g -I/usr/local/include/ -L/usr/local/lib64/
# NVCC compiler flags
NVCCFLAGS = -arch compute_61 -code sm_61
# CUDA FLAGS
CUFLAGS = -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart 
# OPENCL FLAGS
OPFLAGS = -I/usr/local/cuda/include/   -L/oldroot/root/usr/lib/x86_64-linux-gnu/ -lOpenCL 

# Littelendian and Bigendian flags, by default if value is not set is Littelendian if value is set to -DBIGENDIAN is Bigendian
# -DBIGENDIAN
ENDIANFLAGS = 

# Data type can be FLOAT and DOUBLE
DATATYPE =

# By default BLOCKSIZE equals 4.
BLOCKSIZE = 4
BLOCKSIZESQUARED = $(shell echo $(BLOCKSIZE)\*$(BLOCKSIZE) | bc)

# FOLDERS
# CUDA FOLDER
CUFOLDER = ./cuda/
# OPENCL FOLDER
OPFOLDER = ./opencl/
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

# CUDA part library
.PHONY: cuda
cuda-lib: main_cuda_lib

lib_cuda_lib.o: $(CUFOLDER)lib_cuda_lib.cu
	$(NVCC) -DLIB -D$(DATATYPE) -c $(CUFOLDER)lib_cuda_lib.cu -o $(CUFOLDER)lib_cuda_lib.o $(NVCCFLAGS) 


main_cuda_lib: main.cpp lib_cuda_lib.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -DLIB -D$(DATATYPE) main.cpp $(CUFOLDER)lib_cuda_lib.o $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_lib $(CUFLAGS) $(CFLAGS) -lstdc++ -lcufft

# End CUDA library

# OpenCL Part library
opencl-lib:  main_opencl_lib

lib_opencl_lib.o: $(OPFOLDER)lib_opencl_lib.cpp 
	$(CC) -D$(DATATYPE) -DOPENCL -c $(OPFOLDER)lib_opencl_lib.cpp -o $(OPFOLDER)lib_opencl_lib.o $(CFLAGS) $(OPFLAGS) -I/usr/local/include/ -L/usr/local/lib64/ -lclFFT

main_opencl_lib: main.cpp lib_opencl_lib.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_lib.o  -o $(OUTPUTFOLDER)$(TARGET)_opencl_lib $(CFLAGS) $(OPFLAGS)  -I/usr/local/include/ -L/usr/local/lib64/ -lclFFT

# End OpenCL library

# Clean
.PHONY: clean
clean:
	rm -rf *.o
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(OPFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)_*
