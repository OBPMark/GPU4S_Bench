# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
NVCC = /usr/local/cuda/bin/nvcc
HIP = /opt/rocm/hip/bin/hipcc
# the build target executable:
TARGET = fft
# FLAGS 
# CC compiler flags:
CFLAGS  = -g
# NVCC compiler flags
NVCCFLAGS = -arch compute_61 -code sm_61
# CUDA FLAGS
CUFLAGS = -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart 
# OPENCL FLAGS
OPFLAGS = -I/usr/local/cuda/include/   -L/oldroot/root/usr/lib/x86_64-linux-gnu/ -lOpenCL 
# HIP FLAGS
HIPFLAGS = -I/opt/rocm/hip/include -L/opt/rocm/hip/lib

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
# HIP FOLDER
HIPFOLDER = ./hip/
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
all-bin: cuda cuda-opt cuda-lib opencl opencl-opt opencl-lib openmp openmp-opt openmp-lib hip hip-opt
.PHONY: all-cuda
all-cuda: cuda cuda-opt cuda-lib
.PHONY: all-opencl
all-opencl: opencl opencl-opt opencl-lib
.PHONY: all-openmp
all-openmp: openmp openmp-opt openmp-lib
.PHONY: all-hip
all-hip: hip hip-opt
.PHONY: CUDA
CUDA: cuda
.PHONY: OpenCL
OpenCL: opencl
.PHONY: OpenMP
OpenMP: openmp
.PHONY: Hip
Hip: hip
.PHONY: CUDA-opt
CUDA-opt: cuda-opt
.PHONY: OpenCL-opt
OpenCL-opt: opencl-opt
.PHONY: OpenMP-opt
OpenMP-opt: openmp-opt
.PHONY: Hip-opt
Hip-opt: hip-opt
.PHONY: CUDA-lib
CUDA-lib: cuda-lib
.PHONY: OpenCL-lib
OpenCL-lib: opencl-lib
.PHONY: OpenMP-lib
OpenMP-lib: openmp-lib
# End Shortcuts
# CPU part
lib_cpu.o: $(CPUFOLDER)lib_cpu.cpp
	$(CC) $(ENDIANFLAGS) -D$(DATATYPE) -c $(CPUFOLDER)lib_cpu.cpp -o $(CPUFOLDER)lib_cpu.o $(CFLAGS) 
# End CPU 

# CUDA part
.PHONY: cuda
cuda: main_cuda

lib_cuda.o: $(CUFOLDER)lib_cuda.cu
	$(NVCC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CUFOLDER)lib_cuda.cu -o $(CUFOLDER)lib_cuda.o $(NVCCFLAGS)


main_cuda: main.cpp lib_cuda.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) main.cpp $(CUFOLDER)lib_cuda.o $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CUFLAGS) $(CFLAGS) -lstdc++
# End CUDA

# OpenCL Part
opencl:  main_opencl

lib_opencl.o: $(OPFOLDER)lib_opencl.cpp 
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl.cpp -o $(OPFOLDER)lib_opencl.o $(CFLAGS) $(OPFLAGS)

main_opencl: main.cpp lib_opencl.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(OPFLAGS) 

# End OpenCL

# Hip part
hip: main_hip

lib_hip.o: $(HIPFOLDER)lib_hip.cpp
	$(HIP) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIPFOLDER)lib_hip.cpp -o $(HIPFOLDER)lib_hip.o $(CFLAGS) $(HIPFLAGS)

main_hip: main.cpp lib_hip.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(HIP) -D$(DATATYPE) -DHIP main.cpp $(HIPFOLDER)lib_hip.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_hip_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(HIPFLAGS)
# End Hip 

# CUDA part optimized
.PHONY: cuda
cuda-opt: main_cuda_opt

lib_cuda_opt.o: $(CUFOLDER)lib_cuda_opt.cu
	$(NVCC)  -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CUFOLDER)lib_cuda_opt.cu -o $(CUFOLDER)lib_cuda_opt.o $(NVCCFLAGS)


main_cuda_opt: main.cpp lib_cuda_opt.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) main.cpp $(CUFOLDER)lib_cuda_opt.o $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CUFLAGS) $(CFLAGS) -lstdc++

# End CUDA optimized

# OpenCL Part optimized
opencl-opt:  main_opencl_opt

lib_opencl_opt.o: $(OPFOLDER)lib_opencl_opt.cpp 
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl_opt.cpp -o $(OPFOLDER)lib_opencl_opt.o $(CFLAGS) $(OPFLAGS)

main_opencl_opt: main.cpp lib_opencl_opt.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_opt.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(OPFLAGS) 

# End OpenCL optimized

# Hip part
hip-opt: main_hip_opt

lib_hip_opt.o: $(HIPFOLDER)lib_hip_opt.cpp
	$(HIP) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIPFOLDER)lib_hip_opt.cpp -o $(HIPFOLDER)lib_hip_opt.o $(CFLAGS) $(HIPFLAGS)

main_hip_opt: main.cpp lib_hip_opt.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(HIP) -D$(DATATYPE) -DHIP main.cpp $(HIPFOLDER)lib_hip_opt.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_hip_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(HIPFLAGS)
# End Hip 

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
	$(CC) -D$(DATATYPE) -DOPENCL -c $(OPFOLDER)lib_opencl_lib.cpp -o $(OPFOLDER)lib_opencl_lib.o $(CFLAGS) $(OPFLAGS) -I/home/irodrig/clfft/include/ -L/home/irodrig/clfft/library/ -lclFFT

main_opencl_lib: main.cpp lib_opencl_lib.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_lib.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_lib $(CFLAGS) $(OPFLAGS)  -I/home/irodrig/clfft/include/ -L/home/irodrig/clfft/library/ -lclFFT 

# End OpenCL library

# Clean
.PHONY: clean
clean:
	rm -rf *.o
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(OPFOLDER)*.o
	rm -rf $(OMPFOLDER)*.o
	rm -rf $(HIPFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)_*
