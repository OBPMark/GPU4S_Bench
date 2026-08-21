# CONFIGURATION DIRECTIVES
# Compilers 
CC   = g++
NVCC = /usr/local/cuda/bin/nvcc
HIP  = hipcc #ubuntu : /opt/rocm/hip/bin/hipcc
# the build target executable:
TARGET = matrix_multiplication
# FLAGS 
# CC FUNCTIONS compiler flags:
CFLAGS  = -O3
# CPU compiler flags:
CPUFLAGS  = -O3 
# NVCC compiler flags
# Automatically targets the host machine's local GPU (Supported in CUDA 11.5.1+)
# Note: For cross-compiling or older legacy gpu, set target (e.g., -arch=sm_72)
NVCCFLAGS = -arch=native -O3   
# CUDA FLAGS
CUFLAGS = -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart 
# OPENCL FLAGS
OPFLAGS = -I/usr/local/cuda/include/   -L/oldroot/root/usr/lib/x86_64-linux-gnu/ -lOpenCL 
# OPENMP FLAGS
OMPFLAGS = -fopenmp -lm
# HIP FLAGS
HIPFLAGS = -I/opt/rocm/hip/include -L/opt/rocm/hip/lib -I/usr/include -L/usr/lib64
# OPENBLAS INCLUDE
OBLASFLAG = -I/opt/OpenBLAS/include/ -I/usr/include/openblas -L/opt/OpenBLAS/lib -lopenblas
# ATLAS INCLUDE
ATLASFLAG = -I/usr/local/atlas/include -L/usr/local/atlas/lib -lcblas -latlas

# --- android NDK port ---
NDK_VERSION ?= 27.3.13750724
NDK_PATH = $(ANDROID_HOME)/ndk/$(NDK_VERSION)

LLVM_PREBUILT := $(NDK_PATH)/toolchains/llvm/prebuilt/linux-x86_64/
# android 5 64-Bits clang compilator Or #32 bits : armv7a-linux-androideabi21-clang 
CLANG =	$(LLVM_PREBUILT)/bin/aarch64-linux-android21-clang++	


# Or  32 bits =:  armeabi-v7a	
ABI  ?=	arm64-v8a

#check
$(if $(wildcard $(CLANG)),,$(warning $(shell printf '\033[1;35m') warning:$(shell printf '\033[1;37m') Android NDK Missing, Expected at $(CLANG). See README.md#12 for instruction$(shell printf '\033[0m')))

#android ldflag
ANDROID_LDFLAGS 		= -static-libstdc++ -static-openmp 
#android opencl
ANDROID_OPENCL_FLAGS  	= -I$(ANDROID_INC) -L$(ANDROID_LIB)$(ABI)/ -lOpenCL
#android clblast
ANDROID_CLBLAST_FLAGS 	= -I$(ANDROID_INC) -L$(ANDROID_LIB)$(ABI)/  -lclblast
#android openmp
ANDROID_OPENMP-LIB_FLAG = -I$(ANDROID_INC) -L$(ANDROID_LIB)$(ABI)/  -I$(ANDROID_INC)$(ABI)/ -lopenblas



# LIBRARY: ATLAS or OPENBLAS
# LIBRARY=ATLA
# LIBFLAGS=$(ATLASFLAG)

LIBRARY=OPENBLAS
LIBFLAGS=$(OBLASFLAG)

# Littelendian and Bigendian flags, by default if value is not set is Littelendian if value is set to -DBIGENDIAN is Bigendian
# -DBIGENDIAN
ENDIANFLAGS = 

# Data type can be INT FLOAT HALF and DOUBLE
DATATYPE =

# By default BLOCKSIZE equals 4.
BLOCKSIZE = 4
BLOCKSIZESQUARED = $(shell echo $(BLOCKSIZE)\*$(BLOCKSIZE) | bc)

# FOLDERS
# CPU FUNCTIONS FOLDER
CPUFUNCTIONFOLDER = ./cpu_functions/
# CUDA FOLDER
CUFOLDER = ./cuda/
# OPENCL FOLDER
OPFOLDER = ./opencl/
# OPENMP FOLDER
OMPFOLDER = ./openmp/
# HIP FOLDER
HIPFOLDER = ./hip/
# CPU FOLDER
CPUFOLDER = ./cpu/
# ANDROID FOLDER
ANDROID_LIB = ./android/libs/
ANDROID_INC = ./android/include/

# OUTPUT FOLDER
OUTPUTFOLDER = ./bin/

# COMPILER MAIN
all:
	@echo "YOU NEED TO SELECT A PLATFORM -> make CPU / make CUDA / make OpenCL / make CUDA-opt / make OpenCL-opt / make CUDA-lib / make OpenCL-lib"
# End Main
# Shortcuts
.PHONY: all-cpu
all-cpu: cpu android_cpu
.PHONY: all-cuda
all-cuda: cuda cuda-opt cuda-lib
.PHONY: all-opencl
all-opencl: opencl opencl-opt opencl-lib
.PHONY: all-openmp
all-openmp: openmp openmp-opt openmp-lib android_openmp
.PHONY: all-hip
all-hip: hip hip-opt
.PHONY: CPU
CPU: cpu
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
.PHONY: all-android
all-android: android_cpu android_openmp android_opencl android_opencl-opt android_openmp-opt android_opencl-lib android_openmp-lib

# android OpenCl header dependency
$(ANDROID_INC)CL/opencl.hpp:
	@chmod +x $(OPFOLDER)get_opencl_headers.sh
	@./$(OPFOLDER)get_opencl_headers.sh
# End Android openCL 

# android clblast header dependency
$(ANDROID_INC)CL/clblast.h:
	@chmod +x $(OPFOLDER)get_clblast_headers.sh
	@./$(OPFOLDER)get_clblast_headers.sh
# End Android openCL 


# CPU FUNCTIONS part
cpu_functions.o: $(CPUFUNCTIONFOLDER)cpu_functions.cpp
	$(CC) $(ENDIANFLAGS) -D$(DATATYPE) -fPIC  -c $(CPUFUNCTIONFOLDER)cpu_functions.cpp -o $(CPUFUNCTIONFOLDER)cpu_functions.o $(CFLAGS) 
# End CPU FUNCTIONS

# CPU ANDROID FUNCTIONS part
android_cpu_functions.o: $(CPUFUNCTIONFOLDER)cpu_functions.cpp
	$(CLANG) -D$(DATATYPE) -c $(CPUFUNCTIONFOLDER)cpu_functions.cpp -o $(CPUFUNCTIONFOLDER)android_cpu_functions.o $(CPUFLAGS)
# End CPU ANDROID FUNCTIONS 


# CPU part
.PHONY: cpu
cpu: main_cpu
lib_cpu.o: $(CPUFOLDER)lib_cpu.cpp
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CPUFOLDER)lib_cpu.cpp -o $(CPUFOLDER)lib_cpu.o $(CPUFLAGS)


main_cpu: main.cpp lib_cpu.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) main.cpp $(CPUFOLDER)lib_cpu.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cpu_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CPUFLAGS) $(CFLAGS)

# End CPU 

# ANDROID CPU part
.PHONY: android_cpu
android_cpu: android_main_cpu

android_lib_cpu.o: $(CPUFOLDER)lib_cpu.cpp
	$(CLANG) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CPUFOLDER)lib_cpu.cpp -o $(CPUFOLDER)android_lib_cpu.o $(CPUFLAGS) 

android_main_cpu: main.cpp android_lib_cpu.o android_cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CLANG) -D$(DATATYPE) main.cpp $(CPUFOLDER)android_lib_cpu.o $(CPUFUNCTIONFOLDER)android_cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_android_cpu_$(shell echo $(DATATYPE) | tr 'A-Z' 'a-z')_$(BLOCKSIZESQUARED) $(CPUFLAGS) $(CFLAGS) $(ANDROID_LDFLAGS)
# End CPU 



# OpenCL Part
.PHONY: opencl
opencl:  main_opencl

lib_opencl.o: $(OPFOLDER)lib_opencl.cpp 
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl.cpp -o $(OPFOLDER)lib_opencl.o $(CFLAGS) $(OPFLAGS)

main_opencl: main.cpp lib_opencl.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(OPFLAGS) 
# End OpenCL

# Android OpenCL Part
.PHONY: android_opencl
android_opencl: android_main_opencl

$(OPFOLDER)android_lib_opencl.o: $(OPFOLDER)lib_opencl.cpp  $(ANDROID_INC)CL/opencl.hpp
	$(CLANG) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl.cpp -o $(OPFOLDER)android_lib_opencl.o $(CFLAGS) $(ANDROID_OPENCL_FLAGS)

android_main_opencl: main.cpp $(OPFOLDER)android_lib_opencl.o android_cpu_functions.o $(ANDROID_LIB)$(ABI)/libOpenCL.so $(ANDROID_INC)CL/opencl.hpp
	mkdir -p $(OUTPUTFOLDER) 
	$(CLANG) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)android_lib_opencl.o $(CPUFUNCTIONFOLDER)android_cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_android_opencl_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(ANDROID_OPENCL_FLAGS) $(ANDROID_LDFLAGS) 
# Android End OpenCL


# OpenCL Part optimized
.PHONY: opencl-opt
opencl-opt:  main_opencl_opt

lib_opencl_opt.o: $(OPFOLDER)lib_opencl_opt.cpp 
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl_opt.cpp -o $(OPFOLDER)lib_opencl_opt.o $(CFLAGS) $(OPFLAGS)

main_opencl_opt: main.cpp lib_opencl_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(OPFLAGS) 
# End OpenCL optimized


# Android OpenCL Part optimized
.PHONY: android_opencl-opt
android_opencl-opt: android_main_opencl_opt

$(OPFOLDER)android_lib_opencl_opt.o: $(OPFOLDER)lib_opencl_opt.cpp  $(ANDROID_INC)CL/opencl.hpp
	$(CLANG) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl_opt.cpp -o $(OPFOLDER)android_lib_opencl_opt.o $(CFLAGS) $(ANDROID_OPENCL_FLAGS)

android_main_opencl_opt: main.cpp $(OPFOLDER)android_lib_opencl_opt.o android_cpu_functions.o $(ANDROID_LIB)$(ABI)/libOpenCL.so $(ANDROID_INC)CL/opencl.hpp
	mkdir -p $(OUTPUTFOLDER) 
	$(CLANG) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)android_lib_opencl_opt.o $(CPUFUNCTIONFOLDER)android_cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_android_opencl_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(ANDROID_OPENCL_FLAGS) $(ANDROID_LDFLAGS) 
# Android End OpenCL optimized

# OpenCL Part library
.PHONY: opencl-lib
opencl-lib:  main_opencl_lib

lib_opencl_lib.o: $(OPFOLDER)lib_opencl_lib.cpp 
	$(CC) -D$(DATATYPE) -DOPENCL -c $(OPFOLDER)lib_opencl_lib.cpp -o $(OPFOLDER)lib_opencl_lib.o $(CFLAGS) $(OPFLAGS) -lclblast

main_opencl_lib: main.cpp lib_opencl_lib.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_lib.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OPFLAGS)  -lclblast
# End OpenCL library


# Android OpenCL Part library
.PHONY: android_opencl-lib
android_opencl-lib: android_main_opencl_lib

$(OPFOLDER)android_lib_opencl_lib.o: $(OPFOLDER)lib_opencl_lib.cpp $(ANDROID_INC)CL/opencl.hpp $(ANDROID_INC)CL/clblast.h
	$(CLANG) -D$(DATATYPE) -DOPENCL -DANDROID -c $(OPFOLDER)lib_opencl_lib.cpp -o $(OPFOLDER)android_lib_opencl_lib.o $(CFLAGS) $(ANDROID_OPENCL_FLAGS) $(ANDROID_CLBLAST_FLAGS)

android_main_opencl_lib: main.cpp $(OPFOLDER)android_lib_opencl_lib.o android_cpu_functions.o $(ANDROID_LIB)$(ABI)/libOpenCL.so $(ANDROID_INC)CL/opencl.hpp $(ANDROID_INC)CL/clblast.h
	mkdir -p $(OUTPUTFOLDER)
	$(CLANG) -D$(DATATYPE) -DOPENCL -DANDROID main.cpp $(OPFOLDER)android_lib_opencl_lib.o $(CPUFUNCTIONFOLDER)android_cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_android_opencl_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(ANDROID_OPENCL_FLAGS) $(ANDROID_LDFLAGS) $(ANDROID_CLBLAST_FLAGS)
# End Android OpenCL library

# OpenMP Part
.PHONY: openmp
openmp:  main_openmp

lib_omp.o: $(OMPFOLDER)lib_omp.cpp 
	export OMP_NUM_THREADS=8
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp.cpp -o $(OMPFOLDER)lib_omp.o $(CFLAGS) $(OMPFLAGS)

main_openmp: main.cpp lib_omp.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENMP main.cpp $(OMPFOLDER)lib_omp.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_omp_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS)
# End OpenMP

# Android OpenMP Part
.PHONY: android_openmp
android_openmp:  android_main_openmp

android_lib_omp.o: $(OMPFOLDER)lib_omp.cpp 
	export OMP_NUM_THREADS=8
	$(CLANG) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp.cpp -o $(OMPFOLDER)android_lib_omp.o $(CFLAGS) $(OMPFLAGS)

android_main_openmp: main.cpp android_lib_omp.o android_cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CLANG) -D$(DATATYPE) -DOPENMP main.cpp $(OMPFOLDER)android_lib_omp.o  $(CPUFUNCTIONFOLDER)android_cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_android_omp_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS) $(ANDROID_LDFLAGS)
# End Android OpenMP

# OpenMP Part optimized
.PHONY: openmp-opt
openmp-opt:  main_openmp_opt

lib_omp_opt.o: $(OMPFOLDER)lib_omp_opt.cpp 
	export OMP_NUM_THREADS=8
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp_opt.cpp -o $(OMPFOLDER)lib_omp_opt.o $(CFLAGS) $(OMPFLAGS)

main_openmp_opt: main.cpp lib_omp_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENMP main.cpp $(OMPFOLDER)lib_omp_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_omp_opt_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS)
# End OpenMP optimized

# Android OpenMP Part optimized
.PHONY: android_openmp-opt
android_openmp-opt:  android_main_openmp_opt

android_lib_omp_opt.o: $(OMPFOLDER)lib_omp.cpp 
	export OMP_NUM_THREADS=8
	$(CLANG) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp_opt.cpp -o $(OMPFOLDER)android_lib_omp_opt.o $(CFLAGS) $(OMPFLAGS)

android_main_openmp_opt: main.cpp android_lib_omp_opt.o android_cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CLANG) -D$(DATATYPE) -DOPENMP main.cpp $(OMPFOLDER)android_lib_omp_opt.o  $(CPUFUNCTIONFOLDER)android_cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_android_omp_opt_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS) $(ANDROID_LDFLAGS)
# End Android OpenMP optimized


# OpenMP Part library
.PHONY: openmp-lib
openmp-lib:  main_openmp_lib

lib_omp_lib.o: $(OMPFOLDER)lib_omp_lib.cpp 
	$(CC) -D$(DATATYPE) -D$(LIBRARY) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp_lib.cpp -o $(OMPFOLDER)lib_omp_lib.o $(CFLAGS) $(OMPFLAGS) $(LIBFLAGS)   

main_openmp_lib: main.cpp lib_omp_lib.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -D$(LIBRARY) -DOPENMP main.cpp $(OMPFOLDER)lib_omp_lib.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_omp_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS) $(LIBFLAGS)
# End OpenMP library


# Android OpenMP Part library
.PHONY: android_openmp-lib
android_openmp-lib:  android_main_openmp_lib

android_lib_omp_lib.o: $(OMPFOLDER)lib_omp_lib.cpp 
	$(CLANG) -D$(DATATYPE) -D$(LIBRARY) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp_lib.cpp -o $(OMPFOLDER)android_lib_omp_lib.o $(CFLAGS) $(OMPFLAGS)  $(ANDROID_OPENMP-LIB_FLAG) 

android_main_openmp_lib: main.cpp android_lib_omp_lib.o android_cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CLANG) -D$(DATATYPE) -D$(LIBRARY) -DOPENMP main.cpp $(OMPFOLDER)android_lib_omp_lib.o  $(CPUFUNCTIONFOLDER)android_cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_android_omp_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS) $(ANDROID_LDFLAGS) $(ANDROID_OPENMP-LIB_FLAG)
# End Android OpenMP library

# CUDA part
.PHONY: cuda
cuda: main_cuda

lib_cuda.o: $(CUFOLDER)lib_cuda.cu
	$(NVCC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DCUDA -c $(CUFOLDER)lib_cuda.cu -o $(CUFOLDER)lib_cuda.o $(NVCCFLAGS)

main_cuda: main.cpp lib_cuda.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) -DCUDA main.cpp $(CUFOLDER)lib_cuda.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CUFLAGS) $(CFLAGS) -lstdc++
# End CUDA

# CUDA part optimized
.PHONY: cuda-opt
cuda-opt: main_cuda_opt

lib_cuda_opt.o: $(CUFOLDER)lib_cuda_opt.cu
	$(NVCC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DCUDA -c $(CUFOLDER)lib_cuda_opt.cu -o $(CUFOLDER)lib_cuda_opt.o $(NVCCFLAGS)


main_cuda_opt: main.cpp lib_cuda_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) -DCUDA main.cpp $(CUFOLDER)lib_cuda_opt.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CUFLAGS) $(CFLAGS) -lstdc++

# End CUDA optimized


# CUDA part library
.PHONY: cuda-lib
cuda-lib: main_cuda_lib

lib_cuda_lib.o: $(CUFOLDER)lib_cuda_lib.cu
	$(NVCC) -D$(DATATYPE) -DCUDA -c $(CUFOLDER)lib_cuda_lib.cu -o $(CUFOLDER)lib_cuda_lib.o $(NVCCFLAGS) 


main_cuda_lib: main.cpp lib_cuda_lib.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) -DCUDA main.cpp $(CUFOLDER)lib_cuda_lib.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CUFLAGS) $(CFLAGS) -lstdc++ -lcublas

# End CUDA library

# Hip part
.PHONY: hip
hip: main_hip

lib_hip.o: $(HIPFOLDER)lib_hip.cpp
	$(HIP) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIPFOLDER)lib_hip.cpp -o $(HIPFOLDER)lib_hip.o $(CFLAGS) $(HIPFLAGS)

main_hip: main.cpp lib_hip.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(HIP) -D$(DATATYPE) -DHIP main.cpp -x none $(HIPFOLDER)lib_hip.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_hip_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(HIPFLAGS)
# End Hip 

.PHONY: hip-opt
# Hip part optimized
hip-opt: main_hip_opt

lib_hip_opt.o: $(HIPFOLDER)lib_hip_opt.cpp
	$(HIP) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIPFOLDER)lib_hip_opt.cpp -o $(HIPFOLDER)lib_hip_opt.o $(CFLAGS) $(HIPFLAGS)

main_hip_opt: main.cpp lib_hip_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(HIP) -D$(DATATYPE) -DHIP main.cpp -x none $(HIPFOLDER)lib_hip_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_hip_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(HIPFLAGS)
# End Hip 



# Clean
.PHONY: clean
clean:
	rm -rf *.o 
	rm -rf $(CPUFUNCTIONFOLDER)*.o
	rm -rf $(OPFOLDER)*.o $(OPFOLDER)tmp
	rm -rf $(OMPFOLDER)*.o
	rm -rf $(HIPFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)_*
	rm -rf build/
	rm -rf build-android/
