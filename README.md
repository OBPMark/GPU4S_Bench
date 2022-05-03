# **GPU4S Bench - OBPMark-Kernel**
## **Authors**
- Ivan Rodriguez Ferrandez (UPC-BSC)
- Alvaro Jover-Alvarez (UPC-BSC)
- Leonidas Kosmidis (BSC-UPC)
- David Steenari (ESA)

### **Version: 1.0**  

<br/>

## **Description**
Embedded  GPUs  have  been  identified  from  both private  and  government  space  agencies  as  promising  hardware technologies to satisfy the increased needs of payload processing.The GPU4S (GPU for Space) project funded from the EuropeanSpace  Agency  (ESA)  has  explored  in  detail  the  feasibility  and the  benefit  of  using  them  for  space  workloads.  Currently  at  the closing phases of the project, in this paper we describe the main project outcomes and explain the lessons we learnt. In addition,we  provide  some  guidelines  for  the  next  steps  towards  their adoption  in  space.


## **Implemented Languages**
- Standard C
- CUDA
- OpenCL
- OpenMP
- HIP
  
## **Benchmark List and Basic Description**

For most of the benchmark suite there is a naïve, optimize and library versions. The benchmarks with their implementations are listed below.
- Cifar 10
  - Naïve,optimize and library (only for CUDA)
- Cifar 10 Multiple
  - Naïve,optimize and library (only for CUDA)
- Convolution 2D
  - Naïve,optimize and library (only for CUDA)
- Correlation 2D
  - Naïve,optimize
- Fast fourier transform 2D bench
  - Library
- Fast fourier transform
  - Naïve,optimize and library 
- Fast fourier transform Window
  - Naïve,optimize and library
- Finite impulse response filter
  - Naïve
- Local response normalization (LRN)
  - Naïve,optimize and library (only for CUDA)
- Matrix multiplication
  - Naïve,optimize and library
- Max pooling bench
  - Naïve,optimize and library (only for CUDA)
- Memory Bandwidth 
  - Naïve
- Relu
  - Naïve,optimize and library (only for CUDA)
- Softmax
  - Naïve,optimize and library (only for CUDA)
- Wavelet transform
  - Naïve,optimize


## **Benchmark Compilation**
For compile each of the benchmarks first you need to go to the folder for the specific benchmark that you want to compile.
Inside of the folder you can call the Makefile for compilation. All of the Makefiles behaves the same for compilation. 

There is three parts for the make file. 
First is the type of benchmark that you want to compile, that could be *cuda* (this will compile cuda naïve) will be the same for the rest of the languages, for different version will be will the suffixes -opt, -lib for the optimize and library versions, example cuda-opt, opencl-lib.

The second part is the definition of the data type, for all of the benchmarks float and double is supported and some of the benchmarks supports also integer. For specify the data type you need to add *-DATATYPE=(language)* for the languages the naming is in capital letters and are FLOAT,DOUBLE and INT. 

The last parameter is the block size, this is only needed for the GPU code versions (OpenMP does not need this parameter). For the Makefile you need to provide *-BLOCKSIZE=(SIZE)* the block size is use square of the size that you provide, the recommended values are 4,8,16,32.

A full example will be as follows

``` make opencl-opt DATATYPE=FLOAT BLOCKSIZE=16 ```

The compiled binary will be in the bin folder.

