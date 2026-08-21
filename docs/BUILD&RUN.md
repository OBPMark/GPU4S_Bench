# GPU4S: Build & Usage Guide
This document describes how to build and run all the benchmarks of GPU4S. For general information about the project, check the main [README](../README.md).

## Build Instructions

GPU4S Bench can be built with 2 different methods:

- **globally**: all 17 benchmarks at once, from the `gpu4s` root
- **standalone**: single benchmark, from its own subfolder 

Both follow the same two steps: configure with CMake, then build.

### 1. Configure

**Host (x86_64 / Linux):**
```bash
cmake -B build
```
**Android:**
```bash 
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/27.3.13750724/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-21   
```


>**Notes**: You can name the build directory however you like, which makes it easy to keep multiple configurations side by side:
> ```bash
> cmake -B build-float-256 -DDATATYPE=FLOAT -DBLOCKSIZE=256
> cmake -B build-double-128 -DDATATYPE=DOUBLE -DBLOCKSIZE=128
> ```

<br>


### 2. Build

**Build every target:**
```bash
cmake --build build -j$(nproc)            # host
cmake --build build-android -j$(nproc)    # android
```

>**Notes**: The option `-j$(nproc)` forces your CPU to use all of its core to speed up the process.


**Build a specific target (optional):**
```bash
# Standalone (e.g.: from inside matrix_multiplication_bench/)
cmake --build build --target cpu
cmake --build build --target all-opencl
cmake --build build --target cuda-opt

# Global (from gpu4s_benchmark/ root)
cmake --build build --target matrix_mult-cpu
cmake --build build --target matrix_mult-all-opencl
cmake --build build --target matrix_mult-cuda-opt
```
>**Notes**: The option `--target` can be used to select a specific target.


<details>
<summary><b style="font-size: 1.25em;">📋 List of Target Shortcuts</b></summary>

- **CPU**: `cpu`, `CPU`
- **HIP**: `hip-`, `HIP`, `hip-opt`, `HIP-opt`, `all-hip`
- **CUDA**: `cuda`, `CUDA`, `cuda-opt`, `CUDA-opt`, `cuda-lib`, `CUDA-lib`, `all-cuda`
- **OpenCL**: `cl`, `OpenCL`, `opencl-opt`, `OpenCL-opt`, `opencl-lib`, `OpenCL-lib`, `all-opencl`
- **OpenMP**: `openmp`, `OpenMP`, `openmp-opt`, `OpenMP-opt`, `openmp-lib`, `OpenMP-lib`, `all-openmp`
</details>

<br>

### 3. Compile options

You can customize compilation options during the configuration step by adding `-D<PARAMETER>=<VALUE>` to `cmake -B ...`:

#### Available Parameters

| Parameter | Default | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `DATATYPE` | `FLOAT` | `FLOAT`, `DOUBLE`, `INT` | Data type used in computations |
| `BLOCKSIZE` | `16` | `4`, `8`, `16`, `32`, etc. | 2D tile/block size for GPU kernels |
| `OPT_FLAG` | `-O3` | `-O2`, `-O3`, `-Ofast` | Host compiler optimization level |
| `CUDA_ARCH` | `native` | `native`, `sm_70`, `sm_75`, `sm_80`, `sm_86`, etc. | NVIDIA GPU compute architecture |
| `OPENCL_VERSION` | `300` | `200`, `210`, `220`, `300`, `310` | OpenCL target API version |
| `BLA_VENDOR` | `OpenBLAS` | `OpenBLAS`, `ATLAS`, `Generic` | BLAS library vendor for OpenMP-lib |
| `NSTREAMS` | `4` | Integer $\ge 1$ | Number of concurrent streams (`cifar_10_multiple`) |


<details>
<summary><b style="font-size: 1.25em;">Usage examples</b></summary>

**Simple — change the data type:**

```bash
# e.g. Use double precision instead of float
cmake -B build -DDATATYPE=DOUBLE
```
**Combined — fully customize multiple parameters:**

Chain multiple `-D` flags to customize data type, tuning, and target hardware in a single command. Example: double precision, 32x32 block size, targeting an Ampere GPU with `-Ofast`:

```bash
cmake -B build \
  -DDATATYPE=DOUBLE \
  -DBLOCKSIZE=32 \
  -DCUDA_ARCH=sm_86 \
  -DOPT_FLAG=-Ofast
```
</details>

<br>

### 4. Cleaning up the workspace
In order to reset your build environment, you can remove all generated build directories and start fresh:

```bash
rm -rf build*
```

## Usage

Once compiled, the executables are located in `./<build-directory>/bin`.


### 1. Running on Host (Linux / PC)

To execute a benchmark on your local machine, simply launch the generated executable from the terminal. 

```bash
# General syntax
./build/bin/<target_name> [arguments]

# Example: Running the standard CPU matrix multiplication
./build/bin/matrix_mult_cpu -s 1024

# Example: Running the optimized CUDA version
./build/bin/matrix_mult_cuda_opt -s 2048
```

### 2. Running on Android

The launch command is the same, but you first need to push the executable file to your phone, then you can execute it through `adb shell`.


```bash
# Push all binaries to the device
adb push ./build-android/bin/* /data/local/tmp/

# Grant execution permissions to the binaries
adb shell chmod 755 /data/local/tmp/*

# Execute the benchmarks on the Android device
adb shell /data/local/tmp/matrix_mult_opencl -s 1024 -t -v
```

### Runtime Parameters

All devices share the same set of flags. For most benchmarks, you will need to provide at bare minimum a size flag (`-s`), which indicates how demanding the benchmark will be by changing the size of the calculation, and at least one of the timing/profiling output flags (`-t`, `-c`, `-C`) so the benchmark actually returns some result.

| Flag | Name | Description |
| :--- | :--- | :--- |
| `-s <size>` | **Size** | Sets the X and Y dimensions for the benchmark (e.g., matrix size $N \times N$). |
| `-v` | **Verify** | Executes a CPU baseline calculation and compares it with the GPU output to verify correctness. |
| `-t` | **Timing** | Prints the elapsed execution time to the console. |
| `-c` / `-C` | **CSV format** | Prints timing results in CSV format. Use `-C` to include a timestamp. |
| `-u` | **Unified Memory** | Enables Unified Memory mapping (Zero-Copy). **Highly recommended for Android and Jetson** platforms to prevent unnecessary memory transfers. |
| `-p` | **Profiling** | Enables internal clock profiling for more granular hardware timing. |
| `-d <id>` | **Device ID** | Selects the specific GPU device ID to use (default is usually 0). |
| `-o` | **Print Output** | Prints the resulting matrix or array directly to the terminal. |
| `-f` | **Mute** | Mutes all standard print messages (useful for batch testing scripts). |
| `-i <fileA> <fileB>`| **Input Files** | Loads custom hexadecimal input files instead of using randomized data generation. |
| `-g` / `-e` | **Export** | `-g` exports the GPU result to `gpu_file.out`. `-e` exports both GPU and CPU results in hex format (automatically enables `-v`). |
| `-k <size>` | **Kernel Size** | *(Specific to `convolution_2D`, `FIR_filter`)* Defines the size of the convolution filter / mask ($K \times K$) or FIR tap length. |
| `-l <stride>`| **Stride Size** | *(Specific to `max_pooling`)* Defines the horizontal and vertical step size of the pooling sliding window. |