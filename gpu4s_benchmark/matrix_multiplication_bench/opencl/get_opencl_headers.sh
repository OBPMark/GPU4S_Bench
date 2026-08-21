#!/bin/bash
set -e # Exit immediately if any command fails

# --- Tested on v2026.05.29 ---
# Change this tag if you need a newer OpenCL specifications
OPENCL_RELEASE_TAG=v2026.05.29

ANDROID_INC=./android/include/

# --- Download OpenCL C Headers ---
if [ ! -d "$ANDROID_INC/CL" ]; then
    git clone -b "$OPENCL_RELEASE_TAG" --depth 1 -c advice.detachedHead=false https://github.com/KhronosGroup/OpenCL-Headers.git $ANDROID_INC/tmp
    
    # Structure the target directory
    rm -rf $ANDROID_INC/CL && mkdir -p $ANDROID_INC/CL && mv $ANDROID_INC/tmp/CL/* $ANDROID_INC/CL  && rm -rf $ANDROID_INC/tmp
fi

# --- Download opencl.hpp c++ Header ---
if [ ! -f "$ANDROID_INC/CL/opencl.hpp" ]; then
    curl -o $ANDROID_INC/CL/opencl.hpp https://raw.githubusercontent.com/KhronosGroup/OpenCL-CLHPP/${OPENCL_RELEASE_TAG}/include/CL/opencl.hpp
fi
#For the older version of openCL :
#curl -o $ANDROID_INC/CL/cl2.hpp https://raw.githubusercontent.com/#KhronosGroup/OpenCL-CLHPP/${OPENCL_RELEASE_TAG}/include/CL/cl2.hpp



