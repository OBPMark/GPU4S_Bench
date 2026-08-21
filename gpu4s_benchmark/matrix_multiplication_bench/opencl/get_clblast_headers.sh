#!/bin/bash
set -e # Exit immediately if any command fails

# --- Tested on 1.7.0 ---
# Change this tag if you need a newer cblast specifications
CLBLAST_RELEASE_TAG=1.7.0

ANDROID_INC=./android/include/

# --- Download cblast.h Header ---
if [ ! -f "$ANDROID_INC/clblast.h" ]; then
    curl -o $ANDROID_INC/clblast.h https://raw.githubusercontent.com/CNugteren/CLBlast/${CLBLAST_RELEASE_TAG}/include/clblast.h
fi


