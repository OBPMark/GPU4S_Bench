#!/bin/bash

# BASIC - DEFAULT VALUES
BENCHMARKTYPE=(cuda cuda-opt cuda-lib opencl opencl-opt opencl-lib)
DATATYPES=(INT FLOAT HALF DOUBLE)
BLOCKSIZES=(1024 16)

# ADVANCED - DEFAULT VALUES
BENCHMARKS=("matrix_multiplication_bench cuda-opt INT,FLOAT 1024,16" "max_pooling_bench cuda-opt,opencl FLOAT 1024")

# READ FROM CONFIGURATION FILE
. $PWD/compile.config
case "$1" in
	all)
		# ALL BENCHMARKS
		for d in ./*/ ; do  
			for datatype in ${DATATYPES[@]} ; do
				for blocksize in ${BLOCKSIZES[@]} ; do
					echo -e $"\033[1m=== Benchmark: $d"
					echo -e "=== make ${BENCHMARKTYPE[@]} DATATYPE=$datatype BLOCKSIZE=$blocksize\033[0m";
					(cd "$d" && make ${BENCHMARKTYPE[@]} DATATYPE=$datatype BLOCKSIZE=$blocksize);
				done
			done 
		done
	;;
	spc)
		# SPECIFIC BENCHMARKS	
		for benchmarkdata in "${BENCHMARKS[@]}" ; do
			set -- $benchmarkdata			
			ADV_FOLDER=$1
			ADV_BENCHTYPES=$2
			ADV_DTYPES=$3
			ADV_BLOCKSIZES=$4
			for datatype in ${ADV_DTYPES//,/ }; do
				for blocksize in ${ADV_BLOCKSIZES//,/ }; do
					echo -e $"\033[1m=== Benchmark: $ADV_FOLDER"
					echo -e $"=== make ${ADV_BENCHTYPES//,/ } DATATYPE=$datatype BLOCKSIZE=$blocksize\033[0m";
					(cd "$ADV_FOLDER" && make ${ADV_BENCHTYPES//,/ } DATATYPE=$datatype BLOCKSIZE=$blocksize);
				done
			done
		done

	;;
	clean)
		for d in ./*/ ; do  
			echo -e $"\033[1m=== Benchmark: $d"
			echo -e "=== make clean\033[0m";
			(cd "$d" && make clean);
		done
	;;
	*)
		echo -e $"\033[0;31mERROR:\033[0m Usage: $0 {all|spc|clean}."
		exit 1
esac