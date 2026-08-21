#include <time.h>
#include "benchmark_library.h"
#include "cpu_functions/cpu_functions.h"
#include <sys/time.h>
#include <cstring>

#define NUMBER_BASE 1
// OUTPUT C is N x W matrix
// Print hexadecimal values of result 

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define GPU_FILE "gpu_file.out"
#define CPU_FILE "cpu_file.out"

int arguments_handler(int argc, char ** argv, BenchmarkParameters* arguments_parameters);

int main(int argc, char *argv[]){
	// random init
	srand (21121993);
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	BenchmarkParameters *arguments_parameters = (BenchmarkParameters *)malloc(sizeof(BenchmarkParameters));

	int resolution = arguments_handler(argc,argv,arguments_parameters);
	if (resolution == ERROR_ARGUMENTS)
	{
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// A input vector
	int64_t size_A = arguments_parameters->size;
	int64_t mem_size = sizeof(COMPLEX*) * size_A;
	// initialized to nullptr to prevent wild/dangling pointer references with UMA
	COMPLEX **A = (COMPLEX **)malloc(arguments_parameters->size * sizeof(COMPLEX*));
    for(int64_t i = 0; i < arguments_parameters->size; ++i) A[i] = nullptr;

    COMPLEX **d_B = (COMPLEX **)malloc(arguments_parameters->size * sizeof(COMPLEX*));
    for(int64_t i = 0; i < arguments_parameters->size; ++i) d_B[i] = nullptr;

    COMPLEX **h_B = (COMPLEX **)malloc(arguments_parameters->size * sizeof(COMPLEX*));
    for(int64_t i = 0; i < arguments_parameters->size; ++i) h_B[i] = (COMPLEX *)malloc(arguments_parameters->size * sizeof(COMPLEX));

	// init devices
	char device[100] = "";
	
	// main object init
	GraficCommon* fft_bench = (GraficCommon*)malloc(sizeof(GraficObject));
	
	// --- 1. Init Device & Context ---
	init(fft_bench, 0,arguments_parameters->gpu, device);
	// Update profiling clock mode
	fft_bench->profiling_clock = arguments_parameters->profiling_clock;

	// --- 2. Allocate Device Memory ---
	device_memory_init(fft_bench, size_A);

	// --- 3. Allocate Host Pointers ---
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// map the buffzer to the gpu + cpu take the lead
			// UMA: map buffers between device and cpu (takes the lead)
			get_unified_memory_pointers(fft_bench, A, d_B, size_A);
		#else
			fprintf(stderr, "\033[1;31merror:\033[0m This framework is not compatible with unified memory. Please remove the -u arg!\n");			
			exit(-1);
		#endif
	} else
	{
		// normale malloc
		A = (COMPLEX **)malloc(arguments_parameters->size * sizeof(COMPLEX*));
    	for(int64_t i = 0; i < arguments_parameters->size; ++i) A[i] = (COMPLEX *)malloc(arguments_parameters->size * sizeof(COMPLEX));

		d_B = (COMPLEX **)malloc(arguments_parameters->size * sizeof(COMPLEX*));
    	for(int64_t i = 0; i < arguments_parameters->size; ++i) d_B[i] = (COMPLEX *)malloc(arguments_parameters->size * sizeof(COMPLEX));
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DATA INIT
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (strlen(arguments_parameters->input_file) == 0)
	{
	// inicialice A matrix 
		for (int i=0; i<arguments_parameters->size; ++i){
			for (int j=0; j<arguments_parameters->size; ++j){
				A[i][j].x = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
				A[i][j].y = 0;

				if (arguments_parameters->print_input)
				{
					printf("%f %f,",A[i][j].x, A[i][j].y);
				}
		    	h_B[i][j].x = A[i][j].x;
		    	h_B[i][j].y = A[i][j].y;

	    	}
	    	// if (arguments_parameters->print_input)
			// {	
			// 	printf("\n");
			// }
		}
		
	}
	else
	{	
		// load data
		//get_double_hexadecimal_values(input_file_A, A,size_A);
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (!arguments_parameters->csv_format_timestamp && !arguments_parameters->csv_format && !arguments_parameters->mute_messages )
	{
		printf("Using device: %s\n", device);
	}

	// copy memory to device
	if(arguments_parameters->unified_memory)
	{
		#ifdef UMA_COMPATIBILITY 
			// UMA: unmap shared buffer from host to device
			sync_unified_memory_to_device(fft_bench, A, d_B);
		#endif
	} else
	{	
		copy_memory_to_device(fft_bench, A, size_A);
	}
	
	// execute kernel
	execute_kernel(fft_bench, size_A);
	
	// copy memory to host
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// UMA: map back output buffer to host
			sync_unified_memory_to_host(fft_bench, d_B, size_A);
		#endif
    } else
	{
        copy_memory_to_host(fft_bench, d_B, arguments_parameters->size);
    }

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(fft_bench, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}

	// print output buffer
	if (arguments_parameters->print_output)
	{
		for (int i=0; i<arguments_parameters->size; ++i){
			for (int j=0; j<arguments_parameters->size; ++j){
				printf("%f %f,",d_B[i][j].x, d_B[i][j].y);
		    }
			printf("\n");
		}
	}

	// export gpu buffer
	if (arguments_parameters->export_results_gpu)
	{
		//print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	}

	//check for error
	if (arguments_parameters->verification)
	{
		Clock cpuKernelCLK;
		cpuKernelCLK.start();
		FFT2D(A, size_A, size_A, h_B);
		cpuKernelCLK.end();
		
		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %.0f milliseconds\n", cpuKernelCLK.getElapsedMS());
		}

		if (arguments_parameters->print_output)
		{
	    //result = compare_vectors(h_B, d_B, size_A);
			for (int i=0; i<arguments_parameters->size; ++i){
				for (int j=0; j<arguments_parameters->size; ++j){
					printf("%f %f,",h_B[i][j].x, h_B[i][j].y);
		    	}
				printf("\n");
			}
		} 

	    if (compare_vectors(h_B, d_B, size_A))
		{
	    	printf("OK\n");
	    }

	    if (arguments_parameters->export_results)
		{
	    	//print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	    	//print_double_hexadecimal_values(CPU_FILE, h_B, size_B);
	    }
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(fft_bench);
	// free object memory 
	free(fft_bench);
	free(arguments_parameters);

	if (!arguments_parameters->unified_memory) 
	{
        free(A);
        free(d_B);
    }
	
	free(h_B);
	return 0; 
}



// Arguments part
void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-w] [-v] [-e] [-o] [-t] [-c] [-d] [-i input_file_A_MATRIX ] \n", appName);
	printf(" -s Size : set size of furier transform power of 2 \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -q: prints input\n");
	printf(" -d: selects GPU\n");
	printf(" -h: print help information\n");
	printf(" -p: clock profilling\n");
	printf(" -u: enable unified memory (ANDROID/JETSON)\n");
}

void init_arguments(BenchmarkParameters* arguments_parameters){
	arguments_parameters->size = 0;
	arguments_parameters->gpu = 0;
	arguments_parameters->verification = false;
	arguments_parameters->export_results = false;
	arguments_parameters->export_results_gpu = false;
	arguments_parameters->print_output = false;
	arguments_parameters->print_input = false;
	arguments_parameters->print_timing = false;
	arguments_parameters->csv_format = false;
	arguments_parameters->mute_messages = false;
	arguments_parameters->csv_format_timestamp = false;
	arguments_parameters->unified_memory = false;

	// If android and opencl force profiling clock
	#ifdef FORCE_PROFILING_CLOCK
		arguments_parameters->profiling_clock = true;
	#else
		arguments_parameters->profiling_clock = false;
	#endif
}


int arguments_handler(int argc, char ** argv, BenchmarkParameters* arguments_parameters){
	init_arguments(arguments_parameters);
	if (argc == 1){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	} 
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			// comon part
			case 'v' : arguments_parameters->verification = true;break;
			case 'e' : arguments_parameters->verification = true; arguments_parameters->export_results= true;break;
			case 'o' : arguments_parameters->print_output = true;break;
			case 't' : arguments_parameters->print_timing = true;break;
			case 'c' : arguments_parameters->csv_format   = true;break;
			case 'C' : arguments_parameters->csv_format_timestamp = true;break;  
			case 'g' : arguments_parameters->export_results_gpu = true;break;
			case 'q' : arguments_parameters->print_input = true;break;
			case 'd' : args +=1; arguments_parameters->gpu = atoi(argv[args]);break;
			// specific
			case 'i' : args +=1;
					   strcpy(arguments_parameters->input_file,argv[args]);
					   break;
			case 's' : args +=1; arguments_parameters->size = atol(argv[args]);break;
			case 'p' : arguments_parameters->profiling_clock = true;break;
			case 'u' : arguments_parameters->unified_memory  = true;break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( arguments_parameters->size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	// specific
	return OK_ARGUMENTS;
}
