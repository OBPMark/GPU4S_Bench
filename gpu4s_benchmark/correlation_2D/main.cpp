#include <time.h>
#include "benchmark_library.h"
#include "cpu_functions/cpu_functions.h"
#include <sys/time.h>

#define NUMBER_BASE 1


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
	// CONSTANTS 
	///////////////////////////////////////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// linearizable versions of matrix
	unsigned int size_matrix  =arguments_parameters->size * arguments_parameters->size;
	// A input matrix
    unsigned int mem_size = sizeof(bench_t) * size_matrix;
	// initialized to nullptr to prevent wild/dangling pointer references with UMA
	bench_t* A = nullptr;
	// B input matrix
	bench_t* B = nullptr;
	// Correaltion Value alwais is a float number
	result_bench_t* d_R = nullptr;
	result_bench_t* h_R = (result_bench_t*) malloc(sizeof(result_bench_t));
	// init devices
	char device[100] = "";

	// main object init
	GraficCommon*correlation_bench = (GraficCommon*)malloc(sizeof(GraficObject));

	// --- 1. Init Device & Context ---
	init(correlation_bench, 0,arguments_parameters->gpu, device);
	// Update profiling clock mode
	correlation_bench->profiling_clock = arguments_parameters->profiling_clock;

	// --- 2. Allocate Device Memory ---
	device_memory_init(correlation_bench, size_matrix, size_matrix );

	// --- 3. Allocate Host Pointers ---
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// map the buffzer to the gpu + cpu take the lead
			// UMA: map buffers between device and cpu (takes the lead)
			get_unified_memory_pointers(correlation_bench, A, B, d_R, mem_size);
		#else
			fprintf(stderr, "\033[1;31merror:\033[0m This framework is not compatible with unified memory. Please remove the -u arg!\n");			
			exit(-1);
		#endif
	} else
	{
		// normale malloc
		A = (bench_t*) malloc(mem_size);
		B = (bench_t*) malloc(mem_size);
		d_R = (result_bench_t*) malloc(sizeof(result_bench_t));
	}


	///////////////////////////////////////////////////////////////////////////////////////////////
	// DATA INIT
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (strlen(arguments_parameters->input_file_A) == 0)
	{
		// inicialice A matrix 
		for (int i=0; i<arguments_parameters->size; i++){
			for (int j=0; j<arguments_parameters->size; j++){
				#ifdef INT
				//A[i] = i+1;
				A[i*arguments_parameters->size+j] = rand() % (NUMBER_BASE * 100);
				#else
				A[i*arguments_parameters->size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
				#endif
	    	}
		}
		// iniciate B matrix 
		for (int i=0; i<arguments_parameters->size; i++){
			for (int j=0; j<arguments_parameters->size; j++){
				#ifdef INT
				//A[i] = i+1;
				B[i*arguments_parameters->size+j] = rand() % (NUMBER_BASE * 100);
				#else
				B[i*arguments_parameters->size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
				#endif
			}
		}
		// iniciate R Values
		*h_R = 0;
		*d_R = 0;

	}
	else
	{	
		// load data TODO
		/*get_double_hexadecimal_values(input_file_A, A,size_A);
		get_double_hexadecimal_values(input_file_B, B,size_B);
		
		// iniciate C matrix
		for (int i=0; i<size; i++){
	    	for (int j=0; j<size; j++){
	        	h_C[i*size+j] = 0;
	        	d_C[i*size+j] = 0;
	        	
	    	}
		}*/
	}
	// print input
	if (arguments_parameters->print_input)
	{
		for (int i=0; i<arguments_parameters->size; i++){
			for (int j=0; j<arguments_parameters->size; j++){
				#ifdef INT
				printf("%d ",A[i*arguments_parameters->size+j]);
				#else
				printf("%f ",A[i*arguments_parameters->size+j]);
				#endif
	    	}
			printf("\n");
		}
		printf("\n");
		for (int i=0; i<arguments_parameters->size; i++){
			for (int j=0; j<arguments_parameters->size; j++){
				#ifdef INT
				printf("%d ",B[i*arguments_parameters->size+j]);
				#else
				printf("%f ",B[i*arguments_parameters->size+j]);
				#endif
	    	}
			printf("\n");
		}
		printf("\n");

	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (!arguments_parameters->csv_format_timestamp && !arguments_parameters->csv_format && !arguments_parameters->mute_messages ){
		printf("Using device: %s\n", device);
	}

	// copy memory to device
	if(arguments_parameters->unified_memory)
	{
		#ifdef UMA_COMPATIBILITY 
			// UMA: unmap shared buffer from host to device
			sync_unified_memory_to_device(correlation_bench, A, B, d_R);
		#endif
	}
	else
	{	
		copy_memory_to_device(correlation_bench, A, size_matrix, B, size_matrix );
	}
	
	// execute kernel
	execute_kernel(correlation_bench, arguments_parameters->size);

	// copy memory to host
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// UMA: map back output buffer to host
			sync_unified_memory_to_host(correlation_bench, d_R, sizeof(result_bench_t));
		#endif
    } else
	{
        copy_memory_to_host(correlation_bench, d_R);
    }

	
	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(correlation_bench, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}

	// print output buffer
	if (arguments_parameters->print_output)
	{	
	    printf("%f ", *d_R);
		printf("\n");
	}

	// export gpu buffer
	if (arguments_parameters->export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_R, 1);
	}

	//check for error
	if (arguments_parameters->verification)
	{
		Clock cpuKernelCLK;
		cpuKernelCLK.start();	
		correlation_2D(A,B, h_R ,arguments_parameters->size);
		cpuKernelCLK.end();

		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %.0f milliseconds\n", cpuKernelCLK.getElapsedMS());
		}

		if (arguments_parameters->print_output)
		{
			printf("%f ", *h_R);
	    	printf("\n");
		} 

		
	    if (compare_values(h_R, d_R))
		{
	    	printf("OK\n");
	    }

	    if (arguments_parameters->export_results){
	    	print_double_hexadecimal_values(GPU_FILE, d_R, 1);
	    	print_double_hexadecimal_values(CPU_FILE, h_R, 1);
	    }
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(correlation_bench);
	// free object memory 
	free(correlation_bench);
	free(arguments_parameters);

	if (!arguments_parameters->unified_memory) 
	{
        free(A);
        free(B);
        free(d_R);
    }

	free(h_R);
return 0;
}


// Arguments part

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size -k [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrix A and matrix B\n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -q: prints input values\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
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
			case 'f' : arguments_parameters->mute_messages = true;break;
			// specific
			case 'i' : args +=1;
					   strcpy(arguments_parameters->input_file_A,argv[args]);
					   args +=1;
					   strcpy(arguments_parameters->input_file_B,argv[args]);
					   break;
			case 's' : args +=1; arguments_parameters->size = atoi(argv[args]);break;
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
	if (arguments_parameters->mute_messages){
		arguments_parameters->csv_format = false;
	}
	return OK_ARGUMENTS;
}
