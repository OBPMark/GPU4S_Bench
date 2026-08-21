#include <time.h>
#include "benchmark_library.h"
#include "cpu_functions/cpu_functions.h"
#include <sys/time.h>

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
	// A input matrix
	unsigned int size_A = arguments_parameters->size * arguments_parameters->size;
    unsigned int mem_size_A = sizeof(bench_t) * size_A;
	// initialized to nullptr to prevent wild/dangling pointer references with UMA
	bench_t* A = nullptr;
	// B input matrix
	unsigned int size_lateral = arguments_parameters->size / arguments_parameters->stride;
	unsigned int size_B = size_lateral * size_lateral;
    unsigned int mem_size_B = sizeof(bench_t) * size_B;
	bench_t* d_B = nullptr;
	bench_t* h_B = (bench_t*) malloc(mem_size_B);
	// init devices
	char device[100] = "";

	// main object init
	GraficCommon*max_bench = (GraficCommon*)malloc(sizeof(GraficObject));
	
	// --- 1. Init Device & Context ---
	init(max_bench, 0,arguments_parameters->gpu, device);
	// Update profiling clock mode
	max_bench->profiling_clock = arguments_parameters->profiling_clock;

	// --- 2. Allocate Device Memory ---
	device_memory_init(max_bench, size_A, size_B);

	// --- 3. Allocate Host Pointers ---
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// map the buffzer to the gpu + cpu take the lead
			// UMA: map buffers between device and cpu (takes the lead)
			get_unified_memory_pointers(max_bench, A, mem_size_A);
		#else
			fprintf(stderr, "\033[1;31merror:\033[0m This framework is not compatible with unified memory. Please remove the -u arg!\n");			
			exit(-1);
		#endif
	} else
	{
		// normale malloc
		A = (bench_t*) malloc(mem_size_A);
		d_B = (bench_t*) malloc(mem_size_B);
	}
	

	///////////////////////////////////////////////////////////////////////////////////////////////
	// DATA INIT
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (strlen(arguments_parameters->input_file_A) == 0)
	{
		// inicialice input A matrix 
		for (int i=0; i<arguments_parameters->size; i++){
	    	for (int j=0; j<arguments_parameters->size; j++){
	    		#ifdef INT
	        	A[i*arguments_parameters->size+j] = rand() % (NUMBER_BASE * 100);
	        	#else
	        	A[i*arguments_parameters->size+j] = (double)rand()/RAND_MAX*2.0-1.0;
	        	#endif
	    	}
		}
		// iniciate output B matrix 
		for (int i=0; i<size_lateral; i++){
	    	for (int j=0; j<size_lateral; j++){
	        	h_B[i*size_lateral+j] = 0;
	        	d_B[i*size_lateral+j] = 0;
	    	}
		}
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
		printf("\n\n");

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
			sync_unified_memory_to_device(max_bench, A);
		#endif
	}
	else
	{	
		copy_memory_to_device(max_bench, A, size_A);
	}
	
	// execute kernel
	execute_kernel(max_bench, arguments_parameters->size, arguments_parameters->size, arguments_parameters->size, arguments_parameters->stride, size_lateral);
	
	// copy memory to host
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// UMA: map back output buffer to host
			sync_unified_memory_to_host(max_bench, d_B, size_B);
		#endif
    } else
	{
       	copy_memory_to_host(max_bench, d_B, size_B);
    }

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(max_bench, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}

	// print output buffer
	if (arguments_parameters->print_output)
	{
		#ifdef INT
			for (int i=0; i<size_lateral; i++){
				for (int j=0; j<size_lateral; j++){
					printf("%d ", d_B[i*size_lateral+j]);
				}
				printf("\n");
			}
			printf("\n");
		#else
			for (int i=0; i<size_lateral; i++){
				for (int j=0; j<size_lateral; j++){
					printf("%f ", d_B[i*size_lateral+j]);
					
				}
				printf("\n");
			}
			printf("\n");
		#endif
	}
	
	// export gpu buffer
	if (arguments_parameters->export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	}

	//check for error
	if (arguments_parameters->verification)
	{
		Clock cpuKernelCLK;
		cpuKernelCLK.start();
		max_pooling(A, h_B, arguments_parameters->size, arguments_parameters->stride, size_lateral);
		cpuKernelCLK.end();

		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %.0f milliseconds\n", cpuKernelCLK.getElapsedMS());
		}

		if (arguments_parameters->print_output)
		{
		#ifdef INT
			for (int i=0; i<size_lateral; i++){
		    	for (int j=0; j<size_lateral; j++){
		    		printf("%d ", h_B[i*size_lateral+j]);
		        	
		    	}
	    		printf("\n");
			}
		#else
			for (int i=0; i<size_lateral; i++){
		    	for (int j=0; j<size_lateral; j++){
		    		printf("%f ", h_B[i*size_lateral+j]);
		        	
		    	}
	    		printf("\n");
			}
		#endif
		} 
	    
	    if (compare_vectors(h_B, d_B, size_B))
		{
	    	printf("OK\n");
	    }
	    if (arguments_parameters->export_results)
		{
	    	print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	    	print_double_hexadecimal_values(CPU_FILE, h_B, size_B);
	    }

	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(max_bench);
	// free object memory 
	free(max_bench);
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
	printf("Usage: %s -s Size -k [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -l: size of the stride\n");
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
	printf(" -x: prints the timing of the validation. Only the sequential time of the application will be displayed\n");
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
			case 'l' : args +=1; arguments_parameters->stride = atoi(argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( arguments_parameters->size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if ( arguments_parameters->stride <= 0){
		printf("-l need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (arguments_parameters->mute_messages){
		arguments_parameters->csv_format = false;
	}
	return OK_ARGUMENTS;
}
