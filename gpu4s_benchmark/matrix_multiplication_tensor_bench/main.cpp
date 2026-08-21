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
	// linearizable versions of matrix
	unsigned int size_matrix = arguments_parameters->size  * arguments_parameters->size;
	unsigned int mem_size = sizeof(bench_t) * size_matrix;
	// A input matrix
	// initialized to nullptr to prevent wild/dangling pointer references with UMA
	bench_t* A = nullptr;
	// B input matrix
	bench_t* B = nullptr;
	// C matrix
	bench_t* d_C = nullptr;
	bench_t* h_C = (bench_t*) malloc(mem_size);
	// init devices
	char device[100] = "";

	// main object init
	GraficCommon*matrix_benck = (GraficCommon*)malloc(sizeof(GraficObject));
	
	// --- 1. Init Device & Context ---
	init(matrix_benck, 0, arguments_parameters->gpu, device);
	// Update profiling clock mode
	matrix_benck->profiling_clock = arguments_parameters->profiling_clock;

	// --- 2. Allocate Device Memory ---
	device_memory_init(matrix_benck, size_matrix, size_matrix, size_matrix);


	// --- 3. Allocate Host Pointers ---
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// map the buffzer to the gpu + cpu take the lead
			// UMA: map buffers between device and cpu (takes the lead)
			get_unified_memory_pointers(matrix_benck, A, B, d_C, mem_size);
		#else
			fprintf(stderr, "\033[1;31merror:\033[0m This framework is not compatible with unified memory. Please remove the -u arg!\n");			
			exit(-1);
		#endif
	} else
	{
		// normale malloc
		A = (bench_t*) malloc(mem_size);
		B = (bench_t*) malloc(mem_size);
		d_C = (bench_t*) malloc(mem_size);
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
	        	B[i*arguments_parameters->size+j] = rand() % (NUMBER_BASE * 100);
	        	#else
	        	B[i*arguments_parameters->size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
	        	#endif
	    	}
		}
	}
	else
	{	
		/// load data
		get_double_hexadecimal_values(arguments_parameters->input_file_A, A,size_matrix);
		get_double_hexadecimal_values(arguments_parameters->input_file_B, B,size_matrix);
		//get_values_file(input_file, A, B);

	}

	// reset output C matrix
	for (int i=0; i<arguments_parameters->size; i++){
		for (int j=0; j<arguments_parameters->size; j++){
			h_C[i*arguments_parameters->size+j] = 0;
			d_C[i*arguments_parameters->size+j] = 0;
			
		}
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
			sync_unified_memory_to_device(matrix_benck, A, B, d_C);
		#endif
	}
	else
	{	
		copy_memory_to_device(matrix_benck, A, B, size_matrix, size_matrix);
	}
	
	// execute kernel
	execute_kernel(matrix_benck, arguments_parameters->size, arguments_parameters->size, arguments_parameters->size);
	
	// copy memory to host
	if (arguments_parameters->unified_memory)
	{	
		#ifdef UMA_COMPATIBILITY
			// UMA: map back output buffer to host
			sync_unified_memory_to_host(matrix_benck, d_C, mem_size);
		#endif
    } else
	{
        copy_memory_to_host(matrix_benck, d_C, size_matrix);
    }

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(matrix_benck, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}

	// print output buffer
	if (arguments_parameters->print_output)
	{
		#ifdef INT
			for (int i=0; i<arguments_parameters->size; i++){
				for (int j=0; j<arguments_parameters->size; j++){
					printf("%d ", d_C[i*arguments_parameters->size+j]);	
				}
				printf("\n");
			}
		#else
			for (int i=0; i<arguments_parameters->size; i++){
				for (int j=0; j<arguments_parameters->size; j++){
					printf("%f ", d_C[i*arguments_parameters->size+j]);
				}
				printf("\n");
			}
		#endif
	}
	
	// export gpu buffer
	if (arguments_parameters->export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_C, size_matrix);
		//set_values_file(output_file, d_C, size);
	}

	//check for error
	if (arguments_parameters->verification)
	{
		Clock kernelCLK;
		kernelCLK.start();
		matrix_multiplication(A, B, h_C, arguments_parameters->size,  arguments_parameters->size, arguments_parameters->size);
		kernelCLK.end();

		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %.0f milliseconds\n", kernelCLK.getElapsedMS() );
		}

		if (arguments_parameters->print_output)
		{
		#ifdef INT
			for (int i=0; i<arguments_parameters->size; i++){
		    	for (int j=0; j<arguments_parameters->size; j++){
		    		printf("%d ", h_C[i*arguments_parameters->size+j]);
		        	
		    	}
	    		printf("\n");
			}
		#else
			for (int i=0; i<arguments_parameters->size; i++){
		    	for (int j=0; j<arguments_parameters->size; j++){
		    		printf("%f ", h_C[i*arguments_parameters->size+j]);
		        	
		    	}
	    		printf("\n");
			}
		#endif
		} 
	    
	    if (compare_vectors(h_C, d_C, size_matrix)){
	    	printf("OK\n");
	    }
	    if (arguments_parameters->export_results){
	    	//set_values_file(output_file, d_C, size);
	    	print_double_hexadecimal_values(GPU_FILE, d_C, size_matrix);
	    	print_double_hexadecimal_values(CPU_FILE, h_C, size_matrix);
	    }
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(matrix_benck);
	free(arguments_parameters);
	// free object memory 
	free(matrix_benck);

	if (!arguments_parameters->unified_memory) 
	{
        free(A);
        free(B);
        free(d_C);
    }
	free(h_C);
	return 0;
}


// Arguments part
void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
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
	// --- Properly clear character arrays ---
	arguments_parameters->input_file_A[0] = '\0';
	arguments_parameters->input_file_B[0] = '\0';
	arguments_parameters->output_file[0] = '\0';
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
			case 'g' : arguments_parameters->export_results_gpu   = true;break;
			case 'd' : args +=1; arguments_parameters->gpu = atoi(argv[args]);break;
			case 'f' : arguments_parameters->mute_messages = true;break;
					   args +=1;
					   strcpy(arguments_parameters->output_file,argv[args]);
					   break;
			case 'i' : args +=1;
					strcpy(arguments_parameters->input_file_A,argv[args]);
					args +=1;
					strcpy(arguments_parameters->input_file_B,argv[args]); //TODO FIX with final version of input files
					break;
			case 's' : args +=1; arguments_parameters->size  = atoi(argv[args]);break;
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