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
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CONSTANTS 
	///////////////////////////////////////////////////////////////////////////////////////////////
	
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// linearizable versions of matrix
	unsigned int size_matrix =arguments_parameters->size * arguments_parameters->size;
	// A input matrix
	unsigned int size_A = arguments_parameters->size * arguments_parameters->size;
    unsigned int mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);
	// B input matrix
	unsigned int size_B = arguments_parameters->size * arguments_parameters->size ;
    unsigned int mem_size_B = sizeof(bench_t) * size_B;
	bench_t* B = (bench_t*) malloc(mem_size_B);
	// Correaltion Value alwais is a float number
	result_bench_t* h_R = (result_bench_t*) malloc(sizeof(result_bench_t));
	result_bench_t* d_R = (result_bench_t*) malloc(sizeof(result_bench_t));
	
	// comparation result
	bool result = false;
	// strucs for CPU timing
	struct timespec start, end;
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
		// iniciate C Values
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

	// base object init
	GraficObject *correlation_bench = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(correlation_bench, 0,arguments_parameters->gpu, device);
	if (!arguments_parameters->csv_format_timestamp && !arguments_parameters->csv_format && !arguments_parameters->mute_messages ){
		printf("Using device: %s\n", device);
	}
	// init memory
	device_memory_init(correlation_bench, size_A , size_B );
	// copy memory to device
	copy_memory_to_device(correlation_bench, A, size_A, B, size_B );
	// execute kernel
	execute_kernel(correlation_bench, arguments_parameters->size);
	// copy memory to host
	copy_memory_to_host(correlation_bench, d_R);

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(correlation_bench, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}
	if (arguments_parameters->print_output)
	{	
	    printf("%f ", *d_R);
		printf("\n");
	}
	
	if (arguments_parameters->verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		correlation_2D(A,B, h_R ,arguments_parameters->size);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %lu milliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (arguments_parameters->print_output)
		{
			printf("%f ", *h_R);
	    	printf("\n");
		} 
		result = compare_values(h_R, d_R);
	    if (result){
	    	printf("OK\n");
	    }
	    if (arguments_parameters->export_results){
	    	print_double_hexadecimal_values(GPU_FILE, d_R, 1);
	    	print_double_hexadecimal_values(CPU_FILE, h_R, 1);
	    }

	}
	if (arguments_parameters->export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_R, 1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(correlation_bench);
	// free object memory 
	free(correlation_bench);
	free(arguments_parameters);
	free(A);
	free(B);
	free(h_R);
	free(d_R);
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