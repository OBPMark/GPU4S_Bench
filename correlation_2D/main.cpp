#include <time.h>
#include "benchmark_library.h"
#include "cpu/lib_cpu.h"
#include <sys/time.h>

#define NUMBER_BASE 1


#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define GPU_FILE "gpu_file.out"
#define CPU_FILE "cpu_file.out"

int arguments_handler(int argc, char ** argv,unsigned int *size ,unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format,bool *print_input, bool *validation_timing, bool *mute_messages,char *input_file_A, char *input_file_B);

int main(int argc, char *argv[]){
	// random init
	srand (21121993);
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	unsigned int size = 0, gpu = 0;
	bool verification  = false, export_results = false, print_output = false, print_timing = false, export_results_gpu = false, csv_format = false, print_input = false, validation_timing = false, mute_messages = false;
	char input_file_A[100] = "";
	char input_file_B[100] = "";

	int resolution = arguments_handler(argc,argv, &size,&gpu, &verification, &export_results, &export_results_gpu,&print_output, &print_timing, &csv_format, &print_input, &validation_timing, &mute_messages,input_file_A, input_file_B);
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
	unsigned int size_matrix =size * size;
	// A input matrix
	unsigned int size_A = size * size;
    unsigned int mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);
	// B input matrix
	unsigned int size_B = size * size ;
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
	if (strlen(input_file_A) == 0)
	{
		// inicialice A matrix 
		for (int i=0; i<size; i++){
			for (int j=0; j<size; j++){
				#ifdef INT
				//A[i] = i+1;
				A[i*size+j] = rand() % (NUMBER_BASE * 100);
				#else
				A[i*size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
				#endif
	    	}
		}
		// iniciate B matrix 
		for (int i=0; i<size; i++){
			for (int j=0; j<size; j++){
				#ifdef INT
				//A[i] = i+1;
				B[i*size+j] = rand() % (NUMBER_BASE * 100);
				#else
				B[i*size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
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
	if (print_input)
	{
		for (int i=0; i<size; i++){
			for (int j=0; j<size; j++){
				#ifdef INT
				printf("%d ",A[i*size+j]);
				#else
				printf("%f ",A[i*size+j]);
				#endif
	    	}
			printf("\n");
		}
		printf("\n");
		for (int i=0; i<size; i++){
			for (int j=0; j<size; j++){
				#ifdef INT
				printf("%d ",B[i*size+j]);
				#else
				printf("%f ",B[i*size+j]);
				#endif
	    	}
			printf("\n");
		}
		printf("\n");

	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE FOR ONLY TIMING  OF THE VALIDATION
	///////////////////////////////////////////////////////////////////////////////////////////////
	if(validation_timing){
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		correlation_2D(A,B, h_R ,size);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (!mute_messages){
			printf("CPU Time %lu miliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (print_output)
		{
			printf("%f ", *h_R);
			printf("\n");
		} 
		exit(0);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////

	// base object init
	GraficObject *correlation_bench = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(correlation_bench, 0,gpu, device);
	if (!csv_format && !mute_messages ){
		printf("Using device: %s\n", device);
	}
	// init memory
	device_memory_init(correlation_bench, size_A , size_B );
	// copy memory to device
	copy_memory_to_device(correlation_bench, A, size_A, B, size_B );
	// execute kernel
	execute_kernel(correlation_bench, size);
	// copy memory to host
	copy_memory_to_host(correlation_bench, d_R);

	// get time
	if (print_timing || csv_format)
	{
		get_elapsed_time(correlation_bench, csv_format);
	}
	if (print_output)
	{	
	    printf("%f ", *d_R);
		printf("\n");
	}
	
	if (verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		correlation_2D(A,B, h_R ,size);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (print_timing)
		{
			printf("CPU Time %lu miliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (print_output)
		{
			printf("%f ", *h_R);
	    	printf("\n");
		} 
		result = compare_values(h_R, d_R);
	    if (result){
	    	printf("OK\n");
	    }
	    if (export_results){
	    	print_double_hexadecimal_values(GPU_FILE, d_R, 1);
	    	print_double_hexadecimal_values(CPU_FILE, h_R, 1);
	    }

	}
	if (export_results_gpu)
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
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verificaction of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -q: prints input values\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -x: prints the timing of the validation. Only the sequential time of the application will be displayed\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}


int arguments_handler(int argc, char ** argv,unsigned int *size ,unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format,bool *print_input, bool *validation_timing, bool *mute_messages, char *input_file_A, char *input_file_B){
	if (argc == 1){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	} 
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			// comon part
			case 'v' : *verification = true;break;
			case 'e' : *verification = true; *export_results= true;break;
			case 'o' : *print_output = true;break;
			case 't' : *print_timing = true;break;
			case 'c' : *csv_format   = true;break;
			case 'g' : *export_results_gpu = true;break;
			case 'q' : *print_input = true;break;
			case 'd' : args +=1; *gpu = atoi(argv[args]);break;
			case 'x' : *validation_timing = true;break;
			case 'f' : *mute_messages = true;break;
			// specific
			case 'i' : args +=1;
					   strcpy(input_file_A,argv[args]);
					   args +=1;
					   strcpy(input_file_B,argv[args]);
					   break;
			case 's' : args +=1; *size = atoi(argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( *size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (*mute_messages){
		*csv_format = false;
	}
	return OK_ARGUMENTS;
}