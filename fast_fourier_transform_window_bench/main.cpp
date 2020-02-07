#include <time.h>
#include "benchmark_library.h"
#include "cpu/lib_cpu.h"
#include <sys/time.h>

#define NUMBER_BASE 1
// OUTPUT C is N x W matrix
// Print hexadecimal values of result 

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define GPU_FILE "gpu_file.out"
#define CPU_FILE "cpu_file.out"

int arguments_handler(int argc, char ** argv,int64_t *size, int64_t *window, unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format,bool *print_input,char *input_file_A, char *input_file_B);

int main(int argc, char *argv[]){
	// random init
	srand (21121993);
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	int64_t size = 0, window = 8;
	unsigned int gpu = 0;
	bool verification  = false, export_results = false, print_output = false, print_timing = false, export_results_gpu = false, print_input = false, csv_format = false;
	char input_file_A[100] = "";
	char input_file_B[100] = "";

	int resolution = arguments_handler(argc,argv, &size, &window, &gpu, &verification, &export_results, &export_results_gpu,&print_output, &print_timing, &csv_format, &print_input,input_file_A, input_file_B);
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// A input vector
	int64_t size_A = size;
    int64_t mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);
	// B output vector
	int64_t size_B = ((size - window) + 1) * window;
    int64_t mem_size_B = sizeof(bench_t) * size_B;
	bench_t* h_B = (bench_t*) malloc(mem_size_B);
	bench_t* d_B = (bench_t*) malloc(mem_size_B);

	bench_t aux_value = 0;
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
		for (int i=0; i<size_A; i++)
		{
			if (i % 2 == 0)
			{
				aux_value = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
			}	
			else
			{
				aux_value = 0;
			}
			if (print_input)
			{
				printf("%f ",aux_value);
			}
	    	A[i] = aux_value;
		}
		if (print_input)
		{
			printf("\n");
		}
	}
	else
	{	
		// load data
		get_double_hexadecimal_values(input_file_A, A,size_A);
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////
	/*for (unsigned int i = 0; i < size_A; ++i)
	{
		h_B[i] = A[i];
		d_B[i] = A[i];
	}*/
	// base object init
	GraficObject *fft_benck = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(fft_benck, 0,gpu, device);
	if (!csv_format){
		printf("Using device: %s\n", device);
	}
	// init memory
	device_memory_init(fft_benck, size_A ,size_B);
	// copy memory to device
	copy_memory_to_device(fft_benck, A, size_A);
	// execute kernel
	execute_kernel(fft_benck, window, size>>1);
	// copy memory to host
	copy_memory_to_host(fft_benck, d_B, size_B);

	// get time
	if (print_timing || csv_format)
	{
		get_elapsed_time(fft_benck, csv_format);
	}
	if (print_output)
	{
		for (int i=0; i<size_B; i++){
	    	printf("%f ", d_B[i]);	
    		
		}
		printf("\n");
		// re print for get the same result of matlab
		/*for (int i=0; i<size_B; i++){
	    	printf("%f ", d_B[i]);	
    		
		}
		printf("\n");*/
	}
	


	if (verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		fft_function(A ,h_B , window ,size>>1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (print_timing)
		{
			printf("CPU Time %lu miliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (print_output)
		{
			for (int i=0; i<size_B; i++){
		    	printf("%f ", h_B[i]);
			}
			printf("\n");
			/*// re print for get the same result of matlab
			for (int i=0; i<size_B; i++){
		    	printf("%f ", h_B[i]);
			}
			printf("\n");*/
		} 
	    result = compare_vectors(h_B, d_B, size_B);
	    if (result){
	    	printf("OK\n");
	    }
	    if (export_results){
	    	print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	    	print_double_hexadecimal_values(CPU_FILE, h_B, size_B);
	    }

	}
	if (export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(fft_benck);
	// free object memory 
	free(fft_benck);
	free(A);
	free(d_B);
	free(h_B);
	return 0; 
}


// Arguments part

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-w] [-v] [-e] [-o] [-t] [-c] [-d] [-i input_file_A_MATRIX ] \n", appName);
	printf(" -s Size : set size of furier transform power of 2 \n");
	printf(" -w: window size power of 2 and smaller than size\n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verificaction of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -q: prints input\n");
	printf(" -d: selects GPU\n");
	printf(" -h: print help information\n");
}



int arguments_handler(int argc, char ** argv,int64_t *size,int64_t *window, unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format, bool *print_input, char *input_file_A, char *input_file_B){
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
			// specific
			case 'w' : args +=1; *window = atol(argv[args]);break;
			case 'i' : args +=1;
					   strcpy(input_file_A,argv[args]);
					   args +=1;
					   strcpy(input_file_B,argv[args]); //TODO FIX with final version of input files
					   break;
			case 's' : args +=1; *size = atol(argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( *size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	// specific
	if (*size < *window){
		printf("-w need to samller than size\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	return OK_ARGUMENTS;
}