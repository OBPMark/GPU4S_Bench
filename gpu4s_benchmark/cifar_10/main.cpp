#include <time.h>
#include "benchmark_library.h"
#include "cpu_functions/cpu_functions.h"
#include <sys/time.h>

#define NUMBER_BASE 1
#define MIN_VALUE -0.6
#define MAX_VALUE 0.6
// OUTPUT C is N x W matrix
// Print hexadecimal values of result 

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define CIFAR_10_INPUT 32
#define CIFAR_10_OUTPUT 10
#define KERNEL_CON_1 3
#define KERNEL_CON_2 3
#define STRIDE_1 2
#define STRIDE_2 2
#define DENSE_1 384
#define DENSE_2 10

#define GPU_FILE "gpu_file.out"
#define CPU_FILE "cpu_file.out"

int arguments_handler(int argc, char ** argv, BenchmarkParameters* arguments_parameters);
bench_t RandomNumber();


int main(int argc, char *argv[]){
	// random init
	//srand (time(NULL));
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
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// linearizable versions of matrix
	unsigned int size_matrix =CIFAR_10_INPUT * CIFAR_10_INPUT;
	// A input matrix
	unsigned int size_A = CIFAR_10_INPUT * CIFAR_10_INPUT;
    unsigned int mem_size_A = sizeof(bench_t) * size_A;
	bench_t* input_data = (bench_t*) malloc(mem_size_A);
	// B output matrix
	unsigned int size_B = CIFAR_10_INPUT * CIFAR_10_INPUT;
    unsigned int mem_size_B = sizeof(bench_t) * size_B;
	bench_t* d_output = (bench_t*) malloc(mem_size_B);
	// kernel matrix 1
	unsigned int size_k_1 = KERNEL_CON_1 * KERNEL_CON_2;
    unsigned int mem_size_k_1 = sizeof(bench_t) * size_k_1;
	bench_t* kernel_1 = (bench_t*) malloc(mem_size_k_1);
	// kernel matrix 2
	unsigned int size_k_2 = KERNEL_CON_2 * KERNEL_CON_2;
    unsigned int mem_size_k_2 = sizeof(bench_t) * size_k_2;
	bench_t* kernel_2 = (bench_t*) malloc(mem_size_k_2);
	// weights  1
	unsigned int size_w_1 = DENSE_1 * (((CIFAR_10_INPUT / STRIDE_1)/STRIDE_2)*((CIFAR_10_INPUT / STRIDE_1)/STRIDE_2));
    unsigned int mem_size_w_1 = sizeof(bench_t) * size_w_1;
	bench_t* weights_1 = (bench_t*) malloc(mem_size_w_1);
	// weights  1
	unsigned int size_w_2 = DENSE_1 * DENSE_2;
    unsigned int mem_size_w_2 = sizeof(bench_t) * size_w_2;
	bench_t* weights_2 = (bench_t*) malloc(mem_size_w_2);
	// Outputs 
	const unsigned int size_pooling_1 = CIFAR_10_INPUT / STRIDE_1;
    const unsigned int size_pooling_2 = size_pooling_1 / STRIDE_2;
	bench_t* conv_1_output = (bench_t*) malloc ( CIFAR_10_INPUT * CIFAR_10_INPUT * sizeof(bench_t*));
	bench_t* pooling_1_output = (bench_t*) malloc ( size_pooling_1 * size_pooling_1 * sizeof(bench_t));
	bench_t* conv_2_output = (bench_t*) malloc ( size_pooling_1 * size_pooling_1 * sizeof(bench_t*));
	bench_t* pooling_2_output = (bench_t*) malloc ( size_pooling_2 * size_pooling_2 * sizeof(bench_t));
   	bench_t* dense_layer_1_output = (bench_t*) malloc ( DENSE_1 * sizeof(bench_t));
	bench_t* dense_layer_2_output = (bench_t*) malloc ( DENSE_2 * sizeof(bench_t));
	bench_t* output_data = (bench_t*) malloc ( DENSE_2 * sizeof(bench_t));
	
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
		for (int i=0; i<CIFAR_10_INPUT; i++){
	    	for (int j=0; j<CIFAR_10_INPUT; j++){
	    		#ifdef INT
	        	input_data[i*CIFAR_10_INPUT+j] = rand() % (NUMBER_BASE * 100);

	        	#else
	        	input_data[i*CIFAR_10_INPUT+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
	        	if (arguments_parameters->print_input)
	        	{
	        		printf("%f ", input_data[i*CIFAR_10_INPUT+j]);
	        	}
	        	#endif

	    	}
		}
		if (arguments_parameters->print_input)
	    {
	    	printf("\n");
	    }
	// reseed por compasion reasons
	//srand (time(NULL));
	srand (21121993);
	// inicialice kernel 1
		for (int i=0; i<size_k_1; i++)
		{
			#ifdef INT
	        kernel_1[i] = rand() % (NUMBER_BASE * 100);
	        #else
	        kernel_1[i] = RandomNumber();
	        #endif
		}
	// inicialice kernel 2
		for (int i=0; i<size_k_2; i++)
		{
			#ifdef INT
	        kernel_2[i] = rand() % (NUMBER_BASE * 100);
	        #else
	        kernel_2[i] = RandomNumber();
	        #endif
		}
	// inicialice weights 1
		for (int i=0; i<size_w_1; i++)
		{
			#ifdef INT
	        weights_1[i] = rand() % (NUMBER_BASE * 100);
	        #else
	        weights_1[i] = RandomNumber();
	        #endif
		}
	// inicialice weights 2
		for (int i=0; i<size_w_2; i++)
		{
			#ifdef INT
	        weights_2[i] = rand() % (NUMBER_BASE * 100);
	        #else
	        weights_2[i] = RandomNumber();
	        #endif
		}
	// inicialice output
		for(int i=0; i<size_B; ++i){
			d_output[i] = 0;
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



	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////

	// base object init
	GraficObject *cifar10_bench = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(cifar10_bench, 0,arguments_parameters->gpu, device);
	if (!arguments_parameters->csv_format_timestamp && !arguments_parameters->csv_format && !arguments_parameters->mute_messages ){
		printf("Using device: %s\n", device);
	}
	
	// init memory
	bool mem_result = true;
	mem_result = device_memory_init(cifar10_bench, CIFAR_10_INPUT, CIFAR_10_OUTPUT, KERNEL_CON_1, KERNEL_CON_2, STRIDE_1, STRIDE_2, DENSE_1, DENSE_2);
	if (!mem_result)
	{
		printf("ERROR MEMORY INIT\n");
		exit(-1);
	}
	// copy memory to device
	copy_memory_to_device(cifar10_bench, input_data, kernel_1, kernel_2, weights_1, weights_2, CIFAR_10_INPUT, KERNEL_CON_1, KERNEL_CON_2, size_w_1, size_w_2);
	// execute kernel
	execute_kernel(cifar10_bench, CIFAR_10_INPUT, CIFAR_10_OUTPUT, KERNEL_CON_1, KERNEL_CON_2, STRIDE_1, STRIDE_2, DENSE_1, DENSE_2);
	// copy memory to host
	copy_memory_to_host(cifar10_bench, d_output, CIFAR_10_OUTPUT);

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(cifar10_bench, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}
	if (arguments_parameters->print_output)
	{
		#ifdef INT
		for (int i=0; i<CIFAR_10_OUTPUT; i++){
	    		printf("%d ", d_output[i]);
		}
		printf("\n");
		#else
		for (int i=0; i<CIFAR_10_OUTPUT; i++){
	    		printf("%f ", d_output[i]);
	        	
    		
		}
		printf("\n");
		#endif
	}
	

	if (arguments_parameters->verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		cifar10(output_data, conv_1_output, pooling_1_output, conv_2_output, pooling_2_output, dense_layer_1_output, dense_layer_2_output, input_data, kernel_1, kernel_2, weights_1 , weights_2, CIFAR_10_INPUT, CIFAR_10_OUTPUT, KERNEL_CON_1, KERNEL_CON_2, STRIDE_1,STRIDE_2, DENSE_1, DENSE_2);		
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %lu milliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (arguments_parameters->print_output)
		{
		#ifdef INT
		for (int i=0; i<CIFAR_10_OUTPUT; i++){
	    		printf("%d ", output_data[i]);
		}
		printf("\n");
		#else
		for (int i=0; i<CIFAR_10_OUTPUT; i++){
	    		printf("%f ", output_data[i]);
	        	
    		
		}
		printf("\n");
		#endif
		} 
	    result = compare_vectors(output_data, d_output, CIFAR_10_OUTPUT);
	    if (result){
	    	printf("OK\n");
	    }
	    if (arguments_parameters->export_results){
	    	print_double_hexadecimal_values(GPU_FILE, d_output, CIFAR_10_OUTPUT);
	    	print_double_hexadecimal_values(CPU_FILE, output_data, CIFAR_10_OUTPUT);
	    }

	}


	if (arguments_parameters->export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_output, size_B);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(cifar10_bench);
	// free object memory 
	free(arguments_parameters);
	free(cifar10_bench);
	free(input_data);
	free(d_output);
	free(kernel_1);
	free(kernel_2);
	free(weights_1);
	free(weights_2);
	free(conv_1_output);
	free(pooling_1_output);
	free(conv_2_output);
	free(pooling_2_output);
   	free(dense_layer_1_output);
	free(dense_layer_2_output);
	free(output_data);
return 0;
}


// Arguments part

void print_usage(const char * appName)
{
	printf("Usage: %s [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
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
					   args +=1;
					   strcpy(arguments_parameters->input_file_B,argv[args]);
					   break;
			// specific
			case 'i' : args +=1;
					   strcpy(arguments_parameters->input_file_A,argv[args]);
					   args +=1;
					   strcpy(arguments_parameters->input_file_B,argv[args]);
					   break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if (arguments_parameters->mute_messages){
		arguments_parameters->csv_format = false;
	}
	return OK_ARGUMENTS;
}
bench_t RandomNumber()
{
    return ((bench_t(rand()) / bench_t(RAND_MAX)) * (MAX_VALUE - MIN_VALUE)) + MIN_VALUE;
}