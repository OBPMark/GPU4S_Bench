#include "lib_cpu.h"

void aux_fft_function(bench_t* data, int64_t nn, int64_t start_pos){
    int64_t loop_w = 0, loop_for_1 = 0, loop_for_2 = 0; 
    int64_t n, mmax, m, j, istep, i , window = nn;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
    // reverse-binary reindexing for all data 
    nn = nn>>1;

    n = nn<<1;
    //printf(" nn %ld n %ld window %ld start_pos %ld,\n",nn, n, window, start_pos);
    j=1;
    for (i=1; i<n; i+=2) {
        if (j>i) {
            std::swap(data[(start_pos * window) + (j-1)], data[(start_pos * window) + (i-1)]);
            std::swap(data[(start_pos * window) + j], data[(start_pos * window) + i]);
            //printf("i %lu j %lu data %f \n",i ,j, data[(start_pos * window) + (j-1)] );
        }
        m = nn;
        while (m>=2 && j>m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    };
    
    // here begins the Danielson-Lanczos section for each window
    mmax=2;
    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
                j=i+mmax;
                tempr = wr*data[(start_pos * window) + j-1] - wi*data[(start_pos * window) +j];
                tempi = wr * data[(start_pos * window) + j] + wi*data[(start_pos * window) + j-1];
                
                data[(start_pos * window) + j-1] = data[(start_pos * window) + i-1] - tempr;
                data[(start_pos * window) +j] = data[(start_pos * window) + i] - tempi;
                data[(start_pos * window) + i-1] += tempr;
                data[(start_pos * window) +i] += tempi;
                ++loop_for_1;
                //printf("wr %f wi %f\n", wr, wi);
            }
            loop_for_1 = 0;
            
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            ++loop_for_2;

        }
        loop_for_2 = 0;
        mmax=istep;
    ++loop_w;    
    }
}


void fft_function(bench_t* data ,bench_t* output,const int64_t window,const int64_t nn){
    // do for all window
    for (unsigned int i = 0; i < (nn * 2 - window + 1); i+=2){
        // copy values of the  window to output
        for(unsigned int j = 0; j < window ; ++j){
            output[i * window + j] = data[i+j];
        }
        aux_fft_function(output, window, i);
    }
    
	
	
}

bool compare_vectors(const bench_t* host,const bench_t* device, const int64_t size){
		for (int i = 0; i < size; ++i){
			if (fabs(host[i] - device[i]) > 1e-4){
				printf("Error in element %d is %f but was %f\n", i,device[i], host[i]);
				return false;
			}
		}
		return true;
}

void print_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size){
	FILE *output_file = fopen(filename, "w");
  	// file created
  	for (unsigned int i = 0; i < size; ++i){
  		binary_float.f = float_vector[i];
		fprintf(output_file, "%02x", binary_float.binary_values.a );
		fprintf(output_file, "%02x", binary_float.binary_values.b );
		fprintf(output_file, "%02x", binary_float.binary_values.c );
		fprintf(output_file, "%02x", binary_float.binary_values.d );
		fprintf(output_file, "%02x", binary_float.binary_values.e );
		fprintf(output_file, "%02x", binary_float.binary_values.f );
		fprintf(output_file, "%02x", binary_float.binary_values.g );
		fprintf(output_file, "%02x", binary_float.binary_values.h );
		fprintf(output_file, "\n"); 
  	}
  	fclose(output_file);	

}

void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size){
	// open file
	FILE *file = fopen(filename, "r");
	// read line by line
	char * line = NULL;
    size_t len = 0;
    

	for (unsigned int i = 0; i < size; ++i){
		getline(&line, &len, file);
		// delete /n
		line[strlen(line)-1] = 0;
		// strip for each char
		char *temp = (char*) malloc(sizeof(char) * 2);
		char *ptr;
    	temp[0] = line[0];
		temp[1] = line[1];
    	binary_float.binary_values.a = (char)strtol(temp, &ptr, 16);
		temp[0] = line[2];
		temp[1] = line[3];
		binary_float.binary_values.b = (char)strtol(temp, &ptr, 16);
		temp[0] = line[4];
		temp[1] = line[5];
		binary_float.binary_values.c = (char)strtol(temp, &ptr, 16);
		temp[0] = line[6];
		temp[1] = line[7];
		binary_float.binary_values.d = (char)strtol(temp, &ptr, 16);
		temp[0] = line[8];
		temp[1] = line[9];
		binary_float.binary_values.e = (char)strtol(temp, &ptr, 16);
		temp[0] = line[10];
		temp[1] = line[11];
		binary_float.binary_values.f = (char)strtol(temp, &ptr, 16);
		temp[0] = line[12];
		temp[1] = line[13];
		binary_float.binary_values.g = (char)strtol(temp, &ptr, 16);
		temp[0] = line[14];
		temp[1] = line[15];
		binary_float.binary_values.h = (char)strtol(temp, &ptr, 16);

		float_vector[i] = binary_float.f;
	}
  	fclose(file);	

}
