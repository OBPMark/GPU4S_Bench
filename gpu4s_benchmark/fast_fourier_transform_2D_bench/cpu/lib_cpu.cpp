#include "lib_cpu.h"
#include <complex.h> 
#include <fftw3.h>

bool FFT2D(COMPLEX **c,int nx,int ny,COMPLEX **out)
{
  fftw_plan plan;

  fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
  fftw_complex *out_data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*nx*ny);
  for (int i=0; i<nx; ++i)
      {
          for (int j=0; j<nx; ++j)
          {
                  in[j+i*nx][0] = c[i][j].x ;
                  in[j+i*nx][1] = c[i][j].y;
          }
      }

  plan =  fftw_plan_dft_2d(nx, nx, in, out_data, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  for (int i=0; i<nx; ++i)
      {
          for (int j=0; j<nx; ++j)
          {
                  out[i][j].x = out_data[j+i*nx][0];
                  out[i][j].y = out_data[j+i*nx][1];
          }
      }

  return true;
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

bool compare_vectors( COMPLEX **host, COMPLEX **device, const int64_t size){
   for (unsigned int i=0; i<size; ++i)
      {
         for (unsigned int j=0; j<size; ++j)
         {
               if (fabs(host[i][j].x - device[i][j].x) > 1e-4){
                  printf("Error in element %d %d is %f but was %f\n", i, j,device[i][j].x, host[i][j].x);
                  return false;
               }
                if (fabs(host[i][j].y - device[i][j].y) > 1e-4){
                  printf("Error in element %d %d is %f but was %f\n", i, j,device[i][j].y, host[i][j].y);
                  return false;
               }
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
