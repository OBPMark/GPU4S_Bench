#ifndef SHARED_LIB_H
#define SHARED_LIB_H

#ifdef FLOAT
typedef float bench_t;
#elif DOUBLE 
typedef double bench_t;
#endif
struct COMPLEX{
	bench_t x;
	bench_t y;
};
#endif