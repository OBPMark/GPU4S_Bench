
#ifdef FLOAT
std::string atomic_code = 
"void atomic_add_global(volatile global float *source, const float operand) {\n"
"union {\n"
"unsigned int intVal;\n"
"float floatVal;\n"
"} newVal;\n"
"union {\n"
"unsigned int intVal;\n"
"float floatVal;\n"
"} prevVal;\n"
"do {\n"
"prevVal.floatVal = *source;\n"
"newVal.floatVal = prevVal.floatVal + operand;\n"
"} while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);\n"
"}\n"
;
#else
std::string string atomic_code = 
"#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n"
"void atomic_add_global(volatile global double *source, const double operand) {\n"
"union {\n"
"unsigned long int intVal;\n"
"double floatVal;\n"
"} newVal;\n"
"union {\n"
"unsigned long int intVal;\n"
"double floatVal;\n"
"} prevVal;\n"
"do {\n"
"prevVal.floatVal = *source;\n"
"newVal.floatVal = prevVal.floatVal + operand;\n"
"} while (atomic_cmpxchg((volatile global unsigned long int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);\n"
"}\n"
;
#endif
