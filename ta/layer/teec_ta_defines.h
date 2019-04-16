// universal param used by both teec and ta
// *** this file both in teec and ta, you should modify both of them
#ifndef TEEC_TA_DEFINES_H
#define TEEC_TA_DEFINES_H

#include <stddef.h>

#include <arm_neon.h>

typedef struct {
	size_t elemsize;
	int dims;
	
	int w;
	int h;
	int c;
	
	size_t cstep;
} Mat_C;

// see init_mat_c_from_mat() in "teec.h"

float* row(void* data, Mat_C* mat, int y);  // only support float32
float* channel(void* data, Mat_C* mat, int c); // only support float32


#endif // TEEC_TA_DEFINES_H
