#include "teec_ta_defines.h"

float* row(void* data, Mat_C* mat, int y){  // only support float32
	return (float*)data + mat->w * y;
}
float* channel(void* data, Mat_C* mat, int c){ // only support float32
	return (float*)data + mat->cstep * c;
}
