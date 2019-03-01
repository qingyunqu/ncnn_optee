// pooling param defines used by both teec and ta
// *** this file both in teec and ta, you should modify both of them

#ifndef POOLING_TEEC_TA_DEFINES_H
#define POOLING_TEEC_TA_DEFINES_H

#ifdef __cplusplus
extern "C"{
#endif

#include "teec_ta_defines.h"

typedef struct {
	Mat_C bottom_blob;
	Mat_C top_blob;
	int pooling_type; // enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };
	int kernel_w;
	int kernel_h;
	int stride_w;
	int stride_h;
	int pad_left;
	int pad_right;
	int pad_top;
	int pad_bottom;
	int global_pooling;
	int pad_mode; // 0=full 1=valid 2=SAME
}Pooling_params;

#ifdef __cplusplus
}
#endif

#endif // POOLING_TEEX_TA_DEFINES_H
