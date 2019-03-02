// relu param defines used by both teec and ta
// *** this file both in teec and ta, you should modify both of them

#ifndef RELU_TEEC_TA_DEFINES_H
#define RELU_TEEC_TA_DEFINES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "teec_ta_defines.h"

typedef struct {
	Mat_C bottom_top_blob;
	float slope;
}ReLU_params;

#ifdef __cplusplus
}
#endif

#endif // RELU_TEEC_TA_DEFINES_H
