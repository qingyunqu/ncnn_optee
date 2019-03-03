// softmax param defines used by both teec and ta
// *** this file both in teec and ta, you should modify both of them

#ifndef SOFTMAX_TEEC_TA_DEFINES_H
#define SOFTMAX_TEEC_TA_DEFINES_H

#include "teec_ta_defines.h"

typedef struct {
	Mat_C bottom_top_blob;
	int axis;
}Softmax_params;

#endif // SOFTMAX_TEEC_TA_DEFINES_H
