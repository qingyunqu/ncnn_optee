// flatten param defines used by both teec and ta
// *** this file both in teec and ta, you should modify both of them

#ifndef FLATTEN_TEEC_TA_DEFINES_H
#define FLATTEN_TEEC_TA_DEFINES_H

#include "teec_ta_defines.h"

typedef struct {
	Mat_C bottom_blob;
	Mat_C top_blob;
}Flatten_params;

#endif // FLATTEN_TEEC_TA_DEFINES_H
