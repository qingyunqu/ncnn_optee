// scale param defines used by both teec and ta
// *** this file both in teec and ta, you should modify both of them

#ifndef SCALE_TEEC_TA_DEFINES_H
#define SCALE_TEEC_TA_DEFINES_H


#include "teec_ta_defines.h"

typedef struct {
	Mat_C bottom_top_blob;
	Mat_C scale_blob;
	Mat_C bias_data;
	int scale_data_size;
	int bias_term;
}Scale_params;


#endif // SCALE_TEEC_TA_DEFINES_H
