// LRN param defines used by both teec and ta
// *** this file both in teec and ta, you should modify both of them

#ifndef LRN_TEEC_TA_DEFINES_H
#define LRN_TEEC_TA_DEFINES_H

#include "teec_ta_defines.h"

typedef struct {
	Mat_C bottom_top_blob;
	int region_type; // enum { NormRegion_ACROSS_CHANNELS = 0, NormRegion_WITHIN_CHANNEL = 1 };
	int local_size;
	float alpha;
	float beta;
	float bias;
}LRN_params;

enum { NormRegion_ACROSS_CHANNELS = 0, NormRegion_WITHIN_CHANNEL = 1 };

#endif // LRN_TEEC_TA_DEFIINES_H
