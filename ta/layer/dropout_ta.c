#include "layer_registered.h"
#include "dropout_teec_ta_defines.h"

#include <stdio.h>

TEE_Result dropout_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("dropout_ta\n");
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: Dropout_params* dp;
	  * params[2]: NONE
	  * params[3]: NONE
	  */
	const uint32_t exp_param_types = TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
													TEE_PARAM_TYPE_MEMREF_INPUT,
													TEE_PARAM_TYPE_NONE,
													TEE_PARAM_TYPE_NONE);
	if(param_types != exp_param_types){
		printf("error params!\n");
		return TEE_ERROR_BAD_PARAMETERS;
	}
	Dropout_params* dp = (Dropout_params*)params[1].memref.buffer;
	Mat_C* btb = &dp->bottom_top_blob;
	float* bottom_top_blob = (float*)params[0].memref.buffer;
	
	int w = btb->w;
	int h = btb->h;
	int channels = btb->c;
	int size = w * h;
	float scale = dp->scale;
	
	//#pragma omp parallel for num_threads(xxx)
	for(int q=0; q<channels; q++){
		float* ptr = channel(bottom_top_blob,btb,q);
		for(int i=0; i<size; i++){
			ptr[i] = ptr[i] * scale;
		}
	}
	
	dprintf("dropout_ta success\n");
	return TEE_SUCCESS;
}
