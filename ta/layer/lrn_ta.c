#include "layer_registered.h"
#include "lrn_teec_ta_defines.h"
#include "math.h" // my implementation

#include <stdio.h>

TEE_Result lrn_ta(uint32_t param_types, TEE_Param params[4])
{
	dprintf("lrn_ta\n");
	/**
	  * params[0]: void* bottom_top_blob.data
	  * params[1]: LRN_params* lrnp;
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
	LRN_params* lrnp = (LRN_params*)params[1].memref.buffer;
	Mat_C* btb = &lrnp->bottom_top_blob;
	float* bottom_top_blob = (float*)params[0].memref.buffer;

	int w = btb->w;
	int h = btb->h;
	int channels = btb->c;
	size_t elemsize = btb->elemsize;
	int size = w * h;
	
	
	
	dprintf("softmax_ta failed\n");
	return TEE_ERROR_BAD_PARAMETERS;
}
